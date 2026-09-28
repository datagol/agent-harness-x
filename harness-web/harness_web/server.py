"""Local-only application server. API keys stay in the server environment."""

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import re
import shlex
from typing import Literal
from urllib.parse import quote, urlsplit
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from harnessx import PermissionLevel

from .catalog import (
    CATALOG,
    PROVIDERS,
    ROOT,
    arguments,
    provider_missing,
    public_catalog,
    requirements,
)
from .chat import DOWNLOADS_DIR, Chat, close_chat, create_chat, rebuild_chat, run_chat, set_tool_enabled
from .files import open_workspace_file
from .runs import RunManager, run_example


class ExampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    example_id: str
    prompt: str = Field(default="", max_length=32000)
    mode: Literal["offline", "live"] = "offline"
    provider: Literal["anthropic", "openai", "gemini", "openrouter"] = "anthropic"
    model: str = Field(default="", max_length=200)
    streaming: bool = True
    server: str = Field(default="", max_length=200)
    command: str = Field(default="", max_length=2000)
    args: str = Field(default="", max_length=8000)
    url: str = Field(default="", max_length=2000)


class InputRequest(BaseModel):
    prompt_id: str
    value: str = Field(max_length=64000)


class ChatRequest(BaseModel):
    provider: Literal["anthropic", "openai", "gemini", "openrouter", "demo"] = (
        "anthropic"
    )
    model: str = Field(default="", max_length=200)


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=32000)


class ChatSetupRequest(BaseModel):
    system_prompt: str = Field(min_length=1, max_length=20000)


class SkillUploadRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=262144)


class MCPConnectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    command: str = Field(default="", max_length=1000)
    args: str = Field(default="", max_length=8000)
    url: str = Field(default="", max_length=2000)
    permission: Literal["ask", "allow", "deny"] = "ask"


def create_app(*, data_dir=None, load_env=True, chat_factory=create_chat):
    if load_env:
        load_dotenv(ROOT / ".env")
    directory = Path(
        data_dir or os.environ.get("HARNESS_WEB_DATA_DIR", ROOT / ".harness-web")
    ).resolve()
    manager = RunManager(directory / "runs")
    chats: dict[str, Chat] = {}

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await manager.close()
            await asyncio.gather(
                *(close_chat(chat) for chat in chats.values()),
                return_exceptions=True,
            )

    app = FastAPI(title="harness-web", lifespan=lifespan)
    app.state.manager = manager
    app.state.chats = chats
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def local_origin(request, call_next):
        # Local tools must not be triggered by an unrelated website.
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).hostname not in (
            "127.0.0.1",
            "localhost",
            "::1",
            "testserver",
        ):
            return JSONResponse(
                {"detail": "This local application only accepts local origins"},
                status_code=403,
            )
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse(
                {"detail": "Cross-site requests are not allowed"}, status_code=403
            )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def get_run(run_id):
        if run_id not in manager.runs:
            raise HTTPException(
                404, "Run not found; history lasts until the server restarts"
            )
        return manager.runs[run_id]

    def get_chat(chat_id):
        if chat_id not in chats:
            raise HTTPException(404, "Conversation not found")
        return chats[chat_id]

    @app.get("/api/health")
    async def health():
        return {
            "app": "harness-web",
            "status": "ready",
            "providers": [
                {
                    "id": key,
                    "name": value[0],
                    "missing": provider_missing(key),
                    "model": os.getenv("AGENT_MODEL", value[3])
                    if os.getenv("AGENT_PROVIDER", "anthropic") == key
                    else value[3],
                }
                for key, value in PROVIDERS.items()
            ]
            + [{"id": "demo", "name": "Local demo", "missing": [], "model": "demo"}],
        }

    @app.get("/api/examples")
    async def examples():
        return public_catalog()

    @app.post("/api/runs", status_code=201)
    async def start_example(body: ExampleRequest):
        example = CATALOG.get(body.example_id)
        if not example:
            raise HTTPException(404, "Unknown example")
        missing = requirements(example, body.mode, provider=body.provider)
        if missing:
            raise HTTPException(422, "Configure before running: " + ", ".join(missing))
        try:
            # Validate all configuration before creating a run.
            config = body.model_dump()
            arguments(example, config, directory)
            run = manager.create(example.title, example.id)
            run.mode = (
                "offline"
                if example.offline
                and not (example.id == "run_evals" and body.mode == "live")
                else "live"
            )
            args = arguments(example, config, run.workdir)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        run.task = asyncio.create_task(run_example(run, example, args, body.prompt))
        return run.public()

    @app.get("/api/runs")
    async def runs():
        return [run.public() for run in reversed(list(manager.runs.values()))]

    @app.get("/api/runs/{run_id}")
    async def run_status(run_id: str):
        return get_run(run_id).public()

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request, after: int = 0):
        run = get_run(run_id)
        try:
            cursor = max(0, after, int(request.headers.get("last-event-id", "0")))
        except ValueError as exc:
            raise HTTPException(422, "Invalid event cursor") from exc

        async def generate():
            async for event in run.subscribe(cursor):
                if await request.is_disconnected():
                    break
                if event["type"] == "heartbeat":
                    yield ": heartbeat\n\n"
                else:
                    yield f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{run_id}/input")
    async def send_input(run_id: str, body: InputRequest):
        try:
            await get_run(run_id).reply(body.prompt_id, body.value)
        except (ValueError, BrokenPipeError, ConnectionResetError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return get_run(run_id).public()

    @app.post("/api/runs/{run_id}/stop")
    async def stop_run(run_id: str):
        run = get_run(run_id)
        await run.stop()
        return run.public()

    def files_for(run):
        # Downloads are limited to ordinary, non-hidden files in the run workspace.
        paths = []
        for folder, dirs, files in os.walk(run.workdir, followlinks=False):
            dirs[:] = [
                name
                for name in dirs
                if not name.startswith(".")
                and name != "__pycache__"
                and not (Path(folder) / name).is_symlink()
            ]
            for name in files:
                path = Path(folder) / name
                if (
                    not name.startswith(".")
                    and not path.is_symlink()
                    and path.is_file()
                ):
                    paths.append(path)
                if len(paths) >= 200:
                    return paths
        return paths

    @app.get("/api/runs/{run_id}/files")
    async def files(run_id: str):
        run = get_run(run_id)
        return [
            {"name": str(path.relative_to(run.workdir)), "size": path.stat().st_size}
            for path in files_for(run)
        ]

    @app.get("/api/runs/{run_id}/files/{name:path}")
    async def download(run_id: str, name: str):
        run = get_run(run_id)
        try:
            stream = open_workspace_file(run.workdir, name)
        except (OSError, ValueError) as exc:
            raise HTTPException(404, "File not found") from exc

        async def chunks():
            try:
                while data := await asyncio.to_thread(stream.read, 65536):
                    yield data
            finally:
                stream.close()

        return StreamingResponse(
            chunks(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename*=UTF-8''"
                + quote(name.split("/")[-1], safe=""),
            },
        )

    @app.post("/api/chats", status_code=201)
    async def new_chat(body: ChatRequest):
        missing = provider_missing(body.provider)
        if missing:
            raise HTTPException(422, "Configure before chatting: " + ", ".join(missing))
        if len(chats) >= 50:
            raise HTTPException(
                409, "Delete a conversation before creating another (limit: 50)"
            )
        model = (
            body.model.strip()
            or (PROVIDERS.get(body.provider, ("", "", "", "demo"))[3])
        )
        if not model:
            raise HTTPException(422, "Enter a model ID for this provider")
        workspace = directory / "chats" / uuid.uuid4().hex
        workspace.mkdir(parents=True, exist_ok=True)
        chat = chat_factory(body.provider, model, workspace)
        chats[chat.id] = chat
        return chat.public()

    @app.get("/api/chats")
    async def list_chats():
        return [
            chat.public(include_messages=False)
            for chat in reversed(list(chats.values()))
        ]

    @app.get("/api/chats/{chat_id}")
    async def conversation(chat_id: str):
        return get_chat(chat_id).public()

    def ensure_chat_idle(chat: Chat) -> None:
        if chat.active and chat.active.task and not chat.active.task.done():
            raise HTTPException(409, "Stop the active response before changing conversation setup")

    @app.patch("/api/chats/{chat_id}/setup")
    async def update_chat_setup(chat_id: str, body: ChatSetupRequest):
        chat = get_chat(chat_id)
        ensure_chat_idle(chat)
        system_prompt = body.system_prompt.strip()
        if not system_prompt:
            raise HTTPException(422, "System prompt cannot be empty")
        previous = chat.system_prompt
        chat.system_prompt = system_prompt
        try:
            await rebuild_chat(chat)
        except Exception as exc:
            chat.system_prompt = previous
            raise HTTPException(422, f"Could not apply system prompt: {exc}") from exc
        return chat.public()

    @app.post("/api/chats/{chat_id}/skills", status_code=201)
    async def upload_skill(chat_id: str, body: SkillUploadRequest):
        chat = get_chat(chat_id)
        ensure_chat_idle(chat)
        filename = Path(body.name).name
        if filename != body.name or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,123}\.md", filename
        ):
            raise HTTPException(422, "Skill file names must end in .md and use letters, numbers, ., _, or -")
        target = chat.directory / "skills" / filename
        if target.exists():
            raise HTTPException(409, "A skill with that file name already exists")
        target.write_text(body.content, encoding="utf-8")
        try:
            await rebuild_chat(chat)
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(422, f"Could not load skill: {exc}") from exc
        return chat.public()

    @app.delete("/api/chats/{chat_id}/skills/{filename}")
    async def delete_skill(chat_id: str, filename: str):
        chat = get_chat(chat_id)
        ensure_chat_idle(chat)
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name.endswith(".md"):
            raise HTTPException(404, "Skill not found")
        target = chat.directory / "skills" / safe_name
        if not target.is_file() or target.is_symlink():
            raise HTTPException(404, "Skill not found")
        original = target.read_text(encoding="utf-8")
        target.unlink()
        try:
            await rebuild_chat(chat)
        except Exception as exc:
            target.write_text(original, encoding="utf-8")
            raise HTTPException(422, f"Could not remove skill: {exc}") from exc
        return chat.public()

    class ToolSwitchRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str = Field(min_length=1, max_length=200)
        enabled: bool

    @app.patch("/api/chats/{chat_id}/tools")
    async def switch_chat_tool(chat_id: str, body: ToolSwitchRequest):
        """Enable or disable one tool for this conversation's next turns."""
        chat = get_chat(chat_id)
        ensure_chat_idle(chat)
        try:
            set_tool_enabled(chat, body.name, body.enabled)
        except KeyError:
            raise HTTPException(404, "Unknown tool") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            await rebuild_chat(chat)
        except Exception as exc:
            raise HTTPException(422, f"Could not apply tool change: {exc}") from exc
        return chat.public()

    @app.get("/api/chats/{chat_id}/downloads/{file_id}/{name}")
    async def chat_download(chat_id: str, file_id: str, name: str):
        """Serve a file that generate_file published for this conversation."""
        chat = get_chat(chat_id)
        if not re.fullmatch(r"[0-9a-f]{6,32}", file_id):
            raise HTTPException(404, "File not found")
        try:
            stream = open_workspace_file(chat.directory / DOWNLOADS_DIR, f"{file_id}/{name}")
        except (OSError, ValueError) as exc:
            raise HTTPException(404, "File not found") from exc

        async def chunks():
            try:
                while data := await asyncio.to_thread(stream.read, 65536):
                    yield data
            finally:
                stream.close()

        return StreamingResponse(
            chunks(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename*=UTF-8''" + quote(name, safe="")},
        )

    @app.post("/api/chats/{chat_id}/mcp", status_code=201)
    async def connect_chat_mcp(chat_id: str, body: MCPConnectRequest):
        chat = get_chat(chat_id)
        ensure_chat_idle(chat)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", body.name):
            raise HTTPException(422, "MCP server names use letters, numbers, _ or -")
        if body.name in chat.mcp.list_servers():
            raise HTTPException(409, "An MCP server with that name is already connected")
        command, url = body.command.strip(), body.url.strip()
        if bool(command) == bool(url):
            raise HTTPException(422, "Provide exactly one MCP command or HTTP URL")
        if url:
            parsed = urlsplit(url)
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                raise HTTPException(422, "MCP URLs must be a plain http or https URL")
        try:
            args = shlex.split(body.args)
        except ValueError as exc:
            raise HTTPException(422, f"Invalid MCP arguments: {exc}") from exc
        if len(args) > 64 or any(len(arg) > 1000 for arg in args):
            raise HTTPException(422, "MCP arguments are too large")
        try:
            tools = await chat.mcp.connect(
                body.name,
                command=command or None,
                args=args,
                url=url or None,
                permission=PermissionLevel(body.permission),
            )
            await rebuild_chat(chat)
        except Exception as exc:
            await chat.mcp.disconnect(body.name)
            raise HTTPException(422, f"Could not connect MCP server: {exc}") from exc
        return {
            **chat.public(),  # includes "tools", every tool the agent can now call
            "connected": body.name,
            "discovered": [
                {"name": tool.tool_name, "description": tool.description}
                for tool in tools
            ],
        }

    @app.delete("/api/chats/{chat_id}/mcp/{name}")
    async def disconnect_chat_mcp(chat_id: str, name: str):
        chat = get_chat(chat_id)
        ensure_chat_idle(chat)
        if name not in chat.mcp.list_servers():
            raise HTTPException(404, "MCP server not found")
        await chat.mcp.disconnect(name)
        try:
            await rebuild_chat(chat)
        except Exception as exc:
            raise HTTPException(422, f"Could not apply MCP change: {exc}") from exc
        return chat.public()

    @app.post("/api/chats/{chat_id}/messages", status_code=201)
    async def chat_message(chat_id: str, body: MessageRequest):
        chat = get_chat(chat_id)
        if not body.message.strip():
            raise HTTPException(422, "Enter a message")
        if chat.active and chat.active.task and not chat.active.task.done():
            raise HTTPException(409, "This conversation already has a turn in progress")
        try:
            run = manager.create(chat.title, "chat", workdir=chat.directory)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        run.chat_id = chat.id
        run.mode = "offline" if chat.provider == "demo" else "live"
        chat.active = run
        run.task = asyncio.create_task(run_chat(chat, run, body.message.strip()))
        return run.public()

    @app.delete("/api/chats/{chat_id}")
    async def delete_chat(chat_id: str):
        chat = get_chat(chat_id)
        await close_chat(chat)
        del chats[chat_id]
        return {"deleted": chat_id}

    dist = ROOT / "harness-web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
