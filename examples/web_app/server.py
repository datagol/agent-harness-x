"""FastAPI wrapper for the agent harness.

Exposes the agent as an HTTP API with streaming SSE support.

Run from the project root:
    pip install -e '.[server]'
    uvicorn examples.web_app.server:app --host 127.0.0.1 --reload --port 8000

Requires ANTHROPIC_API_KEY. This is a trusted local, single-conversation demo.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from harnessx import (
    Limits,
    AgentConfig,
    MCPManager,
    PermissionLevel,
    SkillManager,
    Agent,
    RunEventType,
    ToolCall,
    ToolResult,
    PermissionManager,
    ToolRegistry,
)
from harnessx.builtin import register_all_tools

HERE = os.path.dirname(__file__)
STATIC_DIR = os.path.join(HERE, "static")
OUTPUT_DIR = os.environ.get("AGENT_OUTPUT_DIR", os.path.join(os.getcwd(), "output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _resolve_skill_paths() -> list[str]:
    """Discover skills via AGENT_SKILLS env var, or fall back to a default dir.

    AGENT_SKILLS is a colon-separated list of paths (files or folders). Each
    entry is either a SKILL.md file, a folder containing SKILL.md, or a folder
    of skill folders (auto-expanded one level).
    """
    raw = os.environ.get("AGENT_SKILLS", "")
    if raw:
        entries = [p.strip() for p in raw.split(":") if p.strip()]
    else:
        # Default: a sibling examples/skills/ next to the repo
        default = os.path.join(os.getcwd(), "examples", "skills")
        entries = [default] if os.path.isdir(default) else []

    resolved: list[str] = []
    for entry in entries:
        if not os.path.exists(entry):
            print(f"Warning: skill path does not exist: {entry}")
            continue
        if os.path.isfile(entry):
            resolved.append(entry)
            continue
        # Folder: SKILL.md directly inside, or one-level expand into subfolders
        if os.path.isfile(os.path.join(entry, "SKILL.md")):
            resolved.append(entry)
        else:
            for child in sorted(os.listdir(entry)):
                child_path = os.path.join(entry, child)
                if os.path.isfile(child_path) and child.endswith(".md"):
                    resolved.append(child_path)
                elif os.path.isdir(child_path) and os.path.isfile(
                    os.path.join(child_path, "SKILL.md")
                ):
                    resolved.append(child_path)
    return resolved


# ── Global state ─────────────────────────────────────────────────────────

streaming_agent: Agent | None = None
mcp_manager: MCPManager | None = None
agent_lock = asyncio.Lock()


def agent_bindings() -> dict[str, Any]:
    """Rebind tools and skills for both new agents and restored snapshots."""
    tools = ToolRegistry()
    register_all_tools(tools, output_dir=OUTPUT_DIR)
    permissions = PermissionManager()
    for name in [
        "read_file",
        "list_directory",
        "run_bash",
        "fetch_url",
        "write_file",
        "generate_file",
    ]:
        permissions.set_permission(name, PermissionLevel.ALLOW)
    if mcp_manager and mcp_manager.tool_count:
        mcp_manager.register_tools(tools, permission=PermissionLevel.ALLOW)
    paths = _resolve_skill_paths()
    return {
        "tools": tools,
        "permissions": permissions,
        "skills": SkillManager.from_paths([*paths]) if paths else None,
    }


def create_streaming_agent() -> Agent:
    return Agent(
        config=AgentConfig(
            system_prompt=(
                "You are a helpful assistant with access to tools. "
                "When a request matches an available skill, load it first. "
                "Be concise and direct."
            ),
            limits=Limits(max_iterations=25),
        ),
        **agent_bindings(),
    )


@asynccontextmanager
async def use_agent():
    # This demo shares one conversation. Serialize runs and session/tool changes.
    async with agent_lock:
        if streaming_agent is None:
            raise HTTPException(status_code=503, detail="Agent is not running")
        yield streaming_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    global streaming_agent, mcp_manager
    mcp_manager = MCPManager()

    try:
        # Connect to MCP servers from environment
        mcp_config = os.environ.get("MCP_SERVERS")
        if mcp_config:
            try:
                servers = json.loads(mcp_config)
                await mcp_manager.connect_from_config(servers)
            except Exception as e:
                print(f"Warning: Failed to connect MCP servers: {e}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        streaming_agent = create_streaming_agent()
        yield
    finally:
        try:
            if streaming_agent is not None:
                await streaming_agent.aclose()
        finally:
            streaming_agent = None
            await mcp_manager.disconnect_all()
            mcp_manager = None


# The UI is served from this same origin, so no CORS middleware is installed:
# this server has no authentication and exposes shell and file tools, and a
# wildcard origin would let any open web page drive it.
app = FastAPI(title="Agent Harness API", lifespan=lifespan)


# ── Request/Response models ──────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    response: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class MCPConnectRequest(BaseModel):
    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str


# ── API routes ───────────────────────────────────────────────────────────


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    async with use_agent() as agent:
        if req.system_prompt is not None:
            agent.config.system_prompt = req.system_prompt
        calls: dict[str, dict[str, Any]] = {}
        async with agent.run_stream(req.message) as events:
            async for event in events:
                if event.type == RunEventType.TOOL_CALL_START and isinstance(
                    event.data, ToolCall
                ):
                    call = event.data
                    calls[call.id] = {
                        "id": call.id,
                        "name": call.name,
                        "input": call.input,
                        "status": "running",
                    }
                elif event.type == RunEventType.TOOL_RESULT and isinstance(
                    event.data, ToolResult
                ):
                    result = event.data
                    if result.tool_call_id in calls:
                        calls[result.tool_call_id].update(
                            status="error" if result.is_error else "done",
                            result=result.content,
                        )
            result = await events.result()
        if result.status != "completed":
            raise HTTPException(
                status_code=502, detail=result.error or {"status": result.status.value}
            )
        return ChatResponse(
            response=result.output,
            tool_calls=list(calls.values()),
            usage=agent.guardrails.usage_summary,
        )


@app.post("/api/stream")
async def stream(req: ChatRequest):
    async def event_generator():
        async with use_agent() as agent:
            if req.system_prompt is not None:
                agent.config.system_prompt = req.system_prompt
            async with agent.run_stream(req.message) as events:
                async for event in events:
                    payload = None
                    if event.type in (
                        RunEventType.TEXT_DELTA,
                        RunEventType.TEXT_COMPLETE,
                    ):
                        payload = {"type": event.type.value, "content": event.data}
                    elif event.type == RunEventType.ATTEMPT_RESET:
                        payload = {"type": "attempt_reset"}
                    elif event.type == RunEventType.TOOL_CALL_START and isinstance(
                        event.data, ToolCall
                    ):
                        call = event.data
                        payload = {
                            "type": "tool_call_start",
                            "name": call.name,
                            "input": call.input,
                            "id": call.id,
                        }
                    elif event.type == RunEventType.TOOL_RESULT and isinstance(
                        event.data, ToolResult
                    ):
                        result = event.data
                        payload = {
                            "type": "tool_result",
                            "tool_call_id": result.tool_call_id,
                            "content": result.content,
                            "is_error": result.is_error,
                        }
                    elif event.type == RunEventType.TURN_COMPLETE:
                        payload = {
                            "type": "turn_complete",
                            "usage": agent.guardrails.usage_summary,
                        }
                    elif event.type == RunEventType.ERROR:
                        payload = {"type": "error", "content": str(event.data)}
                    if payload is not None:
                        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/status")
async def status():
    """Get agent status, tools, and MCP servers."""
    sa = streaming_agent
    tools = sa.tools.list_tools() if sa else []
    mcp_servers = mcp_manager.list_servers() if mcp_manager else {}
    mcp_tools = [
        {
            "server": t.server_name,
            "name": t.tool_name,
            "description": t.description[:100],
        }
        for t in (mcp_manager.list_tools() if mcp_manager else [])
    ]

    skills = (
        [{"name": s.name, "description": s.description} for s in sa.skills.list()]
        if sa and sa.skills
        else []
    )

    return {
        "status": "running" if sa else "stopped",
        "model": sa.config.model if sa else "",
        "system_prompt": sa.config.system_prompt[:200] if sa else "",
        "session_id": sa.session_id if sa else "",
        "tools": tools,
        "mcp_servers": mcp_servers,
        "mcp_tools": mcp_tools,
        "skills": skills,
        "usage": sa.guardrails.usage_summary if sa else {},
    }


@app.get("/api/skills")
async def list_skills():
    """List all skills available to the agent."""
    sa = streaming_agent
    if not sa or not sa.skills:
        return {"skills": []}
    return {
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "source_path": str(s.source_path),
                "body_chars": len(s.body),
            }
            for s in sa.skills.list()
        ]
    }


@app.post("/api/mcp/connect")
async def mcp_connect(req: MCPConnectRequest):
    """Connect to an MCP server at runtime."""
    global streaming_agent
    async with use_agent() as previous:
        assert mcp_manager is not None
        try:
            tools = await mcp_manager.connect(
                req.name,
                command=req.command,
                args=req.args,
                url=req.url,
                headers=req.headers,
                permission=PermissionLevel.ALLOW,
            )
        except Exception as e:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=400,
                content={
                    "error": str(e),
                    "detail": f"Failed to connect to MCP server '{req.name}'",
                },
            )
        # Re-create agent with new tools
        streaming_agent = create_streaming_agent()
        await previous.aclose()
        return {
            "server": req.name,
            "tools": [
                {"name": t.tool_name, "description": t.description[:100]} for t in tools
            ],
        }


@app.post("/api/mcp/disconnect/{name}")
async def mcp_disconnect(name: str):
    """Disconnect an MCP server."""
    global streaming_agent
    async with use_agent() as previous:
        assert mcp_manager is not None
        await mcp_manager.disconnect(name)
        streaming_agent = create_streaming_agent()
        await previous.aclose()
        return {"disconnected": name}


@app.post("/api/session/save", response_model=SessionResponse)
async def save_session():
    async with use_agent() as agent:
        return SessionResponse(session_id=await agent.save_session())


@app.post("/api/session/load/{session_id}")
async def load_session(session_id: str):
    global streaming_agent
    async with use_agent() as previous:
        restored = await Agent.load_session(session_id, **agent_bindings())
        streaming_agent = restored
        await previous.aclose()
        return {"loaded": restored.session_id}


@app.post("/api/session/clear")
async def clear_session():
    global streaming_agent
    async with use_agent() as previous:
        streaming_agent = create_streaming_agent()
        await previous.aclose()
        return {"status": "cleared"}


@app.get("/api/sessions")
async def list_sessions():
    from harnessx import PersistentMemory

    storage = PersistentMemory()
    return {"sessions": storage.list_sessions()}


# ── Serve the UI ─────────────────────────────────────────────────────────


@app.get("/")
async def serve_ui():
    with open(os.path.join(STATIC_DIR, "index.html")) as f:
        return HTMLResponse(f.read())


# ── Serve generated files ───────────────────────────────────────────────

app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="generated_files")
