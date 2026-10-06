"""Independent chat sessions over the same HarnessX streaming engine."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import re
from typing import Any
import uuid

from harnessx import (
    Agent,
    AgentConfig,
    HookEvent,
    Limits,
    Middleware,
    PermissionLevel,
    PermissionManager,
    ProviderResponse,
    RunEventType,
    SkillManager,
    ToolCall,
    ToolResult,
)
from harnessx.builtin.ask import register_ask_user_tool
from harnessx.builtin.bash import register_bash_tools
from harnessx.builtin.filesystem import register_filesystem_tools
from harnessx.execution import wire
from harnessx.mcp import MCPManager
from harnessx.providers import LLMProvider
from harnessx.subagents import SubAgent, install_subagent, prepare_subagents
from .calculator import calculate
from .runs import Run


class DemoProvider(LLMProvider):
    """An explicitly labelled local fixture for trying chat without credentials."""

    async def count_tokens(self, **kwargs):
        return 0

    async def create(self, *, messages, **kwargs):
        last = messages[-1]["content"]
        if isinstance(last, list) and last and last[0].get("type") == "tool_result":
            return ProviderResponse(
                text=f"The calculator returned: {last[0]['content']}\n\nThis is the local demo, with no model calls."
            )
        text = last if isinstance(last, str) else ""
        match = re.search(r"[\d(][\d\s.+*/()%\-]*", text)
        if match and any(char in match.group() for char in "+-*/%"):
            return ProviderResponse(
                tool_calls=[
                    ToolCall(
                        uuid.uuid4().hex,
                        "calculate",
                        {"expression": match.group().strip()},
                    )
                ],
                stop_reason="tool_use",
            )
        return ProviderResponse(
            text="You're connected to the local HarnessX demo. Try `What is 48 * 12?` to see a real calculator tool call.\n\nFor open-ended conversation, start a new chat with a configured model provider. This demo does not use a language model."
        )


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful general assistant. Be clear and concise. Use your "
    "calculator for arithmetic. Files are relative to the conversation "
    "workspace. File changes require the user's approval. When you save or "
    "generate a file, the app attaches a download link to your reply: refer to "
    "the file by name and never invent a URL for it. When a request is ambiguous "
    "-- a term you do not recognize or that could mean several things -- use "
    "ask_user to check before starting substantial work."
)

DOWNLOADS_DIR = ".downloads"  # generate_file copies land here, served by /api/chats/{id}/downloads
_MARKER = re.compile(r"__FILE__:/files/([0-9a-f]+)/([^:\n]+):([^\n]*)")


class _DownloadLinks(Middleware):
    """Turn generate_file's UI marker into a link the model can quote."""

    def __init__(self, chat: "Chat") -> None:
        self._chat = chat

    async def after_tool_execution(self, result: Any) -> Any:
        if isinstance(result, ToolResult) and isinstance(result.content, str) and "__FILE__:" in result.content:
            content = _MARKER.sub(
                lambda m: f"Saved {m.group(3) or m.group(2)}. Download link: "
                f"/api/chats/{self._chat.id}/downloads/{m.group(1)}/{m.group(2)}",
                result.content,
            )
            return ToolResult(result.tool_call_id, content, result.is_error)
        return result


def today_line() -> str:
    return f"Today's date is {date.today():%A, %B %-d, %Y}. Treat it as the present when reasoning about recent events or dates."


def workspace_relative(chat: "Chat", path: str) -> str | None:
    """A workspace-relative POSIX path for a tool's path argument, or None if outside."""
    root = chat.directory.resolve()
    try:
        target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        return target.relative_to(root).as_posix()
    except (OSError, ValueError):
        return None


@dataclass
class Chat:
    id: str
    provider: str
    model: str
    agent: Agent | None
    directory: Path
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    mcp: MCPManager = field(default_factory=MCPManager)
    title: str = "New conversation"
    messages: list = field(default_factory=list)
    active: Run | None = None
    disabled_tools: set = field(default_factory=set)
    tool_catalog: list = field(default_factory=list)  # every tool, enabled or not
    # Specialists the agent can delegate to: {name, description, instructions, tools}.
    subagents: list = field(default_factory=list)

    def public(self, *, include_messages=True):
        assert self.agent is not None
        result: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "provider": self.provider,
            "model": self.model,
            "latest_run": self.active.id if self.active else None,
            "active_run": self.active.id
            if self.active
            and self.active.status not in {"completed", "failed", "cancelled"}
            else None,
            "usage": self.agent.guardrails.usage_summary,
            "system_prompt": self.system_prompt,
            "skills": skill_records(self),
            "mcp_servers": self.mcp.list_servers(),
            "tools": tool_records(self),
            "subagents": self.subagents,
        }
        if include_messages:
            result["messages"] = self.messages
        return result


def _skill_paths(chat: Chat) -> list[Path]:
    folder = chat.directory / "skills"
    if not folder.is_dir():
        return []
    return [
        path
        for path in sorted(folder.glob("*.md"))
        if path.is_file() and not path.is_symlink() and not path.name.startswith(".")
    ]


_SOURCE_ORDER = {"builtin": 0, "skill": 1, "mcp": 2, "subagent": 3}


def tool_records(chat: Chat) -> list[dict[str, Any]]:
    """Every tool the conversation knows, enabled or not, with where it came from."""
    return [
        {**record, "enabled": record["name"] not in chat.disabled_tools}
        for record in chat.tool_catalog
    ]


def catalog_tools(chat: Chat, agent: Agent) -> list[dict[str, Any]]:
    """Describe the agent's registered tools before any disabled ones are removed."""
    bridged = set(getattr(agent, "mcp_tools", ()) or ())
    servers = list(chat.mcp.list_servers())
    records: list[dict[str, Any]] = []
    for definition in agent.tools.get_tools():
        name = definition.name
        tag = getattr(definition.handler, "__mcp_tool__", None)
        if tag:
            source, server, tool = "mcp", tag[0], tag[1]
        elif name in bridged:
            server = next((s for s in servers if name.startswith(f"{s}_")), None)
            source, tool = "mcp", name[len(server) + 1:] if server else name
        elif name == "Skill":
            source, server, tool = "skill", None, name
        elif name.startswith("delegate_") and name[len("delegate_"):] in {s["name"] for s in chat.subagents}:
            source, server, tool = "subagent", None, name
        else:
            source, server, tool = "builtin", None, name
        records.append({
            "name": name,
            "tool": tool,
            "description": definition.description,
            "source": source,
            "server": server,
            "permission": agent.permissions.get_effective_permission(name, definition).value,
        })
    records.sort(key=lambda r: (_SOURCE_ORDER[r["source"]], r["server"] or "", r["name"]))
    return records


def set_tool_enabled(chat: Chat, name: str, enabled: bool) -> None:
    """Record a per-conversation tool switch; rebuild_chat applies it."""
    known = {record["name"] for record in chat.tool_catalog}
    if name not in known:
        raise KeyError(name)
    if name == "Skill":
        raise ValueError("The Skill tool cannot be disabled")
    if enabled:
        chat.disabled_tools.discard(name)
    else:
        chat.disabled_tools.add(name)


def skill_records(chat: Chat) -> list[dict[str, Any]]:
    paths = _skill_paths(chat)
    if not paths:
        return []
    manager = SkillManager.from_paths([str(path) for path in paths])
    return [
        {
            "name": skill.name,
            "description": skill.description,
            "file": skill.source_path.name,
            "body_chars": len(skill.body),
        }
        for skill in manager.list()
    ]


def _build_agent(chat: Chat, *, provider_instance=None) -> Agent:

    async def approve(call, definition):
        if chat.active is None:
            return False
        answer = await chat.active.question(
            f"Allow {call.name}?",
            "approval",
            tool={"name": call.name, "input": call.input},
        )
        return answer == "y"

    skills = _skill_paths(chat)
    agent = Agent(
        config=AgentConfig(
            provider="anthropic" if chat.provider == "demo" else chat.provider,
            model=chat.model,
            system_prompt=chat.system_prompt,
            limits=Limits(max_iterations=20),
        ),
        provider=provider_instance
        or (DemoProvider() if chat.provider == "demo" else None),
        permissions=PermissionManager(approval_callback=approve),
        mcp=chat.mcp,
        skills=[str(path) for path in skills] or None,
    )
    agent.tools.register_tool(
        calculate, permission=PermissionLevel.ALLOW, replay_policy="safe"
    )
    register_filesystem_tools(
        agent.tools, base_path=str(chat.directory), output_dir=str(chat.directory / DOWNLOADS_DIR),
    )
    # The agent can ask before it guesses. The question appears in the
    # conversation as its own prompt; with no run attached nobody can answer,
    # and the tool tells the model to proceed on a stated assumption.
    async def ask(question, choices):
        if chat.active is None:
            return None
        return await chat.active.question(question, "question", choices=choices)

    register_ask_user_tool(agent.tools, ask)
    # Shell commands start in this conversation's own directory, like the file
    # tools, and every one is approved first: a command can reach anything the
    # server process can.
    register_bash_tools(agent.tools, cwd=str(chat.directory), permission=PermissionLevel.ASK)
    for name in ("write_file", "generate_file", "run_bash"):
        agent.permissions.set_permission(name, PermissionLevel.ASK)
    for name in ("read_file", "list_directory"):
        agent.permissions.set_permission(name, PermissionLevel.ALLOW)
    install_subagents(chat, agent)
    agent.middleware.add(_DownloadLinks(chat))
    # The model has no clock. Day granularity keeps the cached prompt prefix
    # stable within a day while stopping searches for last year's news.
    agent.prompt_providers.append(today_line)
    # Agent(mcp=chat.mcp) already bridged the connected servers' tools. Record
    # the full catalog, then take the switched-off ones away from the model.
    chat.tool_catalog = catalog_tools(chat, agent)
    chat.disabled_tools &= {record["name"] for record in chat.tool_catalog}
    for name in chat.disabled_tools:
        agent.tools.unregister(name)
    return agent


SUBAGENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,54}")


def validate_subagents(chat: Chat, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean, checked subagent definitions, or ValueError saying what is wrong."""
    available = {record["name"] for record in chat.tool_catalog if record["source"] != "subagent"}
    cleaned, seen = [], set()
    for item in items:
        name = str(item.get("name", "")).strip()
        if not SUBAGENT_NAME.fullmatch(name):
            raise ValueError(f"Subagent name {name!r}: use 1-55 letters, digits, _ or -, starting with a letter")
        if name in seen:
            raise ValueError(f"Two subagents are named {name!r}")
        seen.add(name)
        description = str(item.get("description", "")).strip()
        instructions = str(item.get("instructions", "")).strip()
        if not description:
            raise ValueError(f"Subagent {name!r} needs a description: when should the agent delegate to it?")
        if not instructions:
            raise ValueError(f"Subagent {name!r} needs instructions")
        tools = [str(tool) for tool in item.get("tools", [])]
        unknown = sorted(set(tools) - available)
        if unknown:
            raise ValueError(f"Subagent {name!r} lists tools this conversation does not have: {', '.join(unknown)}")
        cleaned.append({"name": name, "description": description, "instructions": instructions,
                        "tools": sorted(set(tools))})
    return cleaned


def install_subagents(chat: Chat, agent: Agent) -> None:
    """Give the agent a delegate_<name> tool per subagent.

    Each delegation runs a fresh child agent on the conversation's provider and
    model, with the subagent's instructions as its system prompt and only the
    tools it was given, taken from this conversation (so file tools stay in its
    directory and approvals still come to the browser).
    """
    if not chat.subagents:
        return
    definitions = [
        SubAgent(
            name=spec["name"],
            description=spec["description"],
            config=AgentConfig(
                provider="anthropic" if chat.provider == "demo" else chat.provider,
                model=chat.model,
                system_prompt=spec["instructions"],
                limits=Limits(max_iterations=15),
                planning=False,
            ),
            tools=[agent.tools.get_tool(name) for name in spec["tools"] if agent.tools.has_tool(name)],
        )
        for spec in chat.subagents
    ]
    prepared = prepare_subagents(definitions, agent.tools)
    for definition in prepared:
        install_subagent(agent, definition)
    agent.subagents = tuple(agent.subagents) + prepared


def create_chat(provider, model, directory, *, provider_instance=None):
    resolved_model = model or ("demo" if provider == "demo" else "claude-sonnet-4-6")
    chat = Chat(
        uuid.uuid4().hex,
        provider,
        resolved_model,
        None,
        Path(directory),
    )
    (chat.directory / "skills").mkdir(parents=True, exist_ok=True)
    chat.agent = _build_agent(chat, provider_instance=provider_instance)
    return chat


async def rebuild_chat(chat: Chat) -> None:
    """Apply configuration changes while retaining validated conversation history."""
    if chat.active and chat.active.task and not chat.active.task.done():
        raise RuntimeError("Stop the active response before changing conversation setup")
    assert chat.agent is not None
    previous = chat.agent
    messages = previous.memory.get_messages()
    replacement = _build_agent(chat)
    replacement.memory.set_messages(messages)
    chat.agent = replacement
    await previous.aclose()


async def close_chat(chat: Chat) -> None:
    if chat.active:
        await chat.active.stop()
    if chat.agent is not None:
        await chat.agent.aclose()
    await chat.mcp.disconnect_all()


async def run_chat(chat, run, message):
    import asyncio

    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tools": [],
        "files": [],
        "segments": [],  # text and tool calls in the order they happened
        "todos": None,  # the agent's plan, as last announced by TODOS_UPDATED
        "run_id": run.id,
        "status": "running",
    }

    def append_text(delta: str) -> None:
        segments = assistant["segments"]
        if segments and segments[-1]["type"] == "text":
            segments[-1]["content"] += delta
        else:
            segments.append({"type": "text", "content": delta})

    def record_file(entry: dict[str, Any]) -> bool:
        if any(item["url"] == entry["url"] for item in assistant["files"]):
            return False
        assistant["files"].append(entry)
        return True
    chat.messages.extend([{"role": "user", "content": message}, assistant])
    chat.title = message[:65]
    assert chat.agent is not None
    agent = chat.agent

    # The browser shows what the turn is doing between visible events: a model
    # call can stream a long tool argument with nothing else to display.
    async def model_started(ctx):
        await run.emit("phase", phase="model", message_count=ctx.data.get("message_count"))

    async def model_finished(ctx):
        await run.emit("phase", phase="waiting")

    registrations = [
        agent.hooks.on(HookEvent.LLM_REQUEST, model_started),
        agent.hooks.on(HookEvent.LLM_RESPONSE, model_finished),
    ]
    try:
        await run.state("running")
        async with chat.agent.run_stream(message) as stream:
            async for event in stream:
                if event.type == RunEventType.TEXT_DELTA:
                    assistant["content"] += event.data
                    append_text(event.data)
                elif event.type == RunEventType.TODOS_UPDATED:
                    assistant["todos"] = event.data
                elif event.type == RunEventType.ATTEMPT_RESET:
                    assistant["content"] = ""
                    # A retried model call restarts its answer: drop the provisional text.
                    if assistant["segments"] and assistant["segments"][-1]["type"] == "text":
                        assistant["segments"].pop()
                elif event.type == RunEventType.TOOL_CALL_START:
                    assistant["tools"].append(
                        {
                            "id": event.data.id,
                            "name": event.data.name,
                            "input": event.data.input,
                            "status": "running",
                        }
                    )
                    assistant["segments"].append({"type": "tool", "id": event.data.id})
                elif event.type == RunEventType.TOOL_RESULT:
                    for tool in assistant["tools"]:
                        if tool["id"] == event.data.tool_call_id:
                            tool.update(
                                status="failed" if event.data.is_error else "completed",
                                content=event.data.content,
                            )
                            if not event.data.is_error and tool["name"] in ("write_file", "generate_file"):
                                added = False
                                for link in re.findall(r"Download link: (/api/chats/\S+)", str(event.data.content)):
                                    added |= record_file({"name": link.rsplit("/", 1)[-1], "url": link, "kind": "download"})
                                rel = workspace_relative(chat, str((tool.get("input") or {}).get("path", "")))
                                if rel and not rel.startswith(DOWNLOADS_DIR):
                                    added |= record_file({"name": rel, "url": f"/api/runs/{run.id}/files/{rel}", "kind": "workspace"})
                                if added:
                                    await run.emit("chat_files", files=list(assistant["files"]))
                await run.emit("agent", event=wire(event))
            result = await stream.result()
        assistant["content"] = result.output or assistant["content"]
        assistant["status"] = result.status.value
        if result.error:
            assistant["error"] = result.error["message"]
        await run.emit(
            "chat_result", message=assistant, usage=chat.agent.guardrails.usage_summary
        )
        await run.state("completed" if result.status == "completed" else "failed")
    except asyncio.CancelledError:
        assistant["status"] = "cancelled"
        for tool in assistant["tools"]:
            if tool["status"] == "running":
                tool["status"] = "cancelled"
        await run.emit(
            "chat_result", message=assistant, usage=chat.agent.guardrails.usage_summary
        )
        await run.state("cancelled")
        raise
    except Exception as exc:
        assistant.update(status="failed", error=str(exc))
        await run.emit("error", content=str(exc))
        await run.state("failed")
    finally:
        for registration in registrations:
            registration.close()
        run.pending = None
