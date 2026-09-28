"""Independent chat sessions over the same HarnessX streaming engine."""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any
import uuid

from harnessx import (
    Agent,
    AgentConfig,
    Limits,
    PermissionLevel,
    PermissionManager,
    ProviderResponse,
    RunEventType,
    SkillManager,
    ToolCall,
)
from harnessx.builtin.filesystem import register_filesystem_tools
from harnessx.execution import wire
from harnessx.mcp import MCPManager
from harnessx.providers import LLMProvider
from examples._calculator import calculate
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
    "workspace. File changes require the user's approval."
)


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


_SOURCE_ORDER = {"builtin": 0, "skill": 1, "mcp": 2}


def tool_records(chat: Chat) -> list[dict[str, Any]]:
    """Every tool the conversation's agent can call, with where it came from."""
    agent = chat.agent
    assert agent is not None
    bridged = set(getattr(agent, "mcp_tools", ()) or ())
    servers = list(chat.mcp.list_servers())
    records = []
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
    register_filesystem_tools(agent.tools, base_path=str(chat.directory))
    for name in ("write_file", "generate_file"):
        agent.permissions.set_permission(name, PermissionLevel.ASK)
    for name in ("read_file", "list_directory"):
        agent.permissions.set_permission(name, PermissionLevel.ALLOW)
    # Agent(mcp=chat.mcp) already bridged the connected servers' tools.
    return agent


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

    assistant = {
        "role": "assistant",
        "content": "",
        "tools": [],
        "run_id": run.id,
        "status": "running",
    }
    chat.messages.extend([{"role": "user", "content": message}, assistant])
    chat.title = message[:65]
    try:
        await run.state("running")
        assert chat.agent is not None
        async with chat.agent.run_stream(message) as stream:
            async for event in stream:
                if event.type == RunEventType.TEXT_DELTA:
                    assistant["content"] += event.data
                elif event.type == RunEventType.ATTEMPT_RESET:
                    assistant["content"] = ""
                elif event.type == RunEventType.TOOL_CALL_START:
                    assistant["tools"].append(
                        {
                            "id": event.data.id,
                            "name": event.data.name,
                            "input": event.data.input,
                            "status": "running",
                        }
                    )
                elif event.type == RunEventType.TOOL_RESULT:
                    for tool in assistant["tools"]:
                        if tool["id"] == event.data.tool_call_id:
                            tool.update(
                                status="failed" if event.data.is_error else "completed",
                                content=event.data.content,
                            )
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
        run.pending = None
