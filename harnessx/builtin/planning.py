"""A task list the agent keeps, and keeps seeing.

Long runs lose the thread. The usual remedy is to let the agent write down what
it intends to do and tick items off, which works only if the list is still in
front of it fifty iterations later. OpenCode persists todos to a table and never
puts them back into the conversation, so the list survives in storage while
quietly vanishing from the model's view, taken out by the same compaction that
truncates every other tool result. An agent that believes it has a plan it can
no longer read is worse off than one with no plan at all.

So two halves. The tools write the list onto the run state, where it is
checkpointed with every phase and comes back on a durable resume. And the engine
rebuilds a short reminder at the tail of the conversation each turn, which the
condensation step preserves verbatim.

The reminder goes at the tail, not into the system prompt. The system prompt
heads the cacheable prefix, so rewriting it every time a task is ticked off would
invalidate the prompt cache for the whole conversation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from harnessx.types import PermissionLevel
from ._registration import select_tools

if TYPE_CHECKING:
    from harnessx.tools import ToolRegistry

STATUSES = ("pending", "in_progress", "completed")
MAX_TODOS = 50
MAX_CONTENT_CHARS = 200

_MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def _agent():
    """The agent running the current tool call, when there is one."""
    from harnessx.execution import current_agent

    return current_agent()


def normalize(todos: Any) -> tuple[list[dict[str, str]], list[str]]:
    """Validate and clean a list the model supplied. Raises ValueError on junk.

    One item may be in progress. When the model marks several, the first stays
    in progress and the rest go back to pending: the list is still usable, so it
    is fixed rather than refused, and the second value names what was moved so
    the model can be told.
    """
    if not isinstance(todos, list):
        raise ValueError("todos must be a list of {content, status} objects")
    if len(todos) > MAX_TODOS:
        raise ValueError(f"too many todos: {len(todos)}, the limit is {MAX_TODOS}")
    cleaned: list[dict[str, str]] = []
    for index, item in enumerate(todos):
        if not isinstance(item, dict):
            raise ValueError(f"todo {index} must be an object with content and status")
        content = str(item.get("content", "")).strip()
        if not content:
            raise ValueError(f"todo {index} has no content")
        status = str(item.get("status", "pending")).strip().lower()
        if status not in STATUSES:
            raise ValueError(f"todo {index} has status {status!r}; use one of {', '.join(STATUSES)}")
        cleaned.append({"content": content[:MAX_CONTENT_CHARS], "status": status})
    active = [item for item in cleaned if item["status"] == "in_progress"]
    for item in active[1:]:
        item["status"] = "pending"
    return cleaned, [item["content"] for item in active[1:]]


def render(todos: list[dict[str, str]]) -> str:
    """The reminder the model reads each turn."""
    if not todos:
        return ""
    lines = [f"  {_MARK.get(item['status'], '[ ]')} {item['content']}" for item in todos]
    done = sum(1 for item in todos if item["status"] == "completed")
    return f"[Task list, {done}/{len(todos)} complete]\n" + "\n".join(lines)


def register_planning_tools(
    registry: ToolRegistry, *, include: list[str] | None = None,
    exclude: list[str] | None = None, permission: PermissionLevel | None = None,
    replace: bool = False,
) -> list[str]:
    """Register the task-list tools. The list itself lives on the run state."""
    names = select_tools(registry, ["write_todos", "read_todos"], include, exclude, replace=replace)
    if not names:
        return []

    async def write_todos(todos: list[dict]) -> str:
        """Record the task list, replacing it entirely.

        Use this for work with several distinct steps, and keep it current as you
        go. Skip it for anything you can finish in one or two steps: a task list
        for trivial work is noise.

        Send the whole list every time, including items already completed. Keep
        exactly one item in_progress while you work on it; if you send more, the
        first is kept and the rest go back to pending.

        Args:
            todos: Objects of {"content": str, "status": "pending"|"in_progress"|"completed"}.
        """
        cleaned, demoted = normalize(todos)
        agent = _agent()
        if agent is not None:
            agent.todos = cleaned
        done = sum(1 for item in cleaned if item["status"] == "completed")
        reply = f"Task list recorded: {done}/{len(cleaned)} complete."
        if demoted:
            kept = next(item["content"] for item in cleaned if item["status"] == "in_progress")
            moved = ", ".join(f'"{content}"' for content in demoted)
            reply += f' Only one item can be in progress: kept "{kept}", set {moved} back to pending.'
        return reply

    async def read_todos() -> str:
        """Read the current task list."""
        agent = _agent()
        return render(getattr(agent, "todos", []) or []) or "The task list is empty."

    handlers = {"write_todos": write_todos, "read_todos": read_todos}
    for name in names:
        handler = handlers[name]
        # Tagged so an Agent can tell its own default apart from one the
        # application registered under the same name, and leave that one alone.
        handler.__harnessx_builtin__ = True  # type: ignore[attr-defined]
        registry.register_tool(
            handler, name=name, permission=permission,
            concurrent=False, replay_policy="safe", replace=replace,
        )
    return names
