"""Planning with a task list: `write_todos` and `read_todos`, and the events they raise.

Every agent has two planning tools by default (`AgentConfig(planning=False)` removes them). The model records its
plan with `write_todos`, sending the whole list each time with one item `in_progress`, and the harness rebuilds the
current list into every request so the plan survives condensing. Each change fires `HookEvent.TODOS_UPDATED` and,
on the run stream, `RunEventType.TODOS_UPDATED` with the whole list, the completed count and the active item.
Afterwards the list is on `agent.todos`. A scripted model works through a three-step database migration.

Run:
    python examples/03-context/planning_todos.py
Needs: Nothing: a scripted model, no network.
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from harnessx import Agent, ProviderResponse, RunEventType, StopReason, ToolCall, ToolResult
from harnessx.providers import LLMProvider

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

STEPS = ["Back up the orders table", "Rename user_id to account_id", "Verify row counts match"]
MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


class ScriptedProvider(LLMProvider):
    """Fixed replies, so the example runs the real engine without a model."""

    def __init__(self, replies):
        self.replies = list(replies)

    async def create(self, **kwargs):
        return self.replies.pop(0)

    async def count_tokens(self, **kwargs):
        return 0


def run_step(step: str) -> str:
    """Carry out one step of the migration."""
    return f"{step}: done"


def plan(done: int) -> list[dict]:
    """The whole list with the first `done` steps completed and the next one in progress."""
    return [
        {"content": step, "status": "completed" if i < done else "in_progress" if i == done else "pending"}
        for i, step in enumerate(STEPS)
    ]


def turn(n: int) -> ProviderResponse:
    """Update the plan, then work on the step now in progress; once all are done, read the list back."""
    calls = [ToolCall(f"todos_{n}", "write_todos", {"todos": plan(n)})]
    if n < len(STEPS):
        calls.append(ToolCall(f"step_{n}", "run_step", {"step": STEPS[n]}))
    else:
        calls.append(ToolCall("check", "read_todos", {}))
    return ProviderResponse(tool_calls=calls, stop_reason=StopReason.TOOL_USE)


SCRIPT = [turn(0), turn(1), turn(2), turn(3), ProviderResponse(text="Migration finished: backup taken, column "
                                                                      "renamed, row counts verified.")]


async def main() -> None:
    async with Agent(provider=ScriptedProvider(SCRIPT), tools=[run_step]) as agent:
        print(f"Planning tools registered: {[n for n in ('write_todos', 'read_todos') if agent.tools.has_tool(n)]}")
        async with agent.run_stream("Rename orders.user_id to account_id safely.") as stream:
            async for event in stream:
                if isinstance(event.data, ToolCall) and event.data.name == "run_step":
                    print(f"  > {event.data.name}({json.dumps(event.data.input)[:120]})")
                elif isinstance(event.data, ToolResult) and event.data.tool_call_id == "check":
                    print("  read_todos returned:\n    " + event.data.content.replace("\n", "\n    "))
                elif event.type == RunEventType.TODOS_UPDATED and isinstance(event.data, dict):
                    plan_now = dict(event.data)  # {todos, completed, total, in_progress}
                    print(f"TODOS_UPDATED [{plan_now['completed']}/{plan_now['total']}] "
                          f"in progress: {plan_now['in_progress']}")
            result = await stream.result()

        if result.error:
            print(f"Error: {result.error['message']}")
            return
        print(f"\nAnswer: {result.output}")
        print("Final list (agent.todos):")
        for item in agent.todos:
            print(f"  {MARK[item['status']]} {item['content']}")


if __name__ == "__main__":
    asyncio.run(main())
