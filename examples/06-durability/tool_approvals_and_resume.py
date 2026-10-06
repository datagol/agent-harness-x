"""Pause a run for a human decision, persist it, and finish the same run after approval.

A tool registered with PermissionLevel.ASK does not run on its own: the run stops as
``awaiting_input`` and the pending call is saved in the runtime's store (SQLite here, in
a temporary directory). ``runtime.approve(pending, allow=..., resume=True)`` records the
decision and runs the turn to its end, so approving later, even from another process,
continues the run instead of starting over.

Run:   python examples/06-durability/tool_approvals_and_resume.py
Needs: Nothing: a scripted model, no network. It asks one question before writing a note.
"""

import asyncio
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from harnessx import Agent, AgentRuntime, PermissionLevel, ProviderResponse, RunResult, SQLiteBackend, ToolCall
from harnessx.providers import LLMProvider

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables


class ScriptedProvider(LLMProvider):
    """Fixed replies, so the example runs the real engine without a model."""

    def __init__(self, replies):
        self.replies = list(replies)

    async def create(self, **kwargs):
        return self.replies.pop(0)

    async def count_tokens(self, **kwargs):
        return 0


async def decide(runtime: AgentRuntime, result: RunResult) -> RunResult:
    """Ask about the paused tool, record the decision, and finish the run."""
    pending = result.pending[0]
    if pending.status != "approval":
        print("Tool outcome needs reconciliation:", pending.execution_key)
        return result
    # input() runs in a thread so the runtime keeps renewing its session lease while a person decides.
    try:
        answer = await asyncio.to_thread(input, f"Allow {pending.call.name}({pending.call.input})? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        answer = ""
    # resume=True persists the decision and runs the turn to its end.
    # With several pending tools, approve each without resume, then resume once.
    return await runtime.approve(pending, allow=answer.strip().lower() == "y", resume=True)


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="harnessx-approval-") as directory:
        note = Path(directory) / "note.txt"
        provider = ScriptedProvider([
            ProviderResponse(
                tool_calls=[ToolCall("note", "create_note", {"text": "Reviewed fixture"})], stop_reason="tool_use",
            ),
            ProviderResponse(text="The approval decision has been processed."),
        ])
        async with Agent(provider=provider) as agent:

            @agent.tools.register(permission=PermissionLevel.ASK, replay_policy="manual")
            def create_note(text: str) -> str:
                """Create a note in this example's temporary directory."""
                note.write_text(text)
                return "Note created"

            async with SQLiteBackend(str(Path(directory) / "runtime.db")) as backend:
                async with AgentRuntime(agent, backend=backend) as runtime:
                    result = await runtime.run("Create a review note")
                    print("Before approval:", result.status.value, "| Note exists:", note.exists())
                    if result.needs_input:
                        result = await decide(runtime, result)
                    print("After approval:", result.status.value, "| Note exists:", note.exists())
                    if result.error:
                        print("Error:", result.error["message"])
                    else:
                        print("Answer:", result.output)


if __name__ == "__main__":
    asyncio.run(main())
