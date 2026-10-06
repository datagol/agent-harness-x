"""Persisted tool approval with a scripted provider and temporary SQLite store.

Run: python -m examples.runtime_approvals
No model API key is required. The terminal prompts before creating a fixture note.
"""

import asyncio
from pathlib import Path
import tempfile

from harnessx import (
    Agent,
    AgentRuntime,
    PermissionLevel,
    ProviderResponse,
    RunResult,
    SQLiteBackend,
    ToolCall,
)
from examples._fixtures import ScriptedProvider


async def decide(runtime: AgentRuntime, result: RunResult) -> RunResult:
    """Ask about the paused tool, record the decision, and finish the run."""
    pending = result.pending[0]
    if pending.status != "approval":
        print("Tool outcome needs reconciliation:", pending.execution_key)
        return result
    answer = await asyncio.to_thread(
        input, f"Allow {pending.call.name} {pending.call.input}? [y/N] "
    )
    # approve(..., resume=True) persists the decision and runs the turn to its end.
    # With several pending tools, approve each without resume, then resume once.
    return await runtime.approve(
        pending, allow=answer.strip().lower() == "y", resume=True
    )


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="harnessx-approval-") as directory:
        note = Path(directory) / "note.txt"
        provider = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[
                        ToolCall("note", "create_note", {"text": "Reviewed fixture"})
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(text="The approval decision has been processed."),
            ]
        )
        agent = Agent(provider=provider)

        @agent.tools.register(permission=PermissionLevel.ASK, replay_policy="manual")
        def create_note(text: str) -> str:
            """Create a note in this example's temporary directory."""
            note.write_text(text)
            return "Note created"

        backend = SQLiteBackend(str(Path(directory) / "runtime.db"))
        try:
            async with AgentRuntime(agent, backend=backend) as runtime:
                result = await runtime.run("Create a review note")
                print(
                    "Before approval:",
                    result.status.value,
                    "| Note exists:",
                    note.exists(),
                )
                if result.needs_input:
                    result = await decide(runtime, result)
                print(
                    "After approval:",
                    result.status.value,
                    "| Note exists:",
                    note.exists(),
                )
                if result.error:
                    raise RuntimeError(result.error["message"])
        finally:
            await backend.aclose()


if __name__ == "__main__":
    asyncio.run(main())
