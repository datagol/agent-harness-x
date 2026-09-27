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


async def approve_pending(runtime: AgentRuntime, result: RunResult):
    unresolved = False
    for pending in result.pending:
        if pending["status"] != "approval":
            print("Tool outcome needs reconciliation:", pending["execution_key"])
            unresolved = True
            continue
        call = pending["call"]
        response = await asyncio.to_thread(
            input, f"Allow {call['name']} {call['input']}? [y/N] "
        )
        await runtime.approve(
            pending["execution_key"], allow=response.strip().lower() == "y"
        )
    return None if unresolved else await runtime.resume(runtime.session_id)


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
                result = await runtime.execute("Create a review note")
                print(
                    "Before approval:",
                    result.status.value,
                    "| Note exists:",
                    note.exists(),
                )
                if result.status == "awaiting_input":
                    handle = await approve_pending(runtime, result)
                    if handle:
                        result = await handle.result()
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
