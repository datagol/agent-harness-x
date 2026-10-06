"""Save a conversation to disk, load it into a fresh agent, and keep talking.

``agent.save_session(storage_dir)`` writes a snapshot of the conversation, its configuration,
usage and extension state, and returns the session ID. ``Agent.load_session(session_id,
storage_dir, provider=...)`` builds a new Agent from that snapshot; tools, provider and
extensions are code, so you pass them again. The second agent continues the same history.

This is the step before AgentRuntime: a snapshot is taken between turns and never resumes
a run that was interrupted halfway. When a run must survive a crash, a pending approval, or a
restart in the middle of a tool call, use AgentRuntime (see the other examples in this folder).

Run:   python examples/06-durability/session_snapshots.py
Needs: Nothing: a scripted model, no network. Snapshots go to a temporary directory.
"""

import asyncio
import tempfile

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, ProviderResponse
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


def show_history(agent: Agent) -> None:
    for message in agent.memory.get_messages():
        content = message["content"]
        if not isinstance(content, str):
            content = " ".join(block.get("text", "") for block in content if isinstance(block, dict))
        print(f"  {message['role']:>9}: {content}")


async def main() -> None:
    config = AgentConfig(system_prompt="You are a concise trip-planning assistant.")
    with tempfile.TemporaryDirectory(prefix="harnessx-sessions-") as storage:
        first = ScriptedProvider([ProviderResponse(text="Noted: Lisbon, five days in May, on a mid-range budget.")])
        async with Agent(config=config, provider=first) as agent:
            await agent.run("I'm planning five days in Lisbon in May, mid-range budget.")
            session_id = await agent.save_session(storage)
        print("Saved session", session_id)

        # A new process would do exactly this: same config and bindings, history from disk.
        second = ScriptedProvider([ProviderResponse(text="Day 1 in Lisbon: Alfama and the castle at sunset.")])
        async with await Agent.load_session(session_id, storage, config=config, provider=second) as restored:
            print("Loaded session", restored.session_id, "with", restored.memory.message_count, "messages")
            result = await restored.run("Start the itinerary with day one.")
            if result.error:
                print("Error:", result.error["message"])
                return
            print("History after continuing:")
            show_history(restored)


if __name__ == "__main__":
    asyncio.run(main())
