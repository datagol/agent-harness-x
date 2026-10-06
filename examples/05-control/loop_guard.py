"""Loop guard: notice an agent that keeps making the same call and getting the same answer.

`Limits(loop_guard=LoopGuard(...))` watches the tool calls a run makes. When a call (or a short cycle of calls)
repeats `threshold` times with the same arguments *and* the same result, the guard fires `HookEvent.REPETITION` and
puts a note in front of the tool result the model reads next. It never fails the run; `Limits.max_iterations` is
the hard stop. Here a scripted model polls a build that never leaves the queue, reads the note, and gives up
cleanly.

Run:
    python examples/05-control/loop_guard.py
Needs: Nothing: a scripted model, no network.
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, HookEvent, Limits, LoopGuard, ProviderResponse, StopReason, ToolCall
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


def check_build(job: str) -> str:
    """Report the status of a CI build."""
    return f"build {job}: queued (position 4)"


def poll(n: int) -> ProviderResponse:
    return ProviderResponse(
        tool_calls=[ToolCall(f"poll_{n}", "check_build", {"job": "1234"})], stop_reason=StopReason.TOOL_USE
    )


SCRIPT = [
    poll(1),
    poll(2),
    poll(3),  # the third identical call with the identical result trips the guard
    ProviderResponse(text="Build 1234 has not moved from the queue in three checks. Stopping here; the CI runner "
                          "looks stalled, so someone should look at the build queue."),
]


async def main() -> None:
    config = AgentConfig(limits=Limits(max_iterations=10, loop_guard=LoopGuard(threshold=3, max_period=4)))
    async with Agent(config=config, provider=ScriptedProvider(SCRIPT), tools=[check_build]) as agent:
        agent.hooks.on(HookEvent.TOOL_CALL_START, lambda ctx: print(
            f"  > {ctx.data['tool_call'].name}({json.dumps(ctx.data['tool_call'].input)[:120]})"))
        agent.hooks.on(HookEvent.REPETITION, lambda ctx: print(
            f"REPETITION: cycle of {ctx.data['period']} call(s), {ctx.data['laps']} laps, ending in {ctx.data['tool']}"))

        result = await agent.run("Wait for build 1234 to finish, then tell me whether it passed.")

        # The note the model read, in front of the third result.
        last_result = next(
            block["content"]
            for message in reversed(agent.memory.get_messages()) if isinstance(message["content"], list)
            for block in message["content"] if block.get("type") == "tool_result"
        )

    print("\nWhat the model read after the third call:")
    print("  " + last_result.replace("\n", "\n  "))
    if result.error:
        print(f"Error: {result.error['message']}")
        return
    print(f"\nRun status: {result.status.value} (the guard warns; it does not fail the run)")
    print(f"Answer: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
