"""Watch a long run's progress from the caller's side, including the silence of a slow model call.

`agent.run_stream()` yields one typed `RunEvent` per step of the run. A model that is slow rather than broken sends
nothing at all, which looks exactly like a hung agent, so the engine fills that gap with `RunEventType.WAITING`
events: `{"on": "model", "seconds": ...}`. `AgentConfig(progress=ProgressPolicy(...))` sets when the first one
arrives and how often it repeats (defaults: 10s, then every 15s; `first_after_seconds=None` turns them off). They
are notices, not deadlines: nothing is cancelled. Here the delays are shortened so the run takes about five seconds.

Run: python examples/01-basics/progress_and_waiting.py
Needs: Nothing: a scripted model, no network.
"""

import asyncio
import json
import time

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, ProgressPolicy, ProviderResponse, RunEventType, ToolCall
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


class SlowProvider(ScriptedProvider):
    """Takes a while before each reply, like a large model thinking before its first token."""

    def __init__(self, replies, delay):
        super().__init__(replies)
        self.delay = delay

    async def create(self, **kwargs):
        await asyncio.sleep(self.delay)
        return await super().create(**kwargs)


async def build_report(region: str) -> str:
    """Build the quarterly sales report for a region.

    Args:
        region: Sales region, such as 'north'.
    """
    await asyncio.sleep(1.0)
    return f"{region}: 1,204 orders, revenue 88,410"


def describe(event) -> str:
    kind = event.type
    if kind == RunEventType.WAITING:
        return f"still waiting on the {event.data['on']}, {event.data['seconds']}s so far"
    if kind in (RunEventType.TOOL_CALL_START, RunEventType.TOOL_CALL_COMPLETE):
        return f"{event.data.name}({json.dumps(event.data.input)[:120]})"
    if kind == RunEventType.TOOL_RESULT:
        return f"{'! ' if event.data.is_error else ''}{event.data.content}"
    if kind == RunEventType.RUN_RESULT:
        return f"status={event.data.status.value}"
    return repr(event.data)[:120] if isinstance(event.data, str) else ""


async def main() -> None:
    provider = SlowProvider(
        [
            ProviderResponse(
                tool_calls=[ToolCall("call_1", "build_report", {"region": "north"})], stop_reason="tool_use"
            ),
            ProviderResponse(text="North had 1,204 orders and 88,410 in revenue this quarter."),
        ],
        delay=1.6,
    )
    config = AgentConfig(
        system_prompt="You write short sales summaries. Use build_report for figures.",
        progress=ProgressPolicy(first_after_seconds=0.5, repeat_every_seconds=0.5),
    )
    started = time.monotonic()
    async with Agent(config=config, provider=provider, tools=[build_report]) as agent:
        async with agent.run_stream("Summarize this quarter for the north region.") as stream:
            async for event in stream:
                print(f"[{time.monotonic() - started:4.1f}s] {event.type.value:<18} {describe(event)}")
            result = await stream.result()

    # WAITING covers model calls. A slow tool is already visible to the caller: TOOL_CALL_START names it before it
    # runs and TOOL_RESULT follows when it returns; its own timeout bounds it.
    if result.error:
        print(f"Error: {result.error['message']}")
    else:
        print(f"\nAnswer: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
