"""A provider that is slow rather than broken must not go silent.

29 September 2026, in a shopping assistant built on this harness: the model
returned 503s and took 7-18 seconds for calls that normally take 1.4-4.6. Every
call eventually succeeded, so nothing failed, nothing retried, and nothing was
logged -- and the app showed the user nothing at all for the duration. From
the outside that is indistinguishable from a hung agent, and the first
conclusion drawn was that the last deploy had broken it.
"""

import asyncio

from harnessx import Agent
from harnessx.types import (
    AgentConfig,
    ProgressPolicy,
    ProviderResponse,
    RetryPolicy,
    StopReason,
    StreamChunk,
    TokenUsage,
)


def _response(text="hello"):
    return ProviderResponse(
        text=text, tool_calls=[], thinking=None,
        stop_reason=StopReason.END_TURN, usage=TokenUsage(), raw=None,
    )


class _SlowProvider:
    """Answers correctly, just late."""

    name = "slow"

    def __init__(self, delay=0.25):
        self.delay = delay

    async def create(self, **kwargs):
        await asyncio.sleep(self.delay)
        return _response()

    def stream(self, **kwargs):
        delay = self.delay

        async def gen():
            await asyncio.sleep(delay)
            yield StreamChunk(kind="text_delta", data="hello")
            yield StreamChunk(kind="response", data=_response())

        return gen()

    async def count_tokens(self, **kwargs):
        return 0

    async def aclose(self):
        pass


def _agent(provider, **config):
    return Agent(config=AgentConfig(model="m", **config), provider=provider)


async def _events(agent, message="hi"):
    out = []
    async with agent.run_stream(message) as stream:
        async for event in stream:
            out.append(event)
    return out


def test_a_slow_stream_says_it_is_still_working():
    async def _run():
        provider = _SlowProvider(delay=0.3)
        async with _agent(
            provider,
            progress=ProgressPolicy(first_after_seconds=0.05, repeat_every_seconds=0.05),
        ) as agent:
            events = await _events(agent)
        waiting = [e for e in events if e.type.value == "waiting"]
        assert waiting, "a slow call produced no notice at all"
        assert all(e.data["on"] == "model" for e in waiting)
        assert waiting[0].data["seconds"] > 0
        # Still a normal turn: the answer is delivered, nothing is cancelled.
        assert [e.data for e in events if e.type.value == "text_delta"] == ["hello"]

    asyncio.run(_run())


def test_the_notices_repeat_while_the_wait_continues():
    async def _run():
        provider = _SlowProvider(delay=0.4)
        async with _agent(
            provider,
            progress=ProgressPolicy(first_after_seconds=0.05, repeat_every_seconds=0.05),
        ) as agent:
            events = await _events(agent)
        seconds = [e.data["seconds"] for e in events if e.type.value == "waiting"]
        assert len(seconds) >= 2, f"expected repeats, got {seconds}"
        assert seconds == sorted(seconds), "elapsed time must only grow"

    asyncio.run(_run())


def test_a_prompt_answer_says_nothing():
    async def _run():
        provider = _SlowProvider(delay=0)
        async with _agent(
            provider, progress=ProgressPolicy(first_after_seconds=0.5)
        ) as agent:
            events = await _events(agent)
        assert not [e for e in events if e.type.value == "waiting"]

    asyncio.run(_run())


def test_notices_can_be_switched_off():
    async def _run():
        provider = _SlowProvider(delay=0.2)
        async with _agent(
            provider, progress=ProgressPolicy(first_after_seconds=None)
        ) as agent:
            events = await _events(agent)
        assert not [e for e in events if e.type.value == "waiting"]

    asyncio.run(_run())


def test_the_non_streaming_call_is_covered_too():
    async def _run():
        provider = _SlowProvider(delay=0.3)
        async with _agent(
            provider,
            progress=ProgressPolicy(first_after_seconds=0.05, repeat_every_seconds=0.05),
        ) as agent:
            result = await agent.run("hi")  # agent.run does not stream
        assert result.output == "hello"

    asyncio.run(_run())


def test_waiting_is_a_notice_and_never_a_deadline():
    """The call is bounded by the retry policy's timeout, not by this."""

    async def _run():
        provider = _SlowProvider(delay=0.3)
        async with _agent(
            provider,
            progress=ProgressPolicy(first_after_seconds=0.02, repeat_every_seconds=0.02),
            retry=RetryPolicy(call_timeout_seconds=30),
        ) as agent:
            events = await _events(agent)
        assert not [e for e in events if e.type.value == "error"]
        assert [e.data for e in events if e.type.value == "text_delta"] == ["hello"]

    asyncio.run(_run())
