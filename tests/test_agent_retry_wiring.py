"""Retry is on by default, and reaches the agent — not just the helper.

A helper nothing calls is the failure mode this test exists to prevent.
"""

import asyncio

from harnessx import Agent
from harnessx.types import (
    RetryPolicy,
    AgentConfig,
    ProviderResponse,
    StopReason,
    StreamChunk,
    TokenUsage,
)


class _Boom(Exception):
    def __init__(self, code=503):
        super().__init__(f"status {code}")
        self.code = code


class _FlakyProvider:
    """Fails the first stream and the first create, succeeds afterwards."""

    name = "flaky"

    def __init__(self, fail_times=1):
        self.opens = 0
        self.creates = 0
        self.fail_times = fail_times

    async def create(self, **kwargs):
        self.creates += 1
        if self.creates <= self.fail_times:
            raise _Boom()
        return ProviderResponse(
            text="hello", tool_calls=[], thinking=None,
            stop_reason=StopReason.END_TURN, usage=TokenUsage(), raw=None,
        )

    def stream(self, **kwargs):
        self.opens += 1
        should_fail = self.opens <= self.fail_times

        async def gen():
            if should_fail:
                raise _Boom()
            yield StreamChunk(kind="text_delta", data="hello")
            yield StreamChunk(
                kind="response",
                data=ProviderResponse(
                    text="hello", tool_calls=[], thinking=None,
                    stop_reason=StopReason.END_TURN, usage=TokenUsage(), raw=None,
                ),
            )

        return gen()

    async def count_tokens(self, **kwargs):
        return 0

    async def aclose(self):
        pass


def _agent(provider, **config):
    return Agent(config=AgentConfig(model="m", **config), provider=provider)


async def _stream_kinds_and_text(agent, message="hi"):
    kinds, text = [], []
    async with agent.run_stream(message) as stream:
        async for event in stream:
            kind = getattr(event.type, "value", event.type)
            kinds.append(kind)
            if kind == "text_delta":
                text.append(event.data)
    return kinds, text


def test_the_default_config_retries_a_transient_failure():
    async def _run():
        provider = _FlakyProvider()
        async with _agent(provider, retry=RetryPolicy(backoff_seconds=0)) as agent:
            kinds, text = await _stream_kinds_and_text(agent)
        assert text == ["hello"]
        assert "error" not in kinds
        assert provider.opens == 2, "the first stream failed and must be retried"

    asyncio.run(_run())


def test_retry_also_covers_the_non_streaming_call():
    async def _run():
        provider = _FlakyProvider()
        async with _agent(provider, retry=RetryPolicy(backoff_seconds=0)) as agent:
            result = await agent.run("hi")
        assert result.output == "hello"
        assert provider.creates == 2

    asyncio.run(_run())


def test_retry_can_be_switched_off():
    async def _run():
        provider = _FlakyProvider()
        async with _agent(provider, retry=RetryPolicy(attempts=1)) as agent:
            kinds, _ = await _stream_kinds_and_text(agent)
        assert "error" in kinds
        assert provider.opens == 1, "retry disabled means one attempt only"

    asyncio.run(_run())


def test_a_persistent_failure_still_surfaces():
    async def _run():
        provider = _FlakyProvider(fail_times=99)
        async with _agent(provider, retry=RetryPolicy(backoff_seconds=0)) as agent:
            kinds, _ = await _stream_kinds_and_text(agent)
        assert "error" in kinds, "a persistent failure must still surface"
        assert provider.opens == 2, "bounded by RetryPolicy.attempts"

    asyncio.run(_run())


def test_a_deterministic_failure_is_not_retried():
    async def _run():
        class _Bad(_FlakyProvider):
            def stream(self, **kwargs):
                self.opens += 1

                async def gen():
                    raise _Boom(400)
                    yield  # pragma: no cover

                return gen()

        provider = _Bad(fail_times=99)
        async with _agent(provider, retry=RetryPolicy(backoff_seconds=0)) as agent:
            kinds, _ = await _stream_kinds_and_text(agent)
        assert "error" in kinds
        assert provider.opens == 1, "a 4xx other than 429 must not be retried"

    asyncio.run(_run())
