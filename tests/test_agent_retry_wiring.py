"""Retry is on by default, and reaches the agent — not just the helper.

A helper nothing calls is the failure mode this test exists to prevent.
"""

import asyncio

import pytest

from datagol_agent_harness.streaming import StreamingAgent
from datagol_agent_harness.types import (
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
    """Fails the first stream, succeeds on the second."""

    name = "flaky"

    def __init__(self, fail_times=1):
        self.opens = 0
        self.fail_times = fail_times

    async def create(self, **kwargs):  # pragma: no cover - streaming path only
        raise NotImplementedError

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

    def format_tools(self, tools):
        return tools


def test_the_default_config_retries_a_transient_failure():
    async def _run():
        provider = _FlakyProvider()
        agent = StreamingAgent(
            config=AgentConfig(provider="flaky", model="m", llm_retry_backoff_seconds=0),
            provider=provider,
        )
        text = [e.data async for e in agent.run_stream("hi")
                if getattr(e.type, "value", e.type) == "text_delta"]
        assert text == ["hello"]
        assert provider.opens == 2, "the first stream failed and must be retried"

    asyncio.run(_run())


def test_retry_can_be_switched_off():
    async def _run():
        provider = _FlakyProvider()
        agent = StreamingAgent(
            config=AgentConfig(provider="flaky", model="m", llm_max_attempts=1),
            provider=provider,
        )
        kinds = [getattr(e.type, "value", e.type) async for e in agent.run_stream("hi")]
        assert "error" in kinds
        assert provider.opens == 1, "retry disabled means one attempt only"

    asyncio.run(_run())


def test_a_persistent_failure_still_surfaces():
    async def _run():
        provider = _FlakyProvider(fail_times=99)
        agent = StreamingAgent(
            config=AgentConfig(provider="flaky", model="m", llm_retry_backoff_seconds=0),
            provider=provider,
        )
        kinds = [getattr(e.type, "value", e.type) async for e in agent.run_stream("hi")]
        assert "error" in kinds, "a persistent failure must still surface"
        assert provider.opens == 2, "bounded by llm_max_attempts"

    asyncio.run(_run())
