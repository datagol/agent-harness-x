"""Per-run framework overhead: no needless network calls, and a time budget.

The 2026-10-10 benchmarks measured 0.2-0.6 s of HarnessX overhead per run
against 0.03-0.1 s for other stacks. Most of it was HTTP: a token count before
every model call once the history had six messages, and a Models API lookup
per Agent. These tests pin both down with the real SDK over a mock transport.
"""

import collections
import json
import os
import time

import anthropic
import pytest

if int(anthropic.__version__.split(".")[0]) >= 1:
    import httpx2 as httpx
else:
    import httpx
from anthropic import AsyncAnthropic

from harnessx import Agent, AgentConfig, PermissionLevel
from harnessx.memory import ConversationMemory
from harnessx.providers.anthropic import AnthropicProvider
from tests.test_anthropic_provider import message, sse_message

MODEL = "claude-sonnet-4-6"


@pytest.fixture(autouse=True)
def _fresh_limits():
    AnthropicProvider.clear_model_limits_cache()
    yield
    AnthropicProvider.clear_model_limits_cache()


def tool_reply(i):
    return message(
        [{"type": "tool_use", "id": f"t{i}", "name": "lookup", "input": {"i": i}, "caller": {"type": "direct"}}],
        stop_reason="tool_use",
    )


def make_client(paths, *, input_tokens=8, base_url="https://anthropic.invalid"):
    # `input_tokens` is what each reply reports in usage; an exact count is
    # always small, so a run that does count still has room and never condenses.
    replies = [tool_reply(i) for i in range(5)] + [message([{"type": "text", "text": "done"}])]
    for r in replies:
        r["usage"] = {"input_tokens": input_tokens, "output_tokens": 10}
    state = {"n": 0}

    def handle(request):
        paths[request.url.path] += 1
        if request.url.path.endswith("/count_tokens"):
            return httpx.Response(200, json={"input_tokens": 800})
        if request.url.path.startswith("/v1/models"):
            return httpx.Response(200, json={
                "id": MODEL, "type": "model", "display_name": "x", "created_at": "2025-01-01T00:00:00Z",
                "max_input_tokens": 200_000, "max_tokens": 64_000,
            })
        body = json.loads(request.content)
        reply = replies[state["n"]]
        state["n"] += 1
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_message(reply))
        return httpx.Response(200, json=reply)

    return AsyncAnthropic(
        api_key="k", base_url=base_url, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )


async def five_call_run(client, *, streaming=False):
    agent = Agent(config=AgentConfig(model=MODEL), provider=AnthropicProvider(client=client))
    agent.tools.register_tool(lambda i: f"r{i}" * 50, name="lookup", permission=PermissionLevel.ALLOW)
    async with agent:
        if streaming:
            async with agent.run_stream("go") as stream:
                [e async for e in stream]
                return await stream.result()
        return await agent.run("go")


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_a_five_call_run_with_room_never_counts_tokens(streaming):
    paths = collections.Counter()
    async with make_client(paths) as client:
        result = await five_call_run(client, streaming=streaming)
    assert result.ok
    assert paths["/v1/messages"] == 6
    assert paths["/v1/messages/count_tokens"] == 0


@pytest.mark.asyncio
async def test_a_request_near_the_window_is_still_counted():
    paths = collections.Counter()
    # The provider reports 150K tokens per request: above 60% of the room,
    # so the estimate is not trusted and the count goes over the network.
    async with make_client(paths, input_tokens=150_000) as client:
        result = await five_call_run(client)
    assert result.ok
    assert paths["/v1/messages/count_tokens"] >= 1


@pytest.mark.asyncio
async def test_model_limits_are_fetched_once_per_process_and_endpoint():
    paths = collections.Counter()
    async with make_client(paths) as one, make_client(paths) as two:
        assert (await five_call_run(one)).ok
        assert (await five_call_run(two)).ok
    assert sum(n for p, n in paths.items() if p.startswith("/v1/models")) == 1

    other = collections.Counter()
    async with make_client(other, base_url="https://gateway.invalid") as three:
        assert (await five_call_run(three)).ok
    assert sum(n for p, n in other.items() if p.startswith("/v1/models")) == 1


@pytest.mark.asyncio
async def test_a_failed_limits_lookup_is_not_shared():
    calls = collections.Counter()

    def handle(request):
        calls[request.url.path] += 1
        return httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "x"}})

    for _ in range(2):
        async with AsyncAnthropic(
            api_key="k", base_url="https://anthropic.invalid", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ) as client:
            assert await AnthropicProvider(client=client).model_limits("claude-future-9") is None
    assert calls["/v1/models/claude-future-9"] == 2


class CountingProvider:
    def __init__(self):
        self.counted = 0

    async def count_tokens(self, **_):
        self.counted += 1
        return 1_000


@pytest.mark.asyncio
async def test_needs_condensing_trusts_a_measured_size_only_with_clear_room():
    memory = ConversationMemory()
    for i in range(4):
        memory.add_user_message(f"question {i}")
        memory.add_assistant_message([{"type": "text", "text": f"answer {i}"}])
    provider = CountingProvider()

    assert not await memory.needs_condensing(provider, MODEL, "", [], max_context_tokens=100_000, measured=10_000)
    assert provider.counted == 0
    assert memory.last_prompt_tokens and memory.last_prompt_tokens >= 10_000

    await memory.needs_condensing(provider, MODEL, "", [], max_context_tokens=100_000, measured=70_000)
    assert provider.counted == 1

    await memory.needs_condensing(provider, MODEL, "", [], max_context_tokens=100_000)
    assert provider.counted == 2


@pytest.mark.asyncio
async def test_framework_overhead_budget():
    """A five-call run on a zero-latency mock stays well under a quarter second."""
    paths = collections.Counter()
    async with make_client(paths) as client:
        await five_call_run(client)  # warm imports and caches
    times = []
    for _ in range(3):
        async with make_client(collections.Counter()) as client:
            started = time.perf_counter()
            assert (await five_call_run(client)).ok
            times.append(time.perf_counter() - started)
    budget = float(os.environ.get("HARNESSX_OVERHEAD_BUDGET", "0.25"))
    assert min(times) < budget, f"five-call run took {min(times):.3f}s of framework time"


def limits_client(paths, release):
    """A Models API that answers only once ``release`` is set."""
    async def handle(request):
        paths[request.url.path] += 1
        await release.wait()
        return httpx.Response(200, json={
            "id": "m", "type": "model", "display_name": "x", "created_at": "2025-01-01T00:00:00Z",
            "max_input_tokens": 300_000, "max_tokens": 40_000,
        })

    return AsyncAnthropic(api_key="k", base_url="https://limits.invalid", max_retries=0,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))


@pytest.mark.asyncio
async def test_a_known_model_does_not_wait_for_the_models_api():
    """The lookup took 260-320 ms before the first call of every fresh process,
    which was all of HarnessX's measured overhead. The table answers until the
    API does."""
    import asyncio

    from harnessx.models import ModelLimits, resolve_model_limits

    paths, release = collections.Counter(), asyncio.Event()
    async with limits_client(paths, release) as client:
        provider = AnthropicProvider(client=client)
        first = await asyncio.wait_for(resolve_model_limits(provider, MODEL), 0.5)
        assert first.context_window == 1_000_000, "the built-in table, without waiting"
        release.set()
        await asyncio.sleep(0.05)
        assert await provider.model_limits(MODEL) == ModelLimits(300_000, 40_000), "the API's figures once known"
        await provider.aclose()
    assert sum(paths.values()) == 1


@pytest.mark.asyncio
async def test_an_unknown_model_still_waits_for_the_models_api():
    import asyncio

    from harnessx.models import ModelLimits

    paths, release = collections.Counter(), asyncio.Event()
    async with limits_client(paths, release) as client:
        provider = AnthropicProvider(client=client)
        asking = asyncio.ensure_future(provider.model_limits("claude-future-9"))
        await asyncio.sleep(0.05)
        assert not asking.done(), "no table entry to fall back on"
        release.set()
        assert await asking == ModelLimits(300_000, 40_000)


@pytest.mark.asyncio
async def test_closing_the_provider_stops_a_lookup_still_running():
    import asyncio

    paths, release = collections.Counter(), asyncio.Event()
    async with limits_client(paths, release) as client:
        provider = AnthropicProvider(client=client)
        assert await provider.model_limits(MODEL) is None
        [asking] = provider._asking.values()
        await provider.aclose()
        await asyncio.sleep(0)
        assert asking.cancelled()

