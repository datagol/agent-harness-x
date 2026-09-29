"""Retry seams (0.4.2): provider classification, Retry-After, capped backoff,
the RETRY hook, FallbackProvider, per-tool ToolRetry, TransientToolError, and
MCP transient classification. One engine loop, several seams."""

from __future__ import annotations

import json
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from harnessx import (
    Agent,
    AgentConfig,
    Fallback,
    FallbackProvider,
    HookEvent,
    PendingTool,
    PermissionLevel,
    ProviderResponse,
    RetryPolicy,
    StreamChunk,
    ToolCall,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolRetry,
    TransientToolError,
)
from harnessx import AgentRegistry
from harnessx.backends import SQLiteBackend
from harnessx.engine import new_state
from harnessx.errors import ConfigurationError, HarnessError
from harnessx.mcp import DEFAULT_MCP_RETRY, MCPConnection, MCPServerConfig, mcp_result_transient
from harnessx.providers import LLMProvider
from harnessx.providers.retry import is_transient_text, retry_after_seconds
from harnessx.runtime import AgentRuntime


class _Boom(Exception):
    def __init__(self, code=503, headers=None, body=None):
        super().__init__(f"status {code}")
        self.code = code
        if headers is not None:
            self.headers = headers
        if body is not None:
            self.body = body


class Scripted(LLMProvider):
    """Returns responses in order; an Exception entry is raised instead. Streams too."""

    name = "scripted"

    def __init__(self, responses=None, *, label=None):
        self.responses = list(responses or [ProviderResponse(text="done")])
        self.calls: list[dict[str, Any]] = []
        self.opens = 0
        self.closed = False
        if label:
            self.name = label

    def _next(self):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._next()

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        self.opens += 1
        provider = self

        async def gen():
            response = provider._next()
            if response.text:
                yield StreamChunk(kind="text_delta", data=response.text)
            yield StreamChunk(kind="response", data=response)

        return gen()

    async def count_tokens(self, **kwargs):
        return 7

    async def aclose(self):
        self.closed = True


def tool_call_then_text(name: str, text: str = "done", **inputs) -> list[ProviderResponse]:
    return [
        ProviderResponse(tool_calls=[ToolCall("c1", name, inputs)], stop_reason="tool_use"),
        ProviderResponse(text=text),
    ]


# Tests never sleep for real: every policy below either sets backoff_seconds=0
# or caps the wait with max_backoff_seconds, and the wait the engine computed is
# asserted from the RETRY hook payload rather than from the clock. (Patching
# asyncio.sleep is not an option: the engine reaches it through the module, so a
# patch would be global and stall the loop.)


def waits_from(records: list[dict[str, Any]]) -> list[float]:
    return [r["wait_seconds"] for r in records]


# ── classification and Retry-After ───────────────────────────────────────────


def test_retry_after_is_read_from_headers_and_bodies():
    class _Headers(dict):
        def get(self, key, default=None):  # httpx-style case-insensitive
            for k, v in self.items():
                if k.lower() == key.lower():
                    return v
            return default

    class _Response:
        headers = _Headers({"Retry-After": "12"})

    with_response = _Boom(429)
    with_response.response = _Response()
    assert retry_after_seconds(with_response) == 12.0
    assert retry_after_seconds(_Boom(429, headers=_Headers({"retry-after": "2.5"}))) == 2.5
    assert retry_after_seconds(_Boom(429, body={"error": {"retry_after": 3}})) == 3.0
    when = datetime.now(timezone.utc) + timedelta(seconds=30)
    seconds = retry_after_seconds(_Boom(429, headers=_Headers({"Retry-After": format_datetime(when)})))
    assert seconds is not None and 25 <= seconds <= 30
    past = datetime.now(timezone.utc) - timedelta(seconds=30)
    assert retry_after_seconds(_Boom(429, headers=_Headers({"Retry-After": format_datetime(past)}))) == 0.0
    assert retry_after_seconds(_Boom(429, headers=_Headers({"Retry-After": "soon"}))) is None
    assert retry_after_seconds(_Boom(500)) is None


def test_transient_text_recognizes_throttling_phrases():
    assert is_transient_text("HTTP 429 Too Many Requests")
    assert is_transient_text("upstream service unavailable")
    assert not is_transient_text("invalid api key")


def test_wait_for_backs_off_honors_retry_after_and_caps():
    policy = RetryPolicy(backoff_seconds=1, max_backoff_seconds=5)
    assert policy.wait_for(1) == 1 and policy.wait_for(2) == 2 and policy.wait_for(3) == 4
    assert policy.wait_for(4) == 5, "capped"
    assert policy.wait_for(1, retry_after=3) == 3, "the server's hint wins over a shorter backoff"
    assert policy.wait_for(1, retry_after=60) == 5, "and is capped too"
    with pytest.raises(ConfigurationError):
        RetryPolicy(max_backoff_seconds=0)


@pytest.mark.asyncio
async def test_model_retry_emits_the_hook_and_honors_the_servers_retry_after():
    # Retry-After says 4s; the cap keeps the test fast and proves both are applied.
    provider = Scripted([_Boom(429, headers={"retry-after": "4"}), ProviderResponse(text="ok")])
    seen: list[dict[str, Any]] = []
    agent = Agent(
        config=AgentConfig(model="m", retry=RetryPolicy(backoff_seconds=0, max_backoff_seconds=0.01)),
        provider=provider,
    )
    agent.hooks.on(HookEvent.RETRY, lambda ctx: seen.append(dict(ctx.data)))
    served: list[str] = []
    agent.hooks.on(HookEvent.LLM_RESPONSE, lambda ctx: served.append(ctx.data["provider"]))
    async with agent:
        assert (await agent.run("hi")).output == "ok"
    assert seen == [{
        "kind": "model", "name": "m", "attempt": 1, "next_attempt": 2,
        "wait_seconds": 0.01, "error": "status 429", "provider": "scripted",
    }], "backoff_seconds=0 alone would wait 0; the server's hint raised it, the cap bounded it"
    assert served == ["scripted"]


@pytest.mark.asyncio
async def test_a_provider_can_reclassify_failures():
    class Picky(Scripted):
        def is_transient(self, exc):
            return "quota" in str(exc)

    class _Quota(Exception):
        pass

    provider = Picky([_Quota("quota exceeded for now"), ProviderResponse(text="ok")])
    async with Agent(config=AgentConfig(model="m", retry=RetryPolicy(backoff_seconds=0)), provider=provider) as agent:
        assert (await agent.run("hi")).output == "ok", "the provider said quota errors are transient"

    strict = Picky([_Boom(503), ProviderResponse(text="never")])
    async with Agent(config=AgentConfig(model="m", retry=RetryPolicy(backoff_seconds=0)), provider=strict) as agent:
        result = await agent.run("hi")
    assert result.failed and len(strict.calls) == 1, "and that a 503 is not, so the engine did not retry"


# ── FallbackProvider ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fallback_moves_to_the_next_member_within_one_call():
    primary = Scripted([_Boom(503)], label="primary")
    backup = Scripted([ProviderResponse(text="from backup")], label="backup")
    chain = FallbackProvider(primary, Fallback(backup, model="backup-model", max_tokens=99))
    response = await chain.create(model="m", messages=[], system=None, tools=[], max_tokens=5)
    assert response.text == "from backup" and chain.last_served == "backup"
    assert backup.calls[0]["model"] == "backup-model" and backup.calls[0]["max_tokens"] == 99
    assert primary.calls[0]["model"] == "m" and primary.calls[0]["max_tokens"] == 5


@pytest.mark.asyncio
async def test_fallback_does_not_fail_over_a_deterministic_error():
    primary = Scripted([_Boom(400)], label="primary")
    backup = Scripted([ProviderResponse(text="never")], label="backup")
    chain = FallbackProvider(primary, backup)
    with pytest.raises(_Boom):
        await chain.create(model="m", messages=[], system=None, tools=[], max_tokens=5)
    assert backup.calls == [] and chain.is_transient(_Boom(400)) is False


@pytest.mark.asyncio
async def test_switch_after_counts_strikes_across_calls_and_success_resets_them():
    primary = Scripted([_Boom(503), _Boom(503), ProviderResponse(text="p"), _Boom(503)], label="primary")
    backup = Scripted([ProviderResponse(text="b1"), ProviderResponse(text="b2")], label="backup")
    chain = FallbackProvider(primary, backup, switch_after=2)
    request = dict(model="m", messages=[], system=None, tools=[], max_tokens=5)
    with pytest.raises(_Boom):
        await chain.create(**request)  # strike 1: the engine would retry
    assert (await chain.create(**request)).text == "b1", "strike 2 switches within the call"
    assert (await chain.create(**request)).text == "p", "primary is tried first again (no cooldown)"
    with pytest.raises(_Boom):
        await chain.create(**request), "strikes were reset by the success"


@pytest.mark.asyncio
async def test_cooldown_keeps_a_struck_member_out_of_rotation():
    primary = Scripted([_Boom(503), ProviderResponse(text="never yet")], label="primary")
    backup = Scripted([ProviderResponse(text="b1"), ProviderResponse(text="b2")], label="backup")
    chain = FallbackProvider(primary, backup, cooldown_seconds=60)
    request = dict(model="m", messages=[], system=None, tools=[], max_tokens=5)
    assert (await chain.create(**request)).text == "b1"
    assert (await chain.create(**request)).text == "b2", "primary is cooling down"
    assert len(primary.calls) == 1


@pytest.mark.asyncio
async def test_stream_fails_over_only_before_the_first_chunk():
    primary = Scripted([_Boom(503)], label="primary")
    backup = Scripted([ProviderResponse(text="streamed")], label="backup")
    chain = FallbackProvider(primary, backup)
    chunks = [c async for c in chain.stream(model="m", messages=[], system=None, tools=[], max_tokens=5)]
    assert [c.kind for c in chunks] == ["text_delta", "response"] and chain.last_served == "backup"

    class _Midway(Scripted):
        def stream(self, **kwargs):
            self.calls.append(kwargs)

            async def gen():
                yield StreamChunk(kind="text_delta", data="partial")
                raise _Boom(503)

            return gen()

    chain = FallbackProvider(_Midway(label="primary"), Scripted([ProviderResponse(text="never")], label="backup"))
    delivered = []
    with pytest.raises(_Boom):
        async for chunk in chain.stream(model="m", messages=[], system=None, tools=[], max_tokens=5):
            delivered.append(chunk.data)
    assert delivered == ["partial"], "output already reached the caller; the engine's ATTEMPT_RESET handles it"


@pytest.mark.asyncio
async def test_fallback_through_an_agent_reports_the_serving_member():
    primary = Scripted([_Boom(503), ProviderResponse(text="p")], label="primary")
    backup = Scripted([ProviderResponse(text="b")], label="backup")
    served: list[str] = []
    agent = Agent(config=AgentConfig(model="m", retry=RetryPolicy(attempts=1)), provider=FallbackProvider(primary, backup))
    agent.hooks.on(HookEvent.LLM_RESPONSE, lambda ctx: served.append(ctx.data["provider"]))
    async with agent:
        assert (await agent.run("one")).output == "b"
        assert (await agent.run("two")).output == "p"
    assert served == ["backup", "primary"]


@pytest.mark.asyncio
async def test_fallback_owns_members_built_from_names_only(monkeypatch):
    built = Scripted(label="built")
    monkeypatch.setattr("harnessx.providers.fallback.make_provider", lambda name, **kw: built)
    injected = Scripted(label="injected")
    chain = FallbackProvider("anthropic", injected)
    assert chain.labels == ("anthropic", "injected") and chain.members[0].provider == "anthropic"
    await chain.aclose()
    assert built.closed and not injected.closed
    assert await chain.count_tokens(model="m", messages=[], system=None, tools=[]) == 7


def test_fallback_validates_its_arguments():
    with pytest.raises(ValueError):
        FallbackProvider(Scripted(), switch_after=0)
    with pytest.raises(ValueError):
        Fallback(Scripted(), max_tokens=0)
    with pytest.raises(TypeError):
        Fallback(object())  # type: ignore[arg-type]


# ── ToolRetry and TransientToolError ─────────────────────────────────────────


def test_tool_retry_validates_and_round_trips():
    policy = ToolRetry(attempts=4, backoff_seconds=1, max_backoff_seconds=3, retry_error_results=True)
    assert ToolRetry.from_dict(policy.to_dict()) == policy
    assert ToolRetry.from_dict(None) == ToolRetry() and ToolRetry.from_dict({"attempts": 2, "unknown": 1}).attempts == 2
    assert policy.wait_for(1) == 1 and policy.wait_for(2) == 2 and policy.wait_for(3) == 3
    with pytest.raises(ConfigurationError):
        ToolRetry(attempts=0)
    assert issubclass(TransientToolError, HarnessError)


def test_tool_policy_retry_is_adopted_and_explicit_registrations_win():
    registry = ToolRegistry()

    @registry.register(permission=PermissionLevel.ALLOW)
    def inherits() -> str:
        return "x"

    @registry.register(permission=PermissionLevel.ALLOW, retry=ToolRetry(attempts=5))
    def explicit() -> str:
        return "y"

    assert registry.get_tool("inherits").retry is None and registry.policy.retry is None
    agent = Agent(config=AgentConfig(tools=ToolPolicy(retry=ToolRetry(attempts=2))), tools=registry, provider=Scripted())
    assert agent.tools.get_tool("inherits").retry == ToolRetry(attempts=2)
    assert agent.tools.get_tool("explicit").retry == ToolRetry(attempts=5)
    assert agent.tools.policy.retry == ToolRetry(attempts=2)
    assert ToolRegistry(retry=ToolRetry(attempts=9)).policy.retry == ToolRetry(attempts=9)


@pytest.mark.asyncio
async def test_a_safe_tool_raising_transient_error_is_retried_and_journaled():
    tries = 0

    def flaky() -> str:
        nonlocal tries
        tries += 1
        if tries < 3:
            raise TransientToolError(f"429 from upstream (try {tries})")
        return "third time lucky"

    provider = Scripted(tool_call_then_text("flaky"))
    agent = Agent(provider=provider)
    agent.tools.register_tool(
        flaky, permission=PermissionLevel.ALLOW, replay_policy="safe",
        retry=ToolRetry(attempts=3, backoff_seconds=1, max_backoff_seconds=0.01),
    )
    retries: list[dict[str, Any]] = []
    agent.hooks.on(HookEvent.RETRY, lambda ctx: retries.append(dict(ctx.data)))
    async with agent:
        result = await agent.run("go")
    assert result.ok and tries == 3
    assert waits_from(retries) == [0.01, 0.01], "exponential backoff, bounded by the cap"
    assert [(r["kind"], r["name"], r["attempt"], r["next_attempt"]) for r in retries] == [("tool", "flaky", 1, 2), ("tool", "flaky", 2, 3)]
    assert provider.calls[1]["messages"][-1]["content"][0]["content"] == "third time lucky"


@pytest.mark.asyncio
async def test_exhausted_attempts_hand_the_failure_to_the_model():
    def always() -> str:
        raise TransientToolError("still throttled")

    provider = Scripted(tool_call_then_text("always"))
    agent = Agent(provider=provider)
    agent.tools.register_tool(always, permission=PermissionLevel.ALLOW, replay_policy="safe", retry=ToolRetry(attempts=2, backoff_seconds=0))
    async with agent:
        result = await agent.run("go")
    assert result.ok, "the run continues; the model saw the error result"
    block = provider.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] and "still throttled" in block["content"]


@pytest.mark.asyncio
async def test_a_manual_tool_is_never_retried_but_a_declared_failure_is_not_a_recovery_stop():
    calls = 0

    def manual() -> str:
        nonlocal calls
        calls += 1
        raise TransientToolError("rate limited")

    provider = Scripted(tool_call_then_text("manual"))
    agent = Agent(provider=provider)
    agent.tools.register_tool(manual, permission=PermissionLevel.ALLOW, retry=ToolRetry(attempts=3))
    async with agent:
        result = await agent.run("go")
    assert result.ok and calls == 1 and not result.pending
    assert provider.calls[1]["messages"][-1]["content"][0]["is_error"]


@pytest.mark.asyncio
async def test_retry_if_result_treats_a_successful_result_as_transient():
    bodies = iter([json.dumps({"error": "rate limited", "status": 429}), "real data"])

    def search() -> str:
        return next(bodies)

    provider = Scripted(tool_call_then_text("search"))
    agent = Agent(provider=provider)
    agent.tools.register_tool(
        search, permission=PermissionLevel.ALLOW, replay_policy="idempotent",
        retry=ToolRetry(attempts=2, backoff_seconds=0), retry_if_result=mcp_result_transient,
    )
    async with agent:
        assert (await agent.run("go")).ok
    assert provider.calls[1]["messages"][-1]["content"][0]["content"] == "real data"


@pytest.mark.asyncio
async def test_retry_error_results_retries_error_results_that_read_transient():
    outcomes = iter([ToolResult("c1", "HTTP 503 service unavailable", is_error=True), ToolResult("c1", "fine")])

    def fetch() -> ToolResult:
        return next(outcomes)

    provider = Scripted(tool_call_then_text("fetch"))
    agent = Agent(provider=provider)
    agent.tools.register_tool(
        fetch, permission=PermissionLevel.ALLOW, replay_policy="safe",
        retry=ToolRetry(attempts=2, backoff_seconds=0, retry_error_results=True),
    )
    async with agent:
        assert (await agent.run("go")).ok
    assert provider.calls[1]["messages"][-1]["content"][0]["content"] == "fine"


@pytest.mark.asyncio
async def test_standalone_execute_applies_the_tools_retry():
    tries = 0

    async def flaky(**_):
        nonlocal tries
        tries += 1
        if tries < 2:
            raise TransientToolError("busy")
        return "ok"

    registry = ToolRegistry()
    registry.register_with_schema(
        "t", "t", {"type": "object", "properties": {}}, flaky, permission=PermissionLevel.ALLOW,
        replay_policy="safe", retry=ToolRetry(attempts=3, backoff_seconds=0),
    )
    assert (await registry.execute(ToolCall("1", "t", {}))).content == "ok" and tries == 2

    registry.register_with_schema(
        "m", "m", {"type": "object", "properties": {}}, flaky, permission=PermissionLevel.ALLOW,
        retry=ToolRetry(attempts=3, backoff_seconds=0),
    )
    tries = 0
    result = await registry.execute(ToolCall("2", "m", {}))
    assert result.is_error and "busy" in result.content and tries == 1, "manual: one try, error to the caller"


def test_pending_tool_carries_the_retry_policy_when_present():
    entry = {
        "call": {"id": "t", "name": "write", "input": {}}, "status": "approval", "attempt": 0,
        "execution_key": "r:1:t", "policy": "manual", "concurrent": False, "timeout": 5,
        "retry": {"attempts": 2, "backoff_seconds": 0.5, "max_backoff_seconds": 30.0, "retry_error_results": False},
    }
    pending = PendingTool.from_dict(entry)
    assert pending.retry == ToolRetry(attempts=2) and pending.to_dict()["retry"] == entry["retry"]
    legacy = PendingTool.from_dict({k: v for k, v in entry.items() if k != "retry"})
    assert legacy.retry is None and "retry" not in legacy.to_dict()


@pytest.mark.asyncio
async def test_a_prepared_tool_carries_its_retry_policy_on_the_wire(tmp_path):
    # An ASK tool pauses the run, so the persisted entry is readable as a PendingTool.
    provider = Scripted(tool_call_then_text("ask_first"))
    agent = Agent(provider=provider)
    agent.tools.register_tool(
        lambda: "done", name="ask_first", description="asks", permission=PermissionLevel.ASK,
        replay_policy="safe", retry=ToolRetry(attempts=4, backoff_seconds=0),
    )
    backend = await SQLiteBackend.connect(tmp_path / "r.db")
    async with backend, AgentRuntime(agent, backend=backend) as runtime:
        result = await runtime.run("go")
        assert result.needs_input
        assert result.pending[0].retry == ToolRetry(attempts=4, backoff_seconds=0)
        saved = (await backend.get_run(result.run_id))["tools"][0]
        assert saved["retry"] == {
            "attempts": 4, "backoff_seconds": 0, "max_backoff_seconds": 30.0, "retry_error_results": False,
        }


@pytest.mark.asyncio
async def test_durable_runs_retry_a_transient_tool_and_old_states_default(tmp_path):
    tries = 0

    def flaky() -> str:
        nonlocal tries
        tries += 1
        if tries < 2:
            raise TransientToolError("busy")
        return "ok"

    provider = Scripted(tool_call_then_text("flaky"))
    agent = Agent(provider=provider)
    agent.tools.register_tool(flaky, permission=PermissionLevel.ALLOW, replay_policy="safe", retry=ToolRetry(attempts=2, backoff_seconds=0))
    backend = await SQLiteBackend.connect(tmp_path / "r.db")
    async with backend, AgentRuntime(agent, backend=backend) as runtime:
        result = await runtime.run("go")
        assert result.ok and tries == 2


@pytest.mark.asyncio
async def test_a_run_state_written_before_0_4_2_still_retries(tmp_path):
    # No "retry" key on the entry: the engine falls back to the default policy
    # rather than failing on a missing key.
    tries = 0
    registry = AgentRegistry()

    def factory():
        agent = Agent(provider=Scripted([ProviderResponse(text="resumed")]))

        @agent.tools.register(permission=PermissionLevel.ALLOW, replay_policy="safe")
        def flaky() -> str:
            nonlocal tries
            tries += 1
            if tries < 2:
                raise TransientToolError("busy")
            return "ok"

        return agent

    ref = registry.register("flaky-agent", factory)
    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(ref, registry=registry, backend=store)
    sid = await runtime.start()
    state = new_state(runtime.agent, "again", durable=True)
    state.update(phase="tools", tools=[{
        "call": {"id": "t", "name": "flaky", "input": {}}, "status": "prepared", "attempt": 0,
        "execution_key": state["run_id"] + ":1:t", "policy": "safe", "concurrent": True, "timeout": 5,
    }])
    lease = await store.claim(sid)
    await store.create_run(sid, "r2", state, lease)
    await store.save_run(state, [], lease)
    await store.release(sid, lease)
    handle = await runtime.resume(sid)
    assert (await handle.result()).ok and tries == 2
    await runtime.stop()
    await store.aclose()


# ── MCP ───────────────────────────────────────────────────────────────────────


def test_mcp_result_transient_reads_status_codes_out_of_json_bodies():
    assert mcp_result_transient(ToolResult("c", json.dumps({"error": "Too Many Requests", "status": 429})))
    assert mcp_result_transient(ToolResult("c", json.dumps({"error": {"code": 503, "message": "down"}})))
    assert mcp_result_transient(ToolResult("c", json.dumps({"error": "rate limit exceeded"})))
    assert not mcp_result_transient(ToolResult("c", json.dumps({"results": [{"status": 503}]})))
    assert not mcp_result_transient(ToolResult("c", json.dumps({"error": "invalid query", "status": 400})))
    assert not mcp_result_transient(ToolResult("c", "The server returned 429 results in total"))
    assert mcp_result_transient(ToolResult("c", "upstream timeout", is_error=True))
    assert not mcp_result_transient(ToolResult("c", "x" * 3000, is_error=True))


@pytest.mark.asyncio
async def test_mcp_call_tool_raises_transient_error_for_throttling():
    class _Block:
        def __init__(self, text):
            self.text = text

    class _Result:
        def __init__(self, text, is_error):
            self.content = [_Block(text)]
            self.isError = is_error

    class _Session:
        def __init__(self, result):
            self.result = result

        async def call_tool(self, name, arguments):
            return self.result

    conn = MCPConnection(MCPServerConfig.stdio("s", "cmd"))
    conn.session = _Session(_Result("429 rate limit exceeded", True))
    with pytest.raises(TransientToolError):
        await conn.call_tool("t", {})
    conn.session = _Session(_Result("no such tool", True))
    with pytest.raises(RuntimeError):
        await conn.call_tool("t", {})
    conn.session = _Session(_Result("fine", False))
    assert await conn.call_tool("t", {}) == "fine"


def test_mcp_server_config_carries_replay_policy_and_retry():
    config = MCPServerConfig.http("search", "http://x/mcp", replay_policy="idempotent", retry=ToolRetry(attempts=5))
    assert config.replay_policy == "idempotent" and config.retry == ToolRetry(attempts=5)
    assert MCPServerConfig.stdio("s", "cmd").replay_policy == "manual"
    with pytest.raises(ConfigurationError):
        MCPServerConfig.stdio("s", "cmd", replay_policy="sometimes")
    assert DEFAULT_MCP_RETRY.attempts == 3 and DEFAULT_MCP_RETRY.retry_error_results


def test_bridged_mcp_tools_get_the_servers_policy():
    from harnessx.mcp import MCPManager, MCPToolInfo

    manager = MCPManager()
    config = MCPServerConfig.http("search", "http://x/mcp", replay_policy="safe", permission=PermissionLevel.ALLOW)
    conn = MCPConnection(config)
    conn._tools = [MCPToolInfo("search", "find", "find things", {"type": "object", "properties": {}})]
    manager._connections["search"] = conn
    registry = ToolRegistry()
    assert manager.register_tools(registry) == ["search_find"]
    tool = registry.get_tool("search_find")
    assert tool.replay_policy == "safe" and tool.retry == DEFAULT_MCP_RETRY and tool.retry_if_result is mcp_result_transient


# ── SDK clients ───────────────────────────────────────────────────────────────


def test_owned_sdk_clients_do_not_retry_on_their_own(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    from harnessx.providers.anthropic import AnthropicProvider

    assert AnthropicProvider().client.max_retries == 0
    from anthropic import AsyncAnthropic

    injected = AsyncAnthropic(max_retries=4)
    assert AnthropicProvider(client=injected).client.max_retries == 4
    try:
        from harnessx.providers.openai import OpenAIProvider
        from harnessx.providers.openrouter import OpenRouterProvider
        from harnessx.providers.azure_openai import AzureOpenAIProvider
    except ImportError:
        pytest.skip("openai SDK not installed")
    assert OpenAIProvider().client.max_retries == 0
    assert OpenRouterProvider(api_key="k").client.max_retries == 0
    assert OpenRouterProvider(api_key="k", max_retries=3).client.max_retries == 3
    azure = AzureOpenAIProvider(azure_endpoint="https://x.openai.azure.com", api_version="2024-06-01", api_key="k")
    assert azure.client.max_retries == 0
