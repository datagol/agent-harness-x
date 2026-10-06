"""Regression cases from the reliability review, at execution boundaries."""
import asyncio
import copy
import json
import os
import shlex
import sys
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest

from harnessx import Agent, AgentConfig, Limits, ProviderResponse, RetryPolicy, TokenUsage, ToolCall
from harnessx.engine import await_with_notices, new_state, restore, snapshot
from harnessx.errors import IncompleteStreamError
from harnessx.models import ModelLimits, learn_model_limits, resolve_model_limits
from harnessx.providers import LLMProvider
from harnessx.providers.fallback import FallbackProvider


class Scripted(LLMProvider):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def create(self, **request):
        self.calls.append(copy.deepcopy(request))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response

    async def count_tokens(self, **request):
        return 0


def config(**kw):
    return AgentConfig(model="fixture", planning=False, prompt_cache=None,
                       retry=RetryPolicy(attempts=1, backoff_seconds=0), **kw)


@pytest.mark.asyncio
@pytest.mark.parametrize("name,registered,executes", [
    ("disable_user_account", "enable_user_account", False),
    ("serch_web", "search_web", False),
    ("search-web", "search_web", True),
    ("searchWeb", "search_web", True),
    ("functions.search_web", "search_web", True),
    ("tools.search_web", "search_web", True),
    ("default_api:search_web", "search_web", True),
    ("Search_Web", "search_web", True),
])
async def test_tool_alias_policy_at_dispatch(name, registered, executes):
    ran = []
    provider = Scripted([ProviderResponse(tool_calls=[ToolCall("t", name, {})]), ProviderResponse(text="done")])
    async with Agent(config=config(), provider=provider) as agent:
        agent.tools.register_tool(lambda: ran.append(True), name=registered)
        result = await agent.run("go")
    assert result.ok and bool(ran) == executes
    if not executes:
        content = provider.calls[-1]["messages"][-1]["content"][0]
        assert content["is_error"] and "Did you mean" in content["content"]
        assert registered in content["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asked,expected", [("do_it", ["do_it"]), ("do-it", ["do-it"]), ("doIt", []), ("DO_IT", [])])
async def test_alias_collisions_only_allow_exact_names(asked, expected):
    ran = []
    provider = Scripted([ProviderResponse(tool_calls=[ToolCall("t", asked, {})]), ProviderResponse(text="done")])
    async with Agent(config=config(), provider=provider) as agent:
        agent.tools.register_tool(lambda: ran.append("do_it"), name="do_it")
        agent.tools.register_tool(lambda: ran.append("do-it"), name="do-it")
        await agent.run("go")
    assert ran == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("arguments", ['{"value":"abc"}', '{"value":"abc","label":"custom'])
async def test_real_anthropic_sdk_incomplete_stream_never_dispatches(monkeypatch, implicit, arguments):
    from test_anthropic_provider import AsyncAnthropic, httpx, message
    from harnessx.providers.anthropic import AnthropicProvider
    events = [
        {"type": "message_start", "message": message([], stop_reason=None)},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "t", "name": "record_value", "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {
            "type": "input_json_delta", "partial_json": arguments}},
    ]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    def handle(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    sdk = AsyncAnthropic(api_key="fixture", max_retries=0,
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    provider = AnthropicProvider(client=sdk)
    async def limits(model):
        return ModelLimits(100000, 10000)
    monkeypatch.setattr(provider, "model_limits", limits)
    if implicit:
        async def require_stream(**kw):
            raise ValueError("Streaming is required")
        monkeypatch.setattr(sdk.messages, "create", require_stream)
    ran = []
    async with Agent(config=config(), provider=provider) as agent:
        def record_value(value: str, label: str = "default"):
            ran.append((value, label))
        agent.tools.register_tool(record_value)
        if implicit:
            result = await agent.run("go")
        else:
            async with agent.run_stream("go") as stream:
                async for _ in stream:
                    pass
                result = await stream.result()
    await sdk.close()
    assert result.failed and result.error["type"] == "IncompleteStreamError"
    assert not ran


@pytest.mark.asyncio
@pytest.mark.parametrize("compat,tool,allowed", [(False, False, False), (True, False, True), (True, True, False)])
async def test_openai_completion_marker_policy_and_close(compat, tool, allowed):
    from harnessx.providers.openai import OpenAIProvider
    closed = []
    async def chunks():
        try:
            calls = [NS(index=0, id="t", function=NS(name="t", arguments="{}"))] if tool else None
            yield NS(usage=None, choices=[NS(finish_reason=None, delta=NS(content="hello", tool_calls=calls, refusal=None))])
        finally:
            closed.append(True)
    async def create(**kw):
        return chunks()
    provider = OpenAIProvider(client=NS(chat=NS(completions=NS(create=create))), allow_missing_finish_reason_for_text=compat)
    request = dict(model="fixture", messages=[], tools=[], system=None, max_tokens=100)
    if allowed:
        results = [c async for c in provider.stream(**request)]
        assert results[-1].data.text == "hello"
    else:
        with pytest.raises(IncompleteStreamError):
            _ = [c async for c in provider.stream(**request)]
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish,blocked,allowed", [(None, False, False), ("STOP", False, True), (None, True, True)])
async def test_gemini_requires_completion_or_explicit_block(finish, blocked, allowed):
    from harnessx.providers.gemini import GeminiProvider
    closed = []
    async def chunks():
        try:
            yield NS(usage_metadata=None, prompt_feedback=NS(block_reason="SAFETY" if blocked else None),
                     candidates=[] if blocked else [NS(content=NS(parts=[NS(text="hi", thought=False, function_call=None)]), finish_reason=finish)])
        finally:
            closed.append(True)
    async def generate_content_stream(**kw):
        return chunks()
    provider = GeminiProvider(client=NS(aio=NS(models=NS(generate_content_stream=generate_content_stream))))
    request = dict(model="fixture", messages=[], tools=[], system=None, max_tokens=100)
    if allowed:
        chunks = [c async for c in provider.stream(**request)]
        assert chunks[-1].data.stop_reason == ("safety" if blocked else "end_turn")
    else:
        with pytest.raises(IncompleteStreamError):
            _ = [c async for c in provider.stream(**request)]
    assert closed == [True]


@pytest.mark.asyncio
async def test_native_blocks_survive_pause_memory_and_snapshot(tmp_path):
    from test_anthropic_provider import AnthropicMessage, message
    from harnessx.providers.anthropic import _from_anthropic_response, _messages_for_request
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "empty-signed"},
        {"type": "thinking", "thinking": "first", "signature": "sig-a"},
        {"type": "text", "text": "progress"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "thinking", "thinking": "second", "signature": "sig-b"},
        {"type": "server_tool_use", "id": "server", "name": "web_search", "input": {"query": "q"}},
    ]
    response = _from_anthropic_response(AnthropicMessage(**message(blocks, "pause_turn")))
    expected = [ProviderResponse._block_dict(b) for b in response.content]
    provider = Scripted([response, ProviderResponse(text="done")])
    async with Agent(config=config(), provider=provider) as agent:
        assert (await agent.run("go")).ok
        stored = provider.calls[1]["messages"][-1]
        assert stored["content"] == expected
        translated = _messages_for_request([stored])[0]["content"]
        assert [b["type"] for b in translated] == [b["type"] for b in blocks]
        assert [(b["thinking"], b["signature"]) for b in translated if b["type"] == "thinking"] == [("", "empty-signed"), ("first", "sig-a"), ("second", "sig-b")]
        sid = await agent.save_session(str(tmp_path))
    async with await Agent.load_session(sid, str(tmp_path), provider=Scripted([])) as restored:
        assert restored.memory.get_messages()[1]["content"] == expected


@pytest.mark.asyncio
async def test_native_pending_turn_pins_fallback_and_restores():
    response = ProviderResponse(stop_reason="pause_turn", content=[{
        "type": "provider", "provider": "anthropic", "data": {"type": "redacted_thinking", "data": "opaque"}}])
    primary = Scripted([response, ConnectionError("down")])
    fallback = Scripted([ProviderResponse(text="must not run")])
    chain = FallbackProvider(primary, fallbacks=[fallback])
    request = dict(model="fixture", messages=[], tools=[], system=None, max_tokens=100)
    await chain.create(**request)
    saved = json.loads(json.dumps(chain.export_state()))
    chain.restore_state(saved)
    with pytest.raises(ConnectionError):
        await chain.create(**request)
    assert not fallback.calls


@pytest.mark.asyncio
async def test_cancelling_progress_wait_joins_owned_operation():
    from harnessx import ProgressPolicy
    started, stopped = asyncio.Event(), asyncio.Event()
    async def operation():
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()
    async def emit(*args):
        pass
    task = asyncio.create_task(await_with_notices(operation(), ProgressPolicy(first_after_seconds=.01), emit, on="tool"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_bash_kills_descendants_that_keep_pipes_open(tmp_path, cancel):
    from harnessx.builtin.bash import _run_bash
    pidfile = tmp_path / "child.pid"
    script = f"import os,time,pathlib; pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} & wait"
    task = asyncio.create_task(_run_bash(command, timeout=60 if cancel else .3))
    for _ in range(100):
        if pidfile.exists():
            break
        await asyncio.sleep(.01)
    assert pidfile.exists()
    pid = int(pidfile.read_text())
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    else:
        assert "timed out" in await asyncio.wait_for(task, 2)
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(.01)
    else:
        pytest.fail(f"descendant {pid} survived command cleanup")


@pytest.mark.asyncio
async def test_provider_limits_are_scoped_and_durable():
    a, b = Scripted([]), Scripted([])
    learn_model_limits("same-model", provider=a, max_output=2048)
    assert (await resolve_model_limits(a, "same-model")).max_output == 2048
    assert (await resolve_model_limits(b, "same-model")).max_output != 2048
    assert (await resolve_model_limits(a, "same-model-latest")).max_output != 2048
    async with Agent(config=config(), provider=a) as first:
        saved = {**new_state(first, "go"), **await snapshot(first)}
    async with Agent(config=config(), provider=b) as second:
        await restore(second, saved)
        assert (await resolve_model_limits(b, "same-model")).max_output == 2048


@pytest.mark.asyncio
async def test_fallback_repaired_cap_beats_configured_budget():
    primary = Scripted([ConnectionError("down")] * 4)
    backup = Scripted([ValueError("max_tokens: 10000 > 2048"), ProviderResponse(text="done")])
    chain = FallbackProvider(primary, fallbacks=[backup], cooldown_seconds=100)
    chain._members[1].model = "backup-model"
    chain._members[1].max_tokens = 10000
    async with Agent(config=config(max_tokens=10000), provider=chain) as agent:
        result = await agent.run("go")
    assert result.ok
    assert [r["max_tokens"] for r in backup.calls] == [10000, 2048]
    assert (await resolve_model_limits(primary, "fixture")).max_output != 2048
    fresh = FallbackProvider(Scripted([]), fallbacks=[Scripted([])])
    fresh.restore_state(json.loads(json.dumps(chain.export_state())))
    assert fresh._members[1].reply_budget == 2048
    assert (await resolve_model_limits(fresh._members[1].provider, "backup-model")).max_output == 2048


@pytest.mark.asyncio
async def test_custom_provider_reply_budget_hook_is_used():
    class Custom(Scripted):
        def default_max_tokens(self, model):
            return 1234
    provider = Custom([ProviderResponse(text="done")])
    async with Agent(config=config(), provider=provider) as agent:
        assert (await agent.run("go")).ok
    assert provider.calls[0]["max_tokens"] == 1234


@pytest.mark.asyncio
async def test_pause_exhaustion_is_failed_with_partial_output_and_usage():
    provider = Scripted([ProviderResponse(text="part", stop_reason="pause_turn", usage=TokenUsage(input_tokens=3))] * 6)
    async with Agent(config=config(), provider=provider) as agent:
        result = await agent.run("go")
    assert result.failed and not result.ok and result.output == "part" * 6
    assert result.usage.input_tokens == 18 and result.stop_reason == "pause_turn"


@pytest.mark.asyncio
async def test_shared_mcp_readiness_and_cancelled_waiter(monkeypatch):
    from harnessx.mcp import MCPConnection, MCPServerConfig, MCPToolInfo
    client = MCPConnection(MCPServerConfig.stdio("fixture", "unused"))
    gate, entered = asyncio.Event(), asyncio.Event()
    tool = MCPToolInfo("fixture", "tool", "description", {})
    opens = []
    async def open_session(stack):
        opens.append(True)
        entered.set()
        await gate.wait()
        client.session = object()
        client._tools = [tool]
        return [tool]
    monkeypatch.setattr(client, "_open", open_session)
    first = asyncio.create_task(client.connect())
    await entered.wait()
    second = asyncio.create_task(client.connect())
    await asyncio.sleep(0)
    assert not second.done()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    gate.set()
    assert await second == [tool] and len(opens) == 1
    await client.disconnect()
    assert not client.is_connected and not client.tools
    assert await client.connect() == [tool] and len(opens) == 2
    await client.disconnect()


@pytest.mark.asyncio
async def test_disconnect_during_mcp_initialization_releases_all_waiters(monkeypatch):
    from harnessx.mcp import MCPConnection, MCPServerConfig
    client = MCPConnection(MCPServerConfig.stdio("fixture", "unused"))
    entered = asyncio.Event()
    async def opening(stack):
        entered.set()
        await asyncio.Future()
    monkeypatch.setattr(client, "_open", opening)
    callers = [asyncio.create_task(client.connect()) for _ in range(2)]
    await entered.wait()
    await client.disconnect()
    results = await asyncio.wait_for(asyncio.gather(*callers, return_exceptions=True), 1)
    assert all(isinstance(r, RuntimeError) for r in results)


def test_activity_accounting_is_idempotent():
    from harnessx.execution import apply_accounting
    state = {"usage": {}, "total_usage": {"input_tokens": 10}, "estimated_cost": .5}
    entry = {}
    amount = {"usage": asdict(TokenUsage(input_tokens=20, thinking_tokens=3)), "cost": .25}
    apply_accounting(state, entry, amount)
    apply_accounting(state, entry, amount)
    assert state["usage"]["input_tokens"] == 20 and state["total_usage"]["input_tokens"] == 30
    assert state["estimated_cost"] == .75


@pytest.mark.asyncio
async def test_forced_compaction_falls_back_when_counter_underestimates():
    provider = Scripted([ValueError("prompt is too long: 250000 tokens > 200000 maximum"),
                         ValueError("summary unavailable"), ProviderResponse(text="fits")])
    async with Agent(config=config(), provider=provider) as agent:
        for i in range(8):
            agent.memory.add_user_message(f"old {i}: " + "x" * 3000)
            agent.memory.add_assistant_message("y" * 3000)
        result = await agent.run("go")
    assert result.ok and len(provider.calls) == 3
    assert len(str(provider.calls[-1]["messages"])) < len(str(provider.calls[0]["messages"]))


@pytest.mark.asyncio
async def test_child_usage_charged_before_parent_cost_check_and_saved(monkeypatch, tmp_path):
    from harnessx import SubAgent
    child = Scripted([ProviderResponse(tool_calls=[ToolCall("c", "noop", {})], usage=TokenUsage(input_tokens=100, thinking_tokens=3))])
    monkeypatch.setattr("harnessx.core.make_provider", lambda _: child)
    parent_provider = Scripted([ProviderResponse(tool_calls=[ToolCall("d", "delegate_worker", {"task": "go"})])])
    child_config = config(limits=Limits(input_cost_per_m=10000))
    parent_config = config(limits=Limits(max_cost_dollars=.5, input_cost_per_m=1))
    async with Agent(config=parent_config, provider=parent_provider,
                     subagents=[SubAgent("worker", "work", child_config, tools=[lambda: None])]) as parent:
        result = await parent.run("go")
        assert result.failed and result.error["type"] == "CostLimitError"
        assert result.usage.input_tokens == 100 and result.usage.thinking_tokens == 3
        assert parent.guardrails.estimated_cost == 1
        assert len(child.calls) == 1
        sid = await parent.save_session(str(tmp_path))
    async with await Agent.load_session(sid, str(tmp_path), provider=Scripted([])) as saved:
        assert saved.guardrails.estimated_cost == 1
        assert saved.guardrails.total_usage.thinking_tokens == 3


@pytest.mark.asyncio
async def test_child_cancellation_preserves_known_usage(monkeypatch):
    from harnessx import SubAgent
    started = asyncio.Event()
    class Child(Scripted):
        async def create(self, **request):
            if self.calls:
                started.set()
                await asyncio.Future()
            return await super().create(**request)
    child = Child([ProviderResponse(text="partial", stop_reason="max_tokens", usage=TokenUsage(input_tokens=40))])
    monkeypatch.setattr("harnessx.core.make_provider", lambda _: child)
    parent_provider = Scripted([ProviderResponse(tool_calls=[ToolCall("d", "delegate_worker", {"task": "go"})])])
    async with Agent(config=config(), provider=parent_provider, subagents=[SubAgent("worker", "work", config())]) as parent:
        running = asyncio.create_task(parent.run("go"))
        await asyncio.wait_for(started.wait(), 2)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert parent.guardrails.total_usage.input_tokens == 40
        assert parent.guardrails.usage_incomplete


def test_literal_namespaced_tool_alias_does_not_silently_select_short_name():
    from harnessx import ToolRegistry
    from harnessx.tools import ToolNotFoundError
    tools = ToolRegistry()
    tools.register_tool(lambda: None, name="work")
    tools.register_tool(lambda: None, name="tools.work")
    assert tools.resolve_tool("tools.work")[1] == "tools.work"
    with pytest.raises(ToolNotFoundError):
        tools.resolve_tool("Tools.Work")


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["safety", "refusal"])
async def test_refused_or_filtered_reply_never_executes_its_tool_calls(stop):
    ran = []
    provider = Scripted([ProviderResponse(tool_calls=[ToolCall("t", "effect", {})], stop_reason=stop)])
    async with Agent(config=config(), provider=provider) as agent:
        agent.tools.register_tool(lambda: ran.append(True), name="effect")
        result = await agent.run("go")
    assert not ran and result.refused and not result.ok


@pytest.mark.asyncio
async def test_signed_content_cannot_be_replaced_by_response_middleware():
    from harnessx import Middleware
    class Rewrite(Middleware):
        async def after_llm_call(self, response):
            response.content = [{"type": "text", "text": "changed"}]
            return response
    provider = Scripted([ProviderResponse(content=[
        {"type": "thinking", "thinking": "original", "signature": "signature"},
        {"type": "text", "text": "answer"}])])
    async with Agent(config=config(), provider=provider) as agent:
        agent.middleware.add(Rewrite())
        result = await agent.run("go")
    assert result.failed and "signed or opaque" in result.error["message"]


@pytest.mark.asyncio
async def test_native_response_recorder_preserves_block_order(tmp_path):
    from test_flight_recorder import session, export, of_kind
    from harnessx import IncidentRecorder
    content = [
        {"type": "thinking", "thinking": "", "signature": "sig-a"},
        {"type": "provider", "provider": "anthropic", "data": {"type": "redacted_thinking", "data": "opaque"}},
        {"type": "thinking", "thinking": "thought", "signature": "sig-b"},
        {"type": "text", "text": "answer"},
    ]
    async with session(tmp_path, provider=Scripted([ProviderResponse(content=content)]), config=config()) as (runtime, store):
        result = await runtime.run("go")
        assert result.ok
        bundle = await export(runtime, result, tmp_path)
    playback = await IncidentRecorder().playback(bundle)
    assert playback.report.valid and playback.report.complete
    for kind in ("model.response", "model.processed"):
        assert of_kind(playback, kind)[0]["payload"]["content"] == content
