"""Loop hardening found by comparing our loop with Pi, OpenCode, Hermes and
Deep Agents: a transcript repaired before every request, stop reasons that say
what happened, near-miss tool names, stalled streams, a final answer at the
step limit, old tool output cleared before summarizing, cross-provider history,
and subagents that report what they did not finish.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from harnessx import (
    Agent, AgentConfig, HookEvent, Limits, PermissionLevel, ProviderResponse, RetryPolicy,
    RunLimitReached, RunRefused, StreamChunk, SubAgent, TokenUsage, ToolCall,
)
from harnessx import engine, models
from harnessx.engine import UNANSWERED_CALL_NOTICE, sanitize_history
from harnessx.memory import CLEARED_TOOL_RESULT, ConversationMemory
from harnessx.models import register_model_limits
from harnessx.providers import LLMProvider
from harnessx.providers.retry import is_transient
from harnessx.tools import ToolNotFoundError, ToolRegistry
from harnessx.types import StopReason


@pytest.fixture(autouse=True)
def fresh_limits():
    learned, registered = dict(models._learned), dict(models._registered)
    models._learned.clear()
    models._registered.clear()
    yield
    models._learned.clear()
    models._learned.update(learned)
    models._registered.clear()
    models._registered.update(registered)


class Scripted(LLMProvider):
    name = "scripted"

    def __init__(self, responses, *, count=None):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.count = count

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if callable(item):
            item = item(kwargs)
        if isinstance(item, Exception):
            raise item
        return item

    async def count_tokens(self, **kwargs):
        return self.count(kwargs) if self.count else 0


def _agent(provider, **config):
    return Agent(config=AgentConfig(model="m", planning=False, **config), provider=provider)


def _use(*ids):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": i, "name": "t", "input": {}} for i in ids]}


def _result(call_id, text="ok"):
    return {"type": "tool_result", "tool_use_id": call_id, "content": text}


# ── the transcript every request carries ─────────────────────────────────────


def test_a_resumed_call_keeps_its_real_result_not_the_placeholder():
    """A run that failed mid-tools closes its calls with a placeholder; resuming
    re-runs them and appends the real result. Both used to be sent."""
    sent = sanitize_history([
        {"role": "user", "content": "go"}, _use("a"),
        {"role": "user", "content": [{**_result("a", "Not executed"), "is_error": True}]},
        {"role": "user", "content": [_result("a", "real")]},
    ])
    assert sent[-1]["content"] == [_result("a", "real")]


def test_answers_to_nothing_are_dropped_and_calls_with_no_answer_get_one():
    sent = sanitize_history([
        {"role": "user", "content": [_result("ghost")]},
        {"role": "user", "content": "go"}, _use("a", "b"),
        {"role": "user", "content": [_result("b")]},
    ])
    assert sent[0] == {"role": "user", "content": [{"type": "text", "text": "go"}]}
    answers = sent[-1]["content"]
    assert [a["tool_use_id"] for a in answers] == ["a", "b"], "in call order"
    assert answers[0]["content"] == UNANSWERED_CALL_NOTICE and answers[0]["is_error"]


def test_a_call_followed_by_plain_text_is_still_answered():
    sent = sanitize_history([{"role": "user", "content": "go"}, _use("a"), {"role": "user", "content": "next"}])
    assert sent[-1]["content"][0]["tool_use_id"] == "a"
    assert sent[-1]["content"][1] == {"type": "text", "text": "next"}


def test_empty_assistant_turns_from_old_sessions_are_not_sent():
    sent = sanitize_history([
        {"role": "user", "content": "a"}, {"role": "assistant", "content": []},
        {"role": "user", "content": "b"}, {"role": "assistant", "content": [{"type": "text", "text": " "}]},
    ])
    assert [m["role"] for m in sent] == ["user"]


@pytest.mark.asyncio
async def test_a_session_with_an_orphaned_call_can_carry_on():
    provider = Scripted([ProviderResponse(text="fine")])
    agent = _agent(provider)
    agent.memory.set_messages([{"role": "user", "content": "go"}, _use("lost")])
    async with agent:
        assert (await agent.run("continue")).ok
    first = provider.calls[0]["messages"]
    assert first[-1]["content"][0]["tool_use_id"] == "lost"


# ── stop reasons ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_refusal_completes_but_is_not_ok():
    provider = Scripted([ProviderResponse(text="I can't help with that.", stop_reason="refusal")])
    async with _agent(provider) as agent:
        result = await agent.run("go")
    assert result.refused and not result.ok and len(provider.calls) == 1, "not nudged or retried"
    with pytest.raises(RunRefused):
        result.raise_for_status()


def test_providers_map_refusals_and_filters():
    from harnessx.providers.anthropic import _from_anthropic_response
    from harnessx.providers.gemini import _from_gemini_parts
    from harnessx.providers.openai import _from_openai_response

    anthropic = SimpleNamespace(content=[], stop_reason="refusal", usage=None)
    assert _from_anthropic_response(anthropic).stop_reason is StopReason.REFUSAL
    paused = SimpleNamespace(content=[], stop_reason="pause_turn", usage=None)
    assert _from_anthropic_response(paused).stop_reason is StopReason.PAUSE_TURN

    message = SimpleNamespace(content=None, refusal="No.", tool_calls=None)
    openai = _from_openai_response(SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None))
    assert openai.stop_reason is StopReason.REFUSAL and openai.text == "No."

    call = SimpleNamespace(function_call=SimpleNamespace(id="c", name="t", args={}), text=None, thought=None)
    cut = _from_gemini_parts([call], "MAX_TOKENS", None, raw=None)
    assert cut.stop_reason is StopReason.MAX_TOKENS, "a call in a cut-off reply may be the part cut"
    assert _from_gemini_parts([], "RECITATION", None, raw=None).stop_reason is StopReason.SAFETY
    assert _from_gemini_parts([], "MALFORMED_FUNCTION_CALL", None, raw=None).stop_reason is StopReason.OTHER


def test_anthropic_block_types_without_a_canonical_shape_do_not_fail_the_turn():
    from harnessx.providers.anthropic import _from_anthropic_response

    blocks = [
        SimpleNamespace(type="redacted_thinking", data="opaque"),
        SimpleNamespace(type="text", text="answer", citations=None),
    ]
    response = _from_anthropic_response(SimpleNamespace(content=blocks, stop_reason="end_turn", usage=None))
    assert response.text == "answer"


@pytest.mark.asyncio
async def test_a_paused_turn_is_resumed_and_joined():
    provider = Scripted([
        ProviderResponse(text="Searching... ", stop_reason="pause_turn"),
        ProviderResponse(text="found it."),
    ])
    async with _agent(provider) as agent:
        result = await agent.run("go")
    assert result.ok and result.output == "Searching... found it."
    assert provider.calls[1]["messages"][-1]["role"] == "assistant", "sent back as it stands"


# ── near-miss tool names ─────────────────────────────────────────────────────


def _registry(*names):
    registry = ToolRegistry()
    for name in names:
        registry.register_with_schema(name, name, {"type": "object", "properties": {}}, lambda: "ok",
                                      permission=PermissionLevel.ALLOW)
    return registry


@pytest.mark.parametrize("asked", ["search-web", "searchWeb", "functions.search_web", "default_api:search_web", "Search_Web"])
def test_a_near_miss_name_finds_the_one_tool_it_means(asked):
    assert _registry("search_web", "read_file").resolve_tool(asked)[1] == "search_web"


@pytest.mark.parametrize("asked", ["doit", "browse", "write"])
def test_a_name_that_could_mean_two_tools_or_none_is_not_guessed(asked):
    with pytest.raises(ToolNotFoundError):
        _registry("do_it", "do-it", "read_file", "write_file_now").resolve_tool(asked)


# ── streams that stall or end early ──────────────────────────────────────────


class Stalling(Scripted):
    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            yield StreamChunk(kind="text_delta", data="Hel")
            await asyncio.sleep(10)
        yield StreamChunk(kind="text_delta", data="Hello")
        yield StreamChunk(kind="response", data=ProviderResponse(text="Hello"))


@pytest.mark.asyncio
async def test_a_stream_that_stalls_is_abandoned_and_retried():
    provider = Stalling([])
    agent = _agent(provider, retry=RetryPolicy(stream_idle_timeout_seconds=0.05, backoff_seconds=0))
    async with agent:
        async with agent.run_stream("go") as stream:
            events = [event async for event in stream]
            result = await stream.result()
    assert result.ok and result.output == "Hello" and len(provider.calls) == 2
    assert any(e.type.value == "attempt_reset" for e in events), "consumers are told to drop the stalled text"


class NeverStarts(Scripted):
    """Opens, then sends nothing on its first call: stuck, not thinking."""

    first_event_promptly = True

    def __init__(self, *, first_wait=10.0):
        super().__init__([])
        self.first_wait = first_wait

    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            await asyncio.sleep(self.first_wait)
        yield StreamChunk(kind="progress")
        yield StreamChunk(kind="text_delta", data="Hello")
        yield StreamChunk(kind="response", data=ProviderResponse(text="Hello"))

    async def create(self, **kwargs):
        raise AssertionError("a provider that streams is streamed, even for agent.run()")


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False], ids=["run_stream", "run"])
async def test_a_stream_that_never_starts_is_retried_well_before_the_idle_limit(streaming):
    provider = NeverStarts()
    retry = RetryPolicy(stream_first_event_timeout_seconds=0.05, stream_idle_timeout_seconds=10, backoff_seconds=0)
    agent = _agent(provider, retry=retry)
    async with agent:
        if streaming:
            async with agent.run_stream("go") as stream:
                [event async for event in stream]
                result = await stream.result()
        else:
            result = await asyncio.wait_for(agent.run("go"), 5)
    assert result.ok and result.output == "Hello" and len(provider.calls) == 2


@pytest.mark.asyncio
async def test_a_quiet_start_is_tolerated_where_it_may_be_a_model_thinking():
    """OpenAI's and Gemini's reasoning models say nothing until they answer:
    their providers do not claim a prompt first event, so the first wait gets
    the idle limit like any other gap."""
    provider = NeverStarts(first_wait=0.2)
    provider.first_event_promptly = False
    retry = RetryPolicy(stream_first_event_timeout_seconds=0.05, stream_idle_timeout_seconds=5, backoff_seconds=0)
    agent = _agent(provider, retry=retry)
    async with agent:
        async with agent.run_stream("go") as stream:
            [event async for event in stream]
            result = await stream.result()
    assert result.ok and len(provider.calls) == 1, "not cut, not retried"


@pytest.mark.asyncio
async def test_a_provider_without_a_stream_of_its_own_is_not_cut_by_either_deadline():
    class OnlyCreate(Scripted):
        first_event_promptly = True

        async def create(self, **kwargs):
            await asyncio.sleep(0.2)
            return await super().create(**kwargs)

    provider = OnlyCreate([ProviderResponse(text="done")])
    retry = RetryPolicy(stream_first_event_timeout_seconds=0.05, stream_idle_timeout_seconds=0.1, backoff_seconds=0)
    async with _agent(provider, retry=retry) as agent:
        result = await agent.run("go")
    assert result.ok and result.output == "done"


def test_the_first_event_deadline_grows_with_the_request_and_never_passes_the_idle_limit():
    retry = RetryPolicy()
    assert retry.first_event_timeout(0) == 20.0
    assert retry.first_event_timeout(100_000) == 30.0
    assert RetryPolicy(stream_idle_timeout_seconds=25).first_event_timeout(1_000_000) == 25
    assert RetryPolicy(stream_first_event_timeout_seconds=None).first_event_timeout(10) == 180.0


@pytest.mark.parametrize("name", ["RemoteProtocolError", "ReadError", "IncompleteStreamError"])
def test_a_connection_dropped_mid_reply_is_transient(name):
    assert is_transient(type(name, (Exception,), {})("peer closed connection"))


@pytest.mark.asyncio
async def test_an_openai_stream_with_no_finish_reason_and_a_tool_call_is_not_trusted():
    from harnessx.errors import IncompleteStreamError
    from harnessx.providers.openai import OpenAIProvider

    def chunk(**delta):
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(
            finish_reason=None, delta=SimpleNamespace(**{"content": None, "tool_calls": None, "refusal": None, **delta}))])

    async def chunks():
        yield chunk(tool_calls=[SimpleNamespace(index=0, id="c1", function=SimpleNamespace(name="t", arguments='{"a": '))])

    async def create(**kwargs):
        return chunks()

    provider = OpenAIProvider(client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with pytest.raises(IncompleteStreamError):
        async for _ in provider.stream(model="gpt-4o", messages=[], system=None, tools=[], max_tokens=100):
            pass


# ── the step limit ───────────────────────────────────────────────────────────


def _looping(n):
    return [ProviderResponse(tool_calls=[ToolCall(f"c{i}", "probe", {})], stop_reason="tool_use") for i in range(n)]


def _probe(agent):
    agent.tools.register_with_schema("probe", "probe", {"type": "object", "properties": {}}, lambda: "more",
                                     permission=PermissionLevel.ALLOW)
    return agent


@pytest.mark.asyncio
async def test_the_step_limit_ends_with_an_answer_not_a_failure():
    provider = Scripted([*_looping(2), ProviderResponse(text="Did 2 probes; the rest is unfinished.")])
    agent = _probe(_agent(provider, limits=Limits(max_iterations=2)))
    async with agent:
        result = await agent.run("go")
    assert result.status == "completed" and result.limited and not result.ok
    assert result.output == "Did 2 probes; the rest is unfinished."
    assert "reached the step limit" in str(provider.calls[-1]["messages"][-1])
    with pytest.raises(RunLimitReached):
        result.raise_for_status()


@pytest.mark.asyncio
async def test_a_tool_call_in_the_final_answer_is_answered_not_run():
    provider = Scripted([*_looping(3)])
    agent = _probe(_agent(provider, limits=Limits(max_iterations=2)))
    async with agent:
        result = await agent.run("go")
        stored = agent.memory.get_messages()
    assert result.limited and len(provider.calls) == 3
    assert "step limit" in str(stored[-1]["content"])


@pytest.mark.asyncio
async def test_the_old_failure_is_one_flag_away():
    provider = Scripted(_looping(3))
    agent = _probe(_agent(provider, limits=Limits(max_iterations=2, final_answer_on_limit=False)))
    async with agent:
        result = await agent.run("go")
    assert result.failed and "maximum iterations" in result.error["message"]


# ── a retried call does not store its turn twice ─────────────────────────────


@pytest.mark.asyncio
async def test_a_retried_model_call_does_not_duplicate_the_assistant_turn(monkeypatch):
    real = engine.plan_recovery
    failures = [ConnectionError("lost after the reply arrived")]

    async def flaky(*args, **kwargs):
        if failures:
            raise failures.pop()
        return await real(*args, **kwargs)

    monkeypatch.setattr(engine, "plan_recovery", flaky)
    provider = Scripted([ProviderResponse(text="one"), ProviderResponse(text="one")])
    async with _agent(provider, retry=RetryPolicy(backoff_seconds=0)) as agent:
        assert (await agent.run("go")).ok
        roles = [m["role"] for m in agent.memory.get_messages()]
    assert roles == ["user", "assistant"]


# ── old tool output is cleared before the history is summarized ─────────────


def _results_history(sizes):
    messages: list[dict] = [{"role": "user", "content": "start"}]
    for i, size in enumerate(sizes):
        messages.append(_use(f"c{i}"))
        messages.append({"role": "user", "content": [_result(f"c{i}", "x" * size)]})
    messages.append({"role": "assistant", "content": [{"type": "text", "text": "done"}]})
    return messages


def test_pruning_clears_the_oldest_results_and_keeps_the_newest():
    memory = ConversationMemory()
    memory.set_messages(_results_history([60_000] * 6))
    freed = memory.prune_tool_results()
    bodies = [b["content"] for m in memory.get_messages() if isinstance(m["content"], list)
              for b in m["content"] if b.get("type") == "tool_result"]
    assert freed > 0 and bodies[0] == CLEARED_TOOL_RESULT
    assert bodies[-1] == "x" * 60_000, "the newest output survives"


def test_pruning_does_nothing_for_a_small_saving():
    memory = ConversationMemory()
    memory.set_messages(_results_history([30_000] * 6))
    assert memory.prune_tool_results() == 0


@pytest.mark.asyncio
async def test_clearing_old_output_can_spare_the_summary():
    register_model_limits("prune-model", context_window=100_000, max_output=4_000)
    condensed = []

    def count(kwargs):
        return len(str(kwargs["messages"])) // 4

    def answer(kwargs):
        assert "condense" not in (kwargs.get("system") or ""), "no summary should be needed"
        return ProviderResponse(text="ok")

    provider = Scripted([answer], count=count)
    agent = Agent(config=AgentConfig(model="prune-model", planning=False), provider=provider)
    agent.hooks.on(HookEvent.CONTEXT_CONDENSED, lambda ctx: condensed.append(dict(ctx.data)))
    agent.memory.set_messages(_results_history([60_000] * 6))
    async with agent:
        assert (await agent.run("go")).ok
    assert condensed == [{"messages_dropped": 0, "summary_chars": 0, "summarized": False}]


@pytest.mark.asyncio
async def test_a_huge_error_result_is_spilled_like_any_other():
    memory = ConversationMemory(max_result_chars=1_000)
    from harnessx import ToolResult

    memory.add_tool_results([ToolResult("c1", "E" * 50_000, True)])
    stored = memory.get_messages()[-1]["content"][0]
    assert stored["is_error"] and len(stored["content"]) < 10_000


# ── a conversation moved to Anthropic by a fallback ──────────────────────────


def test_foreign_reasoning_and_private_keys_are_not_sent_to_anthropic():
    from harnessx.providers.anthropic import _messages_for_request

    sent = _messages_for_request([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "gemini reasoning"}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "mine", "signature": "sig"},
            {"type": "tool_use", "id": "c1", "name": "t", "input": {}, "_gemini_thought_signature": "abc"},
        ]},
    ])
    assert len(sent) == 2, "a turn of only foreign reasoning is dropped, not sent empty"
    blocks = sent[1]["content"]
    assert blocks[0]["signature"] == "sig" and "_gemini_thought_signature" not in blocks[1]


# ── subagents ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_child_cut_off_is_flagged_and_its_usage_counted(monkeypatch):
    child = Scripted([ProviderResponse(text="half", stop_reason="max_tokens",
                                       usage=TokenUsage(input_tokens=100, output_tokens=50))])
    monkeypatch.setattr("harnessx.core.make_provider", lambda name: child)
    spec = SubAgent("reviewer", "Review", AgentConfig(limits=Limits(max_truncation_recoveries=0)))
    async with Agent(provider=Scripted([]), subagents=[spec]) as parent:
        from harnessx import ToolCall as Call

        result = await parent.tools.execute(Call("d", "delegate_reviewer", {"task": "review"}), permissions=parent.permissions)
        assert "half" in result.content and "cut off" in result.content
        assert parent.guardrails.total_usage.output_tokens == 50


# ── the plan is announced when it changes ────────────────────────────────────


@pytest.mark.asyncio
async def test_planning_is_on_by_default_and_each_change_is_announced():
    from harnessx.execution import RunEventType

    plan = [{"content": "Read the code", "status": "in_progress"}, {"content": "Write the fix", "status": "pending"}]
    ticked = [{"content": "Read the code", "status": "completed"}, {"content": "Write the fix", "status": "in_progress"}]
    provider = Scripted([
        ProviderResponse(tool_calls=[ToolCall("p1", "write_todos", {"todos": plan})], stop_reason="tool_use"),
        ProviderResponse(tool_calls=[ToolCall("r1", "read_todos", {})], stop_reason="tool_use"),
        ProviderResponse(tool_calls=[ToolCall("p2", "write_todos", {"todos": ticked})], stop_reason="tool_use"),
        ProviderResponse(text="done"),
    ])
    agent = Agent(config=AgentConfig(model="m"), provider=provider)
    hooked = []
    agent.hooks.on(HookEvent.TODOS_UPDATED, lambda ctx: hooked.append(dict(ctx.data)))
    assert agent.tools.has_tool("write_todos") and agent.tools.has_tool("read_todos")
    async with agent:
        async with agent.run_stream("fix it") as stream:
            updates = [e.data for e in [e async for e in stream] if e.type == RunEventType.TODOS_UPDATED]
    assert len(updates) == 2, "announced on each change, not on a read that changes nothing"
    assert updates[0] == {"todos": plan, "completed": 0, "total": 2, "in_progress": "Read the code"}
    assert updates[1]["completed"] == 1 and updates[1]["in_progress"] == "Write the fix"
    assert hooked == updates


@pytest.mark.asyncio
async def test_a_plan_carried_into_the_next_run_is_not_announced_again():
    from harnessx.execution import RunEventType

    plan = [{"content": "Only step", "status": "in_progress"}]
    provider = Scripted([
        ProviderResponse(tool_calls=[ToolCall("p1", "write_todos", {"todos": plan})], stop_reason="tool_use"),
        ProviderResponse(text="planned"),
        ProviderResponse(tool_calls=[ToolCall("r1", "read_todos", {})], stop_reason="tool_use"),
        ProviderResponse(text="still on it"),
    ])
    agent = Agent(config=AgentConfig(model="m"), provider=provider)
    async with agent:
        await agent.run("start")
        async with agent.run_stream("continue") as stream:
            kinds = [e.type async for e in stream]
    assert RunEventType.TODOS_UPDATED not in kinds


# ── a long tool argument is not a stall ──────────────────────────────────────


class ToolWriter(Scripted):
    """Streams a long tool argument: minutes of activity, no visible text."""

    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        for _ in range(6):
            await asyncio.sleep(0.03)  # each gap is under the idle limit, the total is over it
            yield StreamChunk(kind="progress")
        yield StreamChunk(kind="response", data=ProviderResponse(text="written"))


@pytest.mark.asyncio
async def test_a_stream_busy_writing_a_tool_argument_is_not_a_stall():
    provider = ToolWriter([])
    agent = _agent(provider, retry=RetryPolicy(stream_idle_timeout_seconds=0.1, backoff_seconds=0))
    async with agent:
        async with agent.run_stream("write the report") as stream:
            async for _ in stream:
                pass
            result = await stream.result()
    assert result.ok and len(provider.calls) == 1, "no retry: the stream was alive throughout"


@pytest.mark.asyncio
async def test_anthropic_reports_every_stream_event_not_only_text():
    from harnessx.providers.anthropic import _chunks

    class Events:
        def __aiter__(self):
            async def gen():
                for event in (
                    SimpleNamespace(type="input_json", partial_json='{"content": "abc'),
                    SimpleNamespace(type="thinking", thinking="hmm"),
                    SimpleNamespace(type="text", text="Hi"),
                    SimpleNamespace(type="content_block_stop"),
                ):
                    yield event
            return gen()

    kinds = [c.kind async for c in _chunks(Events())]
    assert kinds == ["progress", "thinking_delta", "text_delta", "progress"]


@pytest.mark.asyncio
async def test_still_working_notices_go_out_while_the_model_is_silent():
    from harnessx import ProgressPolicy
    from harnessx.execution import RunEventType

    class Slow(Scripted):
        async def create(self, **kwargs):
            await asyncio.sleep(0.25)
            return ProviderResponse(text="done")

    agent = _agent(Slow([]), progress=ProgressPolicy(first_after_seconds=0.05, repeat_every_seconds=0.05))
    seen_before_text = 0
    async with agent:
        async with agent.run_stream("go") as stream:
            async for event in stream:
                if event.type == RunEventType.WAITING:
                    seen_before_text += 1
                if event.type == RunEventType.TEXT_COMPLETE:
                    break
    assert seen_before_text >= 2, "notices arrive during the wait, not after it"


# ── a run does not end with its own plan left open ───────────────────────────


@pytest.mark.asyncio
async def test_finishing_with_an_open_plan_gets_one_reminder_and_keeps_the_answer():
    plan = [{"content": "Write the report", "status": "in_progress"}]
    provider = Scripted([
        ProviderResponse(tool_calls=[ToolCall("p1", "write_todos", {"todos": plan})], stop_reason="tool_use"),
        ProviderResponse(text="# The report"),
        ProviderResponse(tool_calls=[ToolCall("p2", "write_todos", {"todos": [{**plan[0], "status": "completed"}]})],
                         stop_reason="tool_use"),
        ProviderResponse(text=""),
    ])
    agent = Agent(config=AgentConfig(model="m"), provider=provider)
    async with agent:
        result = await agent.run("write it")
    assert result.ok and result.output == "# The report", "the answer survives; an empty sign-off adds nothing"
    assert "task list still has unfinished items" in str(provider.calls[2]["messages"][-1])
    assert agent.todos[0]["status"] == "completed" and len(provider.calls) == 4


@pytest.mark.asyncio
async def test_the_plan_reminder_is_sent_once_only():
    plan = [{"content": "Step", "status": "pending"}]
    provider = Scripted([
        ProviderResponse(tool_calls=[ToolCall("p1", "write_todos", {"todos": plan})], stop_reason="tool_use"),
        ProviderResponse(text="answer"),
        ProviderResponse(text="Step is left for you."),
    ])
    agent = Agent(config=AgentConfig(model="m"), provider=provider)
    async with agent:
        result = await agent.run("go")
    assert len(provider.calls) == 3 and result.output == "answer\n\nStep is left for you."
