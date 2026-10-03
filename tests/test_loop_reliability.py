"""Loop reliability: the defects found by comparing our loop against OpenCode,
Hermes, Pi and Deep Agents. See doc/agent-loop-reliability.md."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from harnessx import Agent, AgentConfig, HookEvent, Limits, PermissionLevel, ProviderResponse, ToolCall
from harnessx.execution import transition
from harnessx.memory import ConversationMemory
from harnessx.builtin.planning import render
from harnessx.hooks import Middleware
from harnessx.providers import LLMProvider
from harnessx.types import LoopGuard, ToolResult


class Scripted(LLMProvider):
    name = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)

    async def count_tokens(self, **kwargs):
        return 0


def _state(response: dict, *, messages=None, tools=None) -> dict:
    state: dict[str, Any] = {"phase": "model", "status": "running", "response": response}
    if messages is not None:
        state["messages"] = messages
    if tools is not None:
        state["tools"] = tools
    return state


def _assistant_turn(*call_ids):
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": i, "name": "probe", "input": {}} for i in call_ids],
    }


def _unanswered(messages):
    """Tool-use ids in the transcript that nothing answers."""
    used, answered = set(), set()
    for m in messages:
        for b in m.get("content") or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                used.add(b["id"])
            elif b.get("type") == "tool_result":
                answered.add(b["tool_use_id"])
    return used - answered


# ── the turn ends on tool calls, not on the stop reason ──────────────────────


@pytest.mark.parametrize("stop_reason", ["end_turn", "stop", "tool_use", "unknown"])
def test_tool_calls_are_honoured_whatever_the_stop_reason_says(stop_reason):
    """Providers return end_turn or stop while still carrying tool calls.
    Ending the turn there drops them and the run looks like the model ignored
    its tools. OpenCode guards this explicitly; we did not."""
    response = {
        "text": "", "stop_reason": stop_reason,
        "tool_calls": [{"id": "c1", "name": "probe", "input": {}}],
    }
    assert transition(_state(response), "model", {})["phase"] == "prepare_tools"


def test_a_truncated_reply_never_executes_its_tool_calls():
    """Arguments streamed into a reply cut off at max_tokens can parse while
    being incomplete, so running them executes a half-formed call."""
    response = {
        "text": "part", "stop_reason": "max_tokens",
        "tool_calls": [{"id": "c1", "name": "probe", "input": {}}],
    }
    advanced = transition(_state(response), "model", {})
    assert advanced["phase"] == "finish" and advanced["stop_reason"] == "max_tokens"


def test_a_truncated_reply_still_answers_the_tool_calls_it_refused():
    """Refusing to run them does not remove the assistant turn that carries
    them. An unanswered tool_use wedges the session: Anthropic rejects the next
    request outright, and only a fake provider lets it pass."""
    response = {
        "text": "part", "stop_reason": "max_tokens",
        "tool_calls": [{"id": "c1", "name": "probe", "input": {}}],
    }
    advanced = transition(
        _state(response, messages=[_assistant_turn("c1")], tools=[]), "model", {}
    )
    assert not _unanswered(advanced["messages"])
    answer = advanced["messages"][-1]["content"][0]
    assert answer["is_error"] and "token budget" in answer["content"]


def test_closing_open_calls_keeps_the_results_that_do_exist():
    from harnessx.execution import close_open_tool_calls

    state = {
        "messages": [_assistant_turn("c1", "c2")],
        "tools": [{"call": {"id": "c1"}, "result": {"content": "ran fine", "is_error": False}}],
    }
    closed = close_open_tool_calls(state, "never ran")
    answers = {b["tool_use_id"]: b for b in closed["messages"][-1]["content"]}
    assert answers["c1"]["content"] == "ran fine" and not answers["c1"]["is_error"]
    assert answers["c2"]["content"] == "never ran" and answers["c2"]["is_error"]


def test_closing_open_calls_twice_does_not_answer_twice():
    from harnessx.execution import close_open_tool_calls

    state = {"messages": [_assistant_turn("c1")], "tools": []}
    once = close_open_tool_calls(state, "never ran")
    twice = close_open_tool_calls(dict(once), "never ran")
    assert len(twice["messages"]) == len(once["messages"])


def test_closing_open_calls_leaves_a_turn_with_nothing_open_alone():
    from harnessx.execution import close_open_tool_calls

    state = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]}
    assert close_open_tool_calls(state, "never ran")["messages"] == state["messages"]


def test_a_reply_with_no_tool_calls_still_finishes():
    response = {"text": "all done", "stop_reason": "end_turn", "tool_calls": []}
    advanced = transition(_state(response), "model", {})
    assert advanced["phase"] == "finish" and advanced["output"] == "all done"


@pytest.mark.asyncio
async def test_end_turn_with_tool_calls_runs_the_tool_end_to_end():
    ran: list[str] = []
    provider = Scripted([
        ProviderResponse(tool_calls=[ToolCall("c1", "probe", {})], stop_reason="end_turn"),
        ProviderResponse(text="finished"),
    ])
    agent = Agent(config=AgentConfig(model="m"), provider=provider)

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def probe() -> str:
        """Probe."""
        ran.append("yes")
        return "probed"

    async with agent:
        result = await agent.run("go")
    assert ran == ["yes"], "the tool call was dropped because the stop reason was end_turn"
    assert result.output == "finished"


# ── condensing history never fabricates an assistant turn ────────────────────


class _Counter:
    def __init__(self, count): self.count = count
    async def count_tokens(self, **kwargs): return self.count


def _conversation() -> ConversationMemory:
    memory = ConversationMemory()
    memory.add_user_message("first task")
    memory.add_assistant_message([{"type": "text", "text": "working"}])
    memory.add_user_message("more context")
    memory.add_assistant_message([{"type": "tool_use", "id": "t1", "name": "d", "input": {}}])
    memory.add_tool_results([ToolResult(tool_call_id="t1", content="out")])
    memory.add_assistant_message([{"type": "text", "text": "done step"}])
    memory.add_user_message("keep going")
    memory.add_assistant_message([{"type": "text", "text": "final"}])
    return memory


@pytest.mark.asyncio
async def test_condensing_never_puts_words_in_the_models_mouth():
    """The old trim injected an assistant turn reading "Understood ... How can I
    help?". Mid-task that is the model's own voice telling it to stop working."""
    memory = _conversation()
    assert await memory.trim_if_needed(_Counter(10_000), "m", "", [], max_context_tokens=10)

    messages = memory.get_messages()
    assert "How can I help" not in str(messages)
    assert messages[0]["role"] == "user" and "condensed" in str(messages[0]["content"])
    for earlier, later in zip(messages, messages[1:]):
        assert earlier["role"] != later["role"], "roles must still alternate"


@pytest.mark.asyncio
async def test_condensing_is_bounded_by_the_reply_budget_not_the_raw_window():
    """A request can sit under a threshold measured against the whole window and
    still overflow, because the provider reserves the reply from that window."""
    window, reply = 1_000, 900
    # Comfortably under 80% of the window, but not of what is left after the reply.
    memory = _conversation()
    assert not await memory.trim_if_needed(_Counter(700), "m", "", [], max_context_tokens=window)

    memory = _conversation()
    assert await memory.trim_if_needed(
        _Counter(700), "m", "", [], max_context_tokens=window, reply_tokens=reply
    ), "the reply budget leaves 100 tokens; 700 does not fit"


@pytest.mark.asyncio
async def test_a_failed_token_count_condenses_rather_than_assuming_room():
    """A counting failure used to fall back to a character heuristic, which
    silently disabled the guard whenever it read low."""
    class Broken:
        async def count_tokens(self, **kwargs):
            raise RuntimeError("counting unavailable")

    memory = _conversation()
    assert await memory.trim_if_needed(Broken(), "m", "", [], max_context_tokens=10_000_000)


@pytest.mark.asyncio
async def test_a_short_conversation_is_left_alone():
    memory = ConversationMemory()
    memory.add_user_message("hello")
    memory.add_assistant_message([{"type": "text", "text": "hi"}])
    assert not await memory.trim_if_needed(_Counter(10_000), "m", "", [], max_context_tokens=1)


# ── condensing is a real summary, made by the model, as its own loop phase ───


class _Overflowing(LLMProvider):
    """Always reports an overflowing history, so condensing is always due."""

    name = "overflowing"

    def __init__(self, *, summary="TASK: ship it. DONE: a, b. NEXT: c.", fail_summary=False):
        self.summary = summary
        self.fail_summary = fail_summary
        self.summary_calls: list[dict[str, Any]] = []
        self.turns = 0

    def _is_summary(self, kwargs) -> bool:
        return bool(kwargs.get("system")) and "condense" in kwargs["system"]

    async def create(self, **kwargs):
        if self._is_summary(kwargs):
            self.summary_calls.append(kwargs)
            if self.fail_summary:
                raise RuntimeError("summarizer unavailable")
            return ProviderResponse(text=self.summary)
        self.turns += 1
        return ProviderResponse(text=f"reply {self.turns}")

    async def count_tokens(self, **kwargs):
        return 50_000


def _overflowing_agent(provider, **limits):
    from harnessx import Limits

    return Agent(
        config=AgentConfig(model="m", max_tokens=1000, limits=Limits(max_context_tokens=10_000, **limits)),
        provider=provider,
    )


@pytest.mark.asyncio
async def test_condensing_calls_the_model_and_keeps_its_summary():
    provider = _Overflowing()
    agent = _overflowing_agent(provider)
    events: list[dict[str, Any]] = []
    agent.hooks.on(HookEvent.CONTEXT_CONDENSED, lambda ctx: events.append(dict(ctx.data)))

    async with agent:
        for turn in range(4):
            await agent.run(f"turn {turn}")

    assert provider.summary_calls, "the summary is a real model call, not string truncation"
    assert events and events[0]["summarized"] is True
    first = str(agent.memory.get_messages()[0]["content"])
    assert "TASK: ship it" in first, "the model's summary is what the agent carries forward"
    assert "condensed" in first


@pytest.mark.asyncio
async def test_a_failed_summary_falls_back_instead_of_failing_the_run():
    """A summary that cannot be produced must not take the run down with it."""
    provider = _Overflowing(fail_summary=True)
    agent = _overflowing_agent(provider)
    events: list[dict[str, Any]] = []
    agent.hooks.on(HookEvent.CONTEXT_CONDENSED, lambda ctx: events.append(dict(ctx.data)))

    async with agent:
        for turn in range(4):
            result = await agent.run(f"turn {turn}")
            assert result.ok, "the run survives a failed summarization"

    assert events and events[-1]["summarized"] is False
    assert "condensed" in str(agent.memory.get_messages()[0]["content"])


@pytest.mark.asyncio
async def test_a_failed_summary_is_retried_before_the_run_gives_up_on_it():
    """`compact` was added to the driver's retryable commands and then wrapped
    in its own try/except, so the exception never reached the driver and the
    retry never fired. The fallback belongs after the attempts, not instead."""
    from harnessx import RetryPolicy

    provider = _Overflowing(fail_summary=True)
    agent = Agent(
        config=AgentConfig(
            model="m", max_tokens=1000,
            limits=Limits(max_context_tokens=10_000),
            retry=RetryPolicy(attempts=3, backoff_seconds=0.001, max_backoff_seconds=0.001),
        ),
        provider=provider,
    )
    waits: list[dict[str, Any]] = []
    agent.hooks.on(HookEvent.RETRY, lambda ctx: waits.append(dict(ctx.data)))

    async with agent:
        for turn in range(4):
            result = await agent.run(f"turn {turn}")

    assert result.ok, "the run still survives a summary that never succeeds"
    assert len(provider.summary_calls) >= 3, "every attempt is spent before the fallback"
    assert len(waits) >= 2 and {w["kind"] for w in waits} == {"model"}


@pytest.mark.asyncio
async def test_a_summary_that_succeeds_on_the_second_try_is_kept():
    from harnessx import RetryPolicy

    class Flaky(_Overflowing):
        async def create(self, **kwargs):
            if self._is_summary(kwargs) and not self.summary_calls:
                self.summary_calls.append(kwargs)
                raise RuntimeError("summarizer unavailable")
            return await super().create(**kwargs)

    provider = Flaky()
    agent = Agent(
        config=AgentConfig(
            model="m", max_tokens=1000,
            limits=Limits(max_context_tokens=10_000),
            retry=RetryPolicy(attempts=3, backoff_seconds=0.001, max_backoff_seconds=0.001),
        ),
        provider=provider,
    )
    events: list[dict[str, Any]] = []
    agent.hooks.on(HookEvent.CONTEXT_CONDENSED, lambda ctx: events.append(dict(ctx.data)))
    async with agent:
        for turn in range(4):
            await agent.run(f"turn {turn}")

    assert events[0]["summarized"] is True
    assert "TASK: ship it" in str(agent.memory.get_messages()[0]["content"])


@pytest.mark.asyncio
async def test_summary_tokens_are_counted_like_any_other_model_call():
    """The summary is a real call to the agent's model. Leaving its tokens out
    of the accounting hid that spend from `Limits.max_cost_dollars`."""
    from harnessx.types import TokenUsage

    class Metered(_Overflowing):
        async def create(self, **kwargs):
            response = await super().create(**kwargs)
            cost = 500 if self._is_summary(kwargs) else 10
            return ProviderResponse(
                text=response.text, tool_calls=response.tool_calls,
                stop_reason=response.stop_reason,
                usage=TokenUsage(input_tokens=cost, output_tokens=cost),
            )

    provider = Metered()
    agent = _overflowing_agent(provider)
    async with agent:
        for turn in range(4):
            result = await agent.run(f"turn {turn}")

    assert provider.summary_calls, "the summary ran, or this proves nothing"
    assert result.usage.input_tokens >= 500, "summary tokens reach RunResult.usage"
    assert agent.guardrails._total_usage.input_tokens >= 500, "and the cost guardrail"


@pytest.mark.asyncio
async def test_condensing_cannot_loop_forever_within_one_turn():
    """A summary that does not shrink the history would otherwise re-trigger on
    the very next iteration. OpenCode has exactly this open as issue 15533."""
    from harnessx.engine import MAX_CONDENSATIONS

    provider = _Overflowing(summary="x" * 50)
    agent = _overflowing_agent(provider)
    async with agent:
        await agent.run("go")
    assert len(provider.summary_calls) <= MAX_CONDENSATIONS


@pytest.mark.asyncio
async def test_a_history_that_fits_is_never_condensed():
    class Roomy(_Overflowing):
        async def count_tokens(self, **kwargs):
            return 10

    provider = Roomy()
    agent = _overflowing_agent(provider)
    async with agent:
        await agent.run("go")
    assert not provider.summary_calls


def test_the_summary_transcript_labels_roles_and_caps_tool_output():
    from harnessx.engine import render_for_summary

    rendered = render_for_summary([
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "grep", "input": {"q": "x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "content": "y" * 10_000}]},
        {"role": "user", "content": [{"type": "tool_result", "content": "boom", "is_error": True}]},
    ], max_block_chars=100)

    assert "[user] do the thing" in rendered
    assert "[assistant tool call] grep(" in rendered
    assert "[tool error] boom" in rendered
    assert "y" * 101 not in rendered, "tool payloads must not ride into another model call"


# ── a near-miss tool call is repaired, not refused ───────────────────────────


def _agent_with_search(script):
    provider = Scripted(script)
    agent = Agent(config=AgentConfig(model="m"), provider=provider)
    ran: list[str] = []

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def search_web(query: str) -> str:
        """Search the web."""
        ran.append(query)
        return "found"

    return agent, provider, ran


def _last_tool_text(provider: Scripted) -> str:
    last = provider.calls[-1]["messages"][-1]
    content = last["content"]
    block = content[0] if isinstance(content, list) else content
    return str(block.get("content") if isinstance(block, dict) else block)


@pytest.mark.asyncio
async def test_a_tool_name_wrong_only_in_case_is_corrected_and_runs():
    """Models miss a tool name by case far more often than by intent. Refusing
    the call teaches the model nothing it can act on."""
    agent, _provider, ran = _agent_with_search([
        ProviderResponse(tool_calls=[ToolCall("c1", "Search_Web", {"query": "x"})], stop_reason="tool_use"),
        ProviderResponse(text="done"),
    ])
    async with agent:
        assert (await agent.run("go")).ok
    assert ran == ["x"]


@pytest.mark.asyncio
async def test_an_ambiguous_name_is_not_guessed():
    """Two tools differing only by case means the correction is not obvious, so
    the model is told rather than sent somewhere it did not ask for."""
    provider = Scripted([
        ProviderResponse(tool_calls=[ToolCall("c1", "DOIT", {})], stop_reason="tool_use"),
        ProviderResponse(text="done"),
    ])
    agent = Agent(config=AgentConfig(model="m"), provider=provider)
    agent.tools.register_with_schema("doit", "a", {"type": "object", "properties": {}},
                                     lambda: "a", permission=PermissionLevel.ALLOW)
    agent.tools.register_with_schema("DoIt", "b", {"type": "object", "properties": {}},
                                     lambda: "b", permission=PermissionLevel.ALLOW)
    async with agent:
        assert (await agent.run("go")).ok
    assert "No tool named 'DOIT'" in _last_tool_text(provider)


@pytest.mark.asyncio
async def test_bad_arguments_come_back_as_one_actionable_line():
    agent, provider, ran = _agent_with_search([
        ProviderResponse(tool_calls=[ToolCall("c1", "search_web", {"quary": "x"})], stop_reason="tool_use"),
        ProviderResponse(text="done"),
    ])
    async with agent:
        assert (await agent.run("go")).ok
    message = _last_tool_text(provider)
    assert ran == [], "a call that does not validate must not reach the handler"
    assert "Invalid arguments for 'search_web'" in message


@pytest.mark.asyncio
async def test_an_unknown_tool_is_told_what_does_exist():
    agent, provider, _ran = _agent_with_search([
        ProviderResponse(tool_calls=[ToolCall("c1", "nonexistent", {})], stop_reason="tool_use"),
        ProviderResponse(text="done"),
    ])
    async with agent:
        assert (await agent.run("go")).ok
    message = _last_tool_text(provider)
    assert "No tool named 'nonexistent'" in message and "search_web" in message


# ── retry waits are spread, and permanent failures are not retried ──────────


def test_jitter_spreads_waits_downward_and_never_past_the_cap():
    """Concurrent agents sharing a rate limit must not retry in lockstep. The
    spread is downward only, so it cannot exceed the cap or a server's
    Retry-After."""
    from harnessx import RetryPolicy

    policy = RetryPolicy(backoff_seconds=1, max_backoff_seconds=5, jitter=0.25)
    assert policy.wait_for(3, rand=lambda: 0.0) == 4.0, "no draw means the full backoff"
    assert policy.wait_for(3, rand=lambda: 1.0) == 3.0, "a full draw removes the jitter fraction"
    assert policy.wait_for(10, rand=lambda: 0.0) <= 5.0, "still capped"
    assert policy.wait_for(1, retry_after=3, rand=lambda: 1.0) <= 3.0, "never longer than asked"

    exact = RetryPolicy(backoff_seconds=1, jitter=0)
    assert exact.wait_for(3) == 4.0, "jitter=0 restores exact backoff"

    with pytest.raises(Exception):
        RetryPolicy(jitter=1.5)


@pytest.mark.parametrize(
    ("message", "status", "retried"),
    [
        ("Rate limit exceeded", 429, True),
        ("You exceeded your current quota", 429, False),
        ("billing: payment required", 429, False),
        ("Your credit balance is too low", 429, False),
        ("prompt is too long: 250000 tokens", 400, False),
        ("maximum context length exceeded", 429, False),
        ("ThrottlingException: Too many tokens", 429, True),
        ("service unavailable", 503, True),
    ],
)
def test_permanent_and_overflow_failures_are_not_retried(message, status, retried):
    """A quota or billing 429 is not throttling: it will still be exhausted
    after any backoff, so the whole ladder is spent to be refused again. An
    overflowing request needs condensing, not another attempt."""
    from harnessx.providers.retry import is_transient

    error = type("E", (Exception,), {"status_code": status})(message)
    assert is_transient(error) is retried


# ── an agent that stopped making progress is told so ────────────────────────


def test_the_detector_catches_alternation_not_just_repetition():
    """A,B,A,B defeats a consecutive-identical counter entirely, which is what
    OpenCode's doom-loop check is. Cycles are the right unit."""
    from harnessx.loopguard import detect_cycle

    assert detect_cycle(["a", "a", "a"]) == 1
    assert detect_cycle(["a", "a"]) is None, "two is not yet a pattern"
    assert detect_cycle(["a", "b", "a", "b", "a", "b"]) == 2
    assert detect_cycle(["a", "b", "a", "b"]) is None
    assert detect_cycle(["a", "b", "c", "a", "b", "c", "a", "b", "c"]) == 3
    assert detect_cycle(["a", "a", "b", "a", "a"]) is None, "progress breaks the run"
    assert detect_cycle([]) is None
    assert detect_cycle(["a"] * 9, max_period=4, threshold=2) == 1


def test_a_signature_changes_when_the_answer_does():
    """Polling is the same call repeatedly, and is only a loop while the answer
    stays the same."""
    from harnessx.loopguard import call_signature

    same = call_signature("poll", {"id": 1}, "pending")
    assert call_signature("poll", {"id": 1}, "pending") == same
    assert call_signature("poll", {"id": 1}, "ready") != same, "a new answer is progress"
    assert call_signature("poll", {"id": 2}, "pending") != same
    assert call_signature("check", {"id": 1}, "pending") != same
    # Argument order must not matter.
    assert call_signature("t", {"a": 1, "b": 2}, "x") == call_signature("t", {"b": 2, "a": 1}, "x")


def _looping_agent(tool_names, *, replay_policy="safe", turns=10, handler=None):
    class Looper(LLMProvider):
        name = "looper"

        def __init__(self):
            self.n = 0

        async def create(self, **kwargs):
            self.n += 1
            if self.n > turns:
                return ProviderResponse(text="giving up")
            which = tool_names[(self.n - 1) % len(tool_names)]
            return ProviderResponse(
                tool_calls=[ToolCall(f"c{self.n}", which, {})], stop_reason="tool_use"
            )

        async def count_tokens(self, **kwargs):
            return 0

    provider = Looper()
    agent = Agent(config=AgentConfig(model="m"), provider=provider)
    for name in tool_names:
        agent.tools.register_with_schema(
            name, name, {"type": "object", "properties": {}},
            handler or (lambda: "nothing changed"),
            permission=PermissionLevel.ALLOW, replay_policy=replay_policy,
        )
    return agent, provider


async def _run_durable(agent, tmp_path):
    from harnessx import AgentRuntime, SQLiteBackend

    backend = await SQLiteBackend.connect(tmp_path / "r.db")
    async with backend, AgentRuntime(agent, backend=backend) as runtime:
        return await runtime.run("go")


@pytest.mark.asyncio
async def test_an_unattended_run_is_told_when_it_is_going_in_circles(tmp_path):
    agent, _provider = _looping_agent(["check", "poll"])
    seen: list[dict[str, Any]] = []
    agent.hooks.on(HookEvent.REPETITION, lambda ctx: seen.append(dict(ctx.data)))

    await _run_durable(agent, tmp_path)

    assert seen and seen[0]["period"] == 2, "the alternating pair is the cycle"
    notices = [m for m in agent.memory.get_messages() if "[harness]" in str(m.get("content"))]
    assert notices, "the notice has to reach the model, not just a hook"


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_policy", ["manual", "idempotent", "safe"])
async def test_the_guard_watches_every_tool_whatever_its_replay_policy(tmp_path, replay_policy):
    """``manual`` is the default policy, so an agent stuck on ``bash`` or
    ``write_file`` is the common case rather than the exotic one. Clearing the
    history on any non-safe success switched the guard off for exactly those."""
    agent, _provider = _looping_agent(["check", "poll"], replay_policy=replay_policy)
    seen: list[dict[str, Any]] = []
    agent.hooks.on(HookEvent.REPETITION, lambda ctx: seen.append(dict(ctx.data)))

    await _run_durable(agent, tmp_path)

    assert seen, f"{replay_policy} tools go unwatched"


@pytest.mark.asyncio
async def test_a_direct_run_is_watched_too():
    """``durable`` means the state is checkpointed, not that nobody is looking.
    A cron job calling ``Agent.run`` is unattended; an interactive tool using
    ``AgentRuntime`` has a person right there. So the guard does not key on it."""
    agent, _provider = _looping_agent(["check", "poll"])
    seen: list[dict[str, Any]] = []
    agent.hooks.on(HookEvent.REPETITION, lambda ctx: seen.append(dict(ctx.data)))
    async with agent:
        await agent.run("go")
    assert seen


@pytest.mark.asyncio
async def test_opting_out_is_one_flag():
    agent, _provider = _looping_agent(["check", "poll"])
    agent.config = replace(
        agent.config, limits=replace(agent.config.limits, loop_guard=LoopGuard(enabled=False))
    )
    seen: list[dict[str, Any]] = []
    agent.hooks.on(HookEvent.REPETITION, lambda ctx: seen.append(dict(ctx.data)))
    async with agent:
        await agent.run("go")
    assert not seen


@pytest.mark.asyncio
async def test_a_cycle_needs_identical_results_as_well_as_identical_arguments(tmp_path):
    """The shape this protects: change something, re-run the check, repeat. It
    looks like a loop from the call names alone, and must never be reported as
    one. Progress is encoded in the signature, which covers the result, so no
    separate progress rule is needed to tell the two apart."""
    laps = iter(range(100))

    def moving():
        return f"lap {next(laps)}"

    agent, _provider = _looping_agent(["edit", "verify"], replay_policy="manual", handler=moving)
    seen: list[dict[str, Any]] = []
    agent.hooks.on(HookEvent.REPETITION, lambda ctx: seen.append(dict(ctx.data)))

    await _run_durable(agent, tmp_path)

    assert not seen, "a result that keeps changing is progress, not a cycle"


def test_the_guard_decides_its_own_posture():
    assert LoopGuard().applies()
    assert not LoopGuard(enabled=False).applies()
    with pytest.raises(Exception):
        LoopGuard(threshold=1)
    with pytest.raises(TypeError):
        LoopGuard(enabled=None)


# ── a task list the model keeps seeing ──────────────────────────────────────


TODOS = [
    {"content": "research", "status": "completed"},
    {"content": "write the report", "status": "in_progress"},
    {"content": "ship", "status": "pending"},
]


def test_a_task_list_is_validated_before_it_is_believed():
    from harnessx.builtin.planning import normalize

    assert len(normalize(TODOS)) == 3
    assert "[x] research" in render(normalize(TODOS))
    assert "1/3 complete" in render(normalize(TODOS))
    assert render([]) == ""

    for bad, because in [
        ([{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}],
         "two items cannot both be in progress"),
        ([{"content": "", "status": "pending"}], "an item needs content"),
        ([{"content": "a", "status": "done"}], "status must be one of the three"),
        ("not a list", "the whole thing must be a list"),
    ]:
        with pytest.raises(ValueError):
            normalize(bad)
        assert because


def _planning_agent(script):
    # Planning is on by default; nothing to load.
    provider = Scripted(script)
    return Agent(config=AgentConfig(model="m"), provider=provider), provider


@pytest.mark.asyncio
async def test_the_list_is_sent_with_every_later_request():
    """OpenCode persists todos and never re-injects them, so compaction eats the
    tool result and the list silently vanishes from the model's view. An agent
    that believes it has a plan it can no longer read is worse off than one with
    no plan."""
    agent, provider = _planning_agent([
        ProviderResponse(tool_calls=[ToolCall("c1", "write_todos", {"todos": TODOS})], stop_reason="tool_use"),
        ProviderResponse(text="planned"),
        ProviderResponse(text="done"),
    ])
    async with agent:
        await agent.run("plan it")
        await agent.run("carry on")

    tail = provider.calls[-1]["messages"][-1]
    assert tail["role"] == "user", "never an assistant turn"
    assert "write the report" in str(tail["content"])
    assert "1/3 complete" in str(tail["content"])


@pytest.mark.asyncio
async def test_no_request_ever_carries_two_user_turns_in_a_row():
    """A tool-result message is user-role, so appending the reminder after one
    produced ['user', 'assistant', 'user', 'user']. Anthropic tolerates it;
    other providers do not, and a declared fallback can be any of them."""
    agent, provider = _planning_agent([
        ProviderResponse(tool_calls=[ToolCall("c1", "write_todos", {"todos": TODOS})], stop_reason="tool_use"),
        ProviderResponse(tool_calls=[ToolCall("c2", "write_todos", {"todos": TODOS})], stop_reason="tool_use"),
        ProviderResponse(text="planned"),
        ProviderResponse(text="done"),
    ])
    async with agent:
        await agent.run("plan it")
        await agent.run("carry on")

    for call in provider.calls:
        roles = [m["role"] for m in call["messages"]]
        assert all(a != b for a, b in zip(roles, roles[1:])), roles


@pytest.mark.asyncio
async def test_the_reminder_rides_along_with_the_tool_result_it_follows():
    agent, provider = _planning_agent([
        ProviderResponse(tool_calls=[ToolCall("c1", "write_todos", {"todos": TODOS})], stop_reason="tool_use"),
        ProviderResponse(text="planned"),
    ])
    async with agent:
        await agent.run("plan it")

    tail = provider.calls[-1]["messages"][-1]
    kinds = [b["type"] for b in tail["content"]]
    assert kinds == ["tool_result", "text"], "merged into the message, not appended after it"
    assert "1/3 complete" in tail["content"][-1]["text"]
    # The transcript itself is untouched; the reminder is outbound only.
    stored = [m for m in agent.memory.get_messages() if isinstance(m.get("content"), list)]
    assert all(
        b["type"] != "text" for m in stored for b in m["content"] if m["role"] == "user"
    ), "the merge must not write the reminder back into memory"


@pytest.mark.asyncio
async def test_the_reminder_stays_out_of_the_system_prompt():
    """The system prompt heads the cacheable prefix, so putting the list there
    would invalidate the prompt cache every time a task is ticked off."""
    agent, provider = _planning_agent([
        ProviderResponse(tool_calls=[ToolCall("c1", "write_todos", {"todos": TODOS})], stop_reason="tool_use"),
        ProviderResponse(text="planned"),
        ProviderResponse(text="done"),
    ])
    async with agent:
        await agent.run("plan it")
        await agent.run("carry on")

    systems = {str(call["system"]) for call in provider.calls}
    assert len(systems) == 1, "the system prompt must not change as the list changes"
    assert "write the report" not in systems.pop()


@pytest.mark.asyncio
async def test_the_list_outlives_condensing():
    """Condensing rewrites history. The list is rebuilt from the agent each
    turn, so it cannot be summarized away."""
    from harnessx import Limits

    provider = _Overflowing()
    agent = Agent(
        config=AgentConfig(model="m", max_tokens=1000, limits=Limits(max_context_tokens=10_000)),
        provider=provider,
    )
    agent.todos = list(TODOS)

    async with agent:
        for turn in range(5):
            await agent.run(f"turn {turn}")

    assert provider.summary_calls, "this test is only meaningful if condensing happened"
    assert agent.todos == TODOS, "the list is held on the agent, not in the transcript"
    assert "write the report" in render(agent.todos)


@pytest.mark.asyncio
async def test_no_reminder_when_nothing_is_planned():
    agent, provider = _planning_agent([ProviderResponse(text="done")])
    async with agent:
        await agent.run("just answer")
    assert "Task list" not in str(provider.calls[-1]["messages"])


def test_planning_is_on_by_default_and_can_be_turned_off():
    from harnessx import ToolRegistry
    from harnessx.providers import LLMProvider

    class Quiet(LLMProvider):
        name = "quiet"

        async def create(self, **kwargs):
            raise AssertionError("not called")

        async def count_tokens(self, **kwargs):
            return 0

    on = Agent(provider=Quiet())
    assert {"write_todos", "read_todos"} <= set(on.tools.list_tools())

    off = Agent(config=AgentConfig(planning=False), provider=Quiet())
    assert not {"write_todos", "read_todos"} & set(off.tools.list_tools())

    # A registry on its own stays bare; the Agent is what adds them.
    assert "write_todos" not in ToolRegistry().list_tools()
    assert set(ToolRegistry().load_builtin("planning")) == {"write_todos", "read_todos"}


def test_an_application_tool_of_the_same_name_wins():
    from harnessx import ToolRegistry
    from harnessx.providers import LLMProvider

    class Quiet(LLMProvider):
        name = "quiet"

        async def create(self, **kwargs):
            raise AssertionError("not called")

        async def count_tokens(self, **kwargs):
            return 0

    registry = ToolRegistry()
    registry.register_with_schema(
        "write_todos", "the application's own", {"type": "object", "properties": {}},
        lambda: "mine", permission=PermissionLevel.ALLOW,
    )
    agent = Agent(tools=registry, provider=Quiet())
    assert agent.tools.get_tool("write_todos").description == "the application's own"
    assert "read_todos" in agent.tools.list_tools(), "the other default is still added"


# ── a tool-heavy conversation can actually be condensed ─────────────────────


def _tool_heavy(rounds: int = 6) -> ConversationMemory:
    """The shape every agentic run has: each user turn carries a tool result,
    each assistant turn makes a tool call."""
    memory = ConversationMemory()
    memory.add_user_message("do the research")
    for index in range(rounds):
        memory.add_assistant_message(
            [{"type": "tool_use", "id": f"t{index}", "name": "search", "input": {}}]
        )
        memory.add_tool_results([ToolResult(tool_call_id=f"t{index}", content="results " * 50)])
    memory.add_assistant_message([{"type": "text", "text": "done"}])
    return memory


def test_a_tool_heavy_conversation_has_somewhere_to_cut():
    """The old rule wanted a plain user message preceded by a plain assistant
    message. In an agentic run no such point exists, so condensing silently
    never ran for precisely the conversations that needed it."""
    memory = _tool_heavy()
    assert memory._find_safe_trim_boundary(min_keep=4) > 0


def test_condensing_never_separates_a_tool_call_from_its_result():
    """The one rule the providers actually impose."""
    memory = _tool_heavy()
    dropped, split = memory.pending_condensation()
    assert split > 0 and dropped

    def ids(kind: str, key: str) -> set[str]:
        found = set()
        for message in dropped:
            content = message["content"]
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get("type") == kind:
                    found.add(str(block[key]))
        return found

    assert ids("tool_use", "id") <= ids("tool_result", "tool_use_id"), (
        "every call in the dropped prefix must be answered inside it"
    )

    memory.apply_condensation("SUMMARY", split)
    kept = memory.get_messages()
    answered = {
        str(block["tool_use_id"])
        for message in kept
        for block in (message["content"] if isinstance(message["content"], list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    }
    called = {
        str(block["id"])
        for message in kept
        for block in (message["content"] if isinstance(message["content"], list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    }
    assert answered <= called, "the kept tail must not start with an orphaned tool result"


@pytest.mark.asyncio
async def test_a_tool_heavy_conversation_over_budget_does_condense():
    memory = _tool_heavy()
    assert await memory.needs_condensing(
        _Counter(999_999), "m", "", [], max_context_tokens=14_000, reply_tokens=20_000
    ), "over budget with a valid cut point: condensing must be reported as needed"


@pytest.mark.asyncio
async def test_a_mid_turn_failure_leaves_a_transcript_a_provider_will_accept():
    """The generic failure handler marked the run failed and left the assistant
    turn's tool calls unanswered, so resuming the session sent a malformed
    request. Pre-existing; the truncation guard widened the same hole."""
    provider = Scripted([
        ProviderResponse(tool_calls=[ToolCall("c1", "probe", {})], stop_reason="tool_use"),
    ])
    agent = Agent(config=AgentConfig(model="m"), provider=provider)
    agent.tools.register_with_schema(
        "probe", "probe", {"type": "object", "properties": {}}, lambda: "ok",
        permission=PermissionLevel.ALLOW,
    )

    class Dies(Middleware):
        async def before_tool_execution(self, tool_call):
            raise RuntimeError("the worker died")

    agent.middleware.add(Dies())
    async with agent:
        result = await agent.run("go")

    assert result.status.value == "failed"
    assert not _unanswered(agent.memory.get_messages())


@pytest.mark.asyncio
async def test_a_fallback_condensation_that_also_fails_raises_rather_than_looping():
    """The fallback is tried once. Retrying it on its own failure would spin in
    the driver forever with nothing to show for it."""
    from harnessx import RetryPolicy

    provider = _Overflowing(fail_summary=True)
    agent = Agent(
        config=AgentConfig(
            model="m", max_tokens=1000,
            limits=Limits(max_context_tokens=10_000),
            retry=RetryPolicy(attempts=2, backoff_seconds=0.001, max_backoff_seconds=0.001),
        ),
        provider=provider,
    )

    async def refuse(*args, **kwargs):
        raise RuntimeError("cannot trim either")

    agent.memory.trim_if_needed = refuse

    async with agent:
        for turn in range(4):
            result = await agent.run(f"turn {turn}")
    assert result.status.value == "failed"
    assert "cannot trim either" in str(result.error)


class _Truncating(LLMProvider):
    """Cuts the first reply off at the token budget, mid tool call."""

    name = "truncating"

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return ProviderResponse(
                text="part", tool_calls=[ToolCall("c1", "probe", {})], stop_reason="max_tokens",
            )
        return ProviderResponse(text="recovered", stop_reason="end_turn")

    async def count_tokens(self, **kwargs):
        return 0


def _truncating_agent(provider):
    agent = Agent(config=AgentConfig(model="m"), provider=provider)
    agent.tools.register_with_schema(
        "probe", "probe", {"type": "object", "properties": {}}, lambda: "ok",
        permission=PermissionLevel.ALLOW,
    )
    return agent


@pytest.mark.asyncio
async def test_a_truncated_reply_leaves_a_stored_transcript_with_no_orphan():
    """Repairing only the state dict is not enough. `finish` snapshots
    `agent.memory`, and that snapshot put the unanswered tool_use straight back,
    so the stored transcript was still the malformed one. A unit test on
    `transition` alone cannot see this -- it takes a whole run."""
    provider = _Truncating()
    agent = _truncating_agent(provider)
    async with agent:
        result = await agent.run("go")

    assert result.truncated
    assert not _unanswered(agent.memory.get_messages()), agent.memory.get_messages()


@pytest.mark.asyncio
async def test_the_turn_after_a_truncated_reply_is_a_request_a_provider_accepts():
    """The point of the repair: the next request must be well formed, and must
    not carry two same-role turns in a row either."""
    provider = _Truncating()
    agent = _truncating_agent(provider)
    async with agent:
        await agent.run("go")
        await agent.run("carry on")

    sent = provider.calls[-1]["messages"]
    assert not _unanswered(sent), sent
    roles = [m["role"] for m in sent]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles


@pytest.mark.asyncio
async def test_a_durable_run_stores_the_repaired_transcript_too(tmp_path):
    from harnessx import AgentRuntime, SQLiteBackend

    provider = _Truncating()
    agent = _truncating_agent(provider)
    backend = await SQLiteBackend.connect(tmp_path / "r.db")
    async with backend, AgentRuntime(agent, backend=backend) as runtime:
        await runtime.run("go")
        await runtime.run("carry on")

    assert not _unanswered(provider.calls[-1]["messages"])
    assert not _unanswered(agent.memory.get_messages())


def test_coalescing_joins_only_neighbours_that_share_a_role():
    from harnessx.engine import coalesce_same_role

    joined = coalesce_same_role([
        {"role": "user", "content": "one"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "r"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": "two"},
    ])
    assert [m["role"] for m in joined] == ["user", "assistant", "user"]
    assert joined[0]["content"] == [
        {"type": "text", "text": "one"},
        {"type": "tool_result", "tool_use_id": "c1", "content": "r"},
    ]
    assert joined[-1]["content"] == "two", "a lone message is passed through untouched"


def test_coalescing_drops_nothing_a_provider_needs():
    from harnessx.engine import coalesce_same_role

    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "p", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "r"}]},
        {"role": "user", "content": "next"},
    ]
    assert not _unanswered(coalesce_same_role(messages))
