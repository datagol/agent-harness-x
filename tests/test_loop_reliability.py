"""Loop reliability: the defects found by comparing our loop against OpenCode,
Hermes, Pi and Deep Agents. See doc/agent-loop-reliability.md."""

from __future__ import annotations

from typing import Any

import pytest

from harnessx import Agent, AgentConfig, HookEvent, PermissionLevel, ProviderResponse, ToolCall
from harnessx.execution import transition
from harnessx.memory import ConversationMemory
from harnessx.providers import LLMProvider  # noqa: F401
from harnessx.types import ToolResult


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


def _state(response: dict) -> dict:
    return {"phase": "model", "status": "running", "response": response}


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


def _looping_agent(tool_names, *, replay_policy="safe", turns=10):
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
            name, name, {"type": "object", "properties": {}}, lambda: "nothing changed",
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
async def test_a_direct_run_is_left_alone_by_default():
    """Someone at a terminal can see a loop and stop it. An unattended run
    cannot, which is the only reason the guard defaults on there."""
    agent, _provider = _looping_agent(["check", "poll"])
    seen: list[dict[str, Any]] = []
    agent.hooks.on(HookEvent.REPETITION, lambda ctx: seen.append(dict(ctx.data)))
    async with agent:
        await agent.run("go")
    assert not seen


@pytest.mark.asyncio
async def test_progress_clears_the_history_so_edit_then_rerun_is_not_a_loop(tmp_path):
    """The shape this protects: change something, re-run the check, repeat. It
    looks exactly like a loop and must never be reported as one."""
    agent, _provider = _looping_agent(["edit", "verify"], replay_policy="manual")
    seen: list[dict[str, Any]] = []
    agent.hooks.on(HookEvent.REPETITION, lambda ctx: seen.append(dict(ctx.data)))

    await _run_durable(agent, tmp_path)

    assert not seen, "a successful call that is not repeat-safe counts as progress"


def test_the_guard_decides_its_own_posture():
    from harnessx.types import LoopGuard

    assert LoopGuard().applies(durable=True) and not LoopGuard().applies(durable=False)
    assert LoopGuard(enabled=True).applies(durable=False)
    assert not LoopGuard(enabled=False).applies(durable=True)
    with pytest.raises(Exception):
        LoopGuard(threshold=1)
