"""Loop reliability: the defects found by comparing our loop against OpenCode,
Hermes, Pi and Deep Agents. See doc/agent-loop-reliability.md."""

from __future__ import annotations

from typing import Any

import pytest

from harnessx import Agent, AgentConfig, HookEvent, PermissionLevel, ProviderResponse, ToolCall
from harnessx.execution import transition
from harnessx.memory import ConversationMemory
from harnessx.providers import LLMProvider
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
