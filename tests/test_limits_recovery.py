"""Reply budgets resolved from model limits, and a loop that recovers when a
reply or a request hits a limit instead of ending the run.

The defect behind this: a hardcoded 8K budget cut a reply off mid tool call,
the harness refused the half-written call with "call the tool again" -- and
then ended the run, so the model never could.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from harnessx import Agent, AgentConfig, HookEvent, Limits, PermissionLevel, ProviderResponse, ToolCall
from harnessx import models
from harnessx.execution import RunEventType
from harnessx.hooks import HOOK_PAYLOADS, HookManager
from harnessx.models import (
    DEFAULT_OUTPUT_CAP,
    UNKNOWN_MODEL,
    ModelLimits,
    context_window_from_error,
    learn_model_limits,
    lookup_model_limits,
    normalize_model_id,
    register_model_limits,
    reply_limit_from_error,
    resolve_model_limits,
)
from harnessx.providers import LLMProvider
from harnessx.providers.base import INVALID_ARGUMENTS, parse_tool_arguments


@pytest.fixture(autouse=True)
def fresh_limits():
    """Learned and registered limits are process-wide; keep tests independent."""
    learned, registered = dict(models._learned), dict(models._registered)
    models._learned.clear()
    models._registered.clear()
    yield
    models._learned.clear()
    models._learned.update(learned)
    models._registered.clear()
    models._registered.update(registered)


class Scripted(LLMProvider):
    """Returns responses in order; an Exception entry is raised; a callable is called."""

    name = "scripted"

    def __init__(self, responses, *, tokens: int = 0):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.tokens = tokens

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if callable(item):
            item = item(kwargs)
        if isinstance(item, Exception):
            raise item
        return item

    async def count_tokens(self, **kwargs):
        return self.tokens


def _agent(provider, *, model="claude-sonnet-4-6", tools=True, **config):
    agent = Agent(config=AgentConfig(model=model, planning=False, **config), provider=provider)
    if tools:
        agent.ran: list[dict] = []

        def write_file(path: str, content: str) -> str:
            agent.ran.append({"path": path, "content": content})
            return f"wrote {len(content)} chars to {path}"

        agent.tools.register_with_schema(
            "write_file", "Write a file",
            {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
             "required": ["path", "content"]},
            write_file, permission=PermissionLevel.ALLOW,
        )
    return agent


def _cut(text="", *calls) -> ProviderResponse:
    return ProviderResponse(text=text, tool_calls=list(calls), stop_reason="max_tokens")


def _tool_results(request) -> list[dict]:
    last = request["messages"][-1]
    return [b for b in last["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]


# ── resolving limits ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw, expected", [
    ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
    ("anthropic/claude-opus-4-6", "claude-opus-4-6"),
    ("us.anthropic.claude-sonnet-4-6-v1:0", "claude-sonnet-4-6"),
    ("claude-opus-4-5@20251101", "claude-opus-4-5"),
    ("models/gemini-2.5-pro", "gemini-2.5-pro"),
    ("claude-3-5-haiku-latest", "claude-3-5-haiku"),
])
def test_deployment_ids_normalize_to_the_model_name(raw, expected):
    assert normalize_model_id(raw) == expected


def test_every_current_claude_generation_is_known_not_guessed():
    """The regex this replaces knew Claude 4 and nothing after it."""
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-opus-4-8", "claude-sonnet-4-6"):
        assert lookup_model_limits(model) == ModelLimits(1_000_000, 128_000), model
    assert lookup_model_limits("claude-haiku-4-5") == ModelLimits(200_000, 64_000)
    # Longest prefix wins: 3.5 Haiku is not the claude-3 family default.
    assert lookup_model_limits("claude-3-5-haiku-20241022").max_output == 8_192
    assert lookup_model_limits("claude-3-haiku-20240307").max_output == 4_096
    assert lookup_model_limits("gpt-4o-mini").max_output == 16_384


@pytest.mark.asyncio
async def test_an_unknown_model_gets_a_generous_default_not_8k():
    limits = await resolve_model_limits(Scripted([]), "some-new-model-2027")
    assert limits == UNKNOWN_MODEL and limits.max_output == DEFAULT_OUTPUT_CAP


@pytest.mark.asyncio
async def test_registered_limits_beat_everything():
    class Knows(Scripted):
        async def model_limits(self, model):
            return ModelLimits(1_000, 100)

    register_model_limits("claude-sonnet-4-6", context_window=500_000, max_output=50_000)
    assert await resolve_model_limits(Knows([]), "claude-sonnet-4-6") == ModelLimits(500_000, 50_000)


@pytest.mark.asyncio
async def test_a_limit_a_provider_reported_beats_the_table():
    learn_model_limits("claude-sonnet-4-6", max_output=16_000)
    limits = await resolve_model_limits(Scripted([]), "claude-sonnet-4-6")
    assert limits == ModelLimits(1_000_000, 16_000)


@pytest.mark.asyncio
async def test_anthropic_asks_the_models_api_once_and_falls_back_to_the_table():
    from types import SimpleNamespace

    from harnessx import AnthropicProvider

    asked = []

    async def retrieve(model):
        asked.append(model)
        return SimpleNamespace(max_input_tokens=400_000, max_tokens=96_000)

    provider = AnthropicProvider(client=SimpleNamespace(models=SimpleNamespace(retrieve=retrieve)))
    assert await resolve_model_limits(provider, "claude-next-1") == ModelLimits(400_000, 96_000)
    assert await resolve_model_limits(provider, "claude-next-1") == ModelLimits(400_000, 96_000)
    assert asked == ["claude-next-1"], "asked once, then remembered"

    broken = AnthropicProvider(client=object())  # no models API: the table answers
    assert await resolve_model_limits(broken, "claude-haiku-4-5") == ModelLimits(200_000, 64_000)


@pytest.mark.parametrize("message, expected", [
    ("max_tokens: 32000 > 8192, which is the maximum allowed number of output tokens for claude-x", (8192, True)),
    ("max_tokens is too large: 32000. This model supports at most 16384 completion tokens, whereas you provided 32000.", (16384, True)),
    ("input length and `max_tokens` exceed context limit: 190000 + 32000 > 200000, decrease input length or `max_tokens`", (200_000 - 190_000 - 1_024, False)),
    ("prompt is too long: 250000 tokens > 200000 maximum", None),
    ("rate limit exceeded", None),
])
def test_the_limit_a_refusal_names_is_read_back(message, expected):
    assert reply_limit_from_error(Exception(message)) == expected


def test_the_window_an_overflow_names_is_read_back():
    assert context_window_from_error(Exception("prompt is too long: 250000 tokens > 200000 maximum")) == 200_000
    assert context_window_from_error(Exception("This model's maximum context length is 128000 tokens.")) == 128_000
    assert context_window_from_error(Exception("bad request")) is None


# ── the budget sent ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_default_budget_comes_from_the_model_not_a_constant():
    provider = Scripted([ProviderResponse(text="hi")] * 2)
    async with _agent(provider, model="claude-opus-5-5") as agent:
        await agent.run("hi")
    async with _agent(provider, model="claude-3-haiku-20240307") as agent:
        await agent.run("hi")
    assert [c["max_tokens"] for c in provider.calls] == [32_000, 4_096]


@pytest.mark.asyncio
async def test_the_reply_is_fitted_into_what_is_left_of_the_window():
    register_model_limits("tiny-model", context_window=20_000, max_output=16_000)
    provider = Scripted([ProviderResponse(text="ok")], tokens=15_000)
    # Condensing is held off so the clamp is what is under test.
    agent = _agent(provider, model="tiny-model", tools=False, limits=Limits(max_context_tokens=10_000_000))
    for i in range(3):
        agent.memory.add_user_message(f"q{i}")
        agent.memory.add_assistant_message(f"a{i}")
    async with agent:
        await agent.run("go")
    assert provider.calls[0]["max_tokens"] == 20_000 - 15_000 - 1_024


# ── a reply cut off at the budget ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_tool_call_cut_off_is_refused_then_retried_with_a_bigger_budget():
    """The reported failure: the loop must survive the truncated call."""
    half = ToolCall("c1", "write_file", {"path": "big.py"})  # content never arrived
    whole = ToolCall("c2", "write_file", {"path": "big.py", "content": "x" * 50})
    provider = Scripted([
        _cut("Writing it now.", half),
        ProviderResponse(tool_calls=[whole], stop_reason="tool_use"),
        ProviderResponse(text="Done."),
    ])
    agent = _agent(provider)
    async with agent:
        result = await agent.run("write big.py")

    assert result.ok and result.output == "Done.", result
    assert agent.ran == [{"path": "big.py", "content": "x" * 50}], "the half call never ran"
    refused = _tool_results(provider.calls[1])
    assert refused[0]["tool_use_id"] == "c1" and refused[0]["is_error"]
    assert "cut off" in refused[0]["content"]
    budgets = [c["max_tokens"] for c in provider.calls]
    assert budgets == [32_000, 64_000, 64_000], "raised once, and kept for the rest of the run"


@pytest.mark.asyncio
async def test_text_cut_off_is_continued_and_joined():
    provider = Scripted([_cut("The first half, "), _cut("the second half, "), ProviderResponse(text="and the end.")])
    async with _agent(provider, tools=False) as agent:
        result = await agent.run("write an essay")
    assert result.ok and result.output == "The first half, the second half, and the end."
    assert "Continue exactly where it stopped" in str(provider.calls[1]["messages"][-1])
    assert [c["max_tokens"] for c in provider.calls] == [32_000, 64_000, 128_000]


@pytest.mark.asyncio
async def test_the_budget_never_grows_past_what_the_model_can_write():
    provider = Scripted([_cut("a"), _cut("b"), ProviderResponse(text="c")])
    async with _agent(provider, model="claude-3-haiku-20240307", tools=False) as agent:
        await agent.run("go")
    assert [c["max_tokens"] for c in provider.calls] == [4_096, 4_096, 4_096]


@pytest.mark.asyncio
async def test_an_explicit_budget_is_the_starting_point_not_a_ceiling():
    provider = Scripted([_cut("a"), ProviderResponse(text="b")])
    async with _agent(provider, tools=False, max_tokens=600) as agent:
        await agent.run("go")
    assert [c["max_tokens"] for c in provider.calls] == [600, 1_200]


@pytest.mark.asyncio
async def test_recovery_is_bounded_and_then_reports_the_truncation():
    provider = Scripted([_cut(f"part{i} ") for i in range(10)])
    async with _agent(provider, tools=False, limits=Limits(max_truncation_recoveries=2)) as agent:
        result = await agent.run("go")
    assert len(provider.calls) == 3, "the first reply plus two recoveries"
    assert result.truncated and not result.ok
    assert result.output == "part0 part1 part2 ", "nothing written is lost"


@pytest.mark.asyncio
async def test_each_recovery_is_reported_to_hooks():
    seen = []
    hooks = HookManager()
    hooks.on(HookEvent.RECOVERY, lambda ctx: seen.append(dict(ctx.data)))
    provider = Scripted([_cut("a", ToolCall("c1", "write_file", {})), ProviderResponse(text="ok")])
    agent = Agent(config=AgentConfig(planning=False), provider=provider, hooks=hooks)
    async with agent:
        await agent.run("go")
    assert seen == [{"reason": "truncated_tool_call", "attempt": 1, "reply_budget": 64_000}]
    assert set(seen[0]) == set(HOOK_PAYLOADS[HookEvent.RECOVERY].__annotations__)


@pytest.mark.asyncio
async def test_a_durable_run_recovers_the_same_way(tmp_path):
    from harnessx import AgentRuntime, SQLiteBackend

    provider = Scripted([_cut("half "), ProviderResponse(text="whole")])
    backend = await SQLiteBackend.connect(tmp_path / "r.db")
    async with backend, AgentRuntime(_agent(provider, tools=False), backend=backend) as runtime:
        result = await runtime.run("go")
    assert result.ok and result.output == "half whole"


# ── a request refused for asking too much ────────────────────────────────────


@pytest.mark.asyncio
async def test_a_budget_the_model_cannot_give_is_lowered_and_remembered():
    too_big = Exception("max_tokens: 32000 > 8192, which is the maximum allowed number of output tokens for m-2")
    provider = Scripted([too_big, ProviderResponse(text="first"), ProviderResponse(text="second")])
    async with _agent(provider, model="m-2", tools=False) as agent:
        first = await agent.run("go")
        second = await agent.run("again")
    assert first.ok and second.ok
    assert [c["max_tokens"] for c in provider.calls] == [32_000, 8_192, 8_192]


@pytest.mark.asyncio
async def test_a_request_too_long_for_the_window_is_condensed_and_repeated():
    def answer(kwargs):
        if kwargs.get("system") and "condense" in kwargs["system"]:
            return ProviderResponse(text="SUMMARY: earlier work")
        return ProviderResponse(text="fits now")

    overflow = Exception("prompt is too long: 250000 tokens > 200000 maximum")
    provider = Scripted([overflow, answer, answer])
    agent = _agent(provider, model="m-3", tools=False)
    for i in range(5):
        agent.memory.add_user_message(f"question {i}")
        agent.memory.add_assistant_message(f"answer {i}")
    async with agent:
        result = await agent.run("go")
    assert result.ok and result.output == "fits now"
    assert "condense" in provider.calls[1]["system"], "condensed before repeating"
    assert "[Earlier conversation, condensed]" in str(provider.calls[2]["messages"])
    assert (await resolve_model_limits(provider, "m-3")).context_window == 200_000, "the window it named is kept"


@pytest.mark.asyncio
async def test_an_overflow_with_nothing_to_condense_still_fails():
    provider = Scripted([Exception("prompt is too long: 250000 tokens > 200000 maximum")])
    async with _agent(provider, tools=False) as agent:
        result = await agent.run("x" * 100)
    assert result.failed and "too long" in result.error["message"]


# ── an empty reply ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_empty_reply_is_nudged_and_never_stored():
    provider = Scripted([ProviderResponse(text=""), ProviderResponse(text="  "), ProviderResponse(text="answer")])
    async with _agent(provider, tools=False) as agent:
        result = await agent.run("go")
        stored = agent.memory.get_messages()
    assert result.ok and result.output == "answer"
    assert all(m["role"] != "assistant" or m["content"] for m in stored)
    assert [m["role"] for m in stored].count("assistant") == 1
    assert "Your last reply was empty" in str(provider.calls[1]["messages"][-1])


@pytest.mark.asyncio
async def test_nudging_an_empty_reply_is_bounded():
    provider = Scripted([ProviderResponse(text="")] * 5)
    async with _agent(provider, tools=False) as agent:
        result = await agent.run("go")
    assert len(provider.calls) == 3, "the reply, then two nudges, then the run ends"
    assert result.status == "failed" and result.output == ""
    assert result.error["type"] == "RecoveryExhaustedError" and not result.ok


# ── tool calls that cannot run ───────────────────────────────────────────────


@pytest.mark.parametrize("raw, ok", [
    ('{"path": "a.py"}', True),
    ('{"content": "line one\nline two"}', True),  # a raw newline inside a string
    ('{"path": "a.py", "content": "unterminat', False),
    ("[1, 2]", False),
    ("", True),
])
def test_tool_arguments_parse_or_say_why_not(raw, ok):
    parsed = parse_tool_arguments(raw)
    assert isinstance(parsed, dict) and (INVALID_ARGUMENTS not in parsed) == ok


@pytest.mark.asyncio
async def test_unreadable_arguments_reach_the_model_as_the_reason():
    bad = ToolCall("c1", "write_file", parse_tool_arguments('{"path": "a.py", "content": "half'))
    provider = Scripted([
        ProviderResponse(tool_calls=[bad], stop_reason="tool_use"),
        ProviderResponse(text="I'll fix it."),
    ])
    agent = _agent(provider)
    async with agent:
        result = await agent.run("go")
    assert result.ok and not agent.ran
    answer = _tool_results(provider.calls[1])[0]
    assert answer["is_error"] and "not valid JSON" in answer["content"]


@pytest.mark.asyncio
async def test_a_tool_timeout_in_a_direct_run_is_told_to_the_model():
    """It used to stop the run "awaiting input" with an unanswered tool call
    that a direct run had no way to resolve, so every later request failed."""
    provider = Scripted([
        ProviderResponse(tool_calls=[ToolCall("c1", "slow", {})], stop_reason="tool_use"),
        ProviderResponse(text="moved on"),
        ProviderResponse(text="still fine"),
    ])
    agent = Agent(config=AgentConfig(planning=False), provider=provider)

    async def slow() -> str:
        await asyncio.sleep(10)
        return "never"

    agent.tools.register_with_schema(
        "slow", "slow", {"type": "object", "properties": {}}, slow,
        permission=PermissionLevel.ALLOW, timeout_seconds=0.05,
    )
    async with agent:
        result = await agent.run("go")
        after = await agent.run("next")
    assert result.ok and result.output == "moved on" and after.ok
    answer = _tool_results(provider.calls[1])[0]
    assert answer["is_error"] and "timed out" in answer["content"]



@pytest.mark.asyncio
async def test_a_tool_timeout_in_a_durable_run_says_what_timed_out(tmp_path):
    """The bare TimeoutError had no message, so the failed run's error read
    as an empty string."""
    from harnessx import AgentRuntime, SQLiteBackend
    from harnessx.types import ToolRetry

    provider = Scripted([ProviderResponse(tool_calls=[ToolCall("c1", "slow", {})], stop_reason="tool_use")])
    agent = Agent(config=AgentConfig(planning=False), provider=provider)

    async def slow() -> str:
        await asyncio.sleep(10)
        return "never"

    agent.tools.register_with_schema(
        "slow", "slow", {"type": "object", "properties": {}}, slow, permission=PermissionLevel.ALLOW,
        timeout_seconds=0.05, replay_policy="safe", retry=ToolRetry(attempts=1),
    )
    backend = await SQLiteBackend.connect(tmp_path / "r.db")
    async with backend, AgentRuntime(agent, backend=backend) as runtime:
        result = await runtime.run("go")
    assert not result.ok
    assert "Tool 'slow' timed out after 0.05s" in str(result.error), result.error

# ── condensing is not a retry ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_condensing_mid_run_is_not_reported_as_a_retry():
    class Long(Scripted):
        async def count_tokens(self, **kwargs):
            return 10 ** 9 if len(kwargs["messages"]) >= 6 else 0

    def answer(kwargs):
        if kwargs.get("system") and "condense" in kwargs["system"]:
            return ProviderResponse(text="SUMMARY")
        return ProviderResponse(text="ok")

    provider = Long([ProviderResponse(tool_calls=[ToolCall("c1", "write_file", {"path": "a", "content": "b"})],
                                      stop_reason="tool_use"), answer, answer])
    agent = _agent(provider)
    for i in range(3):
        agent.memory.add_user_message(f"q{i}")
        agent.memory.add_assistant_message(f"a{i}")
    async with agent:
        async with agent.run_stream("go") as stream:
            events = [event async for event in stream]
            result = await stream.result()
    assert result.ok
    assert not [e for e in events if e.type == RunEventType.ATTEMPT_RESET]
    assert not result.usage_incomplete


def test_a_fallback_member_keeps_the_budget_within_its_own_model():
    from harnessx.providers.fallback import _Member

    member = _Member(Scripted([]))
    member.model = "gpt-4o"
    assert member.request({"max_tokens": 32_000})["max_tokens"] == 16_384
