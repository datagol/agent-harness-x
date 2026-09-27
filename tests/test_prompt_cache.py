"""Harness-level prompt caching.

The engine owns the one thing every vendor needs: a stable prefix and the
positions where it ends. Providers translate that hint into their vendor's
mechanism and fail open. These tests pin the contract from both sides.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from harnessx import (
    Agent,
    AgentConfig,
    HookEvent,
    PermissionLevel,
    PromptCacheHint,
    PromptCachePolicy,
    ProviderResponse,
    StopReason,
    ToolCall,
)
from harnessx.prompt_cache import DISABLED, accepts_cache, build_hint, hint_from_wire, prefix_key
from harnessx.providers import LLMProvider


# ── Policy and hint types ─────────────────────────────────────────────────────


def test_caching_is_on_by_default_and_none_disables_it():
    assert AgentConfig().prompt_cache == PromptCachePolicy()
    assert AgentConfig(prompt_cache=None).prompt_cache is None


def test_policy_validates_its_fields():
    with pytest.raises(ValueError):
        PromptCachePolicy(ttl_seconds=0)
    with pytest.raises(ValueError):
        PromptCachePolicy(ttl_seconds=True)
    with pytest.raises(TypeError):
        PromptCachePolicy(key_salt=3)
    with pytest.raises(TypeError):
        AgentConfig(prompt_cache="yes")


def test_config_round_trips_through_a_snapshot_dict():
    config = AgentConfig(prompt_cache=PromptCachePolicy(ttl_seconds=3600, key_salt="tenant-a"))
    restored = AgentConfig(**asdict(config))
    assert restored.prompt_cache == config.prompt_cache
    assert AgentConfig(**asdict(AgentConfig(prompt_cache=None))).prompt_cache is None


def test_prefix_key_is_stable_and_sensitive_to_every_input():
    tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    base = prefix_key("m", "sys", tools)
    assert base == prefix_key("m", "sys", [dict(t) for t in tools])
    assert base != prefix_key("m2", "sys", tools)
    assert base != prefix_key("m", "sys ", tools)
    assert base != prefix_key("m", "sys", [])
    assert base != prefix_key("m", "sys", tools, salt="tenant-b")


def test_build_hint_marks_system_tools_and_last_message():
    hint = build_hint(PromptCachePolicy(), "m", "sys", [{"name": "t"}], [{"role": "user"}, {"role": "assistant"}, {"role": "user"}])
    assert hint.enabled and hint.breakpoints == ("system", "tools", "message:2")
    assert build_hint(PromptCachePolicy(cache_history=False), "m", "sys", [], [{"role": "user"}]).breakpoints == ("system",)
    assert build_hint(None, "m", "sys", [], []) is DISABLED


def test_hint_survives_the_wire_and_absent_means_legacy():
    hint = build_hint(PromptCachePolicy(ttl_seconds=60), "m", "sys", [], [{"role": "user"}])
    assert hint_from_wire({"prefix_key": hint.prefix_key, "breakpoints": ["system", "message:0"], "ttl_seconds": 60, "enabled": True}) == hint
    assert hint_from_wire(None) is None
    assert hint_from_wire(hint) is hint


def test_accepts_cache_requires_an_explicit_parameter():
    async def explicit(*, model, cache=None): ...
    async def open_ended(**kwargs): ...
    async def legacy(*, model): ...

    # **kwargs no longer counts: a provider that swallows unknown keywords would
    # otherwise be handed a hint it never reads.
    assert accepts_cache(explicit)
    assert not accepts_cache(open_ended)
    assert not accepts_cache(legacy)


# ── Engine wiring ─────────────────────────────────────────────────────────────


class RecordingProvider(LLMProvider):
    name = "recording"

    def __init__(self, responses):
        self.responses = list(responses)
        self.hints = []

    async def count_tokens(self, **kwargs):
        return 0

    async def create(self, *, model, messages, system, tools, max_tokens, temperature=None, cache=None):
        self.hints.append(cache)
        return self.responses.pop(0)


class LegacyProvider(LLMProvider):
    """Predates the cache keyword; must keep working untouched."""

    name = "legacy"

    async def count_tokens(self, **kwargs):
        return 0

    async def create(self, *, model, messages, system, tools, max_tokens, temperature=None):
        return ProviderResponse(text="ok")


def _tool_call_then_done():
    return [
        ProviderResponse(tool_calls=[ToolCall("c1", "echo", {"value": "x"})], stop_reason=StopReason.TOOL_USE),
        ProviderResponse(text="done"),
    ]


def test_the_engine_sends_a_stable_prefix_key_across_iterations():
    async def _run():
        provider = RecordingProvider(_tool_call_then_done())
        seen = []
        async with Agent(provider=provider) as agent:
            @agent.tools.register(permission=PermissionLevel.ALLOW)
            async def echo(value: str) -> str:
                return value

            agent.hooks.on(HookEvent.LLM_REQUEST, lambda ctx: seen.append(ctx.data.get("prefix_key")))
            result = await agent.run("hi")
        assert result.output == "done"
        first, second = provider.hints
        assert first.enabled and second.enabled
        assert first.prefix_key == second.prefix_key, "prefix drift between iterations defeats every cache"
        assert first.breakpoints == ("system", "tools", "message:0")
        assert second.breakpoints[-1] == "message:2", "history breakpoint tracks the last message"
        assert seen == [first.prefix_key, first.prefix_key]

    asyncio.run(_run())


def test_disabling_the_policy_sends_a_disabled_hint():
    async def _run():
        provider = RecordingProvider([ProviderResponse(text="done")])
        seen = []
        async with Agent(config=AgentConfig(prompt_cache=None), provider=provider) as agent:
            agent.hooks.on(HookEvent.LLM_REQUEST, lambda ctx: seen.append(ctx.data.get("prefix_key")))
            await agent.run("hi")
        assert provider.hints == [DISABLED]
        assert seen == [None]

    asyncio.run(_run())


def test_a_provider_without_the_keyword_is_called_without_it():
    async def _run():
        async with Agent(provider=LegacyProvider()) as agent:
            assert (await agent.run("hi")).output == "ok"

    asyncio.run(_run())


def test_policy_ttl_and_salt_reach_the_hint():
    async def _run():
        provider = RecordingProvider([ProviderResponse(text="done")])
        policy = PromptCachePolicy(ttl_seconds=3600, key_salt="tenant-a")
        async with Agent(config=AgentConfig(prompt_cache=policy), provider=provider) as agent:
            await agent.run("hi")
        hint = provider.hints[0]
        assert hint.ttl_seconds == 3600
        assert hint.prefix_key == prefix_key(agent.config.model, agent._build_system_prompt(), agent.tools.get_tool_params(), "tenant-a")

    asyncio.run(_run())


# ── Anthropic translation ─────────────────────────────────────────────────────


class FakeMessages:
    def __init__(self, reject_cache=False):
        self.calls = []
        self.reject_cache = reject_cache

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_cache and "cache_control" in str(kwargs):
            raise RuntimeError("invalid_request_error: cache_control is not supported here")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0),
        )


def _anthropic(reject_cache=False):
    from harnessx.providers.anthropic import AnthropicProvider

    messages = FakeMessages(reject_cache)
    return AnthropicProvider(client=SimpleNamespace(messages=messages)), messages


TOOLS = [{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}]


def test_anthropic_places_cache_control_at_each_breakpoint():
    async def _run():
        provider, fake = _anthropic()
        hint = PromptCacheHint(prefix_key="k", breakpoints=("system", "tools", "message:1"))
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "text", "text": "yo"}]}]
        await provider.create(model="m", messages=history, system="sys", tools=TOOLS, max_tokens=8, cache=hint)
        call = fake.calls[0]
        assert call["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
        assert call["tools"][-1]["cache_control"] == {"type": "ephemeral"}
        assert call["messages"][0]["content"] == "hi", "only the breakpoint message is marked"
        assert call["messages"][1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in str(TOOLS) and history[1]["content"][0] == {"type": "text", "text": "yo"}, "inputs are not mutated"

    asyncio.run(_run())


def test_anthropic_maps_a_long_ttl_and_wraps_string_messages():
    async def _run():
        provider, fake = _anthropic()
        hint = PromptCacheHint(prefix_key="k", breakpoints=("message:0",), ttl_seconds=7200)
        await provider.create(model="m", messages=[{"role": "user", "content": "hi"}], system=None, tools=[], max_tokens=8, cache=hint)
        assert fake.calls[0]["messages"][0]["content"] == [
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ]
        assert "system" not in fake.calls[0]

    asyncio.run(_run())


def test_anthropic_sends_no_markers_when_disabled_or_absent():
    async def _run():
        for cache in (None, DISABLED):
            provider, fake = _anthropic()
            await provider.create(model="m", messages=[{"role": "user", "content": "hi"}], system="sys", tools=TOOLS, max_tokens=8, cache=cache)
            assert "cache_control" not in str(fake.calls[0])
            assert fake.calls[0]["system"] == "sys"

    asyncio.run(_run())


def test_anthropic_fails_open_when_the_api_rejects_markers():
    async def _run():
        provider, fake = _anthropic(reject_cache=True)
        hint = PromptCacheHint(prefix_key="k", breakpoints=("system",))
        response = await provider.create(model="m", messages=[{"role": "user", "content": "hi"}], system="sys", tools=[], max_tokens=8, cache=hint)
        assert response.text == "ok"
        assert len(fake.calls) == 2 and "cache_control" not in str(fake.calls[1])

    asyncio.run(_run())


# ── OpenAI family ─────────────────────────────────────────────────────────────


class FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content="ok", tool_calls=None, role="assistant")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


def _openai_client():
    completions = FakeCompletions()
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_openai_sends_the_prefix_key_as_prompt_cache_key():
    pytest.importorskip("openai")
    from harnessx.providers.openai import OpenAIProvider

    async def _run():
        client, completions = _openai_client()
        provider = OpenAIProvider(client=client)
        hint = PromptCacheHint(prefix_key="abc", breakpoints=("system",))
        await provider.create(model="m", messages=[{"role": "user", "content": "hi"}], system="sys", tools=[], max_tokens=8, cache=hint)
        assert completions.calls[0]["extra_body"] == {"prompt_cache_key": "abc"}
        await provider.create(model="m", messages=[{"role": "user", "content": "hi"}], system="sys", tools=[], max_tokens=8, cache=DISABLED)
        assert "extra_body" not in completions.calls[1]

    asyncio.run(_run())


def test_azure_and_openrouter_do_not_send_prompt_cache_key():
    pytest.importorskip("openai")
    from harnessx.providers.azure_openai import AzureOpenAIProvider
    from harnessx.providers.openrouter import OpenRouterProvider

    async def _run():
        hint = PromptCacheHint(prefix_key="abc", breakpoints=("system",))
        for make in (lambda c: AzureOpenAIProvider(client=c), lambda c: OpenRouterProvider(client=c)):
            client, completions = _openai_client()
            await make(client).create(model="m", messages=[{"role": "user", "content": "hi"}], system="sys", tools=[], max_tokens=8, cache=hint)
            assert "extra_body" not in completions.calls[0]

    asyncio.run(_run())


# ── Gemini ────────────────────────────────────────────────────────────────────


class FakeGeminiCaches:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        return SimpleNamespace(name=f"cachedContents/{len(self.calls)}")


class FakeGeminiModels:
    def __init__(self):
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        part = SimpleNamespace(text="ok", function_call=None, thought=None, thought_signature=None)
        usage = SimpleNamespace(prompt_token_count=1, candidates_token_count=1, cached_content_token_count=0)
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason="STOP")], usage_metadata=usage)


def _gemini(fail=None, **kwargs):
    pytest.importorskip("google.genai")
    from harnessx.providers.gemini import GeminiProvider

    caches, models = FakeGeminiCaches(fail), FakeGeminiModels()
    client = SimpleNamespace(aio=SimpleNamespace(caches=caches, models=models))
    return GeminiProvider(client=client, **kwargs), caches, models


def _call(provider, cache):
    return provider.create(model="g", messages=[{"role": "user", "content": "hi"}], system="sys " * 50, tools=[], max_tokens=8, cache=cache)


def test_gemini_uses_the_hint_ttl_for_its_explicit_cache():
    async def _run():
        provider, caches, models = _gemini()
        hint = PromptCacheHint(prefix_key="k", breakpoints=("system",), ttl_seconds=3600)
        await _call(provider, hint)
        await _call(provider, hint)
        assert len(caches.calls) == 1 and caches.calls[0]["config"]["ttl"] == "3600s"
        assert all(call["config"]["cached_content"] == "cachedContents/1" for call in models.calls)

    asyncio.run(_run())


def test_gemini_without_a_ttl_relies_on_implicit_caching_only():
    async def _run():
        provider, caches, _ = _gemini()
        await _call(provider, PromptCacheHint(prefix_key="k", breakpoints=("system",)))
        assert caches.calls == []

    asyncio.run(_run())


def test_gemini_constructor_ttl_still_applies_and_a_disabled_hint_wins():
    async def _run():
        provider, caches, _ = _gemini(prompt_cache_ttl=120)
        await _call(provider, None)
        assert len(caches.calls) == 1, "legacy callers keep the constructor behaviour"
        await _call(provider, DISABLED)
        assert len(caches.calls) == 1, "prompt_cache=None must switch explicit caching off"

    asyncio.run(_run())


def test_gemini_holds_off_after_a_failed_create():
    async def _run():
        provider, caches, models = _gemini(fail=RuntimeError("content too small to cache"))
        hint = PromptCacheHint(prefix_key="k", breakpoints=("system",), ttl_seconds=3600)
        await _call(provider, hint)
        await _call(provider, hint)
        assert len(caches.calls) == 1, "a prefix that cannot be cached is not retried on every call"
        assert all("cached_content" not in call["config"] for call in models.calls)

    asyncio.run(_run())
