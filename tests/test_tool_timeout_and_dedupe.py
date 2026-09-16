"""Per-tool timeouts, and opt-in dedupe of identical repeated calls.

A tool that calls a remote API can hang indefinitely and take the whole turn
with it. And a model that re-emits the same call has the work done twice —
two identical side effects, two lots of latency, double the cost.

Both default to off, so nothing that exists today changes behaviour.
"""

import asyncio

import pytest

from datagol_agent_harness.tools import ToolRegistry
from datagol_agent_harness.types import PermissionLevel, ToolCall


def _registry(**kwargs):
    return ToolRegistry(**kwargs)


def _register(registry, fn, *, name="t", timeout=None):
    registry.register_with_schema(
        name=name,
        description="test tool",
        input_schema={"type": "object", "properties": {}},
        handler=fn,
        permission=PermissionLevel.ALLOW,
        **({"timeout_seconds": timeout} if timeout is not None else {}),
    )


# --- timeouts -----------------------------------------------------------------


def test_a_hanging_tool_is_abandoned_and_the_model_is_told():
    async def _run():
        registry = _registry()

        async def slow(**_):
            await asyncio.sleep(10)

        _register(registry, slow, timeout=0.05)
        result = await registry.execute(ToolCall(id="1", name="t", input={}))
        assert result.is_error is True
        assert "timed out" in result.content
        # The model must be able to act on this, so it names the tool.
        assert "'t'" in result.content

    asyncio.run(_run())


def test_a_tool_inside_its_budget_is_untouched():
    async def _run():
        registry = _registry()

        async def quick(**_):
            return "done"

        _register(registry, quick, timeout=5)
        result = await registry.execute(ToolCall(id="1", name="t", input={}))
        assert result.is_error is False
        assert result.content == "done"

    asyncio.run(_run())


def test_a_registry_default_applies_to_tools_without_their_own():
    async def _run():
        registry = _registry(default_timeout_seconds=0.05)

        async def slow(**_):
            await asyncio.sleep(10)

        _register(registry, slow)
        result = await registry.execute(ToolCall(id="1", name="t", input={}))
        assert result.is_error is True
        assert "timed out" in result.content

    asyncio.run(_run())


def test_a_tools_own_timeout_beats_the_registry_default():
    async def _run():
        registry = _registry(default_timeout_seconds=0.01)

        async def medium(**_):
            await asyncio.sleep(0.05)
            return "finished anyway"

        _register(registry, medium, timeout=5)
        result = await registry.execute(ToolCall(id="1", name="t", input={}))
        assert result.content == "finished anyway"

    asyncio.run(_run())


def test_no_timeout_is_the_default():
    """Existing consumers must not suddenly acquire a deadline."""
    async def _run():
        registry = _registry()

        async def slow(**_):
            await asyncio.sleep(0.05)
            return "ok"

        _register(registry, slow)
        assert (await registry.execute(ToolCall(id="1", name="t", input={}))).content == "ok"

    asyncio.run(_run())


# --- dedupe -------------------------------------------------------------------


def test_an_identical_repeated_call_runs_once():
    async def _run():
        registry = _registry(dedupe_calls=True)
        calls = []

        async def counted(**kwargs):
            calls.append(kwargs)
            return f"ran {len(calls)}"

        _register(registry, counted)
        first = await registry.execute(ToolCall(id="1", name="t", input={"q": "milk"}))
        second = await registry.execute(ToolCall(id="2", name="t", input={"q": "milk"}))

        assert len(calls) == 1
        assert second.content == first.content
        # The repeat still answers under its OWN call id, or the provider
        # cannot match the result to the request.
        assert second.tool_call_id == "2"

    asyncio.run(_run())


def test_different_arguments_still_run():
    async def _run():
        registry = _registry(dedupe_calls=True)
        calls = []

        async def counted(**kwargs):
            calls.append(kwargs)
            return "ok"

        _register(registry, counted)
        await registry.execute(ToolCall(id="1", name="t", input={"q": "milk"}))
        await registry.execute(ToolCall(id="2", name="t", input={"q": "bread"}))
        assert len(calls) == 2

    asyncio.run(_run())


def test_argument_order_does_not_defeat_the_dedupe():
    async def _run():
        registry = _registry(dedupe_calls=True)
        calls = []

        async def counted(**kwargs):
            calls.append(kwargs)
            return "ok"

        _register(registry, counted)
        await registry.execute(ToolCall(id="1", name="t", input={"a": 1, "b": 2}))
        await registry.execute(ToolCall(id="2", name="t", input={"b": 2, "a": 1}))
        assert len(calls) == 1

    asyncio.run(_run())


def test_dedupe_is_off_by_default():
    """A tool meant to be called repeatedly with the same arguments must not
    be silently collapsed."""
    async def _run():
        registry = _registry()
        calls = []

        async def counted(**kwargs):
            calls.append(kwargs)
            return "ok"

        _register(registry, counted)
        await registry.execute(ToolCall(id="1", name="t", input={"q": "milk"}))
        await registry.execute(ToolCall(id="2", name="t", input={"q": "milk"}))
        assert len(calls) == 2

    asyncio.run(_run())


def test_the_cache_can_be_cleared_between_turns():
    async def _run():
        registry = _registry(dedupe_calls=True)
        calls = []

        async def counted(**kwargs):
            calls.append(kwargs)
            return "ok"

        _register(registry, counted)
        await registry.execute(ToolCall(id="1", name="t", input={"q": "milk"}))
        registry.reset_call_cache()
        await registry.execute(ToolCall(id="2", name="t", input={"q": "milk"}))
        assert len(calls) == 2

    asyncio.run(_run())


def test_a_failing_call_is_cached_as_its_failure_not_retried_silently():
    """Whatever the first call produced is what the repeat sees — including
    an error. Re-running it would be a hidden retry with different rules
    from the provider retry."""
    async def _run():
        registry = _registry(dedupe_calls=True)
        calls = []

        async def boom(**_):
            calls.append(1)
            raise RuntimeError("nope")

        _register(registry, boom)
        first = await registry.execute(ToolCall(id="1", name="t", input={}))
        second = await registry.execute(ToolCall(id="2", name="t", input={}))
        assert first.is_error and second.is_error
        assert len(calls) == 1

    asyncio.run(_run())
