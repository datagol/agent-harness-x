"""Per-tool timeouts, and opt-in dedupe of identical repeated calls.

A tool that calls a remote API can hang indefinitely and take the whole turn
with it. And a model that re-emits the same call has the work done twice —
two identical side effects, two lots of latency, double the cost.

A tool without a timeout of its own gets DEFAULT_TOOL_TIMEOUT_SECONDS (25 s);
built-ins that legitimately run longer declare their own. Dedupe is off by
default.
"""

import asyncio


from harnessx.tools import ToolRegistry
from harnessx.types import PermissionLevel, ToolCall


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


def test_a_tool_well_inside_the_default_deadline_is_untouched():
    async def _run():
        registry = _registry()

        async def slow(**_):
            await asyncio.sleep(0.05)
            return "ok"

        _register(registry, slow)
        assert (await registry.execute(ToolCall(id="1", name="t", input={}))).content == "ok"

    asyncio.run(_run())



# --- the default for tools you register, and the built-ins that run longer ----


def test_a_tool_you_register_without_a_timeout_gets_25_seconds_however_it_is_registered():
    from harnessx.types import ToolDefinition

    registry = _registry()

    @registry.register()
    async def decorated() -> str:
        return "ok"

    async def plain() -> str:
        return "ok"

    registry.register_tool(plain)
    _register(registry, plain, name="schema")
    registry.register_tool(ToolDefinition("defined", "d", {"type": "object"}, plain))
    registry.register_tool(ToolDefinition("own", "d", {"type": "object"}, plain, timeout_seconds=90))
    timeouts = {t.name: t.timeout_seconds for t in registry.get_tools()}
    assert timeouts == {"decorated": 25.0, "plain": 25.0, "schema": 25.0, "defined": 25.0, "own": 90}


def test_a_tool_policy_reaches_a_tool_registered_as_a_definition():
    """register_tool(ToolDefinition) used to keep its own default and ignore
    ToolPolicy(default_timeout_seconds=...)."""
    from harnessx import Agent, AgentConfig
    from harnessx.types import ToolDefinition, ToolPolicy

    async def plain() -> str:
        return "ok"

    tools = [ToolDefinition("defined", "d", {"type": "object"}, plain),
             ToolDefinition("own", "d", {"type": "object"}, plain, timeout_seconds=90)]
    agent = Agent(config=AgentConfig(model="m", tools=ToolPolicy(default_timeout_seconds=7)), tools=tools)
    assert agent.tools.get_tool("defined").timeout_seconds == 7
    assert agent.tools.get_tool("own").timeout_seconds == 90

    registry = _registry()
    registry.register_tool(ToolDefinition("later", "d", {"type": "object"}, plain))
    registry.adopt_policy(ToolPolicy(default_timeout_seconds=3))
    assert registry.get_tool("later").timeout_seconds == 3


def test_built_ins_that_run_long_keep_longer_limits():
    from types import SimpleNamespace

    from harnessx.builtin.bash import register_bash_tools
    from harnessx.builtin.filesystem import register_filesystem_tools
    from harnessx.builtin.web import register_web_tools

    registry = _registry()
    register_bash_tools(registry)
    register_web_tools(registry)
    register_filesystem_tools(registry, include=["read_file"])
    assert registry.get_tool("run_bash").timeout_seconds == 300
    assert registry.get_tool("fetch_url").timeout_seconds == 60
    assert registry.get_tool("read_file").timeout_seconds == 25, "on the host, a file read is quick"

    inside = _registry()
    register_filesystem_tools(inside, include=["read_file"], sandbox=SimpleNamespace(owns_filesystem=True))
    assert inside.get_tool("read_file").timeout_seconds == 300, "the first call may start the sandbox"

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
