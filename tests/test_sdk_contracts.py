"""Acceptance tests for the remaining SDK audit findings."""

import asyncio
import builtins
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harnessx import (
    Agent, AgentConfig, AgentRegistry, AgentRuntime, CliPermissionManager, ConversationMemory, Extension,
    HookEvent, Message, Middleware, PermissionLevel, PermissionManager, PersistentMemory,
    ProviderResponse, ResultSpillExtension, RunEvent, RunEventType, RunResult, RunStatus,
    Limits, RuntimeConfig, Sandbox, SandboxConfig, SQLiteBackend, ToolCall, ToolDefinition, ToolResult,
    TokenUsage,
)
from harnessx.providers import LLMProvider


class Provider(LLMProvider):
    def __init__(self, response=None):
        self.response = response or ProviderResponse(text="done")
        self.closed = 0

    async def create(self, **kwargs):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def count_tokens(self, **kwargs):
        return 0

    async def aclose(self):
        self.closed += 1


def owned_agent(**kwargs):
    agent = Agent(provider=Provider(), **kwargs)
    agent._owns_provider = True
    return agent


@pytest.mark.parametrize("values", [
    {"max_tokens": 0}, {"max_iterations": -1}, {"model_timeout_seconds": float("nan")},
    {"model_timeout_seconds": 0}, {"max_result_chars": -1}, {"max_context_tokens": False},
    {"temperature": -1}, {"max_cost_dollars": float("inf")}, {"provider": "unknown"}, {"model": ""},
])
def test_invalid_agent_configuration_is_rejected(values):
    # from_dict accepts the 0.3 flat keys without warnings and validates them.
    with pytest.raises((ValueError, TypeError)):
        AgentConfig.from_dict(values)


def test_provider_precedence_is_explicit():
    provider = Provider()
    provider.name = "openai"
    with pytest.raises(ValueError, match="requires AgentConfig"):
        Agent(provider=provider)
    with pytest.raises(ValueError, match="does not match"):
        Agent(provider=provider, config=AgentConfig(provider="anthropic"))
    with pytest.raises(TypeError, match="removed in harnessx 0.4"):
        Agent(provider=provider, client=object())
    assert Agent(provider=provider, config=AgentConfig(provider="openai", model="test-model")).provider is provider


def test_budget_and_context_configuration_is_wired():
    agent = Agent(provider=Provider(), config=AgentConfig(
        limits=Limits(max_cost_dollars=2, input_cost_per_m=3, output_cost_per_m=4, max_context_tokens=2048),
    ))
    assert agent.guardrails.max_cost_dollars == 2
    agent.guardrails.track_usage(TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000))
    assert agent.guardrails.estimated_cost == 7
    with pytest.raises(ValueError):
        RuntimeConfig(max_session_duration_seconds=-1)
    with pytest.raises(TypeError):  # removed in 0.4; the runtime checkpoints at every phase boundary
        RuntimeConfig(checkpoint_interval=5)


@pytest.mark.asyncio
async def test_permissions_inherit_defaults_and_never_prompt_implicitly(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *args: pytest.fail("Headless SDK read stdin"))
    definition = ToolDefinition("effect", "effect", {"type": "object"}, lambda: None)
    call = ToolCall("1", "effect", {})
    assert await PermissionManager().check_permission(call, definition)
    assert await PermissionManager(PermissionLevel.ALLOW).check_permission(call, definition)
    assert not await PermissionManager(PermissionLevel.ASK).check_permission(call, definition)
    assert not await PermissionManager(PermissionLevel.DENY).check_permission(call, definition)
    assert CliPermissionManager().get_effective_permission("effect", definition) == PermissionLevel.ASK
    definition.permission_level = PermissionLevel.DENY
    assert not await PermissionManager().check_permission(call, definition)
    definition.permission_level = PermissionLevel.ASK
    assert not await PermissionManager().check_permission(call, definition)
    callback = AsyncMock(return_value=True)
    manager = PermissionManager(approval_callback=callback)
    assert await manager.check_permission(call, definition)
    callback.assert_awaited_once()
    manager.set_permission("effect", PermissionLevel.DENY)
    manager.grant_session("effect")
    assert not await manager.check_permission(call, definition)


@pytest.mark.asyncio
async def test_resume_closes_replaced_factory_agent(tmp_path):
    created = []
    registry = AgentRegistry()
    def factory():
        agent = owned_agent()
        created.append(agent)
        return agent
    ref = registry.register("owned", factory)
    store = SQLiteBackend(tmp_path / "runtime.db")
    runtime = AgentRuntime(ref, backend=store, registry=registry)
    try:
        sid = await runtime.start()
        await runtime.run("hello")
        await runtime.pause()
        await runtime.resume(sid)
        assert len(created) == 2 and created[0].closed and created[0].provider.closed == 1
        assert not created[1].closed
    finally:
        await runtime.stop()
        await store.aclose()
    assert created[1].provider.closed == 1


@pytest.mark.asyncio
async def test_evaluation_factory_owned_agents_close_and_borrowed_stay_open():
    from harnessx.evals.target import AgentTarget
    created = []
    def factory():
        agent = owned_agent()
        created.append(agent)
        return agent
    await AgentTarget(factory, attach_langsmith=False).run_async({"prompt": "hello"})
    assert created[0].closed and created[0].provider.closed == 1
    borrowed = owned_agent()
    await AgentTarget(borrowed, attach_langsmith=False).run_async({"prompt": "hello"})
    assert not borrowed.closed
    await borrowed.aclose()


@pytest.mark.asyncio
async def test_factory_typeerror_is_not_retried():
    from harnessx.evals.target import AgentTarget
    calls = []
    def broken(inputs):
        calls.append(inputs)
        raise TypeError("inside factory")
    with pytest.raises(TypeError, match="inside factory"):
        await AgentTarget(broken).run_async({"prompt": "hello"})
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_tracing_install_failure_closes_owned_evaluation_agent(monkeypatch):
    import harnessx.evals.target as module
    created = owned_agent()

    class BrokenTracing(Extension):
        name = "langsmith"

        def __init__(self, **kwargs):
            pass

        def install(self, ctx):
            ctx.register_tool(lambda: "unused", name="tracing_fixture")
            raise ValueError("tracing install failed")

    monkeypatch.setattr(module, "LangSmithExtension", BrokenTracing)
    monkeypatch.setattr(module, "_LANGSMITH_AVAILABLE", True)
    with pytest.raises(ValueError, match="tracing install failed"):
        await module.AgentTarget(lambda: created).run_async({"prompt": "hello"})
    assert created.provider.closed == 1
    assert not created.tools.has_tool("tracing_fixture")


@pytest.mark.asyncio
async def test_failed_runtime_tool_registration_closes_created_agent(monkeypatch, tmp_path):
    import harnessx.core as module
    provider = Provider()
    monkeypatch.setattr(module, "make_provider", lambda name: provider)

    def registrar(tools):
        raise ValueError("registration failed")

    store = SQLiteBackend(tmp_path / "runtime.db")
    runtime = AgentRuntime(backend=store, tool_registrar=registrar)
    try:
        with pytest.raises(ValueError, match="registration failed"):
            await runtime.start()
        assert provider.closed == 1
    finally:
        await runtime.stop()
        await store.aclose()


@pytest.mark.asyncio
async def test_snapshot_survives_close_and_restores_config_state_and_artifacts(tmp_path):
    directory = str(tmp_path / "sessions")
    spill = ResultSpillExtension(threshold_bytes=10)
    agent = Agent(provider=Provider(), config=AgentConfig(system_prompt="retained", max_tokens=123), extensions=[spill])
    result = spill._maybe_spill(ToolResult("t", json.dumps([{"value": i} for i in range(30)])))
    agent.memory.add_tool_results([result])
    agent.session_metadata["custom"] = {"value": 7}
    original_path = next(iter(spill._results.values())).path
    sid = await agent.save_session(directory)
    saved = json.loads((Path(directory) / f"{sid}.json").read_text())
    assert saved["version"] == 2 and "artifact://" in json.dumps(saved)
    await agent.aclose()
    assert not Path(original_path).exists()
    restored_spill = ResultSpillExtension()
    restored = await Agent.load_session(sid, directory, provider=Provider(), extensions=[restored_spill])
    assert restored.config.system_prompt == "retained" and restored.config.max_tokens == 123
    assert restored.session_metadata["custom"] == {"value": 7}
    restored_path = Path(next(iter(restored_spill._results.values())).path)
    assert json.loads(restored_path.read_text())[3] == {"value": 3}
    await restored.aclose()
    assert not restored_path.exists()
    assert list((Path(directory) / "artifacts" / sid).iterdir())
    PersistentMemory(directory).delete_session(sid)
    assert not (Path(directory) / "artifacts" / sid).exists()


@pytest.mark.asyncio
async def test_snapshot_serialization_failure_preserves_previous_snapshot(tmp_path):
    agent = owned_agent()
    directory = str(tmp_path)
    sid = await agent.save_session(directory)
    original = (tmp_path / f"{sid}.json").read_bytes()
    agent.session_metadata["invalid"] = object()
    with pytest.raises(TypeError):
        await agent.save_session(directory)
    assert (tmp_path / f"{sid}.json").read_bytes() == original
    with pytest.raises(ValueError):
        PersistentMemory(directory).load_session("../escape")
    await agent.aclose()


class ProbeExtension(Extension):
    def __init__(self, name, log, priority=100):
        self.name, self.log, self.priority = name, log, priority

    def install(self, ctx):
        self.log.append((self.name, "install"))
        ctx.register_tool(lambda: "ok", name=self.name)
        ctx.add_middleware(Middleware())
        ctx.on_hook(HookEvent.AGENT_START, lambda ctx: None)
        ctx.register_prompt_provider(lambda: self.name)

    async def on_turn_complete(self, ctx, result):
        self.log.append((self.name, result.status))

    async def teardown(self):
        self.log.append((self.name, "close"))


@pytest.mark.asyncio
async def test_extension_priority_and_disposable_registrations():
    log = []
    agent = owned_agent(extensions=[ProbeExtension("late", log, 100), ProbeExtension("early", log, 1)])
    assert log[:2] == [("early", "install"), ("late", "install")]
    await agent.run("hello")
    assert ("late", RunStatus.COMPLETED) in log
    await agent.aclose()
    assert log[-2:] == [("late", "close"), ("early", "close")]
    assert agent.tools.list_tools() == ["read_tool_result", "write_todos", "read_todos"]
    assert not agent.middleware._middleware and not agent.prompt_providers
    assert not agent.hooks._hooks[HookEvent.AGENT_START]


def test_duplicate_extension_set_is_rejected_before_installation():
    log = []
    with pytest.raises(ValueError, match="Duplicate extension"):
        Agent(provider=Provider(), extensions=[ProbeExtension("same", log), ProbeExtension("same", log)])
    assert log == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_terminal_extension_callback_runs_on_failure_and_cancellation(cancel):
    log = []
    entered = asyncio.Event()
    class Blocking(Provider):
        async def create(self, **kwargs):
            entered.set()
            if cancel:
                await asyncio.Event().wait()
            raise ValueError("provider failed")
    agent = Agent(provider=Blocking(), extensions=[ProbeExtension("terminal", log)])
    task = asyncio.create_task(agent.run("hello"))
    await entered.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert (await task).status == RunStatus.FAILED
    assert ("terminal", RunStatus.CANCELLED if cancel else RunStatus.FAILED) in log
    await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "paused", "awaiting_input"])
async def test_terminal_activity_command_only_observes_terminal_states(status):
    from harnessx.engine import command, new_state
    log = []
    agent = owned_agent(extensions=[ProbeExtension("terminal", log)])
    state = new_state(agent, "fixture")
    state["status"] = status
    endings = []
    agent.hooks.on(HookEvent.AGENT_END, lambda ctx: endings.append(ctx.data["result"]))
    await command(agent, state, "complete", AsyncMock())
    terminal = status in ("completed", "failed", "cancelled")
    assert len(endings) == int(terminal)
    assert (("terminal", status) in log) == terminal
    await agent.aclose()


@pytest.mark.asyncio
async def test_cancelling_pending_approval_notifies_terminal_extension_once(tmp_path):
    log = []
    provider = Provider(ProviderResponse(tool_calls=[ToolCall("t", "ask", {})], stop_reason="tool_use"))
    agent = Agent(provider=provider, extensions=[ProbeExtension("terminal", log)])
    agent.tools.register(name="ask", permission=PermissionLevel.ASK)(lambda: "unused")
    store = SQLiteBackend(tmp_path / "runtime.db")
    runtime = AgentRuntime(agent, backend=store)
    try:
        await runtime.start()
        assert (await runtime.run("hello")).status == RunStatus.AWAITING_INPUT
        assert ("terminal", RunStatus.CANCELLED) not in log
        await runtime.cancel()
        await runtime.cancel()
        assert log.count(("terminal", RunStatus.CANCELLED)) == 1
    finally:
        await runtime.stop()
        await store.aclose()


@pytest.mark.asyncio
async def test_teardown_failure_does_not_skip_owned_provider_cleanup():
    class Broken(ProbeExtension):
        async def teardown(self):
            raise ValueError("cleanup failed")
    agent = owned_agent(extensions=[Broken("broken", [])])
    with pytest.raises(ExceptionGroup):
        await agent.aclose()
    assert agent.provider.closed == 1
    assert agent.tools.list_tools() == ["read_tool_result", "write_todos", "read_todos"]
    with pytest.raises(ExceptionGroup):
        await agent.aclose()
    assert agent.provider.closed == 1


def test_response_and_tool_result_legacy_views_remain_consistent():
    response = ProviderResponse(text="original", tool_calls=[ToolCall("a", "tool", {"value": 1})])
    response.text = "edited"
    response.tool_calls[0].input["value"] = 2
    assert response.content[0].text == "edited" and response.content[1].input == {"value": 2}
    response.content[1].input["value"] = 999
    assert response.tool_calls[0].input == {"value": 2}
    response.content = [{"type": "text", "text": "replaced"}]
    assert response.text == "replaced" and not response.tool_calls
    with pytest.raises(ValueError):
        ProviderResponse(text="mismatch", content=[{"type": "text", "text": "other"}])
    result = ToolResult(tool_use_id="legacy")
    result.tool_call_id = "canonical"
    assert result.tool_use_id == "canonical"
    with pytest.raises(ValueError):
        ToolResult(tool_call_id="a", tool_use_id="b")
    assert "tool_use_id" not in asdict(result)


def test_message_memory_is_validated_and_views_do_not_mutate_it():
    memory = ConversationMemory()
    supplied = [{"role": "assistant", "content": [{"type": "text", "text": "original"}]}]
    memory.set_messages(supplied)
    supplied[0]["content"][0]["text"] = "mutated"
    memory.get_messages()[0]["content"][0]["text"] = "mutated again"
    assert memory.get_messages()[0]["content"][0]["text"] == "original"
    assert isinstance(memory.messages[0], Message)
    with pytest.raises(ValueError):
        memory.set_messages([{"role": "assistant", "content": [{"type": "tool_use", "id": "bad"}]}])
    assert memory.get_messages()[0]["content"][0]["text"] == "original"


def test_event_payload_and_failure_validation():
    with pytest.raises(TypeError):
        RunEvent(RunEventType.TOOL_RESULT, "not a tool result")
    with pytest.raises(ValueError):
        RunResult("session", "run", error={"message": "missing type"})
    event = RunEvent(RunEventType.RUN_RESULT, RunResult("session", "run"))
    from harnessx.execution import wire
    assert RunEvent.from_dict(wire(event)) == event


@pytest.mark.asyncio
async def test_sandbox_selection_never_downgrades(monkeypatch):
    import harnessx.sandbox as module
    with pytest.raises(ValueError):
        SandboxConfig(tier="unknown")
    with pytest.raises(ValueError, match="cannot restrict"):
        SandboxConfig(tier="process", network_enabled=False)
    with pytest.raises(ValueError, match="cannot restrict"):
        SandboxConfig(tier="process", allowed_paths=["/tmp"])
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    sandbox = Sandbox(SandboxConfig(tier="seatbelt"))
    monkeypatch.setattr(sandbox, "_execute_process_command", AsyncMock(side_effect=AssertionError("fallback")))
    with pytest.raises(RuntimeError, match="Seatbelt requires"):
        await sandbox.execute_command("echo fixture")
    sandbox = Sandbox(SandboxConfig(tier="docker"))
    original_import = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "docker":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="no process fallback"):
        await sandbox.execute_command("echo fixture")
    await sandbox.cleanup()


@pytest.mark.asyncio
async def test_docker_timeout_removes_container_and_closes_client(monkeypatch, tmp_path):
    calls = []
    class Container:
        def wait(self, **kwargs):
            raise TimeoutError("fixture timed out")
        def remove(self, **kwargs):
            calls.append(("remove", kwargs))
    class Containers:
        def run(self, *args, **kwargs):
            calls.append(("run", kwargs))
            return Container()
    client = SimpleNamespace(containers=Containers(), close=lambda: calls.append(("close", {})))
    monkeypatch.setitem(sys.modules, "docker", SimpleNamespace(
        from_env=lambda: client, types=SimpleNamespace(Ulimit=lambda **kwargs: kwargs),
    ))
    async with Sandbox(SandboxConfig(tier="docker", allowed_paths=[str(tmp_path)])) as sandbox:
        result = await sandbox.execute_command("fixture")
    assert result.timed_out
    assert calls[0][1]["network_disabled"] is True
    assert calls[0][1]["volumes"][str(tmp_path.resolve())]["mode"] == "ro"
    assert calls[-2:] == [("remove", {"force": True}), ("close", {})]


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="Process sandbox supports POSIX")
async def test_cancelling_process_sandbox_reaps_child():
    async with Sandbox() as sandbox:
        pid_file = Path(sandbox.workdir) / "fixture.pid"
        task = asyncio.create_task(sandbox.execute(
            "import os, pathlib, time\n"
            "pathlib.Path('fixture.pid').write_text(str(os.getpid()))\n"
            "time.sleep(30)\n"
        ))
        try:
            async with asyncio.timeout(5):
                while not pid_file.exists():
                    if task.done():
                        pytest.fail(f"Fixture process failed to start: {await task}")
                    await asyncio.sleep(0.01)
            pid = int(pid_file.read_text())
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def test_offline_evaluation_never_constructs_a_langsmith_client(monkeypatch):
    import langsmith
    from harnessx.evals import build_example, evaluate_agent
    monkeypatch.setattr(langsmith, "Client", lambda *a, **k: pytest.fail("Offline evaluation constructed a client"))
    result = evaluate_agent(lambda: owned_agent(), [build_example("hello")], offline=True, print_summary=False)
    assert result.total_examples == 1 and result.url is None
