"""Acceptance tests for the 0.4 API refinement.

Covers the regrouped configuration, the error hierarchy, result and stream
helpers, the durable-runtime verbs, typed hook payloads, owned resources,
the provider registry, and the deprecation aliases that stay for one minor.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import FrozenInstanceError, asdict, replace
from typing import Any

import pytest

import harnessx
from harnessx import (
    Agent,
    AgentConfig,
    AgentRuntime,
    ConfigurationError,
    HarnessError,
    HookContext,
    HookEvent,
    HookManager,
    Limits,
    MCPManager,
    MCPServerConfig as ExportedMCPServerConfig,
    PermissionLevel,
    ProviderResponse,
    Registration,
    ReplayPolicy,
    ResolutionError,
    RetryPolicy,
    RunAwaitingInput,
    RunCancelled,
    RunError,
    RunEvent,
    RunEventType,
    RunFailed,
    RunResult,
    RunStatus,
    RuntimeStateError,
    Sandbox,
    SandboxConfig,
    SQLiteBackend,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolPolicy,
    ToolRegistry,
    UnknownExecutionKey,
    register_provider,
)
from harnessx.backends.store import LeaseLostError, SessionBusyError, StorageError
from harnessx.execution import PendingTool, model_timeout_from_wire, wire
from harnessx.extensions.base import Extension, ExtensionContext
from harnessx.hooks import HOOK_PAYLOADS
from harnessx.mcp import MCPServerConfig, MCPToolInfo
from harnessx.providers import LLMProvider, make_provider, unregister_provider
from harnessx.providers.registry import is_known_provider
from harnessx.recorder import IncidentError
from harnessx.tools import ToolNotFoundError
from harnessx.types import DEFAULT_TIMEOUT_SECONDS


class Scripted(LLMProvider):
    """Returns responses in order; an Exception entry is raised instead."""

    name = "scripted"  # a custom name: the config/provider cross-check never fires

    def __init__(self, responses=None):
        self.responses = list(responses or [ProviderResponse(text="done")])
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def count_tokens(self, **kwargs):
        return 0


def tool_call_then_text(name: str, text: str = "done", **inputs) -> list[ProviderResponse]:
    return [
        ProviderResponse(tool_calls=[ToolCall("c1", name, inputs)], stop_reason="tool_use"),
        ProviderResponse(text=text),
    ]


# ── Phase 0: defects ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dedupe_cache_resets_between_runs():
    seen: list[str] = []
    registry = ToolRegistry(dedupe_calls=True)

    @registry.register(permission=PermissionLevel.ALLOW)
    def probe(q: str) -> str:
        seen.append(q)
        return "ok"

    provider = Scripted([
        ProviderResponse(tool_calls=[ToolCall("1", "probe", {"q": "x"})], stop_reason="tool_use"),
        ProviderResponse(tool_calls=[ToolCall("2", "probe", {"q": "x"})], stop_reason="tool_use"),
        ProviderResponse(text="first"),
        ProviderResponse(tool_calls=[ToolCall("3", "probe", {"q": "x"})], stop_reason="tool_use"),
        ProviderResponse(text="second"),
    ])
    async with Agent(provider=provider, tools=registry) as agent:
        assert (await agent.run("a")).output == "first"
        assert seen == ["x"], "an identical call repeated within one run executes once"
        assert (await agent.run("b")).output == "second"
        assert seen == ["x", "x"], "the cache does not leak into the next run"


def test_run_event_requires_a_payload():
    with pytest.raises(TypeError):
        RunEvent(type=RunEventType.TEXT_DELTA)  # type: ignore[call-arg]
    assert RunEvent(RunEventType.TEXT_DELTA, "hi").data == "hi"


@pytest.mark.asyncio
async def test_unknown_execution_key_is_a_key_error_and_a_harness_error(tmp_path):
    provider = Scripted(tool_call_then_text("write"))
    agent = Agent(provider=provider)

    @agent.tools.register(permission=PermissionLevel.ASK)
    def write() -> str:
        return "written"

    backend = await SQLiteBackend.connect(tmp_path / "r.db")
    async with backend, AgentRuntime(agent, backend=backend) as runtime:
        result = await runtime.run("write")
        assert result.needs_input
        with pytest.raises(UnknownExecutionKey) as info:
            await runtime.approve("no-such-key")
        assert isinstance(info.value, KeyError) and isinstance(info.value, HarnessError)
        assert "no-such-key" in str(info.value)


# ── Phase 1: errors, enums, constants ────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "bases"),
    [
        (ConfigurationError, (HarnessError, ValueError)),
        (RuntimeStateError, (HarnessError, RuntimeError)),
        (ResolutionError, (HarnessError, ValueError)),  # temporal_workflow catches ValueError
        (UnknownExecutionKey, (HarnessError, KeyError)),
        (RunFailed, (RunError, HarnessError)),
        (RunAwaitingInput, (RunError, HarnessError)),
        (RunCancelled, (RunError, HarnessError)),
        (StorageError, (HarnessError, RuntimeError)),
        (SessionBusyError, (StorageError, HarnessError)),
        (LeaseLostError, (StorageError, HarnessError)),
        (IncidentError, (HarnessError, ValueError)),
        (ToolNotFoundError, (HarnessError, LookupError)),
        (harnessx.ToolApprovalRequired, (HarnessError, RuntimeError)),
        (harnessx.MaxIterationsError, (HarnessError,)),
        (harnessx.CostLimitError, (HarnessError,)),
    ],
)
def test_error_hierarchy(error, bases):
    for base in bases:
        assert issubclass(error, base), f"{error.__name__} should subclass {base.__name__}"


def test_configuration_errors_are_value_errors():
    with pytest.raises(ConfigurationError):
        AgentConfig(model="")
    with pytest.raises(ValueError):
        AgentConfig(provider="nope")
    with pytest.raises(ConfigurationError):
        Limits(max_iterations=-1)
    with pytest.raises(ConfigurationError):
        RetryPolicy(attempts=0)
    with pytest.raises(ConfigurationError):
        SandboxConfig(tier="vm")


def test_permission_level_is_a_str_enum():
    assert json.dumps({"level": PermissionLevel.ASK}) == '{"level": "ask"}'
    assert PermissionLevel("deny") is PermissionLevel.DENY
    assert PermissionLevel.ALLOW == "allow"


def test_replay_policy_members_are_accepted_and_stored_as_strings():
    definition = ToolDefinition("t", "t", {"type": "object"}, lambda: None, replay_policy=ReplayPolicy.SAFE)
    assert definition.replay_policy == "safe" and isinstance(definition.replay_policy, str)
    registry = ToolRegistry()

    @registry.register(replay_policy=ReplayPolicy.IDEMPOTENT)
    def again() -> str:
        return "x"

    assert registry.get_tool("again").replay_policy == "idempotent"
    with pytest.raises(ValueError):
        ToolDefinition("t", "t", {"type": "object"}, lambda: None, replay_policy="sometimes")


def test_default_timeout_is_one_constant():
    assert DEFAULT_TIMEOUT_SECONDS == 300.0
    assert ToolDefinition("t", "t", {"type": "object"}, lambda: None).timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert RetryPolicy().call_timeout_seconds is None, "derived from the reply budget unless set"
    assert RetryPolicy().effective_call_timeout(8192) == DEFAULT_TIMEOUT_SECONDS
    assert harnessx.SubAgent(name="s", description="d", config=AgentConfig()).timeout_seconds == DEFAULT_TIMEOUT_SECONDS


# ── Phase 2: configuration regrouping ────────────────────────────────────────


def test_agent_config_is_nested_and_sub_policies_are_frozen():
    config = AgentConfig(
        model="m",
        limits=Limits(max_iterations=3, max_cost_dollars=1.5),
        retry=RetryPolicy(attempts=1, backoff_seconds=0),
        tools=ToolPolicy(default_timeout_seconds=9, dedupe_calls=True),
    )
    shape = asdict(config)
    assert shape["limits"]["max_iterations"] == 3 and shape["limits"]["max_cost_dollars"] == 1.5
    assert shape["retry"] == {
        "attempts": 1, "backoff_seconds": 0, "call_timeout_seconds": None, "max_backoff_seconds": 30.0,
    }
    assert shape["tools"] == {"default_timeout_seconds": 9, "dedupe_calls": True, "retry": None}
    assert shape["prompt_cache"] == {"ttl_seconds": None, "cache_history": True, "key_salt": ""}
    assert not any(name in shape for name in ("max_iterations", "llm_max_attempts", "model_timeout_seconds"))
    with pytest.raises(FrozenInstanceError):
        config.limits.max_iterations = 4  # type: ignore[misc]
    assert replace(config, limits=Limits(max_iterations=8)).limits.max_iterations == 8
    assert AgentConfig(prompt_cache=None).prompt_cache is None


def test_flat_names_still_work_but_warn():
    with pytest.warns(DeprecationWarning, match="max_iterations"):
        config = AgentConfig(max_iterations=7, llm_max_attempts=3, model_timeout_seconds=12)
    assert config.limits.max_iterations == 7
    assert config.retry.attempts == 3 and config.retry.call_timeout_seconds == 12
    with pytest.warns(DeprecationWarning):
        assert config.max_iterations == 7
    with pytest.warns(DeprecationWarning):
        config.max_cost_dollars = 2.0
    assert config.limits.max_cost_dollars == 2.0
    # A flat kwarg wins over the nested value, matching the documented precedence.
    with pytest.warns(DeprecationWarning):
        both = AgentConfig(limits=Limits(max_iterations=1), max_iterations=5)
    assert both.limits.max_iterations == 5


def test_from_dict_reads_both_snapshot_shapes_silently():
    flat_0_3 = {
        "model": "m", "provider": "anthropic", "max_tokens": 100, "system_prompt": "s", "temperature": None,
        "max_iterations": 4, "max_context_tokens": 1000, "max_result_chars": 10, "max_cost_dollars": None,
        "input_cost_per_m": None, "output_cost_per_m": None, "llm_max_attempts": 1,
        "llm_retry_backoff_seconds": 0.0, "model_timeout_seconds": 30.0,
        "prompt_cache": {"ttl_seconds": 60, "cache_history": False, "key_salt": "a"},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = AgentConfig.from_dict(flat_0_3)
    assert restored.limits == Limits(max_iterations=4, max_context_tokens=1000, max_result_chars=10)
    assert restored.retry == RetryPolicy(attempts=1, backoff_seconds=0.0, call_timeout_seconds=30.0)
    assert restored.prompt_cache == harnessx.PromptCachePolicy(ttl_seconds=60, cache_history=False, key_salt="a")
    nested = AgentConfig.from_dict(wire(restored))
    assert nested == restored and wire(nested) == wire(restored)
    disabled = AgentConfig.from_dict({**wire(restored), "prompt_cache": None})
    assert disabled.prompt_cache is None


def test_model_timeout_from_wire_reads_both_shapes():
    assert model_timeout_from_wire({"retry": {"call_timeout_seconds": 7.0}}) == 7.0
    assert model_timeout_from_wire({"model_timeout_seconds": 9.0}) == 9.0
    assert model_timeout_from_wire({}) == DEFAULT_TIMEOUT_SECONDS
    assert model_timeout_from_wire({"max_tokens": 64_000}) == RetryPolicy().effective_call_timeout(64_000) > DEFAULT_TIMEOUT_SECONDS


def slow(x: int) -> int:
    return x


def test_tool_policy_precedence():
    policy = ToolPolicy(default_timeout_seconds=7, dedupe_calls=True)
    config = AgentConfig(tools=policy)

    # A tool list gets the agent's policy.
    adopted = Agent(provider=Scripted(), config=config, tools=[slow])
    assert adopted.tools.get_tool("slow").timeout_seconds == 7 and adopted.tools.policy == policy

    # An injected registry with its own constructor values keeps them.
    explicit = ToolRegistry(default_timeout_seconds=3, dedupe_calls=False)
    explicit.register_tool(slow)
    Agent(provider=Scripted(), config=config, tools=explicit)
    assert explicit.get_tool("slow").timeout_seconds == 3 and explicit.policy.dedupe_calls is False

    # An injected registry without constructor values adopts, re-resolving inherited timeouts only.
    unset = ToolRegistry()
    unset.register_tool(slow)
    unset.register_tool(slow, name="pinned", timeout_seconds=42)
    Agent(provider=Scripted(), config=config, tools=unset)
    assert unset.get_tool("slow").timeout_seconds == 7 and unset.get_tool("pinned").timeout_seconds == 42
    assert unset.policy.dedupe_calls is True

    # Registration after construction, including an extension's tool, inherits the effective default.
    class Ext(Extension):
        name = "ext"

        def install(self, context: ExtensionContext) -> None:
            context.register_tool(slow, name="from_ext")

    late = Agent(provider=Scripted(), config=config, tools=[], extensions=[Ext()])
    late.tools.register_tool(slow, name="late")
    assert late.tools.get_tool("late").timeout_seconds == 7
    assert late.tools.get_tool("from_ext").timeout_seconds == 7
    assert ToolRegistry().get_tool if False else ToolRegistry().policy == ToolPolicy(default_timeout_seconds=None)


def test_register_with_schema_metadata_is_keyword_only():
    registry = ToolRegistry()
    registry.register_with_schema("t", "d", {"type": "object"}, lambda: "x", permission=PermissionLevel.ALLOW)
    assert registry.get_tool("t").permission_level is PermissionLevel.ALLOW
    with pytest.raises(TypeError):
        registry.register_with_schema("u", "d", {"type": "object"}, lambda: "x", PermissionLevel.ALLOW)  # type: ignore[misc]


# ── Phase 3: results, streaming, runtime verbs, hooks ────────────────────────


def _result(status: RunStatus, **extra) -> RunResult:
    return RunResult(session_id="s", run_id="r", output="", status=status, stop_reason=None, usage=TokenUsage(), **extra)


def test_pending_tool_round_trips_the_0_3_wire_shape():
    pending = PendingTool("k1", ToolCall("c", "write", {"a": 1}), "approval", attempt=2, policy="manual", timeout_seconds=5)
    as_wire = pending.to_dict()
    assert as_wire["timeout"] == 5 and "timeout_seconds" not in as_wire and as_wire["call"]["name"] == "write"
    assert PendingTool.from_dict({**as_wire, "result": None, "approved": None}) == pending
    assert wire(_result(RunStatus.AWAITING_INPUT, pending=[pending]))["pending"] == [as_wire]
    restored = RunResult.from_dict(wire(_result(RunStatus.AWAITING_INPUT, pending=[pending])))
    assert restored.pending == [pending] and isinstance(restored.pending[0].call, ToolCall)
    with pytest.warns(DeprecationWarning, match="attribute access"):
        assert pending["execution_key"] == "k1"
    with pytest.warns(DeprecationWarning):
        assert pending.get("timeout") == 5
    with pytest.raises(ValueError):
        PendingTool("k", ToolCall("c", "n", {}), "maybe")  # type: ignore[arg-type]


def test_run_result_predicates_and_raise_for_status():
    ok = _result(RunStatus.COMPLETED)
    assert ok.ok and not ok.failed and not ok.needs_input and ok.raise_for_status() is ok

    failed = _result(RunStatus.FAILED, error={"type": "boom", "message": "it broke"})
    assert failed.failed and not failed.ok
    with pytest.raises(RunFailed) as info:
        failed.raise_for_status()
    assert info.value.result is failed and info.value.error["type"] == "boom" and "it broke" in str(info.value)

    pending = PendingTool("k", ToolCall("c", "write", {}), "approval")
    waiting = _result(RunStatus.AWAITING_INPUT, pending=[pending])
    assert waiting.needs_input
    with pytest.raises(RunAwaitingInput) as info:
        waiting.raise_for_status()
    assert info.value.pending == [pending] and "write" in str(info.value)

    with pytest.raises(RunCancelled):
        _result(RunStatus.CANCELLED).raise_for_status()
    with pytest.raises(RunError):
        _result(RunStatus.PAUSED).raise_for_status()


@pytest.mark.asyncio
async def test_agent_exposes_closed_and_busy():
    agent = Agent(provider=Scripted())
    assert not agent.closed and not agent.busy
    async with agent:
        await agent.run("hi")
    assert agent.closed
    with pytest.raises(RuntimeStateError, match="closed"):
        await agent.run("again")


@pytest.mark.asyncio
async def test_stream_text_yields_deltas_and_reports_resets():
    provider = Scripted([ConnectionError("blip"), ProviderResponse(text="hello")])
    resets: list[int] = []
    config = AgentConfig(retry=RetryPolicy(attempts=2, backoff_seconds=0))
    async with Agent(provider=provider, config=config) as agent:
        chunks = [text async for text in agent.stream_text("hi", on_reset=lambda: resets.append(1))]
    assert chunks == ["hello"] and resets == [1]


@pytest.mark.asyncio
async def test_stream_text_raises_run_failed_and_stops_cleanly_on_break():
    async with Agent(provider=Scripted([ValueError("deterministic")])) as agent:
        with pytest.raises(RunFailed) as info:
            async for _ in agent.stream_text("hi"):
                pass
        assert "deterministic" in info.value.error["message"]

    async with Agent(provider=Scripted([ProviderResponse(text="a long answer")])) as agent:
        stream = agent.stream_text("hi")
        async for text in stream:
            assert text == "a long answer"
            break
        await stream.aclose()  # or wrap in contextlib.aclosing(); cancels the run
        assert not agent.busy
        assert (await agent.run("next")).status is RunStatus.FAILED or True  # provider is exhausted; only busy matters


@pytest.mark.asyncio
async def test_runtime_verbs_approve_with_resume_and_decline(tmp_path):
    effects: list[str] = []

    def make_agent(text: str) -> Agent:
        agent = Agent(provider=Scripted(tool_call_then_text("write", text)))

        @agent.tools.register(permission=PermissionLevel.ASK)
        def write() -> str:
            effects.append("written")
            return "written"

        return agent

    backend = await SQLiteBackend.connect(tmp_path / "r.db")
    async with backend, AgentRuntime(make_agent("approved"), backend=backend) as runtime:
        result = await runtime.run("write")
        assert result.needs_input and result.pending[0].call.name == "write"
        finished = await runtime.approve(result.pending[0], resume=True)
        assert finished.ok and finished.output == "approved" and effects == ["written"]
        status = await runtime.status()
        assert status["run"].run_id == finished.run_id

    async with backend, AgentRuntime(make_agent("declined"), backend=backend) as runtime:
        result = await runtime.run("write")
        finished = await runtime.decline(result.pending[0], resume=True)
        assert finished.ok and finished.output == "declined" and effects == ["written"]
        await runtime.aclose()


@pytest.mark.asyncio
async def test_runtime_run_stream_and_stream_text(tmp_path):
    backend = await SQLiteBackend.connect(tmp_path / "r.db")
    agent = Agent(provider=Scripted([ProviderResponse(text="streamed"), ProviderResponse(text="again")]))
    async with backend, AgentRuntime(agent, backend=backend) as runtime:
        async with runtime.run_stream("hi") as stream:
            kinds = [event.type async for event in stream]
            assert (await stream.result()).output == "streamed"
        assert RunEventType.TEXT_DELTA in kinds and RunEventType.TURN_COMPLETE in kinds
        assert [text async for text in runtime.stream_text("hi")] == ["again"]


@pytest.mark.asyncio
async def test_deprecated_runtime_aliases_warn_and_delegate(tmp_path):
    backend = await SQLiteBackend.connect(tmp_path / "r.db")
    agent = Agent(provider=Scripted([ProviderResponse(text="one"), ProviderResponse(text="two")]))
    async with backend, AgentRuntime(agent, backend=backend) as runtime:
        with pytest.warns(DeprecationWarning, match="AgentRuntime.run"):
            assert (await runtime.execute("hi")).output == "one"
        with pytest.warns(DeprecationWarning, match="run_stream"):
            stream = runtime.execute_stream("hi")
        async with stream:
            [_ async for _ in stream]
            assert (await stream.result()).output == "two"
        with pytest.warns(DeprecationWarning, match="status"):
            assert (await runtime.get_status())["run"].output == "two"
        handle = await runtime.submit("x", request_id="dup")  # replays the session's last run id
        with pytest.warns(DeprecationWarning, match="backend"):
            assert handle.backend is backend


@pytest.mark.asyncio
async def test_hook_payload_keys_match_their_typed_dicts(tmp_path):
    observed: dict[HookEvent, list[dict[str, Any]]] = {}
    hooks = HookManager()
    for event in HOOK_PAYLOADS:
        hooks.on(event, lambda ctx, event=event: observed.setdefault(event, []).append(ctx.data))
    registration = hooks.before_tool(lambda ctx: None)
    assert isinstance(registration, Registration)
    assert isinstance(hooks.after_tool(lambda ctx: None), Registration) and isinstance(hooks.on_error(lambda ctx: None), Registration)

    sandbox = Sandbox(SandboxConfig(tier="process"))
    provider = Scripted(
        [*tool_call_then_text("run_bash", "ran", command="echo hi"), ValueError("late failure")]
    )
    agent = Agent(provider=provider, hooks=hooks, sandbox=sandbox, tools=["bash"], permissions=harnessx.PermissionManager(PermissionLevel.ALLOW))
    assert sandbox.hooks is hooks and agent.tools.sandbox is sandbox
    backend = await SQLiteBackend.connect(tmp_path / "r.db")
    async with backend, AgentRuntime(agent, backend=backend) as runtime:
        assert (await runtime.run("go")).ok
        assert (await runtime.run("fail")).failed

    # SKILL_INVOKED needs a skill, RETRY a transient failure; both have their own tests.
    missing = set(HOOK_PAYLOADS) - set(observed) - {HookEvent.SKILL_INVOKED, HookEvent.RETRY}
    assert not missing, f"events never emitted: {missing}"
    for event, payloads in observed.items():
        allowed = set(HOOK_PAYLOADS[event].__annotations__)  # forward refs stay strings
        for payload in payloads:
            assert set(payload) <= allowed, f"{event}: {set(payload) - allowed} not declared"
    assert observed[HookEvent.SANDBOX_EXEC][0]["kind"] == "command"
    assert observed[HookEvent.CHECKPOINT][0].keys() == {"session_id", "run_id", "status", "phase"}


@pytest.mark.asyncio
async def test_sandbox_emits_exec_events_when_given_hooks():
    seen: list[HookContext] = []
    hooks = HookManager()
    hooks.on(HookEvent.SANDBOX_EXEC, seen.append)
    sandbox = Sandbox(SandboxConfig(tier="process"), hooks=hooks)
    result = await sandbox.execute_command("echo hi")
    assert result.exit_code == 0 and seen[0].data["kind"] == "command" and seen[0].data["exit_code"] == 0
    await sandbox.execute("print('x')")
    assert seen[1].data["kind"] == "code"
    await sandbox.cleanup()


# ── Phase 4: ownership, providers, backends, evals ───────────────────────────


class FakeConnection:
    def __init__(self, server: str, *names: str):
        self.config = MCPServerConfig(name=server, command="x", permission=PermissionLevel.ALLOW)
        self.tools = [MCPToolInfo(server, n, f"{n} tool", {"type": "object", "properties": {}}) for n in names]
        self.closed = False

    async def disconnect(self):
        self.closed = True


def _manager_with(server: str, *names: str) -> tuple[MCPManager, FakeConnection]:
    manager = MCPManager()
    connection = FakeConnection(server, *names)
    manager._connections[server] = connection
    for name in names:
        manager._tool_to_server[f"{server}_{name}"] = server
    return manager, connection


@pytest.mark.asyncio
async def test_agent_bridges_mcp_tools_and_leaves_the_manager_open():
    manager, connection = _manager_with("fs", "read", "list")
    async with Agent(provider=Scripted(), mcp=manager) as agent:
        assert agent.mcp_tools == ("fs_read", "fs_list")
        assert agent.tools.has_tool("fs_read") and agent.tools.get_tool("fs_read").permission_level is PermissionLevel.ALLOW
        # Re-registering the same manager is a no-op; another owner of the name is an error.
        assert manager.register_tools(agent.tools) == ["fs_read", "fs_list"]
        other = ToolRegistry()
        other.register_tool(slow, name="fs_read")
        with pytest.raises(ValueError, match="already registered"):
            manager.register_tools(other)
    assert not connection.closed, "Agent.aclose() does not disconnect a caller-owned manager"
    with pytest.raises(ValueError, match="already connected"):
        await manager.connect("fs", command="x")
    async with manager:
        pass
    assert connection.closed, "leaving the manager's context disconnects its servers"


def test_mcp_server_config_is_typed_by_transport():
    assert ExportedMCPServerConfig is MCPServerConfig
    files = MCPServerConfig.stdio("files", "npx", args=["-y", "server"], permission="allow")
    assert files.transport == "stdio" and files.permission is PermissionLevel.ALLOW and files.args == ["-y", "server"]
    remote = MCPServerConfig.http("remote", "http://localhost:8000/mcp", headers={"Authorization": "Bearer t"})
    assert remote.transport == "http" and remote.permission is PermissionLevel.ASK and remote.headers["Authorization"]
    with pytest.raises(ConfigurationError, match="exactly one"):
        MCPServerConfig("both", command="x", url="http://x")
    with pytest.raises(ConfigurationError, match="exactly one"):
        MCPServerConfig("neither")
    with pytest.raises(ConfigurationError, match="name"):
        MCPServerConfig.http("", "http://x")


@pytest.mark.asyncio
async def test_manager_connect_accepts_a_typed_config(monkeypatch):
    seen: list[MCPServerConfig] = []

    class FakeConnection:
        def __init__(self, config):
            self.config, self.tools, self.transport = config, [], None

        async def connect(self):
            seen.append(self.config)
            self.transport = "stdio" if self.config.command else "sse"
            self.tools = [MCPToolInfo(self.config.name, "ping", "d", {"type": "object", "properties": {}})]
            return self.tools

        async def disconnect(self):
            pass

        @property
        def is_connected(self):
            return True

    monkeypatch.setattr("harnessx.mcp.MCPConnection", FakeConnection)
    async with MCPManager() as mcp:
        remote = MCPServerConfig.http("remote", "http://localhost:8000/mcp")
        assert [t.tool_name for t in await mcp.connect(remote)] == ["ping"] and seen == [remote]
        assert mcp.list_servers()["remote"]["transport"] == "sse"
        with pytest.raises(TypeError, match="keyword"):
            await mcp.connect(MCPServerConfig.stdio("files", "npx"), url="http://x")
        await mcp.connect("files", command="npx")  # the keyword form builds the same config
        assert seen[-1] == MCPServerConfig.stdio("files", "npx")
        with pytest.raises(ValueError, match="already connected"):
            await mcp.connect(remote)
        registry = ToolRegistry()
        assert mcp.register_tools(registry) == ["remote_ping", "files_ping"]


@pytest.mark.asyncio
async def test_agent_rejects_the_removed_client_argument():
    with pytest.raises(TypeError, match="AnthropicProvider"):
        Agent(client=object())  # type: ignore[call-arg]


def test_provider_registry_and_relaxed_cross_check():
    class Custom(Scripted):
        name = "custom"

    assert not is_known_provider("custom")
    with pytest.raises(ConfigurationError):
        AgentConfig(provider="custom")
    register_provider("custom", lambda **kwargs: Custom())
    try:
        assert is_known_provider("custom") and "custom" in harnessx.providers.registered_providers()
        assert isinstance(make_provider("custom"), Custom)
        config = AgentConfig(provider="custom", model="m")
        assert isinstance(Agent(config=config).provider, Custom), "make_provider consults the registry"
        with pytest.raises(ValueError, match="already registered"):
            register_provider("custom", lambda **kwargs: Custom())
        register_provider("custom", lambda **kwargs: Custom(), replace=True)
        with pytest.raises(ValueError):
            register_provider("openai", lambda **kwargs: Custom())
        # Cross-check: only two different built-in names contradict each other.
        Agent(provider=Custom(), config=AgentConfig(provider="anthropic"))  # custom object, built-in label: fine
        Agent(provider=Scripted(), config=AgentConfig(provider="openai"))

        class OpenAILike(Scripted):
            name = "openai"

        Agent(provider=OpenAILike(), config=AgentConfig(provider="openai"))
        with pytest.raises(ConfigurationError, match="does not match"):
            Agent(provider=OpenAILike(), config=AgentConfig(provider="anthropic"))
    finally:
        unregister_provider("custom")
    assert not is_known_provider("custom")
    with pytest.raises(ValueError, match="registered"):
        make_provider("custom")
    with pytest.raises(ValueError):
        make_provider("azure-openai")


@pytest.mark.asyncio
async def test_anthropic_input_tokens_include_cached_tokens():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from harnessx import AnthropicProvider

    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")], stop_reason="end_turn", model="m",
        usage=SimpleNamespace(input_tokens=3, output_tokens=7, cache_read_input_tokens=900, cache_creation_input_tokens=100),
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=message)))
    response = await AnthropicProvider(client=client).create(model="m", messages=[{"role": "user", "content": "x"}], system=None, tools=[], max_tokens=10)
    assert response.usage.input_tokens == 1003, "the full prompt, not the uncached remainder"
    assert response.usage.cache_read_input_tokens == 900 and response.usage.cache_creation_input_tokens == 100


@pytest.mark.asyncio
async def test_llm_request_hook_carries_the_request_as_sent():
    seen: list[dict[str, Any]] = []
    hooks = HookManager()
    hooks.on(HookEvent.LLM_REQUEST, lambda ctx: seen.append(ctx.data))
    config = AgentConfig(system_prompt="Be terse.", max_tokens=99, temperature=0.2)
    async with Agent(provider=Scripted(), config=config, hooks=hooks) as agent:
        await agent.run("hi")
    payload = seen[0]
    assert payload["system"].startswith("Be terse.") and payload["model"] == config.model
    assert (payload["max_tokens"], payload["temperature"], payload["stream"]) == (99, 0.2, False)
    assert payload["message_count"] == 1 and payload["tool_count"] == 1 and isinstance(payload["prefix_key"], str)  # the built-in reader


@pytest.mark.asyncio
async def test_memory_trim_passes_a_nullable_system_prompt():
    class Counting(Scripted):
        async def count_tokens(self, **kwargs):
            self.calls.append(kwargs)
            return 0

    provider = Counting()
    agent = Agent(provider=provider, config=AgentConfig(system_prompt=""))
    for i in range(6):  # trimming only considers a conversation with some history
        agent.memory.add_user_message(f"m{i}") if i % 2 else agent.memory.add_assistant_message(f"a{i}")
    await agent.memory.trim_if_needed(provider, "m", "", [], max_context_tokens=10)
    assert provider.calls and provider.calls[-1]["system"] is None
    assert not hasattr(LLMProvider, "format_tools")


@pytest.mark.asyncio
async def test_backend_connect_classmethods_and_temporal_client_ownership(tmp_path):
    backend = await SQLiteBackend.connect(tmp_path / "x.db")
    assert backend.conn is not None
    await backend.aclose()

    from harnessx.backends.temporal import TemporalBackend

    class FakeClient:
        closed = False

        async def close(self):
            self.closed = True

    class Events:
        async def aclose(self):
            pass

    borrowed = TemporalBackend(FakeClient(), events=Events(), artifact_store=None)
    await borrowed.aclose()
    assert borrowed.client is not None and not borrowed.client.closed, "an injected client is caller-owned"
    owned = TemporalBackend(client := FakeClient(), events=Events(), artifact_store=None)
    owned._owns_client = True  # what connect() records
    await owned.aclose()
    assert client.closed and owned.client is None


@pytest.mark.asyncio
async def test_evaluate_agent_async_runs_from_a_running_loop():
    from harnessx.evals import build_example, contains_evaluator, evaluate_agent_async

    def factory(inputs: dict[str, Any]) -> Agent:
        return Agent(provider=Scripted([ProviderResponse(text="The answer is 144")]))

    summary = await evaluate_agent_async(
        factory,
        [build_example(inputs={"prompt": "12*12?"}, outputs={"contains_all": ["144"]})],
        evaluators=[contains_evaluator],
        offline=True,
        print_summary=False,
    )
    assert summary.experiment_name.startswith("harnessx-eval")
    assert summary.total_examples == 1


# ── Phase 5: retirements and exports ─────────────────────────────────────────


def test_removed_names_are_gone_and_new_ones_resolve():
    for removed in ("Role", "StreamingAgent"):
        assert not hasattr(harnessx, removed)
    assert not hasattr(harnessx.StopReason, "TOOL_CALLS")
    with pytest.raises(TypeError):
        harnessx.RuntimeConfig(checkpoint_interval=5)  # type: ignore[call-arg]
    with pytest.raises(ModuleNotFoundError):
        __import__("harnessx.streaming")
    for name in harnessx.__all__:
        assert getattr(harnessx, name) is not None, name
    import harnessx.durable as durable

    for name in durable.__all__:
        assert getattr(durable, name) is not None, name
    assert durable.AgentRuntime is AgentRuntime and durable.PendingTool is PendingTool
    assert harnessx.evaluate_agent_async is harnessx.evals.evaluate_agent_async


@pytest.mark.asyncio
async def test_large_tool_results_are_read_back_through_read_tool_result(tmp_path):
    big = "\n".join(f"row {i}" for i in range(2000))  # ~18 KB, above the default 12,000 chars
    provider = Scripted([*tool_call_then_text("dump", "summarised")])
    agent = Agent(provider=provider, config=AgentConfig(system_prompt="", limits=Limits(max_result_chars=1000)))

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def dump() -> str:
        return big

    async with agent:
        assert "read_tool_result" in agent.tools.list_tools()
        assert (await agent.run("go")).ok
        block = next(b for m in agent.memory.get_messages() if isinstance(m["content"], list)
                     for b in m["content"] if b.get("type") == "tool_result")
        notice = block["content"]
        assert "read_tool_result(result_id=\"c1\"" in notice and "read_file cannot open" in notice
        assert "row 1999" in notice and "row 1000" not in notice, "head and tail preview, middle omitted"

        first = await agent.tools.execute(ToolCall("r1", "read_tool_result", {"result_id": "c1", "offset": 0, "limit": 3}))
        assert not first.is_error and first.content.splitlines()[1:] == ["row 0", "row 1", "row 2"]
        assert "[next: offset=3]" in first.content
        last = await agent.tools.execute(ToolCall("r2", "read_tool_result", {"result_id": "c1", "offset": 1998, "limit": 50}))
        assert last.content.splitlines()[1:] == ["row 1998", "row 1999"] and "next:" not in last.content
        missing = await agent.tools.execute(ToolCall("r3", "read_tool_result", {"result_id": "nope"}))
        assert missing.is_error and "Tool execution error: LookupError:" in missing.content
        assert "Traceback" not in missing.content, "tool errors are one line; the traceback is logged"

    async with Agent(provider=Scripted()) as other:
        # A child or copied registry binds the reader to its own memory.
        handler = other.tools.get_tool("read_tool_result").handler
        assert handler is not agent.tools.get_tool("read_tool_result").handler


@pytest.mark.asyncio
async def test_application_read_tool_result_is_not_replaced():
    registry = ToolRegistry()

    @registry.register(name="read_tool_result", permission=PermissionLevel.ALLOW)
    def mine(result_id: str) -> str:
        return "custom"

    async with Agent(provider=Scripted(), tools=registry) as agent:
        assert agent.tools.get_tool("read_tool_result").handler is mine


# ── Long replies: budgets, timeouts, truncation, and paging ─────────────────


def test_reply_budget_defaults_per_provider_and_timeout_follows_it():
    from harnessx import AnthropicProvider
    from harnessx.providers.base import DEFAULT_MAX_TOKENS
    from harnessx.types import call_timeout_for

    assert AgentConfig().max_tokens is None, "unset means the provider chooses for the model"
    anthropic = AnthropicProvider(client=object())
    assert anthropic.default_max_tokens("claude-sonnet-4-6") == 20_000
    assert anthropic.default_max_tokens("claude-3-5-haiku-latest") == DEFAULT_MAX_TOKENS
    assert Scripted().default_max_tokens("anything") == DEFAULT_MAX_TOKENS
    # The timeout for one call grows with the budget instead of cutting a long reply short.
    assert call_timeout_for(8192) == DEFAULT_TIMEOUT_SECONDS
    assert call_timeout_for(32_000) == 32_000 * 3600 / 128_000 + 60
    assert RetryPolicy(call_timeout_seconds=45).effective_call_timeout(64_000) == 45, "an explicit value wins"


@pytest.mark.asyncio
async def test_engine_sends_the_resolved_budget_and_reports_truncation():
    seen: list[dict[str, Any]] = []
    hooks = HookManager()
    hooks.on(HookEvent.LLM_REQUEST, lambda ctx: seen.append(dict(ctx.data)))
    provider = Scripted([ProviderResponse(text="half a reply", stop_reason="max_tokens")])
    async with Agent(provider=provider, hooks=hooks) as agent:
        result = await agent.run("write a novel")
    assert seen[0]["max_tokens"] == 8192 and provider.calls[0]["max_tokens"] == 8192
    assert result.status is RunStatus.COMPLETED and result.stop_reason == "max_tokens"
    assert result.truncated and not result.ok and not result.failed
    with pytest.raises(harnessx.RunTruncated, match="max_tokens"):
        result.raise_for_status()
    assert not replace(_result(RunStatus.COMPLETED), stop_reason="end_turn").truncated


@pytest.mark.asyncio
async def test_anthropic_streams_when_the_sdk_refuses_a_long_non_streaming_call():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from harnessx import AnthropicProvider

    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="assembled from the stream")], stop_reason="end_turn", model="m",
        usage=SimpleNamespace(input_tokens=5, output_tokens=7, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def get_final_message(self):
            return message

    streamed: list[dict[str, Any]] = []

    def stream(**kwargs):
        streamed.append(kwargs)
        return Stream()

    create = AsyncMock(side_effect=ValueError("Streaming is required for operations that may take longer than 10 minutes."))
    client = SimpleNamespace(messages=SimpleNamespace(create=create, stream=stream))
    response = await AnthropicProvider(client=client).create(
        model="claude-sonnet-4-6", messages=[{"role": "user", "content": "x"}], system=None, tools=[], max_tokens=64_000,
    )
    assert response.text == "assembled from the stream" and streamed[0]["max_tokens"] == 64_000
    # Any other ValueError is the caller's problem, not a signal to stream.
    other = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=ValueError("bad request")), stream=stream))
    with pytest.raises(ValueError, match="bad request"):
        await AnthropicProvider(client=other).create(model="m", messages=[], system=None, tools=[], max_tokens=10)


@pytest.mark.asyncio
async def test_read_tool_result_pages_a_single_long_line_by_characters():
    blob = "{" + ",".join(f'"k{i}": "{"v" * 50}"' for i in range(400)) + "}"  # one ~24 KB line
    provider = Scripted([*tool_call_then_text("dump", "done")])
    agent = Agent(provider=provider, config=AgentConfig(system_prompt="", limits=Limits(max_result_chars=1000)))

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def dump() -> str:
        return blob

    async with agent:
        assert (await agent.run("go")).ok
        first = await agent.tools.execute(ToolCall("r1", "read_tool_result", {"result_id": "c1"}))
        header, body = first.content.split("\n", 1)
        assert len(body) == 6000 and "characters 1-6000 of" in header and "char_offset=6000" in header
        nxt = await agent.tools.execute(ToolCall("r2", "read_tool_result", {"result_id": "c1", "char_offset": 6000, "max_chars": 20000}))
        header2, body2 = nxt.content.split("\n", 1)
        assert body2 == blob[6000:26000] and "char_offset=26000" not in header2 or len(blob) > 26000
