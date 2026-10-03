import asyncio
from pathlib import Path
import pytest

from harnessx import (
    Agent,
    AgentRegistry,
    AgentRuntime,
    SQLiteBackend,
    PermissionLevel,
    PermissionManager,
    ProviderResponse,
    ToolCall,
    ToolResult,
    Middleware,
    current_tool_context,
    SessionBusyError,
)
from harnessx.providers import LLMProvider
from harnessx.engine import new_state


class Provider(LLMProvider):
    def __init__(self, responses=None):
        self.responses = list(responses or [ProviderResponse(text="done")])
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def count_tokens(self, **kwargs):
        return 0


@pytest.mark.asyncio
async def test_direct_modes_equal():
    a = Agent(provider=Provider())
    b = Agent(provider=Provider())
    result = await a.run("hello")
    async with b.run_stream("hello") as stream:
        events = [e async for e in stream]
        streamed = await stream.result()
    assert result.output == streamed.output == "done"
    assert a.memory.get_messages() == b.memory.get_messages()
    assert events[-1].data == streamed


@pytest.mark.asyncio
@pytest.mark.parametrize("registration", ["constructor", "after"])
@pytest.mark.parametrize("mode", ["run", "stream", "durable"])
async def test_unspecified_tool_permissions_allow_execution(tmp_path, registration, mode):
    executed = []

    def calculate(a: int, b: int) -> int:
        executed.append((a, b))
        return a * b

    provider = Provider([
        ProviderResponse(
            tool_calls=[ToolCall("math", "calculate", {"a": 48, "b": 12})],
            stop_reason="tool_use",
        ),
        ProviderResponse(text="576"),
    ])
    async with Agent(
        provider=provider, tools=[calculate] if registration == "constructor" else None,
    ) as agent:
        if registration == "after":
            agent.tools.register_tool(calculate)
        if mode == "durable":
            store = SQLiteBackend(tmp_path / "runtime.db")
            runtime = AgentRuntime(agent, backend=store)
            try:
                await runtime.start()
                result = await runtime.run("What is 48 * 12?")
            finally:
                await runtime.stop()
                await store.aclose()
        elif mode == "stream":
            async with agent.run_stream("What is 48 * 12?") as stream:
                async for _ in stream:
                    pass
                result = await stream.result()
        else:
            result = await agent.run("What is 48 * 12?")
        assert result.status == "completed" and result.output == "576"
        assert executed == [(48, 12)]
        outcomes = [
            block
            for message in agent.memory.get_messages()
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert len(outcomes) == 1
        assert outcomes[0]["content"] == "576" and not outcomes[0].get("is_error")


@pytest.mark.asyncio
async def test_middleware_redacts_and_permissions_use_transformed_call():
    class Redact(Middleware):
        async def after_llm_call(self, response):
            response.text = "redacted"
            return response

    agent = Agent(provider=Provider([ProviderResponse(text="secret")]))
    agent.middleware.add(Redact())
    events = [e async for e in agent.run_stream("hello")]
    assert all("secret" not in str(e.data) for e in events)
    assert agent.memory.get_messages()[-1]["content"][0]["text"] == "redacted"
    executed = []

    class Rewrite(Middleware):
        async def before_tool_execution(self, call):
            call.name = "denied"
            return call

    agent = Agent(
        provider=Provider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall("t", "allowed", {})], stop_reason="tool_use"
                ),
                ProviderResponse(text="done"),
            ]
        )
    )
    agent.middleware.add(Rewrite())
    agent.tools.register(name="allowed", permission=PermissionLevel.ALLOW)(lambda: "ok")
    agent.tools.register(name="denied", permission=PermissionLevel.DENY)(
        lambda: executed.append(1)
    )
    agent.permissions.grant_session("denied")
    result = await agent.run("go")
    assert result.status == "completed" and not executed


@pytest.mark.asyncio
async def test_sqlite_roundtrip_dedup_and_binding(tmp_path):
    registry = AgentRegistry()
    ref = registry.register("test", lambda: Agent(provider=Provider()))
    store = SQLiteBackend(tmp_path / "runtime.db")
    runtime = AgentRuntime(ref, backend=store, registry=registry)
    sid = await runtime.start()
    first = await runtime.run("hello", request_id="r1")
    again = await runtime.run("hello", request_id="r1")
    assert first.run_id == again.run_id
    assert len((await store.get_run(first.run_id))["messages"]) == 2
    events = await store.read_events(first.run_id)
    assert events[-1].data.output == "done"
    assert len({e.cursor for e in events}) == len(events)
    await runtime.stop()
    await store.aclose()
    store2 = SQLiteBackend(tmp_path / "runtime.db")
    runtime2 = AgentRuntime(ref, backend=store2, registry=registry)
    assert await runtime2.resume(sid) is None
    result = await runtime2.run("again")
    assert result.output == "done"
    assert len((await store2.get_run(result.run_id))["messages"]) == 4
    await runtime2.stop()
    await store2.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["tool", "manager"])
async def test_approval_persists_and_resume_does_not_repeat_model(tmp_path, policy):
    effects = []
    registry = AgentRegistry()

    def factory():
        class ResponseProvider(Provider):
            async def create(self, **kwargs):
                if len(kwargs["messages"]) == 1:
                    return ProviderResponse(
                        tool_calls=[ToolCall("t", "write", {})], stop_reason="tool_use"
                    )
                return ProviderResponse(text="done")

        a = Agent(
            provider=ResponseProvider(),
            permissions=PermissionManager(PermissionLevel.ASK) if policy == "manager" else None,
        )

        @a.tools.register(name="write", permission=PermissionLevel.ASK if policy == "tool" else None)
        def write():
            effects.append(current_tool_context().execution_key)
            return "written"

        return a

    ref = registry.register("writer", factory)
    store = SQLiteBackend(tmp_path / "runtime.db")
    runtime = AgentRuntime(ref, registry=registry, backend=store)
    sid = await runtime.start()
    result = await runtime.run("write")
    assert result.status == "awaiting_input" and not effects
    key = result.pending[0].execution_key
    await runtime.approve(key)
    handle = await runtime.resume(sid)
    assert (await handle.result()).status == "completed"
    assert effects == [key]
    await runtime.stop()
    await store.aclose()


@pytest.mark.asyncio
async def test_recover_uncertain_manual_call_without_reexecution(tmp_path):
    effects = []
    registry = AgentRegistry()

    def factory():
        a = Agent(provider=Provider())
        a.tools.register(name="write", permission=PermissionLevel.ALLOW)(
            lambda: effects.append(1)
        )
        return a

    ref = registry.register("writer", factory)
    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(ref, registry=registry, backend=store)
    sid = await runtime.start()
    state = new_state(runtime.agent, "write", durable=True)
    state.update(
        phase="tools",
        tools=[
            {
                "call": {"id": "t", "name": "write", "input": {}},
                "status": "started",
                "attempt": 1,
                "execution_key": state["run_id"] + ":1:t",
                "policy": "manual",
                "concurrent": True,
                "timeout": 5,
            }
        ],
    )
    lease = await store.claim(sid)
    await store.create_run(sid, "r", state, lease)
    await store.save_run(state, [], lease)
    await store.release(sid, lease)
    handle = await runtime.resume(sid)
    result = await handle.result()
    assert result.status == "awaiting_input" and not effects
    await runtime.resolve_tool(
        result.pending[0].execution_key, result=ToolResult("t", "already written")
    )
    handle = await runtime.resume(sid)
    assert (await handle.result()).output == "done" and not effects
    await runtime.stop()
    await store.aclose()


@pytest.mark.asyncio
async def test_session_lock_and_disconnected_subscriber(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()

    class Slow(Provider):
        async def create(self, **kwargs):
            started.set()
            await release.wait()
            return ProviderResponse(text="done")

    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(Agent(provider=Slow()), backend=store)
    sid = await runtime.start()
    handle = await runtime.submit("hello")
    await started.wait()
    other = SQLiteBackend(tmp_path / "r.db")
    await other.initialize()
    with pytest.raises(SessionBusyError):
        await other.claim(sid)
    async with handle.stream():
        pass
    assert not runtime._task.done()
    release.set()
    assert (await handle.result()).status == "completed"
    await runtime.stop()
    await store.aclose()
    await other.aclose()


@pytest.mark.asyncio
async def test_artifacts_survive_close_and_rehydrate(tmp_path):
    from harnessx.extensions import ResultSpillExtension

    registry = AgentRegistry()

    def factory():
        class P(Provider):
            async def create(self, **kwargs):
                if len(kwargs["messages"]) == 1:
                    return ProviderResponse(
                        tool_calls=[ToolCall("t", "large", {})], stop_reason="tool_use"
                    )
                return ProviderResponse(text="done")

        a = Agent(
            provider=P(),
            extensions=[
                ResultSpillExtension(
                    threshold_bytes=20, spill_root=str(tmp_path / "spills")
                )
            ],
        )
        a.tools.register(name="large", permission=PermissionLevel.ALLOW)(
            lambda: "x" * 1000
        )
        return a

    ref = registry.register("a", factory)
    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(ref, registry=registry, backend=store)
    sid = await runtime.start()
    result = await runtime.run("hello")
    assert result.status == "completed"
    state = await store.get_run(result.run_id)
    path = next(iter(state["extensions"]["result_spill"]["results"].values()))["path"]
    assert path.startswith("artifact://")
    await runtime.stop()
    runtime2 = AgentRuntime(ref, registry=registry, backend=store)
    await runtime2.resume(sid)
    restored = next(iter(runtime2.agent.extensions[0]._results.values()))
    assert Path(restored.path).read_text() == "x" * 1000
    await runtime2.stop()
    await store.aclose()


@pytest.mark.asyncio
async def test_raw_tool_result_is_committed_before_failing_result_middleware(tmp_path):
    effects = []
    failures = [True]
    registry = AgentRegistry()

    class Transform(Middleware):
        async def after_tool_execution(self, result):
            if failures[0]:
                raise RuntimeError("middleware temporarily unavailable")
            return result

    def factory():
        class P(Provider):
            async def create(self, **kwargs):
                if len(kwargs["messages"]) == 1:
                    return ProviderResponse(
                        tool_calls=[ToolCall("t", "write", {})], stop_reason="tool_use"
                    )
                return ProviderResponse(text="done")

        agent = Agent(provider=P())
        agent.middleware.add(Transform())
        agent.tools.register(name="write", permission=PermissionLevel.ALLOW)(
            lambda: effects.append(1) or "written"
        )
        return agent

    ref = registry.register("a", factory)
    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(ref, registry=registry, backend=store)
    sid = await runtime.start()
    failed = await runtime.run("hello")
    assert failed.status == "failed" and effects == [1]
    saved = await store.get_run(failed.run_id)
    assert saved["tools"][0]["status"] == "raw_completed"
    failures[0] = False
    handle = await runtime.resume(sid)
    assert (await handle.result()).status == "completed" and effects == [1]
    await runtime.stop()
    await store.aclose()


@pytest.mark.asyncio
async def test_pause_during_tool_prevents_next_model_call(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()
    provider = Provider(
        [
            ProviderResponse(
                tool_calls=[ToolCall("t", "slow", {})], stop_reason="tool_use"
            ),
            ProviderResponse(text="done"),
        ]
    )
    agent = Agent(provider=provider)

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def slow():
        entered.set()
        await release.wait()
        return "ok"

    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(agent, backend=store)
    await runtime.start()
    handle = await runtime.submit("go")
    await entered.wait()
    pause = asyncio.create_task(runtime.pause())
    await asyncio.sleep(0)
    release.set()
    await pause
    assert (await handle.result()).status == "paused" and provider.calls == 1
    resumed = await runtime.resume(runtime.session_id)
    assert (await resumed.result()).output == "done"
    await runtime.stop()
    await store.aclose()


@pytest.mark.asyncio
async def test_cancel_preserves_uncertain_tool_and_closes_direct_stream(tmp_path):
    entered = asyncio.Event()
    provider = Provider(
        [
            ProviderResponse(
                tool_calls=[ToolCall("t", "slow", {})], stop_reason="tool_use"
            )
        ]
    )
    agent = Agent(provider=provider)

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def slow():
        entered.set()
        await asyncio.Event().wait()

    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(agent, backend=store)
    await runtime.start()
    handle = await runtime.submit("go")
    await entered.wait()
    await runtime.cancel()
    saved = await store.get_run(handle.run_id)
    assert saved["status"] == "cancelled" and saved["tools"][0]["status"] == "uncertain"
    await runtime.stop()
    await store.aclose()


@pytest.mark.asyncio
async def test_overlapping_direct_runs_are_rejected():
    entered = asyncio.Event()
    release = asyncio.Event()

    class Slow(Provider):
        async def create(self, **kwargs):
            entered.set()
            await release.wait()
            return ProviderResponse(text="done")

    a = Agent(provider=Slow())
    async with a.run_stream("first"):
        await entered.wait()
        with pytest.raises(RuntimeError, match="busy"):
            await a.run("second")
    assert not a.busy


@pytest.mark.asyncio
async def test_completed_request_dedup_during_new_run(tmp_path):
    provider = Provider(
        [ProviderResponse(text="first"), ProviderResponse(text="second")]
    )
    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(Agent(provider=provider), backend=store)
    await runtime.start()
    first = await runtime.run("a", request_id="a")
    second = await runtime.submit("b", request_id="b")
    duplicate = await runtime.submit("a", request_id="a")
    assert (await duplicate.result()).run_id == first.run_id
    assert (await second.result()).output == "second"
    await runtime.stop()
    await store.aclose()


@pytest.mark.asyncio
async def test_fresh_created_run_is_discoverable_before_first_snapshot(tmp_path):
    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(Agent(provider=Provider()), backend=store)
    sid = await runtime.start()
    lease = await store.claim(sid)
    state = new_state(runtime.agent, "go", durable=True)
    await store.create_run(sid, "r", state, lease)
    assert (await store.latest_run(sid))["run_id"] == state["run_id"]
    await store.release(sid, lease)
    await store.aclose()


@pytest.mark.asyncio
async def test_retry_model_marks_usage_incomplete():
    provider = Provider([ConnectionError("interrupted"), ProviderResponse(text="done")])
    # The default RetryPolicy(attempts=2) retries the transient failure once.
    agent = Agent(provider=provider)
    async with agent.run_stream("go") as stream:
        events = [e async for e in stream]
        result = await stream.result()
    assert result.output == "done" and result.usage_incomplete
    assert any(e.type == "attempt_reset" for e in events)


def test_sql_parameters_leave_quoted_schema_intact():
    from harnessx.backends.postgres import postgres_bindings

    assert (
        postgres_bindings('SELECT "schema?%"."sessions" WHERE id=?')
        == 'SELECT "schema?%%"."sessions" WHERE id=%s'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["manual", "safe"])
async def test_real_process_crash_after_effect(tmp_path, policy):
    import subprocess
    import sys
    import textwrap
    import os

    script = tmp_path / "crash.py"
    script.write_text(
        textwrap.dedent("""
        import asyncio, os, sys
        from pathlib import Path
        from harnessx import *
        from harnessx.providers import LLMProvider
        root=Path(sys.argv[1]); policy=sys.argv[2]
        class P(LLMProvider):
            async def count_tokens(self,**kwargs): return 0
            async def create(self,**kwargs):
                return ProviderResponse(tool_calls=[ToolCall('t','effect',{})],stop_reason='tool_use')
        async def main():
            a=Agent(provider=P())
            @a.tools.register(permission=PermissionLevel.ALLOW,replay_policy=policy)
            def effect():
                (root/'effect').write_text('1')
                os._exit(42)
            store=SQLiteBackend(root/'r.db'); runtime=AgentRuntime(a,backend=store)
            (root/'session').write_text(await runtime.start())
            await runtime.run('go')
        asyncio.run(main())
    """)
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()),
        "LANGSMITH_API_KEY": "",
        "LANGSMITH_TRACING": "false",
        "LANGSMITH_TRACING_V2": "false",
    }
    child = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(script), str(tmp_path), policy],
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert child.returncode == 42, child.stderr.decode()
    effects = []
    a = Agent(provider=Provider())
    a.tools.register(
        name="effect", permission=PermissionLevel.ALLOW, replay_policy=policy
    )(lambda: effects.append(1) or "ok")
    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(a, backend=store)
    handle = await runtime.resume((tmp_path / "session").read_text())
    result = await handle.result()
    if policy == "manual":
        assert result.status == "awaiting_input" and not effects
    else:
        assert result.status == "completed" and effects == [1]
    await runtime.stop()
    await store.aclose()


@pytest.mark.asyncio
async def test_cancelled_turn_can_be_followed_by_valid_conversation(tmp_path):
    entered = asyncio.Event()

    class P(Provider):
        async def create(self, **kwargs):
            if len(kwargs["messages"]) == 1:
                return ProviderResponse(
                    tool_calls=[ToolCall("t", "slow", {})], stop_reason="tool_use"
                )
            messages = kwargs["messages"]
            # The cancelled call is answered, so the request is well formed. It
            # now rides in the same user turn as the follow-up text rather than
            # a turn of its own, because consecutive same-role messages are
            # joined on the way out.
            blocks = [b for m in messages for b in (m.get("content") or [])
                      if isinstance(b, dict)]
            used = {b["id"] for b in blocks if b.get("type") == "tool_use"}
            answered = {b["tool_use_id"] for b in blocks if b.get("type") == "tool_result"}
            assert used and not used - answered
            roles = [m["role"] for m in messages]
            assert all(x != y for x, y in zip(roles, roles[1:])), roles
            return ProviderResponse(text="next")

    a = Agent(provider=P())

    @a.tools.register(permission=PermissionLevel.ALLOW)
    async def slow():
        entered.set()
        await asyncio.Event().wait()

    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(a, backend=store)
    await runtime.start()
    await runtime.submit("go")
    await entered.wait()
    await runtime.cancel()
    assert (await runtime.run("next")).output == "next"
    await runtime.stop()
    await store.aclose()


@pytest.mark.asyncio
async def test_hook_cannot_replace_authorized_tool():
    from harnessx import HookEvent

    effects = []
    a = Agent(
        provider=Provider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall("t", "allowed", {})], stop_reason="tool_use"
                ),
                ProviderResponse(text="done"),
            ]
        )
    )
    a.tools.register(name="allowed", permission=PermissionLevel.ALLOW)(
        lambda: effects.append("allowed")
    )
    a.tools.register(name="denied", permission=PermissionLevel.DENY)(
        lambda: effects.append("denied")
    )

    async def rewrite(ctx):
        ctx.data["tool_call"].name = "denied"

    a.hooks.on(HookEvent.TOOL_CALL_START, rewrite)
    assert (await a.run("go")).status == "completed"
    assert effects == ["allowed"]


@pytest.mark.asyncio
async def test_runtime_rechecks_new_ask_policy_before_dispatch(tmp_path):
    effects = []
    a = Agent(
        provider=Provider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall("t", "write", {})], stop_reason="tool_use"
                ),
                ProviderResponse(text="done"),
            ]
        )
    )
    a.tools.register(name="write", permission=PermissionLevel.ALLOW)(
        lambda: effects.append(1)
    )
    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(a, backend=store)
    await runtime.start()
    save = store.save_run

    async def change_policy(state, events, lease):
        if state["phase"] == "tools":
            a.permissions.set_permission("write", PermissionLevel.ASK)
        await save(state, events, lease)

    store.save_run = change_policy
    result = await runtime.run("go")
    assert result.status == "awaiting_input" and not effects
    await runtime.approve(result.pending[0].execution_key)
    handle = await runtime.resume(runtime.session_id)
    assert (await handle.result()).status == "completed" and effects == [1]
    await runtime.stop()
    await store.aclose()


@pytest.mark.asyncio
async def test_owned_provider_closes_once_and_injected_stays_open(monkeypatch):
    class Closeable(Provider):
        def __init__(self):
            super().__init__()
            self.closed = 0

        async def aclose(self):
            self.closed += 1

    owned = Closeable()
    injected = Closeable()
    monkeypatch.setattr("harnessx.core.make_provider", lambda name: owned)
    a = Agent()
    await a.aclose()
    await a.aclose()
    b = Agent(provider=injected)
    await b.aclose()
    assert owned.closed == 1 and injected.closed == 0


@pytest.mark.asyncio
async def test_event_replay_drains_all_pages(tmp_path):
    from harnessx import RunEvent, RunEventType
    from harnessx.runtime import RunHandle

    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(Agent(provider=Provider()), backend=store)
    sid = await runtime.start()
    state = new_state(runtime.agent, "go", durable=True)
    lease = await store.claim(sid)
    await store.create_run(sid, "r", state, lease)
    events = [
        RunEvent(RunEventType.TEXT_DELTA, str(n), sid, state["run_id"])
        for n in range(2100)
    ]
    state.update(status="completed", output="done")
    await store.save_run(state, events, lease)
    await store.release(sid, lease)
    replay = [e async for e in RunHandle(store, sid, state["run_id"]).events()]
    assert sum(e.type == "text_delta" for e in replay) == 2100
    assert replay[-1].data.output == "done"
    await store.aclose()


@pytest.mark.asyncio
async def test_tool_can_read_rehydrated_artifact(tmp_path):
    from harnessx.engine import execute_tool

    store = SQLiteBackend(tmp_path / "r.db")
    await store.initialize()
    key = await store.put_artifact(b"portable content")
    a = Agent(provider=Provider())
    a._artifact_store = store

    @a.tools.register(permission=PermissionLevel.ALLOW)
    def read(path: str):
        return Path(path).read_text()

    state = new_state(a, "go", durable=True)
    entry = {
        "call": {"id": "t", "name": "read", "input": {"path": "artifact://" + key}},
        "attempt": 1,
        "execution_key": "key",
        "timeout": 5,
    }

    async def ignore(event):
        pass

    result = await execute_tool(a, state, entry, ignore)
    assert result["content"] == "portable content"
    await store.aclose()


@pytest.mark.asyncio
async def test_session_deadline_interrupts_provider_and_returns_result(tmp_path):
    from harnessx import RuntimeConfig

    class Slow(Provider):
        async def create(self, **kwargs):
            await asyncio.sleep(5)
            return ProviderResponse(text="late")

    store = SQLiteBackend(tmp_path / "r.db")
    runtime = AgentRuntime(
        Agent(provider=Slow()),
        backend=store,
        runtime_config=RuntimeConfig(max_session_duration_seconds=0.03),
    )
    await runtime.start()
    result = await runtime.run("go")
    assert result.status == "cancelled" and result.stop_reason == "timeout"
    await runtime.stop()
    await store.aclose()
