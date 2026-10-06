"""Temporal/Redis integration tests, enabled by explicit local service addresses."""

import asyncio
import os
import uuid
import pytest

from harnessx import (
    Agent,
    AgentRegistry,
    AgentRuntime,
    SQLiteBackend,
    TemporalBackend,
    RedisEvents,
    ProviderResponse,
    ToolCall,
    Extension,
    PermissionLevel,
)
from harnessx.providers import LLMProvider

TARGET = os.environ.get("HARNESS_TEST_TEMPORAL")
REDIS = os.environ.get("HARNESS_TEST_REDIS")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not TARGET or not REDIS,
        reason="Temporal/Redis integration services not configured",
    ),
]


class P(LLMProvider):
    async def create(self, **kwargs):
        return ProviderResponse(text="temporal works")

    async def count_tokens(self, **kwargs):
        return 0


async def test_worker_restart_replay_and_duplicate_submission(tmp_path):
    from temporalio.worker import Replayer
    from harnessx.backends.temporal_workflow import AgentSessionWorkflow

    registry = AgentRegistry()
    terminal_runs = []
    class TerminalObserver(Extension):
        name = "terminal-observer"

        def install(self, ctx):
            pass

        async def on_turn_complete(self, ctx, result):
            terminal_runs.append((result.run_id, result.status))

    ref = registry.register("test", lambda: Agent(provider=P(), extensions=[TerminalObserver()]))
    artifacts = SQLiteBackend(tmp_path / "artifacts.db")
    await artifacts.initialize()
    backend = await TemporalBackend.connect(
        TARGET,
        events=RedisEvents(REDIS),
        artifact_store=artifacts,
        registry=registry,
        task_queue="test-" + uuid.uuid4().hex,
    )
    runtime = AgentRuntime(ref, backend=backend)
    try:
        async with await backend.worker():
            sid = await runtime.start()
            result = await asyncio.wait_for(
                runtime.run("hello", request_id="r1"), 30
            )
            assert result.output == "temporal works"
        # A fresh worker must reconstruct the session from history.
        async with await backend.worker():
            again = await runtime.run("hello", request_id="r1")
            assert again.run_id == result.run_id
            events = [
                e
                async for e in (await runtime.submit("next", request_id="r2")).events()
            ]
            assert any(
                e.type == "run_result" and e.data.output == "temporal works"
                for e in events
            )
            assert (result.run_id, "completed") in terminal_runs
            history = await backend.handle(sid).fetch_history()
            await Replayer(
                workflows=[AgentSessionWorkflow],
                data_converter=backend.client.config()["data_converter"],
            ).replay_workflow(history)
            await runtime.stop()
    finally:
        if runtime.session_id:
            await backend.handle(runtime.session_id).terminate()
        await backend.aclose()
        await artifacts.aclose()


async def test_approval_survives_worker_restart(tmp_path):
    effects = []
    registry = AgentRegistry()

    def factory():
        class Tools(P):
            async def create(self, **kwargs):
                if len(kwargs["messages"]) == 1:
                    return ProviderResponse(
                        tool_calls=[ToolCall("t", "write", {})], stop_reason="tool_use"
                    )
                return ProviderResponse(text="done")

        agent = Agent(provider=Tools())
        agent.tools.register(name="write", permission=PermissionLevel.ASK)(lambda: effects.append("effect"))
        return agent

    ref = registry.register("writer", factory)
    artifacts = SQLiteBackend(tmp_path / "artifacts.db")
    await artifacts.initialize()
    backend = await TemporalBackend.connect(
        TARGET,
        events=RedisEvents(REDIS),
        artifact_store=artifacts,
        registry=registry,
        task_queue="test-" + uuid.uuid4().hex,
    )
    runtime = AgentRuntime(ref, backend=backend)
    try:
        async with await backend.worker():
            sid = await runtime.start()
            result = await asyncio.wait_for(runtime.run("write"), 30)
            assert result.status == "awaiting_input" and not effects
        async with await backend.worker():
            await runtime.approve(result.pending[0].execution_key)
            handle = await runtime.resume(sid)
            assert (await asyncio.wait_for(handle.result(), 30)).output == "done"
            assert effects == ["effect"]
    finally:
        if runtime.session_id:
            await backend.handle(runtime.session_id).terminate()
        await backend.aclose()
        await artifacts.aclose()


async def test_redis_duplicates_and_expired_cursor():
    from harnessx import RunEvent, RunEventType

    events = RedisEvents(REDIS)
    run = uuid.uuid4().hex
    event = RunEvent(RunEventType.TEXT_DELTA, "hi", run_id=run)
    try:
        await events.publish(event)
        await events.publish(event)
        batch, gap = await events.read(run)
        assert len(batch) == 1 and not gap
        await events.client.delete("harness:events:" + run)
        batch, gap = await events.read(run, batch[0].cursor)
        assert gap
    finally:
        if events.client:
            await events.client.delete(
                "harness:events:" + run, "harness:events:" + run + ":ids"
            )
        await events.aclose()


async def test_reliability_retry_and_child_accounting_replay(tmp_path, monkeypatch):
    from harnessx import AgentConfig, RetryPolicy, SubAgent, TokenUsage
    from temporalio.worker import Replayer
    from harnessx.backends.temporal_workflow import AgentSessionWorkflow

    attempts, children = [], []
    class Parent(P):
        def is_transient(self, exc):
            return isinstance(exc, LookupError)
        async def create(self, **request):
            if len(request["messages"]) == 1:
                attempts.append(True)
                if len(attempts) == 1:
                    raise LookupError("retry this provider-specific failure")
                return ProviderResponse(tool_calls=[ToolCall("d", "delegate_worker", {"task": "go"})],
                                        usage=TokenUsage(input_tokens=2))
            return ProviderResponse(text="done", usage=TokenUsage(input_tokens=3))
    class Child(P):
        async def create(self, **request):
            children.append(True)
            return ProviderResponse(text="child", usage=TokenUsage(input_tokens=17, thinking_tokens=3))
    monkeypatch.setattr("harnessx.core.make_provider", lambda _: Child())
    cfg = AgentConfig(model="fixture", planning=False, prompt_cache=None,
                      retry=RetryPolicy(attempts=2, backoff_seconds=0))
    registry = AgentRegistry()
    ref = registry.register("reliability", lambda: Agent(config=cfg, provider=Parent(),
        subagents=[SubAgent("worker", "work", cfg)]))
    artifacts = await SQLiteBackend.connect(tmp_path / "artifacts.db")
    backend = await TemporalBackend.connect(TARGET, events=RedisEvents(REDIS), artifact_store=artifacts,
        registry=registry, task_queue="test-" + uuid.uuid4().hex)
    runtime = AgentRuntime(ref, backend=backend)
    try:
        async with await backend.worker():
            sid = await runtime.start()
            result = await asyncio.wait_for(runtime.run("go"), 30)
            assert result.ok and result.output == "done"
            assert result.usage.input_tokens == 22 and result.usage.thinking_tokens == 3
            assert result.usage_incomplete and len(attempts) == 2 and len(children) == 1
            history = await backend.handle(sid).fetch_history()
            await Replayer(workflows=[AgentSessionWorkflow],
                data_converter=backend.client.config()["data_converter"]).replay_workflow(history)
            await runtime.stop()
    finally:
        if runtime.session_id:
            await backend.handle(runtime.session_id).terminate()
        await backend.aclose()
        await artifacts.aclose()
