"""Offline Temporal activity contracts and versioned workflow scheduling."""
import asyncio
from dataclasses import asdict, replace
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("temporalio")
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from harnessx import Agent, AgentConfig, AgentRegistry, ProviderResponse, RetryPolicy, SubAgent, TokenUsage, ToolCall
from harnessx.backends.temporal_worker import AgentActivities
from harnessx.engine import command, new_state
from harnessx.execution import apply_accounting, transition, wire
from harnessx.providers import LLMProvider


class Provider(LLMProvider):
    def __init__(self, responses):
        self.responses = iter(responses)
    async def create(self, **request):
        result = next(self.responses)
        if isinstance(result, BaseException):
            raise result
        return result
    async def count_tokens(self, **request):
        return 0
    def is_transient(self, exc):
        return isinstance(exc, LookupError)


async def ignore(*args):
    pass


def config(**kwargs):
    return AgentConfig(model="fixture", planning=False, prompt_cache=None, **kwargs)


async def activity_args(factory, phase="model"):
    registry = AgentRegistry()
    ref = registry.register("fixture", factory)
    async with factory() as agent:
        state = new_state(agent, "go", durable=True)
        state["config"] = wire(agent.config)
        for name in ("start", "prepare_model"):
            state = transition(state, name, await command(agent, state, name, ignore))
    return AgentActivities(registry, None, NS(publish=ignore)), {
        "agent": asdict(ref), "session_id": state["session_id"], "state": state,
        "command": phase, "reliability_v2": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,retryable", [(LookupError("temporary"), True), (ValueError("invalid"), False)])
async def test_activity_uses_provider_error_classifier(failure, retryable):
    activities, args = await activity_args(lambda: Agent(config=config(), provider=Provider([failure])))
    with pytest.raises(ApplicationError) as caught:
        await ActivityEnvironment().run(activities.step, args)
    assert caught.value.type == type(failure).__name__
    assert caught.value.non_retryable != retryable
    assert caught.value.details[0]["accounting"]["incomplete"]


@pytest.mark.asyncio
async def test_failed_summary_activity_uses_forced_trim():
    cfg = config(retry=RetryPolicy(attempts=2, backoff_seconds=0))
    def factory():
        agent = Agent(config=cfg, provider=Provider([LookupError("summary down")]))
        for i in range(8):
            agent.memory.add_user_message(f"old {i}: " + "x" * 3000)
            agent.memory.add_assistant_message("y" * 3000)
        return agent
    activities, args = await activity_args(factory, "compact")
    args["state"]["compact_forced"] = True
    env = ActivityEnvironment()
    env.info = replace(env.info, attempt=2)
    result = await env.run(activities.step, args)
    assert result["outcome"]["phase"] == "prepare_model"
    assert len(str(result["outcome"]["messages"])) < len(str(args["state"]["messages"]))


@pytest.mark.asyncio
@pytest.mark.parametrize("modern", [False, True])
async def test_workflow_schedule_is_versioned_and_uses_effective_budget(monkeypatch, modern):
    from harnessx.backends.temporal_workflow import AgentSessionWorkflow, workflow
    scheduled = []
    monkeypatch.setattr(workflow, "patched", lambda _: modern)
    def start(name, args, **options):
        scheduled.append((name, args, options))
        future = asyncio.get_running_loop().create_future()
        future.set_result({"outcome": {}, "events": []})
        return future
    monkeypatch.setattr(workflow, "start_activity", start)
    coordinator = AgentSessionWorkflow.__new__(AgentSessionWorkflow)
    coordinator.data = {"session_id": "s", "agent": {"name": "fixture"}}
    coordinator.activities = []
    state = {"config": wire(config(retry=RetryPolicy(attempts=5))), "request": {"max_tokens": 64000}}
    await coordinator.step(state, "model")
    _, args, options = scheduled[0]
    assert options["retry_policy"].maximum_attempts == (5 if modern else 3)
    expected = RetryPolicy().effective_call_timeout(64000) * 2 + 10 if modern else 310
    assert options["start_to_close_timeout"] == timedelta(seconds=expected)
    assert args.get("reliability_v2", False) == modern
    assert ("heartbeat_timeout" in options) == modern


@pytest.mark.asyncio
async def test_child_accounting_survives_tool_activity_and_finish(monkeypatch):
    cfg = config()
    monkeypatch.setattr("harnessx.core.make_provider", lambda _: Provider([
        ProviderResponse(text="child answer", usage=TokenUsage(input_tokens=17, thinking_tokens=3))]))
    def factory():
        return Agent(config=cfg, provider=Provider([]), subagents=[SubAgent("worker", "work", cfg)])
    activities, args = await activity_args(factory)
    state = args["state"]
    state["response"] = {"tool_calls": [wire(ToolCall("d", "delegate_worker", {"task": "go"}))]}
    async with factory() as agent:
        prepared = await command(agent, state, "prepare_tools", ignore)
    state.update(prepared)
    entry = state["tools"][0]
    result = await ActivityEnvironment().run(activities.tool, {**args, "entry": entry})
    assert result["accounting"]["usage"]["input_tokens"] == 17
    apply_accounting(state, entry, result["accounting"])
    apply_accounting(state, entry, result["accounting"])
    entry["raw_result"] = result["result"]
    finished = await ActivityEnvironment().run(activities.step, {**args, "command": "finish_tool", "entry": entry})
    assert finished["outcome"]["usage"]["input_tokens"] == 17
    assert finished["outcome"]["total_usage"]["thinking_tokens"] == 3


@pytest.mark.asyncio
async def test_cancelled_tool_activity_returns_known_child_charges(monkeypatch):
    cfg = config()
    waiting = asyncio.Event()
    class Child(Provider):
        def __init__(self):
            super().__init__([ProviderResponse(text="part", stop_reason="max_tokens", usage=TokenUsage(input_tokens=17))])
            self.called = False
        async def create(self, **request):
            if self.called:
                waiting.set()
                await asyncio.Future()
            self.called = True
            return await super().create(**request)
    monkeypatch.setattr("harnessx.core.make_provider", lambda _: Child())
    def factory():
        return Agent(config=cfg, provider=Provider([]), subagents=[SubAgent("worker", "work", cfg)])
    activities, args = await activity_args(factory)
    state = args["state"]
    state["response"] = {"tool_calls": [wire(ToolCall("d", "delegate_worker", {"task": "go"}))]}
    async with factory() as agent:
        state.update(await command(agent, state, "prepare_tools", ignore))
    env = ActivityEnvironment()
    running = asyncio.create_task(env.run(activities.tool, {**args, "entry": state["tools"][0]}))
    await asyncio.wait_for(waiting.wait(), 2)
    env.cancel()
    with pytest.raises(ApplicationError) as caught:
        await asyncio.wait_for(running, 2)
    accounting = caught.value.details[0]["accounting"]
    assert accounting["usage"]["input_tokens"] == 17 and accounting["incomplete"]


@pytest.mark.asyncio
async def test_activity_retry_restores_heartbeat_charges():
    activities, args = await activity_args(lambda: Agent(config=config(), provider=Provider([
        ProviderResponse(text="ok", usage=TokenUsage(input_tokens=5))])))
    env = ActivityEnvironment()
    env.info = replace(env.info, attempt=2, heartbeat_details=[{
        "accounting": {"usage": {"input_tokens": 11}, "cost": .33, "incomplete": True}}])
    result = await env.run(activities.step, args)
    assert result["outcome"]["usage"]["input_tokens"] == 16
    assert result["outcome"]["usage_incomplete"]
    assert result["outcome"]["estimated_cost"] == pytest.approx(.330015)


@pytest.mark.asyncio
async def test_activity_retry_keeps_limits_learned_before_failed_repair():
    seen = []
    class Inspect(Provider):
        async def create(self, **request):
            seen.append(request["max_tokens"])
            return await super().create(**request)
    responses = iter([ValueError("max_tokens: 32000 > 2048"), LookupError("temporary"), ProviderResponse(text="ok")])
    def factory():
        provider = Inspect([])
        provider.responses = responses
        return Agent(config=config(), provider=provider)
    activities, args = await activity_args(factory)
    first = ActivityEnvironment()
    with pytest.raises(ApplicationError) as caught:
        await first.run(activities.step, args)
    checkpoint = caught.value.details[0]
    assert checkpoint["provider_state"]["limits"]["fixture"]["max_output"] == 2048
    retry = ActivityEnvironment()
    retry.info = replace(retry.info, attempt=2, heartbeat_details=[checkpoint])
    result = await retry.run(activities.step, args)
    assert result["outcome"]["response"]["text"] == "ok"
    assert seen == [32000, 2048, 2048]
