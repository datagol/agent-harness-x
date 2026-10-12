"""Tool batches: calls that may overlap run together, serial calls take turns,
and specialists run in parallel within AgentConfig.max_parallel_subagents."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from harnessx import (
    Agent, AgentConfig, PermissionLevel, ProviderResponse, RunEventType, SubAgent, ToolCall,
)
from harnessx.errors import ConfigurationError
from harnessx.permissions import PermissionManager
from harnessx.providers import LLMProvider


class Scripted(LLMProvider):
    name = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.responses.pop(0)
        return item(kwargs) if callable(item) else item

    async def count_tokens(self, **kwargs):
        return 0


def batch(*names):
    return ProviderResponse(tool_calls=[ToolCall(f"c{i}", n, {}) for i, n in enumerate(names)], stop_reason="tool_use")


def timed_tools(agent, spans, *, serial=(), delay=0.05):
    for name in ("note", "fetch_a", "fetch_b", "s1", "s2", "fetch"):
        async def tool(_name=name) -> str:
            started = time.perf_counter()
            await asyncio.sleep(delay)
            spans[_name] = (started, time.perf_counter())
            return _name

        agent.tools.register_tool(tool, name=name, permission=PermissionLevel.ALLOW, concurrent=name not in serial)


@pytest.mark.asyncio
async def test_a_serial_call_no_longer_holds_up_the_concurrent_ones():
    spans: dict[str, tuple[float, float]] = {}
    agent = Agent(config=AgentConfig(model="m", planning=False),
                  provider=Scripted([batch("note", "fetch_a", "fetch_b"), ProviderResponse(text="done")]))
    timed_tools(agent, spans, serial={"note"})
    async with agent:
        began = time.perf_counter()
        assert (await agent.run("go")).ok
        took = time.perf_counter() - began
    assert spans["fetch_a"][0] < spans["note"][1] and spans["fetch_b"][0] < spans["note"][1]
    assert took < 0.12, "three 0.05 s calls overlapped; 0.8.0 ran them one after another"


@pytest.mark.asyncio
async def test_serial_calls_take_turns_in_model_order():
    spans: dict[str, tuple[float, float]] = {}
    agent = Agent(config=AgentConfig(model="m", planning=False),
                  provider=Scripted([batch("s1", "fetch", "s2"), ProviderResponse(text="done")]))
    timed_tools(agent, spans, serial={"s1", "s2"})
    async with agent:
        assert (await agent.run("go")).ok
    assert spans["s2"][0] >= spans["s1"][1], "serial calls never overlap"
    assert spans["fetch"][0] < spans["s1"][1], "but the concurrent call runs beside them"


@pytest.mark.asyncio
async def test_results_and_start_events_keep_model_order():
    spans: dict[str, tuple[float, float]] = {}
    provider = Scripted([batch("fetch_a", "s1", "fetch_b", "s2"), ProviderResponse(text="done")])
    agent = Agent(config=AgentConfig(model="m", planning=False), provider=provider)
    timed_tools(agent, spans, serial={"s1", "s2"})
    async with agent:
        async with agent.run_stream("go") as stream:
            events = [e async for e in stream]
            assert (await stream.result()).ok
    starts = [e.data.name for e in events if e.type is RunEventType.TOOL_CALL_START]
    assert starts[:3] == ["fetch_a", "s1", "fetch_b"] and starts[3] == "s2"
    s1_done = next(i for i, e in enumerate(events) if e.type is RunEventType.TOOL_CALL_COMPLETE and e.data.name == "s1")
    assert events.index(next(e for e in events if e.type is RunEventType.TOOL_CALL_START and e.data.name == "s2")) > s1_done
    results = [b["tool_use_id"] for b in provider.requests[1]["messages"][-1]["content"] if b.get("type") == "tool_result"]
    assert results == ["c0", "c1", "c2", "c3"]


# ── specialists ──────────────────────────────────────────────────────────────


def delegations(*tasks):
    return ProviderResponse(
        tool_calls=[ToolCall(f"d{i}", "delegate_worker", {"task": t}) for i, t in enumerate(tasks)],
        stop_reason="tool_use",
    )


def parent(monkeypatch, child_provider, tasks, *, config=None, worker=None, permissions=None):
    monkeypatch.setattr("harnessx.core.make_provider", lambda name: child_provider())
    return Agent(
        config=config or AgentConfig(model="m", planning=False),
        provider=Scripted([delegations(*tasks), ProviderResponse(text="combined")]),
        subagents=[worker or SubAgent("worker", "Does work", AgentConfig(planning=False))],
        permissions=permissions,
    )


@pytest.mark.asyncio
async def test_specialists_in_one_turn_run_at_the_same_time(monkeypatch):
    barrier = asyncio.Barrier(2)

    class Meets(Scripted):
        async def create(self, **kwargs):
            await asyncio.wait_for(barrier.wait(), 2)  # deadlocks if run one at a time
            return ProviderResponse(text="found it")

    async with parent(monkeypatch, lambda: Meets([]), ["one", "two"]) as agent:
        result = await asyncio.wait_for(agent.run("go"), 5)
    assert result.ok and result.output == "combined"


@pytest.mark.asyncio
@pytest.mark.parametrize(("cap", "concurrent", "peak"), [(2, True, 2), (4, False, 1)])
async def test_the_cap_and_a_serial_specialist_bound_the_fan_out(monkeypatch, cap, concurrent, peak):
    active, seen = [0], []

    class Counts(Scripted):
        async def create(self, **kwargs):
            active[0] += 1
            seen.append(active[0])
            await asyncio.sleep(0.03)
            active[0] -= 1
            return ProviderResponse(text="ok")

    worker = SubAgent("worker", "Does work", AgentConfig(planning=False), concurrent=concurrent)
    config = AgentConfig(model="m", planning=False, max_parallel_subagents=cap)
    async with parent(monkeypatch, lambda: Counts([]), ["a", "b", "c"], config=config, worker=worker) as agent:
        assert (await agent.run("go")).ok
    assert max(seen) == peak


@pytest.mark.asyncio
async def test_parallel_specialists_ask_for_approval_one_at_a_time(monkeypatch):
    asking, overlaps, asked = [0], [], []

    def action() -> str:
        return "done"

    async def approve(call, definition):
        if call.name == "delegate_worker":
            return True
        asking[0] += 1
        overlaps.append(asking[0])
        asked.append(call.name)
        await asyncio.sleep(0.02)
        asking[0] -= 1
        permissions.grant_session("action")  # an "always" answer
        return True

    permissions = PermissionManager(PermissionLevel.ASK, approval_callback=approve)
    child = lambda: Scripted([ProviderResponse(tool_calls=[ToolCall("x", "action", {})], stop_reason="tool_use"),
                              ProviderResponse(text="reviewed")])
    worker = SubAgent("worker", "Does work", AgentConfig(planning=False), tools=[action])
    async with parent(monkeypatch, child, ["a", "b"], worker=worker, permissions=permissions) as agent:
        assert (await agent.run("go")).ok
    assert max(overlaps) == 1 and asked == ["action"], "the second child was covered by the first one's answer"


@pytest.mark.asyncio
async def test_serial_tools_never_overlap_across_parallel_specialists(monkeypatch):
    active, seen = [0], []

    async def write_note() -> str:
        active[0] += 1
        seen.append(active[0])
        await asyncio.sleep(0.03)
        active[0] -= 1
        return "noted"

    from harnessx.tools import ToolRegistry

    tools = ToolRegistry()
    tools.register_tool(write_note, permission=PermissionLevel.ALLOW, concurrent=False)
    child = lambda: Scripted([ProviderResponse(tool_calls=[ToolCall("w", "write_note", {})], stop_reason="tool_use"),
                              ProviderResponse(text="ok")])
    worker = SubAgent("worker", "Does work", AgentConfig(planning=False), tools=tools)
    async with parent(monkeypatch, child, ["a", "b", "c"], worker=worker) as agent:
        assert (await asyncio.wait_for(agent.run("go"), 5)).ok
    assert len(seen) == 3 and max(seen) == 1


def test_max_parallel_subagents_is_validated_and_defaults_for_old_configs():
    with pytest.raises(ConfigurationError):
        AgentConfig(max_parallel_subagents=0)
    assert AgentConfig().max_parallel_subagents == 4
    assert AgentConfig.from_dict({"model": "m"}).max_parallel_subagents == 4
    with pytest.raises(TypeError):
        SubAgent("w", "d", AgentConfig(), concurrent="yes")
