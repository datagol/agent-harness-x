"""Offline contracts for constructor-defined delegation and its recovery boundary."""

import asyncio
from copy import deepcopy

import pytest

from harnessx import (
    Agent, AgentConfig, AgentRegistry, AgentRuntime, Middleware, PermissionLevel,
    PermissionManager, ProviderResponse, RunEventType, SQLiteBackend, SubAgent,
    TokenUsage, ToolCall, ToolRegistry, ToolResult,
)
from harnessx.engine import new_state
from harnessx.providers import LLMProvider


class Provider(LLMProvider):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.closed = False

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    async def count_tokens(self, **kwargs):
        return 0

    async def aclose(self):
        self.closed = True


def delegate(task="Review this code", call_id="review"):
    return ToolCall(call_id, "delegate_reviewer", {"task": task})


def call_response(*calls):
    return ProviderResponse(tool_calls=list(calls), stop_reason="tool_use")


def specialist(**kwargs):
    return SubAgent("reviewer", "Review code", AgentConfig(system_prompt="Review carefully"), **kwargs)


def outcomes(agent):
    return [b for m in agent.memory.get_messages() for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_delegation_isolated_and_serial_with_cleanup(monkeypatch, streaming):
    children = []

    def make_provider(name):
        assert name == "anthropic"  # Child does not inherit parent's OpenAI config.
        assert all(child.closed for child in children)
        child = Provider([ProviderResponse(text="Review finding", usage=TokenUsage(input_tokens=19))])
        children.append(child)
        return child

    monkeypatch.setattr("harnessx.core.make_provider", make_provider)
    parent_provider = Provider([
        call_response(delegate("first", "one"), delegate("second", "two")),
        ProviderResponse(text="Combined findings", usage=TokenUsage(input_tokens=7)),
    ])
    async with Agent(
        config=AgentConfig(provider="openai", model="fixture", system_prompt="Parent-only prompt"),
        provider=parent_provider, subagents=[specialist()],
    ) as parent:
        events = []
        if streaming:
            async with parent.run_stream("Private parent context") as stream:
                events = [event async for event in stream]
                result = await stream.result()
        else:
            result = await parent.run("Private parent context")
        assert result.status == "completed" and result.output == "Combined findings"
        assert result.usage.input_tokens == 7  # Parent-only accounting remains explicit.
        assert len(children) == 2 and all(c.closed for c in children)
        for child, task in zip(children, ("first", "second")):
            request = child.requests[0]
            assert len(request["messages"]) == 1
            assert task in str(request["messages"])
            assert "Private parent context" not in str(request)
            assert "Parent-only prompt" not in str(request)
            assert [t["name"] for t in request["tools"]] == ["read_tool_result"]  # only the built-in reader
        assert [outcome["content"] for outcome in outcomes(parent)] == ["Review finding"] * 2
        assert "Review finding" in str(parent_provider.requests[-1]["messages"])
        if streaming:
            assert len([e for e in events if e.type == RunEventType.TOOL_RESULT]) == 2
            assert all(e.data != "Review finding" for e in events if e.type == RunEventType.TEXT_DELTA)


@pytest.mark.asyncio
@pytest.mark.parametrize("approval", [False, True])
async def test_delegation_approval_does_not_approve_child_tools(monkeypatch, approval):
    effects, approvals, children = [], [], []

    def action() -> str:
        effects.append("executed")
        return "done"

    async def approve(call, definition):
        approvals.append(call.name)
        return call.name == "delegate_reviewer" or approval

    def make_provider(name):
        child = Provider([call_response(ToolCall("action", "action", {})), ProviderResponse(text="reviewed")])
        children.append(child)
        return child

    monkeypatch.setattr("harnessx.core.make_provider", make_provider)
    permissions = PermissionManager(PermissionLevel.ASK, approval_callback=approve)
    async with Agent(
        provider=Provider([call_response(delegate()), ProviderResponse(text="done")]),
        permissions=permissions, subagents=[specialist(tools=[action])],
    ) as parent:
        assert (await parent.run("review")).status == "completed"
    assert approvals == ["delegate_reviewer", "action"]
    assert effects == (["executed"] if approval else [])
    assert children[0].closed


@pytest.mark.asyncio
async def test_denied_delegation_never_constructs_child(monkeypatch):
    def forbidden(name):
        pytest.fail("Denied delegation created a provider")

    monkeypatch.setattr("harnessx.core.make_provider", forbidden)
    permissions = PermissionManager()
    permissions.set_permission("delegate_reviewer", PermissionLevel.DENY)
    permissions.grant_session("delegate_reviewer")
    async with Agent(
        provider=Provider([call_response(delegate()), ProviderResponse(text="denied")]),
        permissions=permissions, subagents=[specialist(permission=PermissionLevel.ALLOW)],
    ) as parent:
        await parent.run("review")
        assert outcomes(parent)[0]["is_error"]


@pytest.mark.asyncio
async def test_child_ask_without_callback_fails_closed(monkeypatch):
    effects = []
    tools = ToolRegistry()
    tools.register_tool(lambda: effects.append(1), name="action", permission=PermissionLevel.ASK)
    child = Provider([call_response(ToolCall("action", "action", {})), ProviderResponse(text="blocked")])
    monkeypatch.setattr("harnessx.core.make_provider", lambda name: child)
    async with Agent(provider=Provider([]), subagents=[specialist(tools=tools)]) as parent:
        result = await parent.tools.execute(delegate(), permissions=parent.permissions)
        assert not result.is_error
    assert not effects and "Permission denied" in str(child.requests[-1]["messages"])


@pytest.mark.asyncio
async def test_failed_child_is_error_and_owned_resources_close(monkeypatch):
    child = Provider([ValueError("bad child response")])
    monkeypatch.setattr("harnessx.core.make_provider", lambda name: child)
    async with Agent(provider=Provider([]), subagents=[specialist()]) as parent:
        result = await parent.tools.execute(delegate(), permissions=parent.permissions)
        assert result.is_error and "bad child response" in result.content
    assert child.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_or_cancellation_closes_child(monkeypatch, cancel):
    entered = asyncio.Event()

    class Waiting(Provider):
        async def create(self, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    child = Waiting([])
    monkeypatch.setattr("harnessx.core.make_provider", lambda name: child)
    async with Agent(
        provider=Provider([call_response(delegate())]),
        subagents=[specialist(timeout_seconds=300 if cancel else 0.05)],
    ) as parent:
        task = asyncio.create_task(parent.run("review"))
        await asyncio.wait_for(entered.wait(), 2)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await asyncio.wait_for(task, 2)
            assert result.status == "awaiting_input" and result.pending[0].status == "uncertain"
        assert child.closed and not parent.busy


@pytest.mark.asyncio
async def test_bindings_and_skills_do_not_mutate_definitions(monkeypatch, tmp_path):
    skill = tmp_path / "review.md"
    skill.write_text("---\nname: review\ndescription: Review carefully\n---\nFollow these instructions.\n")
    tools = ToolRegistry()
    tools.register_tool(lambda: "old", name="sample")
    spec = specialist(tools=tools, skills=[str(skill)])
    children = []

    def make_provider(name):
        child = Provider([ProviderResponse(text="done")])
        children.append(child)
        return child

    monkeypatch.setattr("harnessx.core.make_provider", make_provider)
    async with Agent(provider=Provider([]), subagents=[spec]) as parent:
        tools.unregister("sample")
        spec.config.system_prompt = "changed after registration"
        for _ in range(2):
            assert not (await parent.tools.execute(delegate())).is_error
    assert not tools.has_tool("Skill")
    for child in children:
        assert {t["name"] for t in child.requests[0]["tools"]} == {"sample", "Skill", "read_tool_result"}
        assert "changed after registration" not in str(child.requests)
        assert child.closed


@pytest.mark.parametrize("kwargs", [
    {"name": "bad name"}, {"name": "x" * 56}, {"description": ""},
    {"config": None}, {"permission": "allow"}, {"timeout_seconds": 0},
    {"timeout_seconds": float("nan")}, {"timeout_seconds": True}, {"skills": "path"},
])
def test_invalid_definitions(kwargs):
    values = dict(name="reviewer", description="Review", config=AgentConfig())
    values.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        SubAgent(**values)


def test_collisions_rejected_before_allocating_provider(monkeypatch):
    monkeypatch.setattr("harnessx.core.make_provider", lambda name: pytest.fail("Allocated a provider"))
    with pytest.raises(ValueError, match="Duplicate"):
        Agent(subagents=[specialist(), specialist()])
    tools = ToolRegistry()
    tools.register_tool(lambda: "existing", name="delegate_reviewer")
    with pytest.raises(ValueError, match="Duplicate"):
        Agent(tools=tools, subagents=[specialist()])
    assert tools.get_tool("delegate_reviewer").handler() == "existing"
    with pytest.raises(TypeError, match="SubAgent definitions"):
        Agent(subagents=[object()])


@pytest.mark.asyncio
async def test_durable_raw_result_is_reused_after_processing_failure(monkeypatch, tmp_path):
    children, fail = [], [True]

    def make_provider(name):
        child = Provider([ProviderResponse(text="persisted specialist result")])
        children.append(child)
        return child

    class Transform(Middleware):
        async def after_tool_execution(self, result):
            if fail[0]:
                raise RuntimeError("processing failed")
            return result

    def factory():
        class ParentProvider(Provider):
            async def create(self, **kwargs):
                if len(kwargs["messages"]) == 1:
                    return call_response(delegate())
                return ProviderResponse(text="synthesized")

        parent = Agent(
            provider=ParentProvider([]),
            subagents=[specialist()],
        )
        parent.middleware.add(Transform())
        return parent

    monkeypatch.setattr("harnessx.core.make_provider", make_provider)
    registry = AgentRegistry()
    ref = registry.register("orchestrator", factory)
    store = SQLiteBackend(tmp_path / "runs.db")
    runtime = AgentRuntime(ref, registry=registry, backend=store)
    try:
        sid = await runtime.start()
        failed = await runtime.run("review")
        assert failed.status == "failed" and len(children) == 1
        saved = await store.get_run(failed.run_id)
        assert saved["tools"][0]["status"] == "raw_completed"
        fail[0] = False
        handle = await runtime.resume(sid)
        assert (await handle.result()).status == "completed"
        assert len(children) == 1 and children[0].closed
    finally:
        await runtime.stop()
        await store.aclose()


@pytest.mark.asyncio
async def test_uncertain_durable_delegation_requires_manual_recovery(monkeypatch, tmp_path):
    monkeypatch.setattr("harnessx.core.make_provider", lambda name: pytest.fail("Replayed child"))
    registry = AgentRegistry()
    ref = registry.register("orchestrator", lambda: Agent(
        provider=Provider([ProviderResponse(text="synthesized")]), subagents=[specialist()],
    ))
    store = SQLiteBackend(tmp_path / "runs.db")
    runtime = AgentRuntime(ref, registry=registry, backend=store)
    try:
        sid = await runtime.start()
        state = new_state(runtime.agent, "review", durable=True)
        call = delegate()
        state.update(phase="tools", tools=[{
            "call": {"id": call.id, "name": call.name, "input": call.input},
            "status": "started", "attempt": 1,
            "execution_key": state["run_id"] + ":1:review", "policy": "manual",
            "concurrent": False, "timeout": 300.0,
        }])
        lease = await store.claim(sid)
        await store.create_run(sid, "request", state, lease)
        await store.save_run(state, [], lease)
        await store.release(sid, lease)
        handle = await runtime.resume(sid)
        result = await handle.result()
        assert result.status == "awaiting_input" and result.pending[0].status == "uncertain"
        await runtime.resolve_tool(result.pending[0].execution_key, result=ToolResult(call.id, "verified result"))
        handle = await runtime.resume(sid)
        assert (await handle.result()).output == "synthesized"
    finally:
        await runtime.stop()
        await store.aclose()
