"""Static API acceptance examples; checked by mypy, not executed by pytest."""

from typing import assert_type
from pathlib import Path

from harnessx import Agent, Message, PermissionManager, RunEvent, RunResult, SubAgent, ToolCall, ToolDefinition
from harnessx.extensions.base import ExtensionContext
from harnessx.hooks import HookContext, HookEvent, Registration
from harnessx.types import AgentConfig
from harnessx import AgentRuntime, ExportPolicy, IncidentRecorder, Playback, VerificationReport


async def approve(call: ToolCall, definition: ToolDefinition) -> bool:
    return call.name == definition.name


def hook(context: HookContext) -> None:
    assert_type(context.event, HookEvent)


async def use_sdk(agent: Agent, context: ExtensionContext) -> None:
    permissions = PermissionManager(approval_callback=approve)
    assert_type(await agent.run("hello"), RunResult)
    assert_type(context.config, AgentConfig)
    assert_type(context.on_hook(HookEvent.AGENT_START, hook), Registration)
    assert_type(agent.memory.messages, tuple[Message, ...])
    await agent.tools.execute(ToolCall("id", "tool", {}), permissions=permissions)
    async with agent.run_stream("hello") as stream:
        async for event in stream:
            assert_type(event, RunEvent)
        assert_type(await stream.result(), RunResult)


async def use_subagents() -> None:
    specialist = SubAgent(
        name="reviewer", description="Review code", config=AgentConfig(max_iterations=5),
        tools=[], skills=[], timeout_seconds=60,
    )
    async with Agent(subagents=[specialist]) as orchestrator:
        assert_type(orchestrator.subagents, tuple[SubAgent, ...])
        assert_type(await orchestrator.run("Review this code"), RunResult)


async def inspect_incident(runtime: AgentRuntime, run_id: str) -> None:
    bundle = await runtime.export_incident(
        run_id, destination="incident.hx", policy=ExportPolicy(include_payloads=True),
    )
    assert_type(bundle, Path)
    recorder = IncidentRecorder()
    assert_type(await recorder.verify(bundle), VerificationReport)
    assert_type(await recorder.playback(bundle), Playback)
