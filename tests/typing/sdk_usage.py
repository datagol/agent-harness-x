"""Static API acceptance examples; checked by mypy, not executed by pytest."""

from typing import assert_type
from pathlib import Path

from harnessx import Agent, Fallback, FallbackProvider, Message, PermissionManager, RunEvent, RunResult, SubAgent, ToolCall, ToolDefinition, ToolResult, ToolRetry, TransientToolError
from harnessx.extensions.base import ExtensionContext
from harnessx.hooks import HookContext, HookEvent, Registration
from harnessx.types import AgentConfig, Limits, RetryPolicy, ToolPolicy
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
        name="reviewer", description="Review code", config=AgentConfig(limits=Limits(max_iterations=5)),
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


async def use_0_4_api(agent: Agent, runtime: AgentRuntime) -> None:
    from typing import Any

    from harnessx import (
        HarnessError, Limits, MCPManager, PendingTool, RetryPolicy, RunAwaitingInput, RunFailed,
        RunFailure, SQLiteBackend, ToolPolicy,
    )

    config = AgentConfig(
        limits=Limits(max_iterations=5), retry=RetryPolicy(attempts=1), tools=ToolPolicy(dedupe_calls=True),
    )
    assert_type(config.limits.max_iterations, int)
    assert_type(config.retry.call_timeout_seconds, float | None)
    assert_type(config.retry.effective_call_timeout(config.max_tokens), float)
    assert_type(config.tools.dedupe_calls, bool)

    result = await agent.run("hi")
    assert_type(result.ok, bool)
    assert_type(result.raise_for_status(), RunResult)
    for pending in result.pending:
        assert_type(pending, PendingTool)
        assert_type(pending.call, ToolCall)
        assert_type(pending.execution_key, str)
    async for text in agent.stream_text("hi", on_reset=lambda: None):
        assert_type(text, str)
    assert_type(agent.closed, bool)
    try:
        result.raise_for_status()
    except RunAwaitingInput as waiting:
        assert_type(waiting.pending, list[PendingTool])
    except RunFailed as failed:
        assert_type(failed.error, RunFailure)
    except HarnessError as error:
        assert_type(error, HarnessError)

    outcome = await runtime.run("go")
    if outcome.needs_input:
        assert_type(await runtime.approve(outcome.pending[0], resume=True), RunResult)
        assert_type(await runtime.decline(outcome.pending[0]), None)
    async with runtime.run_stream("events") as stream:
        async for event in stream:
            assert_type(event, RunEvent)
        async for text in stream.text():
            assert_type(text, str)
    assert_type(await runtime.status(), dict[str, Any])

    async with MCPManager() as mcp:
        assert_type(mcp, MCPManager)
        from harnessx import MCPServerConfig, MCPToolInfo

        assert_type(await mcp.connect(MCPServerConfig.http("remote", "http://localhost:8000/mcp")), list[MCPToolInfo])
        assert_type(await mcp.connect("files", command="npx", args=["-y", "server"]), list[MCPToolInfo])
    async with await SQLiteBackend.connect("runtime.db") as backend:
        assert_type(backend, SQLiteBackend)


async def use_retry_seams(agent: Agent) -> None:
    policy = ToolRetry(attempts=3, backoff_seconds=1.0, retry_error_results=True)
    assert_type(policy.wait_for(1), float)
    assert_type(ToolRetry.from_dict(policy.to_dict()), ToolRetry)
    assert_type(RetryPolicy(max_backoff_seconds=10).wait_for(1, retry_after=2.0), float)
    assert_type(ToolPolicy(retry=policy).retry, ToolRetry | None)

    def transient(result: ToolResult) -> bool:
        return result.is_error

    @agent.tools.register(replay_policy="idempotent", retry=policy, retry_if_result=transient)
    def flaky(url: str) -> str:
        """Fetch something flaky."""
        raise TransientToolError("throttled")

    assert_type(agent.tools.get_tool("flaky").retry, ToolRetry | None)

    # The usual way in: names in config, and the Agent builds and owns the chain.
    config = AgentConfig(
        model="claude-sonnet-4-6",
        provider="anthropic",
        fallbacks=[Fallback("openrouter", model="anthropic/claude-sonnet-4.6")],
        retry=RetryPolicy(attempts=3, switch_after=2, cooldown_seconds=30),
    )
    assert_type(config.fallbacks, tuple[Fallback, ...])
    assert_type(config.retry.switch_after, int)
    assert_type(config.retry.cooldown_seconds, float)
    Agent(config=config)

    # The object API, for a member that has to be a live provider.
    chain = FallbackProvider(
        "anthropic",
        fallbacks=[Fallback("openrouter", model="anthropic/claude-sonnet-4.6")],
        switch_after=2,
        cooldown_seconds=30,
    )
    assert_type(chain.labels, tuple[str, ...])
    assert_type(chain.last_served, str | None)
    assert_type(chain.is_transient(RuntimeError("x")), bool)
    assert_type(chain.retry_after(RuntimeError("x")), float | None)
    Agent(config=AgentConfig(model="claude-sonnet-4-6"), provider=chain)
    await chain.aclose()
