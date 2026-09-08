"""AgentTarget adapter for the LangSmith evaluation framework.

Wraps a DataGOL Agent or agent factory into a LangSmith-compatible target function:
    target(inputs: dict) -> dict

Features:
  - Automatically captures telemetry via hooks:
      • Executed tool calls (names, inputs, outputs, errors, durations)
      • Invoked skills
      • Guardrail statistics (iterations, tokens, cost)
      • Exceptions (MaxIterationsError, CostLimitError, API errors)
  - Seamlessly propagates LangSmith run trees so agent child spans (LLM and tool calls)
    nest directly under the LangSmith evaluation run.
  - Supports both fresh agent instances per run (recommended) or stateful agents.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Union

from ..core import Agent
from ..hooks import HookContext, HookEvent
from ..types import ToolCall, ToolResult

try:
    from ..extensions.langsmith import LangSmithExtension, _LANGSMITH_AVAILABLE
except ImportError:
    _LANGSMITH_AVAILABLE = False
    LangSmithExtension = None  # type: ignore[assignment,misc]


AgentFactory = Callable[[dict[str, Any]], Agent]
AgentOrFactory = Union[Agent, AgentFactory]


class AgentTarget:
    """Adapts an Agent or agent factory into a LangSmith target function.

    Usage:
        # With an agent factory (creates a clean agent per example):
        def build_agent(inputs):
            return Agent(config=AgentConfig(model="claude-sonnet-4-6"))

        target = AgentTarget(build_agent)

        # Or with a pre-instantiated agent:
        target = AgentTarget(my_agent)
    """

    def __init__(
        self,
        agent_or_factory: AgentOrFactory,
        *,
        attach_langsmith: bool = True,
        project_name: str | None = None,
        reset_memory_between_runs: bool = True,
    ) -> None:
        self._agent_or_factory = agent_or_factory
        self._attach_langsmith = attach_langsmith
        self._project_name = project_name
        self._reset_memory_between_runs = reset_memory_between_runs

    def _resolve_agent(self, inputs: dict[str, Any]) -> Agent:
        """Resolve or construct the agent instance for this evaluation turn."""
        if callable(self._agent_or_factory) and not isinstance(self._agent_or_factory, Agent):
            # It's an agent factory
            try:
                agent = self._agent_or_factory(inputs)
            except TypeError:
                # Factory might take 0 arguments
                agent = self._agent_or_factory()  # type: ignore[call-arg]
        else:
            # Reusing existing agent
            agent = self._agent_or_factory
            if self._reset_memory_between_runs and hasattr(agent, "memory"):
                agent.memory.clear()

        # Attach LangSmith extension if not present and available
        if self._attach_langsmith and _LANGSMITH_AVAILABLE and LangSmithExtension is not None:
            has_ls = any(isinstance(ext, LangSmithExtension) for ext in getattr(agent, "extensions", []))
            if not has_ls:
                ls_ext = LangSmithExtension(project_name=self._project_name)
                ls_ext.install(agent)
                if hasattr(agent, "extensions"):
                    agent.extensions.append(ls_ext)

        return agent

    async def run_async(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent asynchronously on the given example inputs."""
        agent = self._resolve_agent(inputs)

        # Extract prompt from inputs
        prompt = (
            inputs.get("prompt")
            or inputs.get("input")
            or inputs.get("query")
            or inputs.get("question")
            or ""
        )
        if not prompt and "messages" in inputs:
            # Handle multi-turn message list input
            for m in reversed(inputs["messages"]):
                if m.get("role") == "user":
                    prompt = m.get("content", "")
                    break

        # Telemetry containers
        recorded_tool_calls: list[dict[str, Any]] = []
        recorded_skills: list[str] = []
        active_tools_start: dict[str, float] = {}

        # 1. Wire tool call hooks
        async def on_tool_start(ctx: HookContext) -> None:
            tc = ctx.data.get("tool_call")
            if tc is not None:
                active_tools_start[tc.id] = time.perf_counter()

        async def on_tool_end(ctx: HookContext) -> None:
            tc: ToolCall | None = ctx.data.get("tool_call")
            res: ToolResult | None = ctx.data.get("result")
            if tc is not None and res is not None:
                start_t = active_tools_start.pop(tc.id, None)
                duration_ms = (
                    round((time.perf_counter() - start_t) * 1000, 2)
                    if start_t is not None
                    else None
                )
                recorded_tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                        "output": res.content,
                        "is_error": bool(res.is_error),
                        "duration_ms": duration_ms,
                    }
                )

        # 2. Wire skill invocation hook
        async def on_skill(ctx: HookContext) -> None:
            skill_name = ctx.data.get("skill")
            if skill_name and skill_name not in recorded_skills:
                recorded_skills.append(skill_name)

        agent.hooks.on(HookEvent.TOOL_CALL_START, on_tool_start)
        agent.hooks.on(HookEvent.TOOL_CALL_END, on_tool_end)
        agent.hooks.on(HookEvent.SKILL_INVOKED, on_skill)

        # 3. Execute agent
        error_msg: str | None = None
        output_text = ""
        try:
            output_text = await agent.run(prompt)
        except Exception as e:
            error_msg = str(e)
            output_text = f"Error: {e}"

        # 4. Gather usage stats
        usage_summary = agent.guardrails.usage_summary if hasattr(agent, "guardrails") else {}
        iterations = usage_summary.get("iterations", 0)
        in_tokens = usage_summary.get("input_tokens", 0)
        out_tokens = usage_summary.get("output_tokens", 0)
        cost_str = usage_summary.get("estimated_cost", "$0.0000")

        unique_tools_used = []
        for tc in recorded_tool_calls:
            if tc["name"] not in unique_tools_used:
                unique_tools_used.append(tc["name"])

        return {
            "output": output_text,
            "tool_calls": recorded_tool_calls,
            "tools_used": unique_tools_used,
            "iterations": iterations,
            "token_usage": {
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
            },
            "cost": cost_str,
            "skills_invoked": recorded_skills,
            "error": error_msg,
            "session_id": getattr(agent, "_session_id", None),
        }

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Synchronous target call (compatible with LangSmith's evaluate thread pool)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already in an active event loop, run in a worker thread to avoid loop conflicts
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.run_async(inputs))
                return future.result()
        else:
            return asyncio.run(self.run_async(inputs))


def create_agent_target(
    agent_or_factory: AgentOrFactory,
    *,
    attach_langsmith: bool = True,
    project_name: str | None = None,
    reset_memory_between_runs: bool = True,
) -> AgentTarget:
    """Factory function for creating an AgentTarget."""
    return AgentTarget(
        agent_or_factory,
        attach_langsmith=attach_langsmith,
        project_name=project_name,
        reset_memory_between_runs=reset_memory_between_runs,
    )
