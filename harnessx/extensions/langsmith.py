"""LangSmithExtension — first-class LangSmith tracing for DataGOL agents.

Hooks into the agent lifecycle and middleware pipeline to trace:
  - Agent turns (as root `chain` runs in LangSmith)
  - LLM calls (as child `llm` runs with prompt, tools, outputs, and token usage)
  - Tool executions (as child `tool` runs with arguments, results, and error reporting)
  - Sub-agent calls (automatically nested under parent tool runs in multi-agent workflows)

Usage:
    from harnessx import Agent, AgentConfig
    from harnessx.extensions import LangSmithExtension

    agent = Agent(
        config=AgentConfig(model="claude-sonnet-4-6"),
        extensions=[
            LangSmithExtension(
                project_name="my-project",
                tags=["production"],
            )
        ],
    )
    await agent.run("What is 17 + 25?")
"""

from __future__ import annotations

import contextvars
import logging
import os
from typing import Any

from ..hooks import HookContext, HookEvent, Middleware
from ..types import ToolCall, ToolResult
from .base import Extension

logger = logging.getLogger(__name__)

try:
    from langsmith import Client
    from langsmith.run_trees import RunTree

    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False
    Client = Any  # type: ignore[misc,assignment]
    RunTree = Any  # type: ignore[misc,assignment]


# ── Context Variables for Distributed / Async Tracing ────────────────────────

# Stack of active agent runs (for nesting sub-agents under tool calls or root runs)
_agent_stack: contextvars.ContextVar[tuple[Any, ...]] = contextvars.ContextVar(
    "langsmith_agent_stack", default=()
)

# Currently executing tool run (if a tool invokes a sub-agent, the sub-agent nests here)
_current_tool_run: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "langsmith_current_tool_run", default=None
)

# Active tool runs mapped by tool_call_id
_active_tool_runs: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "langsmith_active_tool_runs", default=None
)

# Currently active LLM call run
_active_llm_run: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "langsmith_active_llm_run", default=None
)


def _safe_post(run: Any) -> None:
    """Post a run tree asynchronously; never raise or crash the agent."""
    if run is None:
        return
    try:
        run.post()
    except Exception as exc:
        logger.debug("Failed to post LangSmith run: %s", exc)


def _safe_patch(run: Any) -> None:
    """Patch a completed run tree asynchronously; never raise or crash the agent."""
    if run is None:
        return
    try:
        run.patch()
    except Exception as exc:
        logger.debug("Failed to patch LangSmith run: %s", exc)


def _serialize_response_content(content: Any) -> Any:
    """Convert provider response content blocks into serializable format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        serialized: list[Any] = []
        for block in content:
            if hasattr(block, "type"):
                b_type = getattr(block, "type")
                if b_type == "text":
                    serialized.append({"type": "text", "text": getattr(block, "text", "")})
                elif b_type == "tool_use":
                    serialized.append(
                        {
                            "type": "tool_use",
                            "id": getattr(block, "id", None),
                            "name": getattr(block, "name", None),
                            "input": getattr(block, "input", None),
                        }
                    )
                else:
                    serialized.append(str(block))
            elif isinstance(block, dict):
                serialized.append(block)
            else:
                serialized.append(str(block))
        return serialized
    return str(content)


# ── LangSmith Extension ─────────────────────────────────────────────────────


class LangSmithExtension(Extension):
    """Extension that integrates DataGOL agent runs with LangSmith tracing."""

    name = "langsmith"

    def __init__(
        self,
        *,
        project_name: str | None = None,
        run_name: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        client: Client | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_llm: bool = True,
        trace_tools: bool = True,
        enabled: bool = True,
    ) -> None:
        if not _LANGSMITH_AVAILABLE:
            raise ImportError(
                "The 'langsmith' package is required to use LangSmithExtension. "
                "Install it with: pip install 'harnessx[langsmith]' or pip install langsmith"
            )

        self.project_name = (
            project_name
            or os.getenv("LANGSMITH_PROJECT")
            or os.getenv("LANGCHAIN_PROJECT")
            or "datagol-agents"
        )
        self.run_name = run_name
        self.tags = list(tags) if tags else []
        self.metadata = dict(metadata) if metadata else {}
        self.trace_llm = trace_llm
        self.trace_tools = trace_tools
        self.enabled = enabled

        # Initialize or capture client
        if client is not None:
            self._client = client
        elif api_key or api_url:
            self._client = Client(api_key=api_key, api_url=api_url)
        else:
            self._client = None  # RunTree creates default client from env vars

    def install(self, agent: Any) -> None:
        """Wire tracing hooks and middleware into the agent."""
        if not self.enabled:
            return

        # Register middleware for LLM and Tool call inspection
        from .base import ExtensionContext
        context = agent if isinstance(agent, ExtensionContext) else ExtensionContext(agent, self.name)
        context.add_middleware(_LangSmithMiddleware(self, context.agent))

        # Register lifecycle hooks for Agent start, end, and error
        context.on_hook(HookEvent.AGENT_START, self._on_agent_start)
        context.on_hook(HookEvent.AGENT_END, self._on_agent_end)
        context.on_hook(HookEvent.ERROR, self._on_agent_error)

    async def teardown(self) -> None:
        """Flush traces on agent close."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass

    # ── Lifecycle Hook Handlers ─────────────────────────────────────────

    async def _on_agent_start(self, ctx: HookContext) -> None:
        if not self.enabled:
            return

        agent = ctx.agent
        # Determine prompt input
        user_message = ctx.data.get("message")
        if not user_message and hasattr(agent, "memory"):
            messages = agent.memory.get_messages()
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content")
                    break
        prompt_text = str(user_message or "")

        # Prepare trace metadata
        meta = {
            "session_id": getattr(agent, "_session_id", None),
            "model": getattr(agent.config, "model", None),
            "provider": getattr(agent.config, "provider", None),
            **self.metadata,
        }

        trace_name = (
            self.run_name
            or (f"{agent.config.model} Agent" if hasattr(agent, "config") else "Agent Run")
        )

        # Check if this agent run is nested under a currently running tool or parent agent
        parent = _current_tool_run.get()
        if parent is None:
            stack = _agent_stack.get()
            if stack:
                parent = stack[-1]

        # If not in an internal stack, check for an active LangSmith evaluation run tree context
        if parent is None and _LANGSMITH_AVAILABLE:
            try:
                from langsmith.run_helpers import get_current_run_tree
                parent = get_current_run_tree()
            except Exception:
                parent = None

        if parent is not None:
            root_run = parent.create_child(
                name=trace_name,
                run_type="chain",
                inputs={"input": prompt_text},
                tags=self.tags,
                extra={"metadata": meta},
            )
        else:
            root_run = RunTree(
                name=trace_name,
                run_type="chain",
                inputs={"input": prompt_text},
                project_name=self.project_name,
                client=self._client,
                tags=self.tags,
                extra={"metadata": meta},
            )

        _safe_post(root_run)

        # Push onto agent stack
        current_stack = _agent_stack.get()
        _agent_stack.set(current_stack + (root_run,))

        # Initialize active tools mapping for this execution branch
        if _active_tool_runs.get() is None:
            _active_tool_runs.set({})

    async def _on_agent_end(self, ctx: HookContext) -> None:
        if not self.enabled:
            return

        stack = _agent_stack.get()
        if not stack:
            return

        current_run = stack[-1]
        _agent_stack.set(stack[:-1])

        result_text = ctx.data.get("result", "")
        result_text = getattr(result_text, "output", result_text)
        current_run.end(outputs={"output": result_text})
        _safe_patch(current_run)

    async def _on_agent_error(self, ctx: HookContext) -> None:
        if not self.enabled:
            return

        error_msg = str(ctx.data.get("error", "Agent run error"))

        # End active LLM run if failing during LLM call
        active_llm = _active_llm_run.get()
        if active_llm is not None:
            _active_llm_run.set(None)
            active_llm.end(error=error_msg)
            _safe_patch(active_llm)

        # End any active tool runs
        active_tools = _active_tool_runs.get() or {}
        for tool_run in list(active_tools.values()):
            tool_run.end(error=error_msg)
            _safe_patch(tool_run)
        _active_tool_runs.set({})

        # End current agent run
        stack = _agent_stack.get()
        if stack:
            current_run = stack[-1]
            _agent_stack.set(stack[:-1])
            current_run.end(error=error_msg)
            _safe_patch(current_run)


# ── Middleware for LLM & Tool Calls ─────────────────────────────────────────


class _LangSmithMiddleware(Middleware):
    """Captures LLM requests/responses and tool executions into LangSmith spans."""

    def __init__(self, ext: LangSmithExtension, agent: Any) -> None:
        self._ext = ext
        self._agent = agent

    def _get_parent(self) -> Any | None:
        stack = _agent_stack.get()
        return stack[-1] if stack else None

    async def before_llm_call(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self._ext.enabled or not self._ext.trace_llm:
            return messages, tools

        parent = self._get_parent()
        if parent is not None:
            model_name = getattr(self._agent.config, "model", "llm")
            provider_name = getattr(self._agent.config, "provider", "")
            llm_name = f"{provider_name}:{model_name}" if provider_name else model_name

            llm_run = parent.create_child(
                name=llm_name,
                run_type="llm",
                inputs={"messages": messages, "tools": tools},
                extra={
                    "metadata": {
                        "model": model_name,
                        "provider": provider_name,
                    }
                },
            )
            _safe_post(llm_run)
            _active_llm_run.set(llm_run)

        return messages, tools

    async def after_llm_call(self, response: Any) -> Any:
        if not self._ext.enabled or not self._ext.trace_llm:
            return response

        llm_run = _active_llm_run.get()
        if llm_run is not None:
            _active_llm_run.set(None)

            usage_dict = None
            if hasattr(response, "usage") and response.usage:
                in_tok = getattr(response.usage, "input_tokens", 0) or 0
                out_tok = getattr(response.usage, "output_tokens", 0) or 0
                tot_tok = getattr(response.usage, "total_tokens", None)
                if tot_tok is None:
                    tot_tok = in_tok + out_tok
                usage_dict = {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": tot_tok,
                }

            content = getattr(response, "content", "")
            outputs = {
                "content": _serialize_response_content(content),
                "stop_reason": getattr(response, "stop_reason", None),
            }

            llm_run.end(
                outputs=outputs,
                metadata={"usage_metadata": usage_dict} if usage_dict else None,
            )
            _safe_patch(llm_run)

        return response

    async def before_tool_execution(self, tool_call: Any) -> Any:
        if not self._ext.enabled or not self._ext.trace_tools:
            return tool_call

        parent = self._get_parent()
        if parent is not None and isinstance(tool_call, ToolCall):
            tool_input = (
                tool_call.input
                if isinstance(tool_call.input, dict)
                else {"input": tool_call.input}
            )
            tool_run = parent.create_child(
                name=tool_call.name,
                run_type="tool",
                inputs=tool_input,
            )
            _safe_post(tool_run)

            # Record in active tools mapping
            active = dict(_active_tool_runs.get() or {})
            active[tool_call.id] = tool_run
            _active_tool_runs.set(active)

            # Set as active tool run in case a sub-agent is invoked within this tool
            _current_tool_run.set(tool_run)

        return tool_call

    async def after_tool_execution(self, result: Any) -> Any:
        if not self._ext.enabled or not self._ext.trace_tools:
            return result

        if isinstance(result, ToolResult):
            active = dict(_active_tool_runs.get() or {})
            tool_run = active.pop(result.tool_use_id, None)
            _active_tool_runs.set(active)

            if tool_run is not None:
                # Reset current tool run context
                if _current_tool_run.get() is tool_run:
                    _current_tool_run.set(None)

                error_msg = result.content if result.is_error else None
                tool_run.end(
                    outputs={"output": result.content},
                    error=error_msg,
                )
                _safe_patch(tool_run)

        return result
