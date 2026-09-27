"""Hooks and middleware: event system for observability + transform pipeline.

Hooks are observe-only side effects (logging, auditing, metrics).
Middleware can transform data passing through the pipeline.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class HookEvent(Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LOOP_ITERATION_START = "loop_iteration_start"
    LOOP_ITERATION_END = "loop_iteration_end"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    SKILL_INVOKED = "skill_invoked"
    SANDBOX_EXEC = "sandbox_exec"
    CHECKPOINT = "checkpoint"
    ERROR = "error"


@dataclass
class HookContext:
    """Data passed to hook callbacks."""

    event: HookEvent
    agent: Any = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


HookCallback = Callable[[HookContext], Awaitable[None] | None]


class Registration:
    """Idempotent handle for removing a hook, middleware, tool, or prompt provider."""

    def __init__(self, remove: Callable[[], None]) -> None:
        self._remove = remove
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class HookManager:
    """Event-based hook system. Supports both sync and async callbacks."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookCallback]] = {}

    def on(self, event: HookEvent, callback: HookCallback) -> Registration:
        self._hooks.setdefault(event, []).append(callback)
        return Registration(lambda: self.off(event, callback))

    def off(self, event: HookEvent, callback: HookCallback) -> None:
        if event in self._hooks:
            self._hooks[event] = [cb for cb in self._hooks[event] if cb is not callback]

    async def emit(self, event: HookEvent, context: HookContext) -> None:
        """Fire all callbacks for an event."""
        for callback in tuple(self._hooks.get(event, [])):
            try:
                result = callback(context)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass  # Hooks should never crash the agent

    # Convenience decorators
    def before_tool(self, callback: HookCallback) -> HookCallback:
        """Decorator: register a TOOL_CALL_START hook."""
        self.on(HookEvent.TOOL_CALL_START, callback)
        return callback

    def after_tool(self, callback: HookCallback) -> HookCallback:
        """Decorator: register a TOOL_CALL_END hook."""
        self.on(HookEvent.TOOL_CALL_END, callback)
        return callback

    def on_error(self, callback: HookCallback) -> HookCallback:
        """Decorator: register an ERROR hook."""
        self.on(HookEvent.ERROR, callback)
        return callback


class Middleware:
    """Base class for middleware that wraps the agentic loop.

    Override any method to transform data at that point in the pipeline.
    Default implementations pass data through unchanged.
    """

    async def before_llm_call(
        self, messages: list[dict], tools: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        return messages, tools

    async def after_llm_call(self, response: Any) -> Any:
        return response

    async def before_tool_execution(self, tool_call: Any) -> Any:
        return tool_call

    async def after_tool_execution(self, result: Any) -> Any:
        return result


class MiddlewarePipeline:
    """Ordered chain of middleware. Each one wraps the next."""

    def __init__(self) -> None:
        self._middleware: list[Middleware] = []

    def add(self, middleware: Middleware) -> Registration:
        self._middleware.append(middleware)
        return Registration(lambda: self.remove(middleware))

    def remove(self, middleware: Middleware) -> None:
        self._middleware = [item for item in self._middleware if item is not middleware]

    async def process_llm_request(
        self, messages: list[dict], tools: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        for mw in self._middleware:
            messages, tools = await mw.before_llm_call(messages, tools)
        return messages, tools

    async def process_llm_response(self, response: Any) -> Any:
        for mw in self._middleware:
            response = await mw.after_llm_call(response)
        return response

    async def process_tool_call(self, tool_call: Any) -> Any:
        for mw in self._middleware:
            tool_call = await mw.before_tool_execution(tool_call)
        return tool_call

    async def process_tool_result(self, result: Any) -> Any:
        for mw in self._middleware:
            result = await mw.after_tool_execution(result)
        return result
