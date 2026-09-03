"""Streaming support: async streaming of LLM responses with typed events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from .extensions.base import Extension, close_extensions, install_extensions
from .hooks import HookContext, HookEvent, HookManager, MiddlewarePipeline
from .memory import ConversationMemory
from .permissions import GuardrailsEngine, PermissionManager
from .skills import SkillManager
from .tools import ToolRegistry
from .types import AgentConfig, ToolCall, ToolResult


class StreamEventType(Enum):
    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    TOOL_RESULT = "tool_result"
    THINKING_DELTA = "thinking_delta"
    TURN_COMPLETE = "turn_complete"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A single event from the streaming agent loop."""

    type: StreamEventType
    data: Any = None


class StreamingAgent:
    """Wraps the core Agent to provide streaming output.

    Uses the Anthropic SDK's streaming API to yield events as they arrive,
    rather than waiting for complete responses.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        client: AsyncAnthropic | None = None,
        tools: ToolRegistry | None = None,
        memory: ConversationMemory | None = None,
        permissions: PermissionManager | None = None,
        hooks: HookManager | None = None,
        middleware: MiddlewarePipeline | None = None,
        skills: SkillManager | list[str] | None = None,
        extensions: list[Extension] | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.client = client or AsyncAnthropic()
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory(max_result_chars=self.config.max_result_chars)
        self.permissions = permissions or PermissionManager()
        self.hooks = hooks or HookManager()
        self.middleware = middleware or MiddlewarePipeline()
        self.guardrails = GuardrailsEngine(max_iterations=self.config.max_iterations)

        if skills is not None:
            self.skills = (
                skills if isinstance(skills, SkillManager) else SkillManager.from_paths(skills)
            )
            self.skills.install(self)
        else:
            self.skills = None

        # Install extensions after skills so they can rely on the Skill tool.
        self.extensions = install_extensions(self, extensions)

    async def aclose(self) -> None:
        """Tear down all extensions in reverse install order. Idempotent."""
        await close_extensions(self.extensions)
        self.extensions = []

    async def run_stream(self, user_message: str) -> AsyncIterator[StreamEvent]:
        """Run the agentic loop, yielding StreamEvents as they happen."""
        self.memory.add_user_message(user_message)

        await self.hooks.emit(
            HookEvent.AGENT_START,
            HookContext(event=HookEvent.AGENT_START, agent=self),
        )

        try:
            async for event in self._streaming_loop():
                yield event
        except Exception as e:
            yield StreamEvent(type=StreamEventType.ERROR, data=str(e))

        await self.hooks.emit(
            HookEvent.AGENT_END,
            HookContext(event=HookEvent.AGENT_END, agent=self),
        )

    async def _streaming_loop(self) -> AsyncIterator[StreamEvent]:
        while True:
            self.guardrails.check_iteration_limit()
            self.guardrails.record_iteration()

            # Trim memory if approaching context limit
            tool_params = self.tools.get_tool_params()
            await self.memory.trim_if_needed(
                self.client,
                self.config.model,
                self.config.system_prompt or "",
                tool_params,
            )

            messages, tool_params = await self.middleware.process_llm_request(
                self.memory.get_messages(),
                tool_params,
            )

            await self.hooks.emit(
                HookEvent.LLM_REQUEST,
                HookContext(event=HookEvent.LLM_REQUEST, agent=self),
            )

            # Stream the response
            collected_text = ""
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            if self.config.system_prompt:
                kwargs["system"] = self.config.system_prompt
            if tool_params:
                kwargs["tools"] = tool_params

            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    collected_text += text
                    yield StreamEvent(type=StreamEventType.TEXT_DELTA, data=text)

                response = await stream.get_final_message()

            if collected_text:
                yield StreamEvent(type=StreamEventType.TEXT_COMPLETE, data=collected_text)

            response = await self.middleware.process_llm_response(response)
            self.guardrails.track_usage(response.usage)
            self.memory.add_assistant_message(response.content)

            await self.hooks.emit(
                HookEvent.LLM_RESPONSE,
                HookContext(event=HookEvent.LLM_RESPONSE, agent=self, data={"response": response}),
            )

            if response.stop_reason == "end_turn":
                yield StreamEvent(type=StreamEventType.TURN_COMPLETE, data="end_turn")
                return

            if response.stop_reason == "tool_use":
                tool_results: list[ToolResult] = []

                for block in response.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue

                    tool_call = ToolCall(id=block.id, name=block.name, input=block.input)
                    yield StreamEvent(type=StreamEventType.TOOL_CALL_START, data=tool_call)

                    tool_def = self.tools.get_tool(tool_call.name)
                    allowed = await self.permissions.check_permission(tool_call, tool_def)

                    if not allowed:
                        result = ToolResult(
                            tool_use_id=tool_call.id,
                            content="Permission denied.",
                            is_error=True,
                        )
                    else:
                        tool_call = await self.middleware.process_tool_call(tool_call)
                        result = await self.tools.execute(tool_call)
                        result = await self.middleware.process_tool_result(result)

                    tool_results.append(result)
                    yield StreamEvent(type=StreamEventType.TOOL_CALL_COMPLETE, data=tool_call)
                    yield StreamEvent(type=StreamEventType.TOOL_RESULT, data=result)

                self.memory.add_tool_results(tool_results)

            if response.stop_reason == "max_tokens":
                yield StreamEvent(type=StreamEventType.TURN_COMPLETE, data="max_tokens")
                return
