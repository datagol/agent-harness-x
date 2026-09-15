"""Streaming support: async streaming of LLM responses with typed events."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator

from .extensions.base import Extension, close_extensions, install_extensions
from .hooks import HookContext, HookEvent, HookManager, MiddlewarePipeline
from .memory import ConversationMemory
from .permissions import GuardrailsEngine, PermissionManager
from .providers import LLMProvider, make_provider
from .skills import SkillManager
from .tools import ToolRegistry, normalize_tool_registry
from .types import AgentConfig, StopReason, StreamChunk, ToolCall, ToolResult


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
    """Wraps the core Agent loop to provide streaming output across providers.

    Uses the LLMProvider streaming interface to yield events as they arrive,
    normalizing text deltas, thinking blocks, and tool executions.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        provider: LLMProvider | None = None,
        client: Any | None = None,  # back-compat shim
        tools: ToolRegistry | list[Any] | None = None,
        memory: ConversationMemory | None = None,
        permissions: PermissionManager | None = None,
        hooks: HookManager | None = None,
        middleware: MiddlewarePipeline | None = None,
        skills: SkillManager | list[str] | None = None,
        extensions: list[Extension] | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.client = client
        if provider is not None:
            self.provider = provider
        elif client is not None:
            from .providers.anthropic import AnthropicProvider
            self.provider = AnthropicProvider(client=client)
        else:
            self.provider = make_provider(self.config.provider)

        self.tools = normalize_tool_registry(tools)
        self.memory = memory or ConversationMemory(max_result_chars=self.config.max_result_chars)
        self.permissions = permissions or PermissionManager()
        self.hooks = hooks or HookManager()
        self.middleware = middleware or MiddlewarePipeline()
        self.guardrails = GuardrailsEngine(max_iterations=self.config.max_iterations)
        self.prompt_providers: list[Any] = []
        self.session_metadata: dict[str, Any] = {}

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

    def _build_system_prompt(self) -> str:
        """Compose static system prompt with dynamic prompt providers."""
        prompt = self.config.system_prompt or ""
        for provider_fn in self.prompt_providers:
            try:
                extra = provider_fn()
                if extra:
                    prompt = f"{prompt}\n\n{extra}" if prompt else extra
            except Exception:
                pass
        return prompt

    async def run_stream(self, user_message: str) -> AsyncIterator[StreamEvent]:
        """Run the agentic loop, yielding StreamEvents as they happen."""
        self.guardrails.reset_turn()
        self.memory.add_user_message(user_message)

        await self.hooks.emit(
            HookEvent.AGENT_START,
            HookContext(
                event=HookEvent.AGENT_START,
                agent=self,
                data={"message": user_message},
            ),
        )

        last_text = ""
        try:
            async for event in self._streaming_loop():
                if event.type == StreamEventType.TEXT_COMPLETE:
                    last_text = str(event.data)
                yield event
        except Exception as e:
            await self.hooks.emit(
                HookEvent.ERROR,
                HookContext(event=HookEvent.ERROR, agent=self, data={"error": str(e)}),
            )
            yield StreamEvent(type=StreamEventType.ERROR, data=str(e))

        await self.hooks.emit(
            HookEvent.AGENT_END,
            HookContext(
                event=HookEvent.AGENT_END,
                agent=self,
                data={"result": last_text},
            ),
        )

    async def _streaming_loop(self) -> AsyncIterator[StreamEvent]:
        while True:
            self.guardrails.check_iteration_limit()
            self.guardrails.check_cost_limit()
            self.guardrails.record_iteration()

            await self.hooks.emit(
                HookEvent.LOOP_ITERATION_START,
                HookContext(
                    event=HookEvent.LOOP_ITERATION_START,
                    agent=self,
                    data={"iteration": self.guardrails.iteration_count},
                ),
            )

            # Trim memory if approaching context limit
            tool_params = self.tools.get_tool_params()
            effective_system = self._build_system_prompt()
            await self.memory.trim_if_needed(
                self.provider,
                self.config.model,
                effective_system,
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

            collected_text = ""
            response: Any = None

            async for chunk in self.provider.stream(
                model=self.config.model,
                messages=messages,
                system=effective_system or None,
                tools=tool_params,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ):
                kind = getattr(chunk, "kind", None)
                if kind == "text_delta":
                    collected_text += str(chunk.data)
                    yield StreamEvent(type=StreamEventType.TEXT_DELTA, data=chunk.data)
                elif kind == "thinking_delta":
                    yield StreamEvent(type=StreamEventType.THINKING_DELTA, data=chunk.data)
                elif kind == "response":
                    response = chunk.data
                elif hasattr(chunk, "content"):
                    response = chunk

            if collected_text:
                yield StreamEvent(type=StreamEventType.TEXT_COMPLETE, data=collected_text)

            if response is None:
                # Fallback ProviderResponse if no final chunk was yielded
                from .types import ProviderResponse
                response = ProviderResponse(text=collected_text)

            response = await self.middleware.process_llm_response(response)
            self.guardrails.track_usage(response.usage)
            self.memory.add_assistant_message(response.content)

            await self.hooks.emit(
                HookEvent.LLM_RESPONSE,
                HookContext(event=HookEvent.LLM_RESPONSE, agent=self, data={"response": response}),
            )

            if response.stop_reason in ("end_turn", StopReason.END_TURN):
                yield StreamEvent(type=StreamEventType.TURN_COMPLETE, data="end_turn")
                return

            if response.stop_reason in ("tool_use", StopReason.TOOL_USE, StopReason.TOOL_CALLS):
                raw_tool_calls: list[ToolCall] = []
                if getattr(response, "tool_calls", None):
                    raw_tool_calls = list(response.tool_calls)
                elif getattr(response, "content", None):
                    for block in response.content:
                        if getattr(block, "type", None) == "tool_use":
                            raw_tool_calls.append(
                                ToolCall(
                                    id=getattr(block, "id", ""),
                                    name=getattr(block, "name", ""),
                                    input=getattr(block, "input", {}) or {},
                                )
                            )

                if not raw_tool_calls:
                    yield StreamEvent(type=StreamEventType.TURN_COMPLETE, data="end_turn")
                    return

                tool_results: list[ToolResult] = []

                for tool_call in raw_tool_calls:
                    yield StreamEvent(type=StreamEventType.TOOL_CALL_START, data=tool_call)

                    try:
                        tool_def = self.tools.get_tool(tool_call.name)
                    except Exception as e:
                        result = ToolResult(
                            tool_call_id=tool_call.id,
                            content=str(e),
                            is_error=True,
                        )
                        tool_results.append(result)
                        yield StreamEvent(type=StreamEventType.TOOL_CALL_COMPLETE, data=tool_call)
                        yield StreamEvent(type=StreamEventType.TOOL_RESULT, data=result)
                        continue

                    allowed = await self.permissions.check_permission(tool_call, tool_def)

                    if not allowed:
                        result = ToolResult(
                            tool_call_id=tool_call.id,
                            content="Permission denied: tool execution was blocked by the user.",
                            is_error=True,
                        )
                    else:
                        await self.hooks.emit(
                            HookEvent.TOOL_CALL_START,
                            HookContext(
                                event=HookEvent.TOOL_CALL_START,
                                agent=self,
                                data={"tool_call": tool_call},
                            ),
                        )
                        transformed_call = await self.middleware.process_tool_call(tool_call)
                        result = await self.tools.execute(transformed_call)
                        result = await self.middleware.process_tool_result(result)
                        await self.hooks.emit(
                            HookEvent.TOOL_CALL_END,
                            HookContext(
                                event=HookEvent.TOOL_CALL_END,
                                agent=self,
                                data={"tool_call": transformed_call, "result": result},
                            ),
                        )

                    tool_results.append(result)
                    yield StreamEvent(type=StreamEventType.TOOL_CALL_COMPLETE, data=tool_call)
                    yield StreamEvent(type=StreamEventType.TOOL_RESULT, data=result)

                self.memory.add_tool_results(tool_results)

            elif response.stop_reason in ("max_tokens", StopReason.MAX_TOKENS):
                yield StreamEvent(type=StreamEventType.TURN_COMPLETE, data="max_tokens")
                return

            else:
                yield StreamEvent(type=StreamEventType.TURN_COMPLETE, data=str(response.stop_reason))
                return

            await self.hooks.emit(
                HookEvent.LOOP_ITERATION_END,
                HookContext(
                    event=HookEvent.LOOP_ITERATION_END,
                    agent=self,
                    data={"iteration": self.guardrails.iteration_count},
                ),
            )
