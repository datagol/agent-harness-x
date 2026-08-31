"""The core agentic loop: LLM reasoning → tool execution → repeat.

This is the heart of the harness. The Agent class orchestrates all other modules:
tool registry, memory, permissions, hooks, middleware, and optionally sandbox.
"""

from __future__ import annotations

import uuid
from typing import Any

from .hooks import HookContext, HookEvent, HookManager, MiddlewarePipeline
from .memory import ConversationMemory, PersistentMemory
from .permissions import GuardrailsEngine, PermissionManager
from .extensions.base import Extension, close_extensions, install_extensions
from .mcp import MCPManager
from .providers import LLMProvider, make_provider
from .sandbox import Sandbox
from .skills import SkillManager
from .tools import ToolRegistry
from .types import AgentConfig, SessionState, TokenUsage, ToolCall, ToolResult


class Agent:
    """The core agentic loop: LLM reasoning -> tool execution -> repeat.

    This is the central orchestrator. It owns the conversation, calls the LLM,
    dispatches tool calls, enforces guardrails, and fires hooks.

    Usage:
        agent = Agent(config=AgentConfig(system_prompt="You are helpful."))

        @agent.tools.register(permission=PermissionLevel.ALLOW)
        async def read_file(path: str) -> str:
            ...

        result = await agent.run("What files are in the current directory?")
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        provider: LLMProvider | None = None,
        client: Any | None = None,  # back-compat shim, wrapped into AnthropicProvider
        tools: ToolRegistry | None = None,
        memory: ConversationMemory | None = None,
        permissions: PermissionManager | None = None,
        hooks: HookManager | None = None,
        middleware: MiddlewarePipeline | None = None,
        sandbox: Sandbox | None = None,
        mcp: MCPManager | None = None,
        skills: SkillManager | list[str] | None = None,
        extensions: list[Extension] | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        if provider is not None:
            self.provider = provider
        elif client is not None:
            # Back-compat: a raw Anthropic client was passed.
            from .providers.anthropic import AnthropicProvider
            self.provider = AnthropicProvider(client=client)
        else:
            self.provider = make_provider(self.config.provider)
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory(max_result_chars=self.config.max_result_chars)
        self.permissions = permissions or PermissionManager()
        self.hooks = hooks or HookManager()
        self.middleware = middleware or MiddlewarePipeline()
        self.sandbox = sandbox
        self.mcp = mcp
        self.guardrails = GuardrailsEngine(max_iterations=self.config.max_iterations)
        self._session_id = str(uuid.uuid4())

        if skills is not None:
            self.skills = (
                skills if isinstance(skills, SkillManager) else SkillManager.from_paths(skills)
            )
            self.skills.install(self)
        else:
            self.skills = None

        # Install extensions after skills so they can rely on the Skill tool
        # existing (and on skill bodies having been appended to the system
        # prompt). Stored for later teardown via aclose().
        self.extensions = install_extensions(self, extensions)

    async def aclose(self) -> None:
        """Tear down all extensions in reverse install order. Idempotent."""
        await close_extensions(self.extensions)
        self.extensions = []

    async def run(self, user_message: str) -> str:
        """Run the full agentic loop until the LLM produces a final text response.

        This is the main entry point. It:
        1. Adds the user message to memory
        2. Enters the agentic loop
        3. Returns the final text output
        """
        self.memory.add_user_message(user_message)

        await self.hooks.emit(
            HookEvent.AGENT_START,
            HookContext(event=HookEvent.AGENT_START, agent=self, data={"message": user_message}),
        )

        try:
            result = await self._agentic_loop()
        except Exception as e:
            await self.hooks.emit(
                HookEvent.ERROR,
                HookContext(event=HookEvent.ERROR, agent=self, data={"error": str(e)}),
            )
            raise

        await self.hooks.emit(
            HookEvent.AGENT_END,
            HookContext(event=HookEvent.AGENT_END, agent=self, data={"result": result}),
        )

        return result

    async def _agentic_loop(self) -> str:
        """The core loop: call LLM → check stop reason → execute tools → repeat."""
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
            await self.memory.trim_if_needed(
                self.provider,
                self.config.model,
                self.config.system_prompt or "",
                tool_params,
            )

            # Prepare request through middleware
            messages = self.memory.get_messages()
            messages, tool_params = await self.middleware.process_llm_request(messages, tool_params)

            await self.hooks.emit(
                HookEvent.LLM_REQUEST,
                HookContext(
                    event=HookEvent.LLM_REQUEST,
                    agent=self,
                    data={"message_count": len(messages), "tool_count": len(tool_params)},
                ),
            )

            response = await self.provider.create(
                model=self.config.model,
                messages=messages,
                system=self.config.system_prompt or None,
                tools=tool_params,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            response = await self.middleware.process_llm_response(response)

            await self.hooks.emit(
                HookEvent.LLM_RESPONSE,
                HookContext(
                    event=HookEvent.LLM_RESPONSE,
                    agent=self,
                    data={"stop_reason": response.stop_reason},
                ),
            )

            # Track usage
            self.guardrails.track_usage(response.usage)

            # Add assistant response to memory
            self.memory.add_assistant_message(response.content)

            # Check stop reason
            if response.stop_reason == "end_turn":
                return self._extract_text(response)

            if response.stop_reason == "tool_use":
                tool_results = await self._handle_tool_calls(response)
                self.memory.add_tool_results(tool_results)
                # Loop continues...

            elif response.stop_reason == "max_tokens":
                text = self._extract_text(response)
                return text + "\n[truncated: max_tokens reached]"

            await self.hooks.emit(
                HookEvent.LOOP_ITERATION_END,
                HookContext(
                    event=HookEvent.LOOP_ITERATION_END,
                    agent=self,
                    data={"iteration": self.guardrails.iteration_count},
                ),
            )

        return ""  # Unreachable, but makes type checker happy

    async def _handle_tool_calls(self, response: Any) -> list[ToolResult]:
        """Process all tool calls in an LLM response."""
        tool_results: list[ToolResult] = []

        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue

            tool_call = ToolCall(id=block.id, name=block.name, input=block.input)

            # Permission check
            try:
                tool_def = self.tools.get_tool(tool_call.name)
            except Exception as e:
                tool_results.append(ToolResult(
                    tool_use_id=tool_call.id,
                    content=str(e),
                    is_error=True,
                ))
                continue

            allowed = await self.permissions.check_permission(tool_call, tool_def)

            if not allowed:
                tool_results.append(ToolResult(
                    tool_use_id=tool_call.id,
                    content="Permission denied: tool execution was blocked by the user.",
                    is_error=True,
                ))
                continue

            # Execute with hooks and middleware
            await self.hooks.emit(
                HookEvent.TOOL_CALL_START,
                HookContext(
                    event=HookEvent.TOOL_CALL_START,
                    agent=self,
                    data={"tool_call": tool_call},
                ),
            )

            tool_call = await self.middleware.process_tool_call(tool_call)
            result = await self.tools.execute(tool_call)
            result = await self.middleware.process_tool_result(result)

            await self.hooks.emit(
                HookEvent.TOOL_CALL_END,
                HookContext(
                    event=HookEvent.TOOL_CALL_END,
                    agent=self,
                    data={"tool_call": tool_call, "result": result},
                ),
            )

            tool_results.append(result)

        return tool_results

    def _extract_text(self, response: Any) -> str:
        """Extract concatenated text from all TextBlock content blocks."""
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts)

    # ── Session persistence ──────────────────────────────────────────────

    async def save_session(self, storage_dir: str = ".agent_sessions") -> str:
        """Save current session state to disk. Returns session_id."""
        storage = PersistentMemory(storage_dir)
        state = SessionState(
            session_id=self._session_id,
            messages=self.memory.get_messages(),
            total_usage=self.guardrails.total_usage,
        )
        storage.save_session(state)
        return self._session_id

    @classmethod
    async def load_session(
        cls,
        session_id: str,
        storage_dir: str = ".agent_sessions",
        **kwargs: Any,
    ) -> Agent:
        """Restore an agent from a saved session."""
        storage = PersistentMemory(storage_dir)
        state = storage.load_session(session_id)

        agent = cls(**kwargs)
        agent._session_id = state.session_id
        agent.memory.set_messages(state.messages)
        agent.guardrails._total_usage = state.total_usage
        return agent

    @property
    def session_id(self) -> str:
        return self._session_id
