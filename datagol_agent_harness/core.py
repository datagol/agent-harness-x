"""The core agentic loop: LLM reasoning → tool execution → repeat.

This is the heart of the harness. The Agent class orchestrates all other modules:
tool registry, memory, permissions, hooks, middleware, and optionally sandbox.
"""

from __future__ import annotations

import asyncio
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
from .tools import ToolRegistry, normalize_tool_registry
from .types import AgentConfig, SessionState, StopReason, TokenUsage, ToolCall, ToolDefinition, ToolResult


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
        tools: ToolRegistry | list[Any] | None = None,
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
        self.tools = normalize_tool_registry(tools)
        self.memory = memory or ConversationMemory(max_result_chars=self.config.max_result_chars)
        self.permissions = permissions or PermissionManager()
        self.hooks = hooks or HookManager()
        self.middleware = middleware or MiddlewarePipeline()
        self.sandbox = sandbox
        self.mcp = mcp
        self.guardrails = GuardrailsEngine(max_iterations=self.config.max_iterations)
        self._session_id = str(uuid.uuid4())
        self.prompt_providers: list[Any] = []
        self.session_metadata: dict[str, Any] = {}

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
        1. Resets turn-level loop guardrails
        2. Adds the user message to memory
        3. Enters the agentic loop
        4. Returns the final text output
        """
        self.guardrails.reset_turn()
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
            effective_system = self._build_system_prompt()
            await self.memory.trim_if_needed(
                self.provider,
                self.config.model,
                effective_system,
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
                system=effective_system or None,
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
            if response.stop_reason in ("end_turn", StopReason.END_TURN):
                return self._extract_text(response)

            elif response.stop_reason in ("tool_use", StopReason.TOOL_USE, StopReason.TOOL_CALLS):
                tool_results = await self._handle_tool_calls(response)
                if not tool_results:
                    return self._extract_text(response) or "[Model completed turn without tool calls]"
                self.memory.add_tool_results(tool_results)
                # Loop continues...

            elif response.stop_reason in ("max_tokens", StopReason.MAX_TOKENS):
                text = self._extract_text(response)
                return text + "\n[truncated: max_tokens reached]"

            else:
                text = self._extract_text(response)
                return text or f"[Completed with stop reason: {response.stop_reason}]"

            await self.hooks.emit(
                HookEvent.LOOP_ITERATION_END,
                HookContext(
                    event=HookEvent.LOOP_ITERATION_END,
                    agent=self,
                    data={"iteration": self.guardrails.iteration_count},
                ),
            )

        return ""  # Unreachable, but makes type checker happy

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

    async def _handle_tool_calls(self, response: Any) -> list[ToolResult]:
        """Process all tool calls in an LLM response with concurrent execution when supported."""
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
            return []

        tool_results: list[ToolResult] = []
        approved_calls: list[tuple[ToolCall, ToolDefinition]] = []

        for tool_call in raw_tool_calls:
            try:
                tool_def = self.tools.get_tool(tool_call.name)
            except Exception as e:
                tool_results.append(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        content=str(e),
                        is_error=True,
                    )
                )
                continue

            allowed = await self.permissions.check_permission(tool_call, tool_def)
            if not allowed:
                tool_results.append(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        content="Permission denied: tool execution was blocked by the user.",
                        is_error=True,
                    )
                )
                continue

            approved_calls.append((tool_call, tool_def))

        if not approved_calls:
            return tool_results

        async def _execute_single(tc: ToolCall) -> ToolResult:
            await self.hooks.emit(
                HookEvent.TOOL_CALL_START,
                HookContext(
                    event=HookEvent.TOOL_CALL_START,
                    agent=self,
                    data={"tool_call": tc},
                ),
            )
            transformed_call = await self.middleware.process_tool_call(tc)
            res = await self.tools.execute(transformed_call)
            transformed_res = await self.middleware.process_tool_result(res)
            await self.hooks.emit(
                HookEvent.TOOL_CALL_END,
                HookContext(
                    event=HookEvent.TOOL_CALL_END,
                    agent=self,
                    data={"tool_call": transformed_call, "result": transformed_res},
                ),
            )
            return transformed_res

        all_concurrent = all(getattr(tdef, "concurrent", True) for _, tdef in approved_calls)
        if all_concurrent and len(approved_calls) > 1:
            parallel_results = await asyncio.gather(*[_execute_single(tc) for tc, _ in approved_calls])
            tool_results.extend(parallel_results)
        else:
            for tc, _ in approved_calls:
                res = await _execute_single(tc)
                tool_results.append(res)

        return tool_results

    def _extract_text(self, response: Any) -> str:
        """Extract concatenated text from all TextBlock content blocks or response.text."""
        if getattr(response, "text", ""):
            return response.text
        parts: list[str] = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "\n".join(parts)

    # ── Session persistence ──────────────────────────────────────────────

    async def save_session(self, storage_dir: str = ".agent_sessions") -> str:
        """Save current session state to disk. Returns session_id."""
        storage = PersistentMemory(storage_dir)

        # Allow extensions to contribute state
        from .extensions.base import ExtensionContext
        ctx = ExtensionContext(self)
        ext_state: dict[str, Any] = {}
        for ext in getattr(self, "extensions", []):
            try:
                s = await ext.on_save_session(ctx)
                if s:
                    ext_state[ext.name] = s
            except Exception as e:
                print(f"[extension {ext.name}] on_save_session failed: {e}")
        if ext_state:
            self.session_metadata.setdefault("extensions", {}).update(ext_state)

        state = SessionState(
            session_id=self._session_id,
            messages=self.memory.get_messages(),
            total_usage=self.guardrails.total_usage,
            metadata=dict(self.session_metadata),
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
        agent.session_metadata = dict(state.metadata) if state.metadata else {}

        # Restore extension state
        if state.metadata and "extensions" in state.metadata:
            from .extensions.base import ExtensionContext
            ctx = ExtensionContext(agent)
            ext_data = state.metadata["extensions"]
            for ext in getattr(agent, "extensions", []):
                if ext.name in ext_data:
                    try:
                        await ext.on_load_session(ctx, ext_data[ext.name])
                    except Exception as e:
                        print(f"[extension {ext.name}] on_load_session failed: {e}")

        return agent

    @property
    def session_id(self) -> str:
        return self._session_id
