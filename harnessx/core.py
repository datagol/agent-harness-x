"""The core agentic loop: LLM reasoning → tool execution → repeat.

This is the heart of the harness. The Agent class orchestrates all other modules:
tool registry, memory, permissions, hooks, middleware, and optionally sandbox.
"""

from __future__ import annotations

import uuid
import asyncio
from dataclasses import asdict
from typing import Any

from .execution import RunResult, RunStream
from .hooks import HookManager, MiddlewarePipeline
from .memory import ConversationMemory, PersistentMemory
from .permissions import GuardrailsEngine, PermissionManager
from .extensions.base import Extension, close_extensions, install_extensions, validate_extensions
from .mcp import MCPManager
from .providers import LLMProvider, make_provider
from .sandbox import Sandbox
from .skills import SkillManager
from .subagents import SubAgent, install_subagent, prepare_subagents
from .tools import ToolRegistry, normalize_tool_registry
from .types import AgentConfig, SessionState


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
        subagents: list[SubAgent] | None = None,
    ) -> None:
        extensions = validate_extensions(extensions)
        if provider is not None and client is not None:
            raise ValueError("Pass provider or client, not both")
        if client is not None and config is not None and config.provider != "anthropic":
            raise ValueError("The legacy client argument requires the anthropic provider")
        provider_name = getattr(provider, "name", "")
        if provider_name in ("anthropic", "openai", "gemini", "openrouter"):
            if config is None and provider_name != "anthropic":
                raise ValueError("An injected non-Anthropic provider requires AgentConfig with its provider and model")
            if config is not None and config.provider != provider_name:
                raise ValueError("Injected provider does not match AgentConfig.provider")
        self.config = config or AgentConfig()
        self.tools = normalize_tool_registry(tools)
        self.subagents = prepare_subagents(subagents, self.tools)
        for subagent in self.subagents:
            install_subagent(self, subagent)
        self._owns_provider = provider is None and client is None
        self._busy = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        if provider is not None:
            self.provider = provider
        elif client is not None:
            # Back-compat: a raw Anthropic client was passed.
            from .providers.anthropic import AnthropicProvider
            self.provider = AnthropicProvider(client=client)
        else:
            self.provider = make_provider(self.config.provider)
        self._owns_memory = memory is None
        self.memory = memory or ConversationMemory(max_result_chars=self.config.max_result_chars)
        self.permissions = permissions or PermissionManager()
        self.hooks = hooks or HookManager()
        self.middleware = middleware or MiddlewarePipeline()
        self.sandbox = sandbox
        self.mcp = mcp
        self.guardrails = GuardrailsEngine(
            max_iterations=self.config.max_iterations, max_cost_dollars=self.config.max_cost_dollars,
            input_cost_per_m=self.config.input_cost_per_m, output_cost_per_m=self.config.output_cost_per_m,
        )
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
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_resources())
        await asyncio.shield(self._close_task)

    async def _close_resources(self) -> None:
        errors = []
        try:
            await close_extensions(self.extensions, self)
        except Exception as exc:
            errors.append(exc)
        self.extensions = []
        if self._owns_provider:
            try:
                await self.provider.aclose()
            except Exception as exc:
                errors.append(exc)
        if self._owns_memory:
            try:
                await self.memory.aclose()
            except Exception as exc:
                errors.append(exc)
        materialized = getattr(self, "_materialized_artifacts", None)
        if materialized is not None:
            materialized.cleanup()
        if errors:
            raise ExceptionGroup("Agent cleanup failed", errors)

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def run(self, user_message: str) -> RunResult:
        from .engine import drive, new_state
        if self._busy or self._closed:
            raise RuntimeError("Agent is busy or closed")
        self._busy = True
        async def discard(event):
            pass
        try:
            return await drive(self, new_state(self, user_message), discard)
        finally:
            self._busy = False

    def run_stream(self, user_message: str) -> RunStream:
        from .engine import drive, new_state
        async def run(emit):
            if self._busy or self._closed:
                raise RuntimeError("Agent is busy or closed")
            self._busy = True
            try:
                state = new_state(self, user_message)
                state["stream"] = True
                return await drive(self, state, emit)
            finally:
                self._busy = False
        return RunStream(run)

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

    # ── Session persistence ──────────────────────────────────────────────

    async def save_session(self, storage_dir: str = ".agent_sessions") -> str:
        """Atomically save a conversation, configuration, extension state, and artifacts.

        Snapshots do not resume in-flight execution; use AgentRuntime for that.
        Artifacts remain until PersistentMemory.delete_session() is called.
        """
        if self._busy or self._closed:
            raise RuntimeError("Cannot snapshot a busy or closed agent")
        from .engine import snapshot

        storage = PersistentMemory(storage_dir)
        previous = getattr(self, "_artifact_store", None)
        self._artifact_store = storage.artifact_store(self.session_id)
        try:
            data = await snapshot(self)
            state = SessionState(
                session_id=self.session_id, messages=data["messages"],
                total_usage=self.guardrails.total_usage, metadata=data["metadata"],
                config=asdict(self.config), extensions=data["extensions"],
                lifetime_iterations=data["lifetime_iterations"],
            )
            await asyncio.to_thread(storage.save_session, state)
        finally:
            self._artifact_store = previous
        return self.session_id

    @classmethod
    async def load_session(
        cls, session_id: str, storage_dir: str = ".agent_sessions", **kwargs: Any,
    ) -> Agent:
        """Restore a validated snapshot. Supply provider/tool/extension bindings separately."""
        from .engine import restore

        storage = PersistentMemory(storage_dir)
        state = await asyncio.to_thread(storage.load_session, session_id)
        if state.config:
            saved_config = AgentConfig(**state.config)
            if "config" in kwargs and kwargs["config"] is not None and asdict(kwargs["config"]) != asdict(saved_config):
                raise ValueError("Explicit configuration differs from saved session")
            kwargs["config"] = saved_config
        agent = cls(**kwargs)
        try:
            missing = set(state.extensions) - {ext.name for ext in agent.extensions}
            if missing:
                raise ValueError(f"Missing extension bindings for restored snapshot: {sorted(missing)}")
            agent._artifact_store = storage.artifact_store(session_id)
            await restore(agent, {
                "session_id": state.session_id, "messages": state.messages,
                "metadata": state.metadata, "extensions": state.extensions,
                "total_usage": asdict(state.total_usage),
                "lifetime_iterations": state.lifetime_iterations,
            })
            return agent
        except BaseException:
            await agent.aclose()
            raise

    @property
    def session_id(self) -> str:
        return self._session_id
