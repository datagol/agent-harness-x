"""The core agentic loop: LLM reasoning → tool execution → repeat.

This is the heart of the harness. The Agent class orchestrates all other modules:
tool registry, memory, permissions, hooks, middleware, and optionally sandbox.
"""

from __future__ import annotations

import uuid
import asyncio
from dataclasses import asdict
from typing import Any, AsyncIterator, Callable

from .errors import ConfigurationError, RuntimeStateError
from .execution import RunResult, RunStream
from .hooks import HookManager, MiddlewarePipeline
from .memory import ConversationMemory, PersistentMemory
from .permissions import GuardrailsEngine, PermissionManager
from .types import PermissionLevel
from .extensions.base import Extension, close_extensions, install_extensions, validate_extensions
from .mcp import MCPManager
from .providers import LLMProvider, make_provider
from .providers.registry import BUILTIN_PROVIDERS
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

    def __new__(cls, *args: Any, **kwargs: Any) -> "Agent":
        # Removed in 0.4; a bare unexpected-keyword error would not say what to do instead.
        if "client" in kwargs:
            raise TypeError(
                "Agent(client=...) was removed in harnessx 0.4. Wrap the SDK client: "
                "Agent(provider=AnthropicProvider(client=client))."
            )
        return super().__new__(cls)

    def __init__(
        self,
        config: AgentConfig | None = None,
        provider: LLMProvider | None = None,
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
        _check_provider_binding(config, provider)
        self.config = config or AgentConfig()
        self.tools = normalize_tool_registry(tools, policy=self.config.tools, sandbox=sandbox)
        # Caller-owned resources with a job: the sandbox is the bash built-in's
        # engine; the MCP manager's discovered tools are bridged into the
        # registry now (register_tools is idempotent for one manager).
        self.sandbox = sandbox
        self.mcp = mcp
        self.mcp_tools: tuple[str, ...] = tuple(mcp.register_tools(self.tools)) if mcp is not None else ()
        self.subagents = prepare_subagents(subagents, self.tools)
        for subagent in self.subagents:
            install_subagent(self, subagent)
        self._owns_provider = provider is None
        self._busy = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self.provider = provider if provider is not None else make_provider(self.config.provider)
        self._owns_memory = memory is None
        self.memory = memory or ConversationMemory(max_result_chars=self.config.limits.max_result_chars)
        self._register_result_reader()
        self.permissions = permissions or PermissionManager()
        self.hooks = hooks or HookManager()
        self.middleware = middleware or MiddlewarePipeline()
        if sandbox is not None and getattr(sandbox, "hooks", None) is None:
            sandbox.hooks = self.hooks  # SANDBOX_EXEC events reach this agent's observers
        limits = self.config.limits
        self.guardrails = GuardrailsEngine(
            max_iterations=limits.max_iterations, max_cost_dollars=limits.max_cost_dollars,
            input_cost_per_m=limits.input_cost_per_m, output_cost_per_m=limits.output_cost_per_m,
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

    def _register_result_reader(self) -> None:
        """Give the model a way back into tool results that were too large for the context.

        Each Agent binds the tool to its own memory (a copied registry may carry
        a parent's); a tool of the same name registered by the application wins.
        """
        existing = self.tools.get_tool("read_tool_result") if self.tools.has_tool("read_tool_result") else None
        if existing is not None and not getattr(existing.handler, "__harnessx_builtin__", False):
            return
        memory = self.memory

        async def read_tool_result(result_id: str, offset: int = 0, limit: int = 200) -> str:
            """Read part of a tool result that was too large for the conversation.

            Args:
                result_id: The tool-use id named in the "[Tool result too large ...]" notice.
                offset: First line to return, counting from 0.
                limit: Number of lines to return, at most 2000.
            """
            return await asyncio.to_thread(memory.read_evicted, result_id, offset, limit)

        read_tool_result.__harnessx_builtin__ = True  # type: ignore[attr-defined]
        self.tools.register_tool(
            read_tool_result, permission=PermissionLevel.ALLOW, replay_policy="safe", replace=existing is not None,
        )

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

    @property
    def closed(self) -> bool:
        """True once aclose() has started; the agent accepts no more runs."""
        return self._closed

    @property
    def busy(self) -> bool:
        """True while a run is in progress; one agent executes one run at a time."""
        return self._busy

    def _check_available(self) -> None:
        if self._closed:
            raise RuntimeStateError("Agent is closed")
        if self._busy:
            raise RuntimeStateError("Agent is busy with another run")

    async def run(self, user_message: str) -> RunResult:
        """Run one turn to completion and return its result. Failures are in result.error."""
        from .engine import drive, new_state
        self._check_available()
        self._busy = True
        async def discard(event):
            pass
        try:
            return await drive(self, new_state(self, user_message), discard)
        finally:
            self._busy = False

    def run_stream(self, user_message: str) -> RunStream:
        """Run one turn as a stream of typed events; the final RunResult is the last event."""
        from .engine import drive, new_state
        async def run(emit):
            self._check_available()
            self._busy = True
            try:
                state = new_state(self, user_message)
                state["stream"] = True
                return await drive(self, state, emit)
            finally:
                self._busy = False
        return RunStream(run)

    async def stream_text(
        self, user_message: str, *, on_reset: Callable[[], Any] | None = None,
    ) -> AsyncIterator[str]:
        """Yield the answer's text as it arrives; raise RunFailed if the run does not complete.

        ``on_reset`` is called when a retried model call restarts the answer, so
        a display can discard the provisional text it has shown.
        """
        async with self.run_stream(user_message) as stream:
            async for text in stream.text(on_reset=on_reset):
                yield text

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
            raise RuntimeStateError("Cannot snapshot a busy or closed agent")
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
            saved_config = AgentConfig.from_dict(state.config)
            if "config" in kwargs and kwargs["config"] is not None and asdict(kwargs["config"]) != asdict(saved_config):
                raise ConfigurationError("Explicit configuration differs from saved session")
            kwargs["config"] = saved_config
        agent = cls(**kwargs)
        try:
            missing = set(state.extensions) - {ext.name for ext in agent.extensions}
            if missing:
                raise ConfigurationError(f"Missing extension bindings for restored snapshot: {sorted(missing)}")
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


def _check_provider_binding(config: AgentConfig | None, provider: LLMProvider | None) -> None:
    """Reject only a genuine contradiction: two different built-in names.

    Custom providers (empty or registered names) are always accepted; the
    injected object is what runs, and the config name is a label.
    """
    if provider is None:
        return
    injected = getattr(provider, "name", "") or ""
    configured = config.provider if config is not None else "anthropic"
    if injected in BUILTIN_PROVIDERS and configured in BUILTIN_PROVIDERS and injected != configured:
        if config is None:
            raise ConfigurationError(
                "An injected non-Anthropic provider requires AgentConfig with its provider and model"
            )
        raise ConfigurationError(
            f"Injected provider {injected!r} does not match AgentConfig.provider {configured!r}"
        )
