"""Extension contract — base class and context facade for pluggable runtime behaviours."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable
import logging
import warnings

from ..hooks import HookCallback, HookEvent, HookManager, Middleware, MiddlewarePipeline, Registration
from ..types import AgentConfig, PermissionLevel, ReplayPolicy
from ..tools import ToolRegistry
from ..memory import ConversationMemory

if TYPE_CHECKING:
    from ..core import Agent
    from ..execution import RunResult

logger = logging.getLogger(__name__)


class ExtensionContext:
    """Capability facade provided to extensions.

    Provides a clean, typed interface to register tools, subscribe to hooks,
    add middleware, inject dynamic prompt providers, and persist extension state.
    Transparently delegates unknown attributes to the underlying agent
    for 100% backward compatibility with legacy extensions.
    """

    def __init__(self, agent: Agent, owner: str | None = None) -> None:
        self._agent = agent
        self._owner = owner

    def _track(self, handle: Registration) -> Registration:
        if self._owner is not None:
            self._agent._extension_registrations.setdefault(self._owner, []).append(handle)
        return handle

    @property
    def agent(self) -> Any:
        return self._agent

    @property
    def session_id(self) -> str:
        return getattr(self._agent, "_session_id", "") or getattr(self._agent, "session_id", "")

    @property
    def config(self) -> AgentConfig:
        return getattr(self._agent, "config", None)

    @property
    def tools(self) -> ToolRegistry:
        return getattr(self._agent, "tools", None)

    @property
    def hooks(self) -> HookManager:
        return getattr(self._agent, "hooks", None)

    @property
    def middleware(self) -> MiddlewarePipeline:
        return getattr(self._agent, "middleware", None)

    @property
    def memory(self) -> ConversationMemory:
        return getattr(self._agent, "memory", None)

    def register_tool(
        self,
        handler: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        permission: PermissionLevel | None = None,
        concurrent: bool = True,
        replay_policy: ReplayPolicy | str = "manual",
        timeout_seconds: float | None = None,
        replace: bool = False,
    ) -> Registration:
        tool_name = name or handler.__name__
        previous = self.tools.get_tool(tool_name) if self.tools.has_tool(tool_name) else None
        perm = permission
        if input_schema is not None:
            definition = self.tools.register_with_schema(
                name=name or handler.__name__,
                description=description or handler.__doc__ or "",
                input_schema=input_schema,
                handler=handler,
                permission=perm,
                concurrent=concurrent,
                replay_policy=replay_policy,
                timeout_seconds=timeout_seconds,
                replace=replace,
            )
        else:
            definition = self.tools.register_tool(
                handler, name=name, description=description, permission=perm, concurrent=concurrent,
                replay_policy=replay_policy, timeout_seconds=timeout_seconds, replace=replace,
            )
        def remove():
            if self.tools.has_tool(tool_name) and self.tools.get_tool(tool_name) is definition:
                self.tools.unregister(tool_name)
                if previous is not None:
                    self.tools._tools[tool_name] = previous
        return self._track(Registration(remove))

    def add_middleware(self, middleware: Middleware) -> Registration:
        return self._track(self.middleware.add(middleware))

    def on_hook(self, event: HookEvent, callback: HookCallback) -> Registration:
        return self._track(self.hooks.on(event, callback))

    def register_prompt_provider(self, provider: Callable[[], str]) -> Registration:
        """Register a callback that returns text to dynamically append to the system prompt."""
        self._agent.prompt_providers.append(provider)
        def remove():
            self._agent.prompt_providers[:] = [item for item in self._agent.prompt_providers if item is not provider]
        return self._track(Registration(remove))

    def get_state(self, key: str, default: Any = None) -> Any:
        meta = getattr(self._agent, "session_metadata", None)
        if meta is None:
            return default
        return meta.setdefault("extensions", {}).get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        meta = getattr(self._agent, "session_metadata", None)
        if meta is not None:
            meta.setdefault("extensions", {})[key] = value

    def __getattr__(self, name: str) -> Any:
        warnings.warn(f"ExtensionContext.{name} compatibility forwarding is deprecated; use ctx.agent explicitly", DeprecationWarning, stacklevel=2)
        return getattr(self._agent, name)


class Extension(ABC):
    """A unit of pluggable behaviour that wires itself into an agent."""

    name: str = ""
    priority: int = 100

    @abstractmethod
    def install(self, agent_or_ctx: Any) -> None:
        """Wire this extension into the agent or ExtensionContext."""

    async def on_turn_start(self, ctx: ExtensionContext, user_message: str) -> None:
        """Invoked before the loop starts for a user turn."""
        return None

    async def on_turn_end(self, ctx: ExtensionContext, final_text: str) -> None:
        """Invoked after the loop completes a user turn."""
        return None

    async def on_turn_complete(self, ctx: ExtensionContext, result: RunResult) -> None:
        """Observe completion, failure, or cancellation. Must tolerate durable retries.

        This terminal cleanup callback is best effort; start/end callbacks remain
        fail-fast. Pauses and approval waits are not terminal completions.
        """
        return None

    async def on_save_session(self, ctx: ExtensionContext) -> dict[str, Any]:
        """Contribute state to be serialized into SessionState.metadata."""
        return {}

    async def on_load_session(self, ctx: ExtensionContext, state: dict[str, Any]) -> None:
        """Restore extension state from saved session metadata."""
        return None

    async def teardown(self) -> None:
        """Release any session-scoped resources held by this extension."""
        return None


def validate_extensions(extensions: list[Extension] | None) -> list[Extension]:
    """Validate the entire set before mutation; lower priorities install first."""
    seen: set[str] = set()
    for ext in extensions or []:
        if not ext.name:
            raise ValueError(
                f"Extension {type(ext).__name__} must set a non-empty `name` class attribute"
            )
        if ext.name in seen:
            raise ValueError(f"Duplicate extension name: {ext.name}")
        if type(ext.priority) is not int:
            raise TypeError("Extension priority must be an integer")
        seen.add(ext.name)
    return sorted(extensions or [], key=lambda ext: ext.priority)


def install_extensions(agent: Any, extensions: list[Extension] | None) -> list[Extension]:
    """Install registrations transactionally. install() must not acquire async resources."""
    ordered = validate_extensions(extensions)
    agent._extension_registrations = {}
    before_tools = dict(agent.tools._tools)
    before_hooks = {key: list(value) for key, value in agent.hooks._hooks.items()}
    before_middleware = list(agent.middleware._middleware)
    before_prompts = list(agent.prompt_providers)
    try:
        for ext in ordered:
            ext.install(ExtensionContext(agent, ext.name))
    except BaseException:
        for handles in agent._extension_registrations.values():
            for handle in reversed(handles):
                handle.close()
        agent.tools._tools = before_tools
        agent.hooks._hooks = before_hooks
        agent.middleware._middleware = before_middleware
        agent.prompt_providers[:] = before_prompts
        raise
    return ordered


async def complete_extensions(agent: Any, result: RunResult) -> None:
    for ext in reversed(agent.extensions):
        try:
            await ext.on_turn_complete(ExtensionContext(agent, ext.name), result)
        except Exception:
            logger.exception("Extension %s terminal callback failed", ext.name)


async def close_extensions(extensions: list[Extension], agent: Any = None) -> None:
    """Fire teardown() on each extension in reverse install order."""
    errors = []
    for ext in reversed(extensions):
        try:
            await ext.teardown()
        except Exception as e:
            errors.append(e)
        finally:
            if agent is not None:
                for handle in reversed(agent._extension_registrations.pop(ext.name, [])):
                    handle.close()
    if errors:
        raise ExceptionGroup("Extension teardown failed", errors)
