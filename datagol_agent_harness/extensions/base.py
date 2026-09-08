"""Extension contract — base class and context facade for pluggable runtime behaviours."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class ExtensionContext:
    """Capability facade provided to extensions.

    Provides a clean, typed interface to register tools, subscribe to hooks,
    add middleware, inject dynamic prompt providers, and persist extension state.
    Transparently delegates unknown attributes to the underlying agent
    for 100% backward compatibility with legacy extensions.
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    @property
    def agent(self) -> Any:
        return self._agent

    @property
    def session_id(self) -> str:
        return getattr(self._agent, "_session_id", "") or getattr(self._agent, "session_id", "")

    @property
    def config(self) -> Any:
        return getattr(self._agent, "config", None)

    @property
    def tools(self) -> Any:
        return getattr(self._agent, "tools", None)

    @property
    def hooks(self) -> Any:
        return getattr(self._agent, "hooks", None)

    @property
    def middleware(self) -> Any:
        return getattr(self._agent, "middleware", None)

    @property
    def memory(self) -> Any:
        return getattr(self._agent, "memory", None)

    def register_tool(
        self,
        handler: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        permission: Any = None,
        concurrent: bool = True,
    ) -> None:
        from ..types import PermissionLevel
        perm = permission or PermissionLevel.ASK
        if input_schema is not None:
            self.tools.register_with_schema(
                name=name or handler.__name__,
                description=description or handler.__doc__ or "",
                input_schema=input_schema,
                handler=handler,
                permission=perm,
                concurrent=concurrent,
            )
        else:
            decorator = self.tools.register(
                name=name, description=description, permission=perm, concurrent=concurrent
            )
            decorator(handler)

    def add_middleware(self, middleware: Any) -> None:
        self.middleware.add(middleware)

    def on_hook(self, event: Any, callback: Callable[..., Any]) -> None:
        self.hooks.on(event, callback)

    def register_prompt_provider(self, provider: Callable[..., Any]) -> None:
        """Register a callback that returns text to dynamically append to the system prompt."""
        if hasattr(self._agent, "prompt_providers"):
            self._agent.prompt_providers.append(provider)

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

    async def on_save_session(self, ctx: ExtensionContext) -> dict[str, Any]:
        """Contribute state to be serialized into SessionState.metadata."""
        return {}

    async def on_load_session(self, ctx: ExtensionContext, state: dict[str, Any]) -> None:
        """Restore extension state from saved session metadata."""
        return None

    async def teardown(self) -> None:
        """Release any session-scoped resources held by this extension."""
        return None


def install_extensions(agent: Any, extensions: list[Extension] | None) -> list[Extension]:
    """Install each extension on the agent, in order. Detects name collisions."""
    if not extensions:
        return []
    ctx = ExtensionContext(agent)
    seen: set[str] = set()
    for ext in extensions:
        if not ext.name:
            raise ValueError(
                f"Extension {type(ext).__name__} must set a non-empty `name` class attribute"
            )
        if ext.name in seen:
            raise ValueError(f"Duplicate extension name: {ext.name}")
        seen.add(ext.name)
        ext.install(ctx)
    return list(extensions)


async def close_extensions(extensions: list[Extension]) -> None:
    """Fire teardown() on each extension in reverse install order."""
    for ext in reversed(extensions):
        try:
            await ext.teardown()
        except Exception as e:
            print(f"[extension {ext.name}] teardown failed: {e}")
