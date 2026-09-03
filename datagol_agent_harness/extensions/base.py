"""Extension contract — base class for pluggable runtime behaviours.

The Extension system mirrors the shape of Pi SDK's ExtensionAPI/ExtensionFactory:
each Extension is a small object that knows how to install itself onto an
agent's existing primitives (tools, hooks, middleware, system prompt) and
how to clean up its session-scoped state when the agent is closed.

Extensions are passed at agent construction time:

    agent = Agent(extensions=[MyExtension(), AnotherExtension()])
    # ... use agent ...
    await agent.aclose()  # fires each extension's teardown() in reverse order
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # Agent / StreamingAgent imported lazily to avoid cycles


class Extension(ABC):
    """A unit of pluggable behaviour that wires itself into an agent.

    Subclasses MUST set the `name` class attribute (used for diagnostics
    and to detect duplicate installs) and implement `install(agent)`.
    They MAY override `teardown()` to release session-scoped resources
    (temp files, background tasks, etc).

    Inside install(), use the agent's existing primitives:
        agent.tools.register_with_schema(...)        # add LLM-callable tools
        agent.hooks.on(HookEvent.X, callback)        # observe events
        agent.middleware.add(SomeMiddleware())       # transform tool results
        agent.config.system_prompt += "..."          # append to system prompt
    """

    name: str = ""

    @abstractmethod
    def install(self, agent: Any) -> None:
        """Wire this extension into the agent. Called once at construction."""

    async def teardown(self) -> None:
        """Release any session-scoped resources held by this extension.

        Default implementation is a no-op. Override when the extension owns
        files, background tasks, sockets, etc.
        """
        return None


def install_extensions(agent: Any, extensions: list[Extension] | None) -> list[Extension]:
    """Install each extension on the agent, in order. Detects name collisions.

    Returns the same list, suitable for stashing on the agent for later
    teardown via close_extensions().
    """
    if not extensions:
        return []
    seen: set[str] = set()
    for ext in extensions:
        if not ext.name:
            raise ValueError(
                f"Extension {type(ext).__name__} must set a non-empty `name` class attribute"
            )
        if ext.name in seen:
            raise ValueError(f"Duplicate extension name: {ext.name}")
        seen.add(ext.name)
        ext.install(agent)
    return list(extensions)


async def close_extensions(extensions: list[Extension]) -> None:
    """Fire teardown() on each extension in reverse install order.

    Errors are swallowed (logged via print) — teardown is best-effort and
    one extension's failure must not prevent others from cleaning up.
    """
    for ext in reversed(extensions):
        try:
            await ext.teardown()
        except Exception as e:
            print(f"[extension {ext.name}] teardown failed: {e}")
