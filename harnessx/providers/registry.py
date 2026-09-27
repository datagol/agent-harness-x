"""Provider names: the built-ins plus factories registered by applications.

``AgentConfig.provider`` may name any of these. Registration is process-local
module state, so a process that resumes a persisted session must register the
same names first, exactly as it must bind the same agent factories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .base import LLMProvider

BUILTIN_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "gemini", "openrouter", "azure")
_registry: dict[str, Callable[..., "LLMProvider"]] = {}


def register_provider(name: str, factory: Callable[..., "LLMProvider"], *, replace: bool = False) -> None:
    """Make ``AgentConfig(provider=name)`` build providers with ``factory(**kwargs)``."""
    key = name.lower()
    if not key or any(c.isspace() for c in key):
        raise ValueError("Provider name must be nonempty and contain no whitespace")
    if key in BUILTIN_PROVIDERS:
        raise ValueError(f"{name!r} is a built-in provider name")
    if key in _registry and not replace:
        raise ValueError(f"Provider {name!r} is already registered; pass replace=True to replace it")
    _registry[key] = factory


def unregister_provider(name: str) -> bool:
    return _registry.pop(name.lower(), None) is not None


def registered_providers() -> tuple[str, ...]:
    return tuple(sorted(_registry))


def provider_factory(name: str) -> Callable[..., "LLMProvider"] | None:
    return _registry.get(name.lower())


def is_known_provider(name: str) -> bool:
    key = name.lower()
    return key in BUILTIN_PROVIDERS or key in _registry
