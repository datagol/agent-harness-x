"""Provider interface + factory.

The internal "canonical" response shape mirrors Anthropic's: an object with
.content (list of blocks with .type/.text or .type/.id/.name/.input),
.stop_reason ('end_turn' | 'tool_use' | 'max_tokens'), and .usage.

Providers for other vendors translate their native shapes into this form.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol


class ProviderResponse(Protocol):
    """Duck-typed canonical response. Anthropic's response already satisfies
    this; OpenAI's is wrapped to match (see providers/openai.py)."""

    content: list[Any]
    stop_reason: str
    usage: Any


class LLMProvider(ABC):
    """Backend that knows how to talk to one LLM vendor."""

    @abstractmethod
    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> ProviderResponse:
        """Single non-streaming completion. Returns canonical-shape response."""

    @abstractmethod
    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
    ) -> int:
        """Estimate prompt token count for trim-decision logic in memory.py."""


def make_provider(name: str, **kwargs: Any) -> LLMProvider:
    """Build a provider by short name. Use this from AgentConfig wiring."""
    name = name.lower()
    if name == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(**kwargs)
    if name == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider(**kwargs)
    raise ValueError(f"Unknown provider: {name!r} (expected 'anthropic' or 'openai')")
