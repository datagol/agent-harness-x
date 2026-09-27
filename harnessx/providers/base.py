"""Provider interface + factory.

Defines the vendor-agnostic LLMProvider abstraction and factory.
Providers translate between provider-specific wire protocols and the
harness's canonical ProviderResponse and RunEvent models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from ..types import ProviderResponse, StopReason, StreamChunk, ToolCall, TokenUsage
from ..types import PromptCacheHint


class LLMProvider(ABC):
    """Backend that knows how to talk to one LLM vendor."""

    name: str = ""

    @abstractmethod
    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        cache: PromptCacheHint | None = None,
    ) -> ProviderResponse:
        """Single non-streaming completion. Returns canonical-shape response.

        ``cache`` is the engine's prompt-cache hint. A provider may ignore it;
        one that honors it must fail open when the vendor rejects the request.
        """

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        cache: PromptCacheHint | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from the provider.
        
        Yields StreamChunk(kind="text_delta", data="..."), StreamChunk(kind="thinking_delta", ...),
        and concludes with StreamChunk(kind="response", data=ProviderResponse(...)).
        Default implementation falls back to create().
        """
        resp = await self.create(
            model=model,
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            cache=cache,
        )
        if resp.text:
            yield StreamChunk(kind="text_delta", data=resp.text)
        yield StreamChunk(kind="response", data=resp)

    async def aclose(self) -> None:
        """Close the provider's owned client. Injected providers are caller-owned."""
        import inspect
        client = getattr(self, "client", None) or getattr(self, "_client", None)
        closer = getattr(client, "aclose", None) or getattr(client, "close", None)
        if closer:
            result = closer()
            if inspect.isawaitable(result):
                await result

    def format_tools(self, tools: list[dict[str, Any]]) -> Any:
        """Convert tools into vendor-specific format. Default passes through."""
        return tools

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
    if name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(**kwargs)
    if name == "openrouter":
        from .openrouter import OpenRouterProvider
        return OpenRouterProvider(**kwargs)
    if name in ("azure", "azure-openai"):
        from .azure_openai import AzureOpenAIProvider
        return AzureOpenAIProvider(**kwargs)
    raise ValueError(
        f"Unknown provider: {name!r} "
        "(expected 'anthropic', 'openai', 'gemini', 'openrouter', or 'azure')"
    )
