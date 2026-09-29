"""Provider interface + factory.

Defines the vendor-agnostic LLMProvider abstraction and factory.
Providers translate between provider-specific wire protocols and the
harness's canonical ProviderResponse and RunEvent models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Self

from ..types import ProviderResponse, StreamChunk
from ..types import PromptCacheHint
from .registry import BUILTIN_PROVIDERS, provider_factory, registered_providers
from .retry import is_transient, retry_after_seconds


DEFAULT_MAX_TOKENS = 8192  # conservative: accepted by every model the built-in providers know


class LLMProvider(ABC):
    """Backend that knows how to talk to one LLM vendor."""

    name: str = ""

    def default_max_tokens(self, model: str) -> int:
        """The reply budget used when AgentConfig.max_tokens is None. Override per vendor."""
        return DEFAULT_MAX_TOKENS

    def is_transient(self, exc: BaseException) -> bool:
        """Whether one more attempt at the same request could succeed.

        The engine's retry loop asks the provider that raised. The default
        treats 429, 5xx, timeouts, and connection loss as transient and every
        other 4xx as final; override to classify vendor-specific errors.
        """
        return is_transient(exc)

    def retry_after(self, exc: BaseException) -> float | None:
        """Seconds the server asked us to wait before retrying, when it said.

        Read from ``Retry-After`` on the failure. The engine waits at least this
        long, capped by ``RetryPolicy.max_backoff_seconds``.
        """
        return retry_after_seconds(exc)

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

    async def __aenter__(self) -> Self:
        """Close this provider's client on the way out.

        An Agent closes only a provider it built itself, so one you construct and
        inject is yours to close. ``async with`` is how::

            async with FallbackProvider("anthropic", Fallback("openai")) as provider:
                async with Agent(provider=provider) as agent:
                    ...

        Naming the provider in ``AgentConfig(provider=...)`` instead leaves it to
        the agent, which then owns and closes it.
        """
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the provider's owned client. Injected providers are caller-owned."""
        import inspect
        client = getattr(self, "client", None) or getattr(self, "_client", None)
        closer = getattr(client, "aclose", None) or getattr(client, "close", None)
        if closer:
            result = closer()
            if inspect.isawaitable(result):
                await result

    @abstractmethod
    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
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
    if name == "azure":
        from .azure_openai import AzureOpenAIProvider
        return AzureOpenAIProvider(**kwargs)
    factory = provider_factory(name)
    if factory is not None:
        return factory(**kwargs)
    registered = ", ".join(registered_providers()) or "none"
    raise ValueError(
        f"Unknown provider: {name!r} (built-in: {', '.join(BUILTIN_PROVIDERS)}; registered: {registered})"
    )
