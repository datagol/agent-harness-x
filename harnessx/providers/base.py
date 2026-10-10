"""Provider interface + factory.

Defines the vendor-agnostic LLMProvider abstraction and factory.
Providers translate between provider-specific wire protocols and the
harness's canonical ProviderResponse and RunEvent models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import inspect
from typing import Any, AsyncIterator, Self

from ..models import DEFAULT_OUTPUT_CAP, ModelLimits, default_reply_budget, lookup_model_limits
from ..types import ProviderResponse, StreamChunk, ToolChoice
from ..types import PromptCacheHint
from .registry import BUILTIN_PROVIDERS, provider_factory, registered_providers
from .retry import is_transient, retry_after_seconds, sdk_connection_failure


# The reply budget for a model no table knows. Kept under its old name for
# callers that import it; the value now comes from ``harnessx.models``.
DEFAULT_MAX_TOKENS = DEFAULT_OUTPUT_CAP


# Set on a tool call's input when the model's arguments could not be read. The
# engine answers such a call with the reason instead of running it, so the model
# sees what was wrong rather than a schema error about a key it never wrote.
INVALID_ARGUMENTS = "__invalid_arguments__"


# One name across every provider; each one spells it differently on the wire.
# See ToolChoice in harnessx.types for what the values mean.
TOOL_CHOICES = tuple(choice.value for choice in ToolChoice)


def normalize_tool_choice(
    value: "str | ToolChoice | None", *, env_var: str | None = None
) -> "ToolChoice | None":
    """Validate a tool_choice, falling back to the environment, else None.

    Returns a ToolChoice, which is a str subclass, so a provider can compare it
    or send it as a plain string without converting.
    """
    import os

    raw = value if value is not None else (os.environ.get(env_var) if env_var else None)
    if raw is None or raw == "":
        return None
    if isinstance(raw, ToolChoice):
        return raw
    # str() on a str-Enum gives "ToolChoice.ANY" on 3.11+, not "any", so the
    # member is returned above rather than round-tripped through str().
    try:
        return ToolChoice(str(raw).strip().lower())
    except ValueError:
        raise ValueError(
            f"tool_choice must be one of {TOOL_CHOICES}, not {raw!r}"
        ) from None


def native_continuation(response) -> bool:
    if response.stop_reason == "pause_turn":
        return True
    native = [b.get("data", {}) for b in response.content if isinstance(b, dict) and b.get("type") == "provider"]
    calls = {b.get("id") for b in native if b.get("type") == "server_tool_use"}
    answered = {b.get("tool_use_id") for b in native}
    return bool(calls - answered)


@asynccontextmanager
async def closing_stream(stream):
    """Close both SDK streams and async generators on completion or cancellation."""
    try:
        yield stream
    finally:
        close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result


def parse_tool_arguments(raw: str | None) -> dict[str, Any]:
    """Tool-call arguments as the model wrote them, or a marker saying why not.

    Raw control characters inside strings (a literal newline in a file body)
    are accepted, as most models emit them. Anything that is not a JSON object
    -- a truncated or malformed document, an array, a bare string -- becomes
    ``{INVALID_ARGUMENTS: reason}``: a dict, so the transcript stays valid, and
    the run goes on instead of failing on the turn that carries it.
    """
    import json

    text = raw or ""
    if not text.strip():
        return {}
    try:
        value = json.loads(text, strict=False)
    except json.JSONDecodeError as exc:
        return {INVALID_ARGUMENTS: f"the arguments are not valid JSON ({exc.msg} at character {exc.pos})"}
    if not isinstance(value, dict):
        return {INVALID_ARGUMENTS: f"the arguments are a JSON {type(value).__name__}, not an object"}
    return value


def streams(provider: Any) -> bool:
    """Whether ``provider`` has a streaming call of its own.

    The base ``stream()`` only awaits ``create()`` and yields the whole reply
    at the end, so the stream deadlines cannot apply to it.
    """
    declared = getattr(provider, "streams_natively", None)  # a chain answers for its members
    if isinstance(declared, bool):
        return declared
    stream = getattr(type(provider), "stream", None)
    return stream is not None and stream is not LLMProvider.stream


class LLMProvider(ABC):
    """Backend that knows how to talk to one LLM vendor."""

    name: str = ""
    _harness_model_limits: dict[str, dict[str, int]]
    # True when ``stream()`` yields something as soon as the model starts,
    # even if it then thinks without showing it (Anthropic sends message_start
    # first). Only then can a quiet start be told from a stuck one, and the
    # engine applies RetryPolicy.stream_first_event_timeout_seconds. A vendor
    # whose reasoning models say nothing until they answer leaves it False.
    first_event_promptly: bool = False

    def default_max_tokens(self, model: str) -> int:
        """The reply budget used when AgentConfig.max_tokens is None.

        The model's own output limit from the built-in table, capped at
        ``DEFAULT_OUTPUT_CAP``. The engine resolves limits through
        ``model_limits`` instead, which can also ask the vendor; this stays for
        callers that want a synchronous answer.
        """
        known = lookup_model_limits(model)
        return default_reply_budget(known) if known is not None else DEFAULT_MAX_TOKENS

    def model_limits(self, model: str) -> ModelLimits | None | Any:
        """The model's context window and output limit, when the vendor can say.

        May be a coroutine function. None means "no better answer than the
        built-in table", which is what the base class gives. Override to ask
        the vendor; the engine caches nothing itself, so cache in the provider.
        """
        return None

    def is_transient(self, exc: BaseException) -> bool:
        """Whether one more attempt at the same request could succeed.

        The engine's retry loop asks the provider that raised. The default
        treats 429, 5xx, timeouts, and connection loss as transient and every
        other 4xx as final; override to classify vendor-specific errors.
        """
        return is_transient(exc) or sdk_connection_failure(exc)

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

        An Agent closes only a provider it built itself, so one you construct
        and inject is yours to close. ``async with`` is how::

            async with FallbackProvider("anthropic", fallbacks=[Fallback("openai")]) as provider:
                async with Agent(provider=provider) as agent:
                    ...

        Naming the provider in ``AgentConfig`` instead leaves it to the agent,
        which then owns and closes it.
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
