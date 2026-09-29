"""Failover across providers, packaged as one provider.

``FallbackProvider`` walks an ordered chain of members. A member that fails
transiently collects a strike; once its strikes reach ``switch_after`` the next
member is tried within the same call, with its own model name and reply budget
(the same model has different ids on different vendors). A deterministic
failure raises at once: a bad request stays bad on every vendor. Streams fail
over only before the first chunk has been delivered; after that the engine's
``ATTEMPT_RESET`` path is the right recovery.

The engine's ``RetryPolicy`` loop still wraps every call: when the whole chain
fails, the last error propagates and the engine decides whether to walk the
chain again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from ..types import PromptCacheHint, ProviderResponse, StreamChunk
from .base import LLMProvider, make_provider
from .retry import is_transient, retry_after_seconds


@dataclass(frozen=True)
class Fallback:
    """One member of a ``FallbackProvider`` chain.

    ``provider`` is a built-in or registered name (built and owned by the chain)
    or an instance (caller-owned). ``model`` and ``max_tokens`` override the
    request's values for this member; ``None`` reuses them.
    """

    provider: LLMProvider | str
    model: str | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.provider, str):
            if not self.provider.strip():
                raise ValueError("Fallback provider name must be nonempty")
        elif not hasattr(self.provider, "create"):
            raise TypeError("Fallback provider must be an LLMProvider or a provider name")
        if self.model is not None and (not isinstance(self.model, str) or not self.model):
            raise ValueError("Fallback model must be a nonempty string")
        if self.max_tokens is not None and (type(self.max_tokens) is not int or self.max_tokens < 1):
            raise ValueError("Fallback max_tokens must be a positive integer")


class _Member:
    __slots__ = ("spec", "provider", "label", "owned", "strikes", "skip_until")

    def __init__(self, spec: Fallback) -> None:
        self.spec = spec
        if isinstance(spec.provider, str):
            self.provider = make_provider(spec.provider)
            self.label = spec.provider.lower()
            self.owned = True
        else:
            self.provider = spec.provider
            self.label = getattr(spec.provider, "name", "") or type(spec.provider).__name__
            self.owned = False
        self.strikes = 0
        self.skip_until = 0.0

    def request(self, request: dict[str, Any]) -> dict[str, Any]:
        mapped = dict(request)
        if self.spec.model is not None:
            mapped["model"] = self.spec.model
        if self.spec.max_tokens is not None:
            mapped["max_tokens"] = self.spec.max_tokens
        return mapped

    def transient(self, exc: BaseException) -> bool:
        classify = getattr(self.provider, "is_transient", None)
        if callable(classify):
            try:
                return bool(classify(exc))
            except Exception:
                pass
        return is_transient(exc)


class FallbackProvider(LLMProvider):
    """An ordered chain of providers that fails over on transient errors.

    ``switch_after`` is how many transient failures a member collects before
    the chain moves past it; ``cooldown_seconds`` keeps a member that was
    switched away from out of rotation for that long. Strikes reset when the
    member succeeds. ``last_served`` names the member that answered the most
    recent call; the engine reports it in the ``LLM_RESPONSE`` hook.
    """

    name = "fallback"

    def __init__(
        self,
        primary: LLMProvider | str | Fallback,
        *fallbacks: LLMProvider | str | Fallback,
        switch_after: int = 1,
        cooldown_seconds: float = 0.0,
    ) -> None:
        if type(switch_after) is not int or switch_after < 1:
            raise ValueError("switch_after must be a positive integer")
        if isinstance(cooldown_seconds, bool) or cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be nonnegative")
        specs = [m if isinstance(m, Fallback) else Fallback(m) for m in (primary, *fallbacks)]
        self._members = [_Member(spec) for spec in specs]
        self.switch_after = switch_after
        self.cooldown_seconds = float(cooldown_seconds)
        self.last_served: str | None = None
        self._failing: _Member | None = None

    # ── introspection ──────────────────────────────────────────────────────
    @property
    def members(self) -> tuple[Fallback, ...]:
        return tuple(m.spec for m in self._members)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(m.label for m in self._members)

    def _eligible(self) -> list[_Member]:
        now = time.monotonic()
        ready = [m for m in self._members if m.skip_until <= now]
        return ready or list(self._members)  # every member cooling down: try them all

    def _struck(self, member: _Member, exc: BaseException, last: bool) -> bool:
        """Record a transient failure; True when the chain should move on."""
        self._failing = member
        member.strikes += 1
        if member.strikes < self.switch_after or last:
            return False
        if self.cooldown_seconds:
            member.skip_until = time.monotonic() + self.cooldown_seconds
        return True

    def _served(self, member: _Member) -> None:
        member.strikes = 0
        self.last_served = member.label
        self._failing = None

    # ── LLMProvider ────────────────────────────────────────────────────────
    def default_max_tokens(self, model: str) -> int:
        return self._members[0].provider.default_max_tokens(model)

    def is_transient(self, exc: BaseException) -> bool:
        member = self._failing or self._members[0]
        return member.transient(exc)

    def retry_after(self, exc: BaseException) -> float | None:
        member = self._failing or self._members[0]
        reader = getattr(member.provider, "retry_after", None)
        if callable(reader):
            try:
                return reader(exc)
            except Exception:
                pass
        return retry_after_seconds(exc)

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
        request = dict(
            model=model, messages=messages, system=system, tools=tools,
            max_tokens=max_tokens, temperature=temperature, cache=cache,
        )
        chain = self._eligible()
        for index, member in enumerate(chain):
            try:
                response = await member.provider.create(**member.request(request))
            except Exception as exc:
                if not member.transient(exc):
                    self._failing = member
                    raise
                if not self._struck(member, exc, last=index == len(chain) - 1):
                    raise
                continue
            self._served(member)
            return response
        raise RuntimeError("FallbackProvider has no members")  # pragma: no cover - constructor requires one

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
        request = dict(
            model=model, messages=messages, system=system, tools=tools,
            max_tokens=max_tokens, temperature=temperature, cache=cache,
        )
        chain = self._eligible()
        for index, member in enumerate(chain):
            iterator = member.provider.stream(**member.request(request))
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                self._served(member)
                return
            except Exception as exc:
                await _aclose(iterator)
                if not member.transient(exc):
                    self._failing = member
                    raise
                if not self._struck(member, exc, last=index == len(chain) - 1):
                    raise
                continue
            # Output has reached the caller: from here a failure is the engine's to handle.
            self.last_served = member.label
            self._failing = member
            yield first
            async for chunk in iterator:
                yield chunk
            self._served(member)
            return

    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
    ) -> int:
        member = self._eligible()[0]
        mapped = member.request({"model": model})
        return await member.provider.count_tokens(
            model=mapped["model"], messages=messages, system=system, tools=tools,
        )

    async def aclose(self) -> None:
        """Close the members this chain built from names; injected instances stay open."""
        for member in self._members:
            if member.owned:
                await member.provider.aclose()


async def _aclose(iterator: Any) -> None:
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception:  # pragma: no cover - best effort
        pass
