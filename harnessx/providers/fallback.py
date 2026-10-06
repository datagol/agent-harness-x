"""Failover across providers, packaged as one provider.

Most callers never name this class. ``AgentConfig(fallbacks=[Fallback(...)])``
is the usual way in, and the Agent builds and owns the chain. Construct one
directly only when a member has to be a live object, such as a provider holding
a pre-configured SDK client, which a config value cannot name.

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

import copy
import time
from typing import Any, AsyncIterator, Sequence

from ..types import Fallback, PromptCacheHint, ProviderResponse, StreamChunk
from ..models import lookup_model_limits, _apply_learned, resolve_model_limits
from .base import LLMProvider, make_provider, native_continuation
from .retry import is_transient, retry_after_seconds

# What a chain accepts as a member: a config record, a provider name, or a live
# provider. Only the first two can appear in AgentConfig, which must serialize.
MemberSpec = Fallback | LLMProvider | str


class _Member:
    __slots__ = ("provider", "label", "owned", "model", "max_tokens", "strikes", "skip_until", "reply_budget")

    def __init__(self, spec: MemberSpec) -> None:
        self.model: str | None = None
        self.max_tokens: int | None = None
        if isinstance(spec, Fallback):
            self.model, self.max_tokens = spec.model, spec.max_tokens
            spec = spec.provider
        if isinstance(spec, str):
            if not spec.strip():
                raise ValueError("Provider name must be nonempty")
            self.provider = make_provider(spec)
            self.label = spec.lower()
            self.owned = True
        elif hasattr(spec, "create"):
            self.provider = spec
            self.label = getattr(spec, "name", "") or type(spec).__name__
            self.owned = False
        else:
            raise TypeError("A chain member must be a Fallback, a provider name, or an LLMProvider")
        self.strikes = 0
        self.reply_budget: int | None = None
        self.skip_until = 0.0

    def request(self, request: dict[str, Any], *, limits=None) -> dict[str, Any]:
        mapped = dict(request)
        if self.model is not None:
            mapped["model"] = self.model
        if self.max_tokens is not None:
            mapped["max_tokens"] = self.max_tokens
        if self.reply_budget is not None and "max_tokens" in request:
            mapped["max_tokens"] = self.reply_budget
        if mapped.get("max_tokens"):
            # The budget was resolved for the primary's model. A member serving a
            # different model with a smaller reply limit would refuse it outright.
            from ..models import UNKNOWN_MODEL
            model = mapped.get("model") or self.model or ""
            known = limits or _apply_learned(model, lookup_model_limits(model) or UNKNOWN_MODEL, self.provider)
            if known is not None and mapped.get("max_tokens"):
                mapped["max_tokens"] = min(int(mapped["max_tokens"]), known.max_output)
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
    _resolves_member_limits = True

    def __init__(
        self,
        primary: MemberSpec,
        *,
        fallbacks: Sequence[MemberSpec] = (),
        switch_after: int = 1,
        cooldown_seconds: float = 0.0,
    ) -> None:
        if type(switch_after) is not int or switch_after < 1:
            raise ValueError("switch_after must be a positive integer")
        if isinstance(cooldown_seconds, bool) or cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be nonnegative")
        self._members = [_Member(spec) for spec in (primary, *fallbacks)]
        self.switch_after = switch_after
        self.cooldown_seconds = float(cooldown_seconds)
        self.last_served: str | None = None
        self._failing: _Member | None = None
        self._active: _Member | None = None
        self._pinned: _Member | None = None
        self._last_request: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: Any) -> FallbackProvider:
        """Build the chain an ``AgentConfig`` describes: its provider, then its fallbacks."""
        return cls(
            config.provider,
            fallbacks=config.fallbacks,
            switch_after=config.retry.switch_after,
            cooldown_seconds=config.retry.cooldown_seconds,
        )

    # ── introspection ──────────────────────────────────────────────────────
    @property
    def labels(self) -> tuple[str, ...]:
        """Member names in order, the primary first."""
        return tuple(m.label for m in self._members)

    def _eligible(self) -> list[_Member]:
        if self._pinned is not None:
            return [self._pinned]
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
        member = self._eligible()[0]
        return member.provider.default_max_tokens(member.model or model)

    async def model_limits(self, model: str):
        """Limits for the member handling this request or recovery."""
        member = self._pinned or self._failing or self._eligible()[0]
        return await resolve_model_limits(member.provider, member.model or model)

    def request_target(self, model):
        member = self._active or self._eligible()[0]
        return member.provider, self._last_request.get("model", member.model or model)

    def set_reply_budget(self, budget):
        (self._active or self._eligible()[0]).reply_budget = budget

    def call_budget(self, budget):
        return max([budget or 0, *(m.reply_budget or m.max_tokens or budget or 0 for m in self._eligible())])

    def export_state(self):
        return {"members": [{"limits": getattr(m.provider, "_harness_model_limits", {}),
                             "reply_budget": m.reply_budget} for m in self._members],
                "pinned": self._members.index(self._pinned) if self._pinned is not None else None,
                "failing": self._members.index(self._failing) if self._failing is not None else None}

    def restore_state(self, state):
        for member, saved in zip(self._members, state.get("members", [])):
            member.provider._harness_model_limits = copy.deepcopy(saved.get("limits", {}))
            member.reply_budget = saved.get("reply_budget")
        index = state.get("pinned")
        self._pinned = self._members[index] if isinstance(index, int) and 0 <= index < len(self._members) else None
        index = state.get("failing")
        self._failing = self._members[index] if isinstance(index, int) and 0 <= index < len(self._members) else None

    async def _request(self, member, request):
        model = member.model or request["model"]
        limits = await resolve_model_limits(member.provider, model)
        mapped = member.request(request, limits=limits)
        mapped["max_tokens"] = min(mapped["max_tokens"], limits.max_output)
        self._active, self._last_request = member, mapped
        return mapped

    def _continuation(self, member, response):
        self._pinned = member if native_continuation(response) else None

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
                response = await member.provider.create(**await self._request(member, request))
            except Exception as exc:
                if not member.transient(exc):
                    self._failing = member
                    raise
                if not self._struck(member, exc, last=index == len(chain) - 1):
                    raise
                continue
            self._served(member)
            self._continuation(member, response)
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
            iterator = member.provider.stream(**await self._request(member, request))
            delivered = False
            try:
                async for chunk in iterator:
                    if chunk.kind == "response":
                        self._continuation(member, chunk.data)
                    # "progress" chunks carry nothing the caller shows, so they
                    # pass straight through -- keeping a stall watchdog fed --
                    # without closing the window in which another member can
                    # still take over.
                    if not delivered and chunk.kind != "progress":
                        # Output has reached the caller: from here a failure is
                        # the engine's to handle.
                        delivered = True
                        self.last_served = member.label
                        self._failing = member
                    yield chunk
            except Exception as exc:
                if delivered:
                    raise
                await _aclose(iterator)
                if not member.transient(exc):
                    self._failing = member
                    raise
                if not self._struck(member, exc, last=index == len(chain) - 1):
                    raise
                continue
            finally:
                await _aclose(iterator)
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
        member = self._pinned or self._failing or self._eligible()[0]
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
