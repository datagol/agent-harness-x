"""Prompt-cache hints: the harness decides what is stable, providers decide how to cache it.

Every vendor caches differently (Anthropic breakpoints, OpenAI automatic prefix
caching with a routing key, Gemini explicit cache resources), so the shared part
is deliberately small: a stable key for the prefix and the positions where the
reusable prefix ends. Providers translate and must fail open.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

from .types import PromptCacheHint, PromptCachePolicy

DISABLED = PromptCacheHint(prefix_key="", breakpoints=(), ttl_seconds=None, enabled=False)


def prefix_key(model: str, system: str | None, tools: list[dict[str, Any]], salt: str = "") -> str:
    """Stable digest of everything that must be byte-identical for a cache hit."""
    payload = json.dumps(
        {"model": model, "system": system or "", "tools": tools, "salt": salt},
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_hint(
    policy: PromptCachePolicy | None,
    model: str,
    system: str | None,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> PromptCacheHint:
    if policy is None:
        return DISABLED
    breakpoints: list[str] = []
    if system:
        breakpoints.append("system")
    if tools:
        breakpoints.append("tools")
    if policy.cache_history and messages:
        breakpoints.append(f"message:{len(messages) - 1}")
    return PromptCacheHint(
        prefix_key=prefix_key(model, system, tools, policy.key_salt),
        breakpoints=tuple(breakpoints),
        ttl_seconds=policy.ttl_seconds,
        enabled=True,
    )


def hint_from_wire(value: Any) -> PromptCacheHint | None:
    """Rebuild a hint from persisted execution state; absent means a pre-hint state."""
    if value is None:
        return None
    if isinstance(value, PromptCacheHint):
        return value
    return PromptCacheHint(
        prefix_key=str(value.get("prefix_key", "")),
        breakpoints=tuple(str(b) for b in value.get("breakpoints", ())),
        ttl_seconds=value.get("ttl_seconds"),
        enabled=bool(value.get("enabled", True)),
    )


def accepts_cache(fn: Any) -> bool:
    """True when a provider method declares a ``cache`` parameter."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "cache" in params
