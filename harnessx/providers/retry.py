"""Retry for transient provider failures.

Every provider here talks to a remote API that rate-limits, returns 5xx and
times out. Without a retry, one such blip ends the caller's turn — so each
consumer ends up writing the same loop, or (more often) shipping without one.

Two rules matter and are easy to get wrong:

* **Only retry what is worth retrying.** A 4xx other than 429 is deterministic:
  a bad request stays bad, and retrying spends a second call to receive the
  same error. 429, 5xx and timeouts are the transient set.

* **Never retry a stream that has already delivered.** Once a text delta has
  reached the caller, re-running the request duplicates output. The engine's
  loop and `FallbackProvider` therefore retry a stream only up to the FIRST
  chunk, and never after.

This module holds the classification the engine's retry loop asks for by
default; providers override it through `LLMProvider.is_transient`.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ATTEMPTS = 2
DEFAULT_BACKOFF_SECONDS = 0.5

_TRANSIENT_MARKERS = (
    "resource_exhausted",
    "rate limit",
    "rate_limit",
    "overloaded",
    "unavailable",
    "timeout",
    "timed out",
    "temporarily",
    "connection reset",
    "server error",
    "too many requests",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
)


# Permanent failures that arrive wearing a retryable status. A quota or billing
# 429 is not throttling: it will still be exhausted after any backoff, so the
# whole ladder is spent to be refused again. Checked before the status.
_PERMANENT_MARKERS = (
    "quota exceeded",
    "insufficient_quota",
    "exceeded your current quota",
    "billing",
    "payment required",
    "spending limit",
    "credit balance is too low",
    "account is not active",
    "suspended",
)

# Overflowing the context window is deterministic: the same request will not fit
# next time either. It needs condensing, not a retry.
_OVERFLOW_MARKERS = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "too many tokens",
    "prompt is too long",
    "reduce the length",
)


def is_permanent(exc: BaseException) -> bool:
    """True when no number of attempts will help, whatever the status says."""
    text = str(exc).lower()
    return any(marker in text for marker in _PERMANENT_MARKERS)


def is_context_overflow(exc: BaseException) -> bool:
    """True when the request did not fit the model's context window."""
    text = str(exc).lower()
    if any(marker in text for marker in _OVERFLOW_MARKERS):
        # "Too many tokens" also appears in some throttling messages; a
        # throttle names a rate, an overflow names the window.
        return not any(marker in text for marker in ("rate limit", "too many requests", "throttl"))
    return False


def is_transient(exc: BaseException) -> bool:
    """True when one more attempt could plausibly succeed."""
    # Permanent and overflow failures can carry a 429 or a 5xx, so they are
    # settled before the status is consulted.
    if is_permanent(exc) or is_context_overflow(exc):
        return False
    # SDKs expose the HTTP status inconsistently; check the usual spellings.
    for attr in ("code", "status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            if value == 429 or value >= 500:
                return True
            if 400 <= value < 500:
                return False           # deterministic; do not retry
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    return is_transient_text(str(exc))


def is_transient_text(text: str) -> bool:
    """True when an error message reads like throttling, overload, or a timeout.

    Used for exceptions without a status attribute and for error results whose
    only signal is their text (an MCP tool reporting a 429 in its body).
    """
    lowered = str(text).lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def sdk_connection_failure(exc: BaseException) -> bool:
    """True when an SDK wrapped a transport failure that is worth another attempt.

    The OpenAI and Anthropic SDKs raise ``APIConnectionError`` for a dropped
    connection, a DNS failure or a refused socket. It carries no status, is not
    an ``OSError``, and stringifies to "Connection error.", so nothing else here
    recognizes it. Their own retry logic does, which is why the engine must:
    harnessx builds those clients with ``max_retries=0`` and owns the count.

    The exception is a proxy rejecting the tunnel with a 4xx. An egress proxy
    that refuses a destination on policy grounds (NVIDIA OpenShell does this,
    answering 403 to the CONNECT) fails the same way a flaky network does, and
    retrying it only spends the budget to be refused again.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    connection_failure = False
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in ("APIConnectionError", "APITimeoutError"):
            connection_failure = True
        if _is_proxy_rejection(current):
            return False
        current = current.__cause__ or current.__context__
    return connection_failure


def _is_proxy_rejection(exc: BaseException) -> bool:
    """A forward proxy refusing the tunnel with a 4xx, rather than a transport failure."""
    if "proxy" not in type(exc).__name__.lower():
        return False
    return any(str(code) in str(exc) for code in range(400, 452))


def retry_after_seconds(exc: BaseException) -> float | None:
    """The server's Retry-After for this failure in seconds, or None when it sent none.

    Reads the header off SDK exceptions (``exc.response.headers`` or ``exc.headers``)
    and a ``retry_after`` field in a JSON error body; accepts delay-seconds or an
    HTTP-date. Never negative.
    """
    value: Any = None
    for headers in (
        getattr(getattr(exc, "response", None), "headers", None),
        getattr(exc, "headers", None),
    ):
        if headers is None:
            continue
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:
            value = None
        if value:
            break
    if not value:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            nested = body.get("error")
            error: dict[str, Any] = nested if isinstance(nested, dict) else body
            value = error.get("retry_after") or error.get("retry-after")
    return _parse_retry_after(value)


def _parse_retry_after(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
