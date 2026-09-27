"""Retry for transient provider failures.

Every provider here talks to a remote API that rate-limits, returns 5xx and
times out. Without a retry, one such blip ends the caller's turn — so each
consumer ends up writing the same loop, or (more often) shipping without one.

Two rules matter and are easy to get wrong:

* **Only retry what is worth retrying.** A 4xx other than 429 is deterministic:
  a bad request stays bad, and retrying spends a second call to receive the
  same error. 429, 5xx and timeouts are the transient set.

* **Never retry a stream that has already delivered.** Once a text delta has
  reached the caller, re-running the request duplicates output. `stream_with_retry`
  therefore retries only up to the FIRST chunk, and never after.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar

from ..types import StreamChunk

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
)

T = TypeVar("T")


def is_transient(exc: BaseException) -> bool:
    """True when one more attempt could plausibly succeed."""
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
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


async def call_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    description: str = "provider call",
) -> T:
    """Run `operation`, retrying transient failures up to `attempts` times."""
    last: BaseException | None = None
    for attempt in range(max(1, attempts)):
        try:
            return await operation()
        except Exception as exc:
            last = exc
            if attempt + 1 >= attempts or not is_transient(exc):
                raise
            logger.warning(
                "%s failed (attempt %d/%d), retrying: %s",
                description, attempt + 1, attempts, exc,
            )
            if backoff_seconds:
                await asyncio.sleep(backoff_seconds * (2 ** attempt))
    raise last  # pragma: no cover - loop always returns or raises


async def stream_with_retry(
    open_stream: Callable[[], AsyncIterator[StreamChunk]],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    description: str = "provider stream",
) -> AsyncIterator[StreamChunk]:
    """Stream, retrying only while nothing has been delivered.

    A failure after the first chunk is raised, not retried: the caller has
    already seen part of the answer and a second run would repeat it.
    """
    for attempt in range(max(1, attempts)):
        iterator = open_stream()
        try:
            first = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except Exception as exc:
            if attempt + 1 >= attempts or not is_transient(exc):
                raise
            logger.warning(
                "%s failed before first chunk (attempt %d/%d), retrying: %s",
                description, attempt + 1, attempts, exc,
            )
            await _aclose(iterator)
            if backoff_seconds:
                await asyncio.sleep(backoff_seconds * (2 ** attempt))
            continue

        yield first
        # Past this point the caller has output; a retry would duplicate it.
        async for chunk in iterator:
            yield chunk
        return


async def _aclose(iterator: Any) -> None:
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception:  # pragma: no cover - best effort
        pass
