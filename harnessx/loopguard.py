"""Detecting an agent that has stopped making progress.

A stuck agent repeats itself. The cheap check is "the same call three times in a
row", which is what most harnesses do, and it misses the common case: a model
alternating between two calls, A, B, A, B, defeats a consecutive-identical
counter entirely. This module looks for a repeating *cycle* at the tail of the
call history instead, up to a bounded period.

The harder problem is not detection but false positives. Re-running a test after
an edit is the normal shape of work and looks exactly like a loop, so detection
that cannot tell them apart gets switched off within a week. The answer, taken
from Hermes, is that progress clears the history: once a call has actually
changed something, whatever came before is no longer evidence of a loop.

Nothing here fails a run. A trip annotates the result the model is about to read,
because a model told it is repeating itself usually stops; terminating the run is
what ``Limits.max_iterations`` is already for.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["call_signature", "detect_cycle", "repetition_notice"]


def call_signature(name: str, arguments: Any, result: str) -> str:
    """A stable digest of one completed call: what was asked, and what came back.

    The result is part of the signature deliberately. Polling a job is the same
    call over and over, and it is only a loop while the answer keeps coming back
    the same; the moment it changes, the agent is making progress.
    """
    try:
        rendered = json.dumps(arguments, sort_keys=True, default=str)
    except Exception:
        rendered = repr(arguments)
    digest = hashlib.sha256(f"{name}\0{rendered}\0{result}".encode("utf-8", "replace"))
    return digest.hexdigest()[:16]


def detect_cycle(signatures: list[str], *, max_period: int = 4, threshold: int = 3) -> int | None:
    """The period of a cycle repeating at the tail, or None.

    A period of 1 is the same call repeated; 2 is A, B, A, B. ``threshold`` is
    how many whole laps must be present before it counts.
    """
    if threshold < 2 or not signatures:
        return None
    for period in range(1, max(1, max_period) + 1):
        needed = period * threshold
        if len(signatures) < needed:
            break
        tail = signatures[-needed:]
        lap = tail[:period]
        if all(tail[index * period:(index + 1) * period] == lap for index in range(1, threshold)):
            return period
    return None


def repetition_notice(period: int, laps: int) -> str:
    """What the model reads when a cycle is detected."""
    what = "This exact call" if period == 1 else f"This cycle of {period} calls"
    return (
        f"[harness] {what} has now run {laps} times with the same result. "
        "Repeating it will not produce anything new. Re-read what you have, then "
        "either take a different approach or stop and explain what is blocking you."
    )
