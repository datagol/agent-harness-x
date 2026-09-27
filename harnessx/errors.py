"""Exception hierarchy. Every failure harnessx raises on purpose derives from HarnessError.

Classes that predate this module keep their previous bases (RuntimeError,
ValueError, LookupError) so existing ``except`` clauses keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .execution import RunResult


class HarnessError(Exception):
    """Base class for exceptions harnessx raises on purpose."""


class ConfigurationError(HarnessError, ValueError):
    """Invalid, conflicting, or mismatched configuration or call arguments."""


class RuntimeStateError(HarnessError, RuntimeError):
    """The operation is not valid in the object's current state."""


class ResolutionError(HarnessError, ValueError):
    """A tool resolution does not apply to the entry's current state."""


class UnknownExecutionKey(HarnessError, KeyError):
    """No pending tool matches the execution key."""

    def __init__(self, execution_key: Any) -> None:
        super().__init__(execution_key)
        self.execution_key = execution_key

    def __str__(self) -> str:  # KeyError quotes its argument; say what it is instead
        return f"No pending tool has execution key {self.execution_key!r}"


class RunError(HarnessError):
    """A run did not complete. ``result`` is the RunResult that describes why."""

    def __init__(self, result: RunResult, message: str | None = None) -> None:
        super().__init__(message or f"Run {result.run_id} ended with status {result.status.value}")
        self.result = result


class RunFailed(RunError):
    """The run failed; ``error`` carries the failure type and message."""

    def __init__(self, result: RunResult) -> None:
        detail = result.error or {"type": "unknown", "message": "no error recorded"}
        super().__init__(result, f"{detail['type']}: {detail['message']}")
        self.error = detail


class RunAwaitingInput(RunError):
    """The run paused for an approval or an uncertain tool outcome; see ``pending``."""

    def __init__(self, result: RunResult) -> None:
        names = ", ".join(p.call.name for p in result.pending) or "no tools listed"
        super().__init__(result, f"Run {result.run_id} is awaiting input for: {names}")
        self.pending = list(result.pending)


class RunCancelled(RunError):
    """The run was cancelled before it completed."""
