"""Internal flight-recorder records. Payloads never enter the public event stream."""

from __future__ import annotations

from .errors import HarnessError

from datetime import datetime, timezone
import uuid

from .execution import wire


class RecordingError(HarnessError, RuntimeError):
    """A required recording boundary could not be committed."""


def record(kind, payload, state, entry=None):
    return {
        "version": 1,
        "id": str(uuid.uuid4()),
        "kind": kind,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "session_id": state["session_id"],
        "run_id": state["run_id"],
        "step_id": str(state.get("iterations", 0)),
        "attempt_id": str((entry or state).get("attempt", 0)),
        "execution_key": (entry or {}).get("execution_key"),
        "payload": wire(payload),
    }


def checkpoint(state):
    return {
        "phase": state["phase"],
        "status": state["status"],
        "error": state.get("error"),
        # Spill envelopes contain result IDs; the extension snapshot binds those
        # IDs to portable artifact URIs without introducing a public reference API.
        "extensions": state.get("extensions", {}),
        "tools": [
            {
                key: tool[key]
                for key in ("execution_key", "status", "attempt", "approved")
                if key in tool
            }
            for tool in state.get("tools", [])
        ],
    }
