"""Serializable execution contracts and deterministic state transitions."""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, TypedDict
import math
import uuid

from .types import TokenUsage, ToolCall, ToolResult


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    AWAITING_INPUT = "awaiting_input"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RunEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"
    THINKING_DELTA = "thinking_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    TOOL_RESULT = "tool_result"
    TURN_COMPLETE = "turn_complete"
    ERROR = "error"
    APPROVAL_REQUIRED = "approval_required"
    RECOVERY_REQUIRED = "recovery_required"
    ATTEMPT_RESET = "attempt_reset"
    RUN_RESULT = "run_result"
    GAP = "gap"


class ReplayPolicy(str, Enum):
    SAFE = "safe"
    IDEMPOTENT = "idempotent"
    MANUAL = "manual"


class RunFailure(TypedDict):
    type: str
    message: str


class ToolCallData(TypedDict):
    id: str
    name: str
    input: dict[str, Any]


class PendingTool(TypedDict):
    call: ToolCallData
    status: Literal["approval", "uncertain"]
    attempt: int
    execution_key: str
    policy: str
    concurrent: bool
    timeout: float


class AttemptReset(TypedDict):
    attempt: int


class EventGap(TypedDict, total=False):
    after: str
    first_available: str
    reason: str


@dataclass
class RunResult:
    session_id: str
    run_id: str
    output: str = ""
    status: RunStatus = RunStatus.COMPLETED
    stop_reason: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    error: RunFailure | None = None
    pending: list[PendingTool] = field(default_factory=list)
    usage_incomplete: bool = False

    def __post_init__(self) -> None:
        self.status = RunStatus(self.status)
        if self.error is not None and not all(isinstance(self.error.get(key), str) for key in ("type", "message")):
            raise ValueError("Run failure requires type and message strings")
        for pending in self.pending:
            if pending.get("status") not in ("approval", "uncertain") or not isinstance(pending.get("execution_key"), str):
                raise ValueError("Invalid pending tool operation")

    @classmethod
    def from_dict(cls, data: dict) -> RunResult:
        return cls(
            **{
                **data,
                "status": RunStatus(data["status"]),
                "usage": TokenUsage(**data.get("usage", {})),
            }
        )


@dataclass
class RunEvent:
    type: RunEventType
    data: str | ToolCall | ToolResult | RunResult | PendingTool | AttemptReset | EventGap | None = None
    session_id: str = ""
    run_id: str = ""
    step_id: str = ""
    attempt_id: str = ""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cursor: str = ""

    def __post_init__(self) -> None:
        self.type = RunEventType(self.type)
        if self.type in (RunEventType.TEXT_DELTA, RunEventType.TEXT_COMPLETE, RunEventType.THINKING_DELTA, RunEventType.TURN_COMPLETE, RunEventType.ERROR):
            valid = isinstance(self.data, str)
        elif self.type in (RunEventType.TOOL_CALL_START, RunEventType.TOOL_CALL_COMPLETE):
            valid = isinstance(self.data, ToolCall)
        elif self.type == RunEventType.TOOL_RESULT:
            valid = isinstance(self.data, ToolResult)
        elif self.type == RunEventType.RUN_RESULT:
            valid = isinstance(self.data, RunResult)
        else:
            valid = isinstance(self.data, dict)
        if not valid:
            raise TypeError(f"Invalid payload for event {self.type.value}")
        if self.type in (RunEventType.APPROVAL_REQUIRED, RunEventType.RECOVERY_REQUIRED) and isinstance(self.data, dict):
            if not isinstance(self.data.get("execution_key"), str) or not isinstance(self.data.get("call"), dict):
                raise ValueError("Pending-tool events require execution_key and call")
        if self.type == RunEventType.ATTEMPT_RESET and isinstance(self.data, dict) and type(self.data.get("attempt")) is not int:
            raise ValueError("Attempt reset requires an integer attempt")

    @classmethod
    def from_dict(cls, data: dict) -> RunEvent:
        data = dict(data)
        data["type"] = RunEventType(data["type"])
        if data["type"] in (
            RunEventType.TOOL_CALL_START,
            RunEventType.TOOL_CALL_COMPLETE,
        ):
            data["data"] = ToolCall(**data["data"])
        elif data["type"] == RunEventType.TOOL_RESULT:
            data["data"] = ToolResult(**data["data"])
        elif data["type"] == RunEventType.RUN_RESULT:
            data["data"] = RunResult.from_dict(data["data"])
        return cls(**data)


def wire(value: Any) -> Any:
    """Strict JSON shape; never pickle callables or stringify unknown objects."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return wire(asdict(value))
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("Execution state dictionary keys must be strings")
        return {k: wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Execution state numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Non-serializable execution state: {type(value).__name__}")


@dataclass(frozen=True)
class ToolExecutionContext:
    session_id: str
    run_id: str
    execution_key: str
    attempt: int


_tool_context: contextvars.ContextVar[ToolExecutionContext | None] = (
    contextvars.ContextVar("harness_tool", default=None)
)


def current_tool_context() -> ToolExecutionContext:
    context = _tool_context.get()
    if context is None:
        raise RuntimeError("No tool execution is active")
    return context


def next_command(state: dict) -> str:
    """Pure, shared by the in-process driver and Temporal workflow replay."""
    if state["status"] != "running":
        return "result"
    return state["phase"]


def transition(state: dict, command: str, outcome: dict) -> dict:
    """Only completed commands advance execution; in-flight tools retain intent."""
    state = {**state, **outcome}
    if command == "prepare_model":
        state["phase"] = "model"
    elif command == "model":
        response = state["response"]
        if response["stop_reason"] == "tool_use" and response["tool_calls"]:
            state["phase"] = "prepare_tools"
        else:
            state.update(
                phase="finish",
                output=response["text"],
                stop_reason=response["stop_reason"],
            )
    elif command == "prepare_tools":
        state["phase"] = "tools"
    elif command == "collect_tools":
        state["phase"] = "prepare_model"
    elif command == "finish":
        state.update(phase="result", status="completed")
    return state


def result_from_state(state: dict) -> RunResult:
    return RunResult(
        session_id=state["session_id"],
        run_id=state["run_id"],
        output=state.get("output", ""),
        status=RunStatus(state["status"]),
        stop_reason=state.get("stop_reason", ""),
        usage=TokenUsage(**state.get("usage", {})),
        error=state.get("error"),
        pending=[
            t
            for t in state.get("tools", [])
            if t["status"] in ("approval", "uncertain")
        ],
        usage_incomplete=state.get("usage_incomplete", False),
    )


class RunStream(AsyncIterator[RunEvent]):
    """Lazy stream. Use async with to guarantee cancellation on early exit."""

    def __init__(
        self,
        runner: Callable[[Callable[[RunEvent], Awaitable[None]]], Awaitable[RunResult]],
    ):
        self._runner = runner
        self._queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        self._task: asyncio.Task[RunResult] | None = None
        self._done = False

    def _start(self) -> asyncio.Task[RunResult]:
        if self._task is None:

            async def drive() -> RunResult:
                try:
                    return await self._runner(self._queue.put)
                finally:
                    await self._queue.put(None)

            self._task = asyncio.create_task(drive())
        return self._task

    def __aiter__(self) -> RunStream:
        return self

    async def __anext__(self) -> RunEvent:
        if self._done:
            raise StopAsyncIteration
        task = self._start()
        event = await self._queue.get()
        if event is None:
            self._done = True
            await task
            raise StopAsyncIteration
        return event

    async def result(self) -> RunResult:
        return await asyncio.shield(self._start())

    async def aclose(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._done = True

    async def __aenter__(self) -> RunStream:
        self._start()
        return self

    async def __aexit__(self, *exc):
        await self.aclose()


class ToolApprovalRequired(RuntimeError):
    """A dispatch-time policy change requires a new exact-call approval."""


def cancel_state(state: dict) -> dict:
    """Close incomplete tool protocol messages while retaining uncertain effects."""
    state["status"] = "cancelled"
    messages = state.get("messages", [])
    if messages and messages[-1].get("role") == "assistant":
        calls = [
            b for b in messages[-1].get("content", []) if b.get("type") == "tool_use"
        ]
        if calls:
            results = []
            recorded = {t["call"]["id"]: t for t in state.get("tools", [])}
            for call in calls:
                entry = recorded.get(call["id"], {})
                result = entry.get("result") or entry.get("raw_result")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": result["content"]
                        if result
                        else "Run cancelled; execution outcome may be unknown.",
                        "is_error": result.get("is_error", False) if result else True,
                    }
                )
            state["messages"] = [*messages, {"role": "user", "content": results}]
    return state
