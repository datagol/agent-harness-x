"""Serializable execution contracts and deterministic state transitions."""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, Mapping, TypedDict
import math
import uuid

from .errors import HarnessError, RunAwaitingInput, RunCancelled, RunError, RunFailed, RunLimitReached, RunRefused, RunTruncated
from .types import DEFAULT_TIMEOUT_SECONDS, ReplayPolicy as ReplayPolicy, TokenUsage, ToolCall, ToolResult, ToolRetry


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
    TODOS_UPDATED = "todos_updated"  # the agent's task list changed: {todos, completed, total}
    WAITING = "waiting"
    RUN_RESULT = "run_result"
    GAP = "gap"


class RunFailure(TypedDict):
    type: str
    message: str


class ToolCallData(TypedDict):
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class PendingTool:
    """A tool call the run stopped on: it needs approval, or its outcome is unknown.

    Resolve it with ``AgentRuntime.approve``/``decline`` (status ``"approval"``)
    or ``AgentRuntime.resolve_tool`` (status ``"uncertain"``).
    """

    execution_key: str
    call: ToolCall
    status: Literal["approval", "uncertain"]
    attempt: int = 0
    policy: str = "manual"
    concurrent: bool = False
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    retry: ToolRetry | None = None  # the tool's retry policy; None on entries persisted before 0.4.3

    def __post_init__(self) -> None:
        if self.status not in ("approval", "uncertain") or not isinstance(self.execution_key, str):
            raise ValueError("Invalid pending tool operation")
        if isinstance(self.call, dict):
            object.__setattr__(self, "call", ToolCall(**self.call))
        if isinstance(self.retry, dict):
            object.__setattr__(self, "retry", ToolRetry.from_dict(self.retry))

    def to_dict(self) -> dict[str, Any]:
        """The persisted shape; unchanged since 0.3 (``timeout``, not ``timeout_seconds``)."""
        return {
            "call": {"id": self.call.id, "name": self.call.name, "input": self.call.input},
            "status": self.status,
            "attempt": self.attempt,
            "execution_key": self.execution_key,
            "policy": self.policy,
            "concurrent": self.concurrent,
            "timeout": self.timeout_seconds,
            **({"retry": self.retry.to_dict()} if self.retry is not None else {}),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PendingTool:
        """Accepts a run-state tool entry, ignoring bookkeeping keys such as ``result``."""
        call = data["call"]
        return cls(
            execution_key=data["execution_key"],
            call=ToolCall(call["id"], call["name"], dict(call.get("input") or {})),
            status=data["status"],
            attempt=int(data.get("attempt", 0)),
            policy=str(data.get("policy", "manual")),
            concurrent=bool(data.get("concurrent", False)),
            timeout_seconds=float(data.get("timeout", data.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))),
            retry=ToolRetry.from_dict(data["retry"]) if data.get("retry") else None,
        )


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
        self.pending = [
            item if isinstance(item, PendingTool) else PendingTool.from_dict(item)
            for item in self.pending
        ]

    @property
    def ok(self) -> bool:
        """True when the run completed with a whole reply. Failures, pauses,
        cancellations, replies cut off at the token budget, and refusals are not ok."""
        return (
            self.status is RunStatus.COMPLETED
            and not self.truncated and not self.refused and not self.limited
            and self.stop_reason not in ("pause_turn", "empty_reply")
        )

    @property
    def truncated(self) -> bool:
        """True when the model stopped at ``max_tokens``: the reply, or a tool call it
        was making, is incomplete even though the run ended normally."""
        return self.status is RunStatus.COMPLETED and self.stop_reason == "max_tokens"

    @property
    def limited(self) -> bool:
        """True when the run reached ``Limits.max_iterations``: the output is the
        model's account of where it got to, not a finished answer."""
        return self.status is RunStatus.COMPLETED and self.stop_reason == "max_iterations"

    @property
    def refused(self) -> bool:
        """True when the model declined or a content filter stopped the reply: the
        run ended normally, but its output is not an answer."""
        return self.status is RunStatus.COMPLETED and self.stop_reason in ("refusal", "safety")

    @property
    def failed(self) -> bool:
        return self.status is RunStatus.FAILED

    @property
    def needs_input(self) -> bool:
        """True when the run is waiting for an approval or an uncertain-outcome resolution."""
        return self.status is RunStatus.AWAITING_INPUT

    def raise_for_status(self) -> RunResult:
        """Return the result if it completed; otherwise raise a RunError describing why."""
        if self.status is RunStatus.COMPLETED:
            if self.truncated:
                raise RunTruncated(self)
            if self.refused:
                raise RunRefused(self)
            if self.limited:
                raise RunLimitReached(self)
            return self
        if self.status is RunStatus.FAILED:
            raise RunFailed(self)
        if self.status is RunStatus.AWAITING_INPUT:
            raise RunAwaitingInput(self)
        if self.status is RunStatus.CANCELLED:
            raise RunCancelled(self)
        raise RunError(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "output": self.output,
            "status": self.status.value,
            "stop_reason": self.stop_reason,
            "usage": asdict(self.usage),
            "error": dict(self.error) if self.error is not None else None,
            "pending": [item.to_dict() for item in self.pending],
            "usage_incomplete": self.usage_incomplete,
        }

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
    data: str | ToolCall | ToolResult | RunResult | PendingTool | AttemptReset | EventGap
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
        if self.type == RunEventType.WAITING and isinstance(self.data, dict):
            if self.data.get("on") not in ("model", "tool"):
                raise ValueError("Waiting events say what is being waited on: 'model' or 'tool'")
            if not isinstance(self.data.get("seconds"), (int, float)):
                raise ValueError("Waiting events require elapsed seconds")

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
        to_dict = getattr(value, "to_dict", None)
        return wire(to_dict() if callable(to_dict) else asdict(value))
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


_agent_context: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "harness_agent", default=None
)


def current_agent() -> Any | None:
    """The agent running the current tool call, or None outside a run.

    Built-in tools that keep run-scoped state use this to reach it. Application
    tools should close over what they need instead; this is not a service
    locator for arbitrary agent access.
    """
    return _agent_context.get()


def next_command(state: dict) -> str:
    """Pure, shared by the in-process driver and Temporal workflow replay."""
    if state["status"] != "running":
        return "result"
    return state["phase"]


def transition(state: dict, command: str, outcome: dict) -> dict:
    """Only completed commands advance execution; in-flight tools retain intent."""
    state = {**state, **outcome}
    if command == "prepare_model":
        # Normally the model call is next, but preparing may decide the history
        # has to be condensed first and say so in its outcome.
        state["phase"] = outcome.get("phase", "model")
    elif command == "model":
        # The command already decided the loop goes on -- a truncated or empty
        # reply it answered, or a request the provider refused as too long.
        recovery = state.pop("recovery", None)
        if recovery == "context_overflow":
            state["phase"] = "compact"
            return state
        if recovery:
            state["phase"] = "prepare_model"
            return state
        response = state["response"]
        failure = state.pop("terminal_failure", None)
        if failure:
            state.update(status="failed", error=failure, phase="finish",
                         output=state.get("partial_output", "") + response["text"])
            return state
        if state.get("final_answer"):
            # The run is at max_iterations and this was its one last call. It
            # ends here whatever the reply holds; a tool call it made anyway is
            # answered, not run.
            if response["tool_calls"]:
                state = close_open_tool_calls(state, "Not executed: the run reached its step limit.")
            state.update(
                phase="finish",
                output=state.get("partial_output", "") + response["text"],
                stop_reason="max_iterations",
            )
            return state
        # Trust the tool calls, not the stop reason. Providers return "end_turn"
        # or "stop" while still carrying tool calls; ending the turn there drops
        # them silently and the run looks like the model ignored its tools.
        # A reply cut off at max_tokens is the exception: its tool arguments may
        # have been truncated mid-JSON and can parse while being incomplete, so
        # those calls are never executed.
        truncated = response["stop_reason"] == "max_tokens"
        blocked = response["stop_reason"] in ("safety", "refusal")
        if response["tool_calls"] and not (truncated or blocked):
            state["phase"] = "prepare_tools"
        else:
            if response["tool_calls"]:
                # Refused above, but the assistant turn carrying them is already
                # in the transcript, so each one still needs an answer or the
                # next request is malformed.
                state = close_open_tool_calls(
                    state,
                    "Not executed: the provider refused or filtered this reply." if blocked else
                    "Not executed: the reply was cut off at the token budget, so "
                    "these arguments may be incomplete. Call the tool again with "
                    "a shorter reply.",
                )
            output = state.get("partial_output", "") + response["text"]
            state.update(
                phase="finish",
                # A reply continued after being cut off arrives in pieces.
                output=output if response["text"].strip() else output.rstrip(),
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
        output=state.get("output", state.get("partial_output", "")),
        status=RunStatus(state["status"]),
        stop_reason=state.get("stop_reason", ""),
        usage=TokenUsage(**state.get("usage", {})),
        error=state.get("error"),
        pending=[
            PendingTool.from_dict(t)
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

    async def text(self, *, on_reset: Callable[[], Any] | None = None) -> AsyncIterator[str]:
        """Yield text deltas only; raise RunError if the run does not complete.

        ``on_reset`` is called on ATTEMPT_RESET, when a retried model call
        restarts the answer and text shown so far should be discarded.
        """
        async for event in self:
            if event.type is RunEventType.TEXT_DELTA:
                yield event.data  # type: ignore[misc]
            elif event.type is RunEventType.ATTEMPT_RESET and on_reset is not None:
                on_reset()
        (await self.result()).raise_for_status()

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


class ToolApprovalRequired(HarnessError, RuntimeError):
    """A dispatch-time policy change requires a new exact-call approval."""


def apply_accounting(state: dict, entry: dict, accounting: dict) -> None:
    """Apply a cumulative activity charge once, including heartbeat recovery."""
    previous = entry.get("accounting", {})
    for key, value in accounting.get("usage", {}).items():
        delta = max(0, value - previous.get("usage", {}).get(key, 0))
        for bucket in ("usage", "total_usage"):
            state.setdefault(bucket, {})[key] = state.get(bucket, {}).get(key, 0) + delta
    state["estimated_cost"] = state.get("estimated_cost", 0) + max(0, accounting.get("cost", 0) - previous.get("cost", 0))
    state["usage_incomplete"] = state.get("usage_incomplete", False) or accounting.get("incomplete", False)
    entry["accounting"] = {
        "usage": {k: max(v, accounting.get("usage", {}).get(k, 0))
                  for k, v in {**accounting.get("usage", {}), **previous.get("usage", {})}.items()},
        "cost": max(previous.get("cost", 0), accounting.get("cost", 0)),
        "incomplete": previous.get("incomplete", False) or accounting.get("incomplete", False),
    }


def model_timeout_from_wire(config: Mapping[str, Any] | None) -> float:
    """Model-call timeout from a persisted config, 0.4 nested or 0.3 flat."""
    config = config or {}
    retry = config.get("retry")
    if isinstance(retry, dict) and retry.get("call_timeout_seconds") is not None:
        return float(retry["call_timeout_seconds"])
    if config.get("model_timeout_seconds") is not None:
        return float(config["model_timeout_seconds"])
    from .types import call_timeout_for

    return call_timeout_for(config.get("max_tokens"))


def close_open_tool_calls(state: dict, unanswered: str) -> dict:
    """Answer every tool call in the last assistant turn that has no result yet.

    An assistant message carrying a ``tool_use`` block that nothing answers is
    not a transcript a provider will accept: Anthropic rejects the next request
    outright, and the session is wedged with no way forward. Any path that stops
    after the assistant turn is saved has to close the calls it leaves open --
    cancellation, a reply truncated at the token budget, or a mid-turn failure.

    Results already recorded are kept, so a tool that did run reports what it
    did; the rest get ``unanswered``. Pure, because ``transition`` is.
    """
    messages = state.get("messages", [])
    if not messages or messages[-1].get("role") != "assistant":
        return state
    content = messages[-1].get("content")
    if not isinstance(content, list):
        return state
    calls = [b for b in content if b.get("type") == "tool_use"]
    if not calls:
        return state

    recorded = {t["call"]["id"]: t for t in state.get("tools", [])}
    answered = {
        b.get("tool_use_id")
        for m in messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    }
    results = []
    for call in calls:
        if call["id"] in answered:
            continue
        entry = recorded.get(call["id"], {})
        result = entry.get("result") or entry.get("raw_result")
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": result["content"] if result else unanswered,
                "is_error": result.get("is_error", False) if result else True,
            }
        )
    if results:
        state["messages"] = [*messages, {"role": "user", "content": results}]
    return state


def cancel_state(state: dict) -> dict:
    """Close incomplete tool protocol messages while retaining uncertain effects."""
    state["status"] = "cancelled"
    return close_open_tool_calls(
        state, "Run cancelled; execution outcome may be unknown."
    )
