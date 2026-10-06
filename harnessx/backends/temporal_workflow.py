"""Deterministic session coordinator. All user code runs in activities."""

from __future__ import annotations
import asyncio
import copy
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from ..execution import (
        model_timeout_from_wire,
        next_command,
        transition,
        result_from_state,
        wire,
        cancel_state,
        apply_accounting,
    )
    from ..types import DEFAULT_TIMEOUT_SECONDS, RetryPolicy as ModelRetryPolicy
    from ..runtime import resolve_entry


# The workflow sandbox re-imports this module's parent package, and importing
# harnessx pulls in provider SDKs and LangSmith, whose HTTP stacks the sandbox
# rejects. Running unsandboxed applies to workers and replayers alike; the
# workflow stays deterministic by delegating every model and tool call to
# activities and keeping only serialized state.
@workflow.defn(name="HarnessAgentSessionV1", sandboxed=False)
class AgentSessionWorkflow:
    @workflow.init
    def __init__(self, data: dict):
        self.data = data
        self.data.setdefault("started_at", workflow.time())
        self.busy = False
        self.activities = []
        self.event_seq = 0

    def state(self):
        return self.data["runs"].get(self.data["active"])

    def args(self, state, **extra):
        return {
            "session_id": self.data["session_id"],
            "agent": self.data["agent"],
            "state": copy.deepcopy(state),
            **extra,
        }

    def event(self, state, kind, payload):
        self.event_seq += 1
        return {
            "type": kind,
            "data": wire(payload),
            "session_id": state["session_id"],
            "run_id": state["run_id"],
            "step_id": str(state["iterations"]),
            "attempt_id": str(state["attempt"]),
            "event_id": f"{workflow.info().run_id}:{self.event_seq}",
            "cursor": "",
        }

    async def publish(self, events):
        if not events:
            return
        try:
            await workflow.execute_activity(
                "harness_publish_v1",
                events,
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
        except ActivityError:
            # Authoritative state is available through queries even if Redis is down.
            pass

    def deadline_exceeded(self):
        limit = self.data.get("runtime_config", {}).get("max_session_duration_seconds")
        return limit is not None and workflow.time() - self.data["started_at"] >= limit

    def activity_timeout(self, seconds):
        limit = self.data.get("runtime_config", {}).get("max_session_duration_seconds")
        if limit is not None:
            seconds = min(
                seconds, max(0.001, self.data["started_at"] + limit - workflow.time())
            )
        return timedelta(seconds=seconds)

    def merge_failure_accounting(self, state, entry, exc):
        cause = exc
        while cause is not None:
            for detail in (*getattr(cause, "details", ()), *getattr(cause, "last_heartbeat_details", ())):
                if isinstance(detail, dict) and "accounting" in detail:
                    apply_accounting(state, entry, detail["accounting"])
                    if "provider_state" in detail:
                        state["provider_state"] = detail["provider_state"]
            cause = getattr(cause, "cause", None)

    async def step(self, state, name, **extra):
        timeout = (
            model_timeout_from_wire((state or {}).get("config"))
            if name == "model"
            else DEFAULT_TIMEOUT_SECONDS
        )
        modern = workflow.patched("harness-reliability-v2")
        policy = RetryPolicy(maximum_attempts=3)
        options = {}
        if modern:
            config = (state or {}).get("config", {})
            retry = ModelRetryPolicy(**config.get("retry", {}))
            if name in ("model", "compact"):
                budget = (state or {}).get("model_call_budget") or (state or {}).get("request", {}).get("max_tokens", config.get("max_tokens"))
                if name == "compact":
                    budget = min(8000, budget or 8000)
                # A model command can immediately heal one refused budget.
                timeout = retry.effective_call_timeout(budget) * (2 if name == "model" else 1)
                policy = RetryPolicy(maximum_attempts=retry.attempts,
                    initial_interval=timedelta(seconds=max(0.001, retry.backoff_seconds)),
                    maximum_interval=timedelta(seconds=max(0.001, retry.max_backoff_seconds)))
            extra["reliability_v2"] = True
            options = {"heartbeat_timeout": timedelta(seconds=20),
                       "cancellation_type": workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED}
        handle = workflow.start_activity(
            "harness_step_v1",
            self.args(state, command=name, **extra),
            start_to_close_timeout=self.activity_timeout(timeout + 10),
            retry_policy=policy,
            **options,
        )
        self.activities.append(handle)
        try:
            return await handle
        except (ActivityError, asyncio.CancelledError) as exc:
            if modern and state is not None:
                self.merge_failure_accounting(state, {}, exc)
            raise
        finally:
            self.activities.remove(handle)

    @workflow.run
    async def run(self, data: dict):
        while True:
            await workflow.wait_condition(
                lambda: self.state() is not None and self.state()["status"] == "running"
            )
            state = self.state()
            self.busy = True
            try:
                while next_command(state) != "result":
                    if self.deadline_exceeded():
                        state.update(status="cancelled", stop_reason="timeout")
                        break
                    if self.data["control"] in ("paused", "cancelled"):
                        state["status"] = self.data["control"]
                        break
                    name = next_command(state)
                    if name == "new":
                        outcome = await self.step(
                            self.data.get("last_state"),
                            "new",
                            message=state["message"],
                            run_id=state["run_id"],
                            stream=state["stream"],
                        )
                        state = outcome["outcome"]
                        self.data["runs"][state["run_id"]] = state
                        continue
                    if name == "tools":
                        if any(
                            t["status"] in ("approval", "uncertain")
                            for t in state["tools"]
                        ):
                            state["status"] = "awaiting_input"
                            break
                        runnable = [
                            t for t in state["tools"] if t["status"] != "completed"
                        ]

                        async def dispatch(entry):
                            if entry["status"] == "raw_completed":
                                return
                            if (
                                entry["status"] == "started"
                                and entry["policy"] == "manual"
                            ):
                                entry["status"] = "uncertain"
                                return
                            entry["status"] = "started"
                            entry["attempt"] += 1
                            modern = workflow.patched("harness-reliability-v2")
                            if modern:
                                entry["accounting"] = {}
                            handle = workflow.start_activity(
                                "harness_tool_v1",
                                self.args(state, entry=entry, **({"reliability_v2": True} if modern else {})),
                                start_to_close_timeout=self.activity_timeout(
                                    entry["timeout"] + 10
                                ),
                                heartbeat_timeout=timedelta(seconds=20),
                                **({"cancellation_type": workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED} if modern else {}),
                                retry_policy=RetryPolicy(
                                    maximum_attempts=1
                                    if entry["policy"] == "manual"
                                    else int((entry.get("retry") or {}).get("attempts", 3))
                                ),
                            )
                            self.activities.append(handle)
                            try:
                                returned = await handle
                                if modern:
                                    apply_accounting(state, entry, returned["accounting"])
                                    entry["raw_result"] = returned["result"]
                                else:
                                    entry["raw_result"] = returned
                                entry["status"] = "raw_completed"
                            except (ActivityError, asyncio.CancelledError) as exc:
                                if modern:
                                    self.merge_failure_accounting(state, entry, exc)
                                    state["usage_incomplete"] = True
                                entry["status"] = (
                                    "uncertain"
                                    if entry["policy"] == "manual"
                                    else "started"
                                )
                                if (
                                    getattr(getattr(exc, "cause", None), "type", "")
                                    == "ToolApprovalRequired"
                                ):
                                    entry["status"] = "approval"
                            finally:
                                self.activities.remove(handle)

                        await self.publish(
                            [
                                self.event(state, "tool_call_start", t["call"])
                                for t in runnable
                            ]
                        )
                        if runnable and all(t["concurrent"] for t in runnable):
                            await asyncio.gather(*(dispatch(t) for t in runnable))
                        else:
                            for entry in runnable:
                                await dispatch(entry)
                        for entry in runnable:
                            if entry["status"] == "raw_completed":
                                outcome = await self.step(
                                    state, "finish_tool", entry=entry
                                )
                                processed = outcome["outcome"]
                                entry["result"] = processed.pop("result")
                                entry["status"] = "completed"
                                state.update(processed)
                                await self.publish(
                                    [
                                        self.event(
                                            state, "tool_call_complete", entry["call"]
                                        ),
                                        self.event(
                                            state, "tool_result", entry["result"]
                                        ),
                                    ]
                                )
                        if self.data["control"] == "cancelled":
                            state["status"] = "cancelled"
                            break
                        if any(t["status"] != "completed" for t in state["tools"]):
                            state["status"] = (
                                "awaiting_input"
                                if any(
                                    t["status"] == "uncertain" for t in state["tools"]
                                )
                                else "failed"
                            )
                            break
                        name = "collect_tools"
                    if name == "model":
                        state["attempt"] += 1
                    outcome = await self.step(state, name)
                    state = transition(state, name, outcome["outcome"])
                    if name in ("model", "compact") and outcome["attempt"] > 1:
                        state["usage_incomplete"] = True
                    self.data["runs"][state["run_id"]] = state
                    # Activity completion is now recorded; committed events may be published.
                    await self.publish(outcome["events"])
            except (ActivityError, asyncio.CancelledError) as exc:
                if state["phase"] == "model":
                    state["usage_incomplete"] = True
                state.update(
                    status="cancelled"
                    if self.data["control"] == "cancelled"
                    else "failed",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            finally:
                if self.deadline_exceeded():
                    state.update(status="cancelled", stop_reason="timeout")
                if state["status"] == "cancelled":
                    state = cancel_state(state)
                if (
                    state["status"] in ("completed", "failed", "cancelled")
                    and workflow.patched("sdk-terminal-callback-v1")
                ):
                    try:
                        # Cleanup has its own bounded grace period, even at the run deadline.
                        await workflow.execute_activity(
                            "harness_step_v1",
                            self.args(state, command="complete"),
                            start_to_close_timeout=timedelta(seconds=30),
                            retry_policy=RetryPolicy(maximum_attempts=3),
                        )
                    except (ActivityError, asyncio.CancelledError):
                        workflow.logger.warning("Terminal observers could not complete")
                self.data["runs"][state["run_id"]] = state
                self.data["last_state"] = state
                # Freeze delivery payloads before accepting the next control/update.
                settled_events = []
                if state["status"] == "completed":
                    settled_events.append(
                        self.event(state, "turn_complete", state.get("stop_reason", ""))
                    )
                for entry in state.get("tools", []):
                    if entry["status"] in ("approval", "uncertain"):
                        settled_events.append(
                            self.event(
                                state,
                                "approval_required"
                                if entry["status"] == "approval"
                                else "recovery_required",
                                entry,
                            )
                        )
                settled_events.append(
                    self.event(state, "run_result", result_from_state(state))
                )
                self.busy = False
            await self.publish(settled_events)
            if workflow.info().is_continue_as_new_suggested():
                await workflow.wait_condition(workflow.all_handlers_finished)
                workflow.continue_as_new(self.data)

    @workflow.update
    def submit(self, request: dict) -> str:
        prior = self.data["requests"].get(request["request_id"])
        if prior:
            if self.data["runs"][prior]["message"] != request["message"]:
                raise ApplicationError(
                    "Request ID already used with different input", non_retryable=True
                )
            return prior
        if self.busy or (
            self.state() and self.state()["status"] not in ("completed", "cancelled")
        ):
            raise ApplicationError(
                "Session already has an unfinished run", non_retryable=True
            )
        run_id = request["run_id"]
        self.data["requests"][request["request_id"]] = run_id
        self.data["runs"][run_id] = {
            "session_id": self.data["session_id"],
            "run_id": run_id,
            "message": request["message"],
            "stream": request["stream"],
            "status": "running",
            "phase": "new",
            "iterations": 0,
            "attempt": 0,
        }
        self.data["active"] = run_id
        self.data["control"] = ""
        return run_id

    @workflow.update
    def control(self, request: dict):
        action = request["action"]
        state = self.state()
        if action == "resume":
            self.data["started_at"] = workflow.time()
            if self.busy:
                raise ApplicationError("Run is already active", non_retryable=True)
            self.data["control"] = ""
            if state and state["status"] not in ("completed", "cancelled"):
                state.update(status="running", error=None)
                return state["run_id"]
            return None
        if action not in ("paused", "cancelled"):
            raise ApplicationError("Invalid control", non_retryable=True)
        self.data["control"] = action
        if action == "cancelled":
            for handle in self.activities:
                handle.cancel()
        if (
            state
            and not self.busy
            and state["status"] not in ("completed", "cancelled")
        ):
            if action == "cancelled" and workflow.patched("sdk-terminal-callback-v1"):
                # Wake the driver so cancellation also runs terminal observers.
                state["status"] = "running"
            else:
                state["status"] = action
                if action == "cancelled":
                    cancel_state(state)
        return self.data["active"]

    @workflow.update
    def resolve(self, request: dict):
        state = self.state()
        if (
            self.busy
            or not state
            or state["status"]
            not in ("awaiting_input", "paused", "failed", "cancelled")
        ):
            raise ApplicationError("Run is not awaiting resolution", non_retryable=True)
        try:
            entry = next(
                t for t in state["tools"] if t["execution_key"] == request["key"]
            )
            resolve_entry(
                state,
                entry,
                **{
                    k: request[k]
                    for k in ("allow", "result", "retry", "abort")
                    if k in request
                },
            )
        except (ValueError, StopIteration) as exc:
            raise ApplicationError(str(exc), non_retryable=True) from None

    @workflow.query
    def get_run(self, run_id: str) -> dict:
        return self.data["runs"][run_id]

    @workflow.query
    def get_session(self) -> dict:
        return {
            "session_id": self.data["session_id"],
            "agent": self.data["agent"],
            "busy": self.busy,
            "active": self.data["active"],
            "status": self.state()["status"] if self.state() else "idle",
        }
