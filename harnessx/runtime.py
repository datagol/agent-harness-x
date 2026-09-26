"""Durable sessions with one shared Agent engine and interchangeable backends."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .recorder import BundleLimits, ExportPolicy

from .core import Agent
from .engine import drive, new_state, restore
from .execution import (
    RunEvent,
    RunEventType,
    RunResult,
    RunStream,
    result_from_state,
    wire,
    cancel_state,
)
from .registry import AgentRef, AgentRegistry, agents
from .backends import SQLiteBackend, SessionBusyError
from .types import AgentConfig, RuntimeConfig, RuntimeState
from ._journal import record as journal_record


class RunHandle:
    def __init__(self, backend, session_id, run_id, task=None):
        self.backend, self.session_id, self.run_id, self._task = (
            backend,
            session_id,
            run_id,
            task,
        )

    async def result(self):
        if self._task:
            try:
                return await asyncio.shield(self._task)
            except asyncio.CancelledError:
                if self._task.cancelled():
                    state = await self.backend.get_run(self.run_id)
                    if state["status"] == "cancelled":
                        return result_from_state(state)
                raise
        while True:
            state = await self.backend.get_run(self.run_id)
            if state["status"] != "running":
                return result_from_state(state)
            await asyncio.sleep(0.1)

    async def events(self, after="0"):
        last_result_status = None
        while True:
            events = await self.backend.read_events(self.run_id, after)
            for event in events:
                after = event.cursor
                if event.type == RunEventType.RUN_RESULT:
                    last_result_status = event.data.status.value
                yield event
            if len(events) == 1000:
                continue
            state = await self.backend.get_run(self.run_id)
            if state["status"] != "running":
                # Status may be committed just before the final event batch.
                while True:
                    tail = await self.backend.read_events(self.run_id, after)
                    for event in tail:
                        after = event.cursor
                        if event.type == RunEventType.RUN_RESULT:
                            last_result_status = event.data.status.value
                        yield event
                    if len(tail) < 1000:
                        break
                if last_result_status != state["status"]:
                    yield RunEvent(
                        RunEventType.RUN_RESULT,
                        result_from_state(state),
                        self.session_id,
                        self.run_id,
                        event_id=f"{self.run_id}:result:{state['status']}",
                    )
                return
            if self._task and self._task.done():
                await self._task
            await asyncio.sleep(0.1)

    def stream(self, after="0"):
        async def observe(emit):
            async for event in self.events(after):
                await emit(event)
            return await self.result()

        return RunStream(observe)


class AgentRuntime:
    def __init__(
        self,
        agent: AgentRef | Agent | None = None,
        *,
        backend=None,
        registry: AgentRegistry | None = None,
        agent_config=None,
        runtime_config=None,
        tool_registrar=None,
        recording: bool | None = None,
        **agent_options,
    ):
        if recording is not None and type(recording) is not bool:
            raise TypeError("recording must be bool or None")
        self._recording_requested = recording
        self.recording = bool(recording)
        if agent is not None and not isinstance(agent, (Agent, AgentRef)):
            raise TypeError("agent must be an Agent or versioned AgentRef")
        if agent is not None and (agent_config is not None or agent_options or tool_registrar is not None):
            raise ValueError("Configure an Agent/AgentRef at its definition; inline construction options cannot override it")
        self.backend = backend or SQLiteBackend(
            str(
                Path((runtime_config or RuntimeConfig()).storage_dir)
                / "runtime.sqlite3"
            )
        )
        self.registry = registry or agents
        self.runtime_config = runtime_config or RuntimeConfig()
        self._named_binding = isinstance(agent, AgentRef)
        self._ref = agent if self._named_binding else AgentRef("inline", "v1")
        self._provided_agent = agent if isinstance(agent, Agent) else None
        self._options = agent_options
        self._config = agent_config or AgentConfig()
        self._registrar = tool_registrar
        self._agent = None
        self._session_id = ""
        self._task = None
        self._active_run_id = None
        self._started_at = 0.0
        self._state = RuntimeState.INITIALIZING
        self._owns_backend = backend is None
        self._last_status = None
        if getattr(self.backend, "remote", False) and not self._named_binding:
            raise ValueError("Remote execution requires a versioned AgentRef")
        if self.recording and not getattr(self.backend, "supports_recording", False):
            raise ValueError("Flight recording currently requires a SQLite or PostgreSQL backend")

    async def _make_agent(self):
        if self._provided_agent is not None:
            if self._provided_agent._closed:
                raise RuntimeError(
                    "Rebind a fresh agent before resuming a closed inline instance"
                )
            return self._provided_agent
        if self._named_binding:
            return self.registry.create(self._ref)
        agent = Agent(config=self._config, **self._options)
        try:
            if self._registrar:
                self._registrar(agent.tools)
        except BaseException:
            await agent.aclose()
            raise
        return agent

    async def start(self, session_id=None):
        if self._session_id:
            raise RuntimeError(
                "Runtime already started; use another runtime for a new session"
            )
        await self.backend.initialize()
        self._session_id = session_id or str(uuid.uuid4())
        try:
            if getattr(self.backend, "remote", False):
                await self.backend.create_session(
                    self._session_id, wire(self._ref), runtime_config=wire(self.runtime_config),
                )
            else:
                self._agent = await self._make_agent()
                self._agent._session_id = self._session_id
                self._agent._artifact_store = self.backend
                await self.backend.create_session(self._session_id, {
                    "version": 1, "agent": wire(self._ref), "config": wire(self._agent.config),
                    "recording": self.recording,
                })
        except BaseException:
            self._session_id = ""
            self._state = RuntimeState.ERROR
            try:
                if self._agent is not None and self._agent is not self._provided_agent:
                    await self._agent.aclose()
            finally:
                self._agent = None
                if self._owns_backend:
                    await self.backend.aclose()
            raise
        self._started_at = time.time()
        self._state = RuntimeState.RUNNING
        return self._session_id

    async def submit(self, message: str, *, request_id=None, stream=True):
        if not self._session_id:
            raise RuntimeError("Call start() or resume() first")
        if self._state != RuntimeState.RUNNING:
            raise RuntimeError("Runtime is not running")
        request_id = request_id or str(uuid.uuid4())
        if getattr(self.backend, "remote", False):
            return await self.backend.submit(
                self._session_id, message, request_id, stream
            )
        existing = await self.backend.find_request(self._session_id, request_id)
        if existing:
            if existing["message"] != message:
                raise ValueError("Request ID already used with different input")
            task = (
                self._task
                if self._task and self._active_run_id == existing["run_id"]
                else None
            )
            return RunHandle(self.backend, self._session_id, existing["run_id"], task)
        if self._task and not self._task.done():
            raise SessionBusyError("Session already executing")
        lease = await self.backend.claim(self._session_id)
        try:
            latest = await self.backend.latest_run(self._session_id)
            if latest:
                await restore(self._agent, latest)
            state = new_state(self._agent, message, durable=True)
            state["stream"] = stream
            state["config"] = wire(self._agent.config)
            state["recording"] = self.recording
            state, created = await self.backend.create_run(
                self._session_id, request_id, state, lease
            )
            if not created:
                await self.backend.release(self._session_id, lease)
                return RunHandle(self.backend, self._session_id, state["run_id"])
            if self.recording:
                await self.backend.save_run(state, [], lease, records=[journal_record(
                    "run.started", {
                        "message": message, "initial_messages": state["messages"],
                        "agent": wire(self._ref), "config": state["config"],
                    }, state,
                )])
            else:
                await self.backend.save_run(state, [], lease)
            self._active_run_id = state["run_id"]
            self._task = asyncio.create_task(self._drive(state, lease))
            return RunHandle(
                self.backend, self._session_id, state["run_id"], self._task
            )
        except BaseException:
            await self.backend.release(self._session_id, lease)
            raise

    async def _drive(self, state, lease):
        lost = None
        driver = asyncio.current_task()

        async def heartbeat():
            nonlocal lost
            try:
                renew_at = asyncio.get_running_loop().time()
                while True:
                    await asyncio.sleep(
                        min(1, getattr(self.backend, "lease_seconds", 30) / 3)
                    )
                    if await self.backend.get_control(self._session_id) == "cancelled":
                        driver.cancel()
                        return
                    now = asyncio.get_running_loop().time()
                    if now >= renew_at:
                        await self.backend.renew(self._session_id, lease)
                        renew_at = now + getattr(self.backend, "lease_seconds", 30) / 3
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                lost = exc
                driver.cancel()

        if self._agent._busy:
            raise SessionBusyError("Agent is already executing")
        self._agent._busy = True
        heart = asyncio.create_task(heartbeat())

        async def persist(state, events, *, records=()):
            if lost:
                raise lost
            if not records and events and all(
                e.type in (RunEventType.TEXT_DELTA, RunEventType.THINKING_DELTA)
                for e in events
            ):
                await self.backend.append_events(state, events, lease)
            else:
                if records:
                    await self.backend.save_run(state, events, lease, records=records)
                else:
                    await self.backend.save_run(state, events, lease)

        async def discard(event):
            pass

        try:
            limit = self.runtime_config.max_session_duration_seconds
            remaining = (
                max(0, limit - (time.time() - self._started_at)) if limit else None
            )
            async with asyncio.timeout(remaining):
                return await drive(
                    self._agent,
                    state,
                    discard,
                    commit=persist,
                    control=lambda: self.backend.get_control(self._session_id),
                )
        except TimeoutError:
            state = await self.backend.get_run(state["run_id"])
            state.update(
                stop_reason="timeout",
                error={
                    "type": "TimeoutError",
                    "message": "Session duration limit exceeded",
                },
            )
            result = result_from_state(state)
            await persist(
                state,
                [
                    RunEvent(
                        RunEventType.RUN_RESULT,
                        result,
                        self._session_id,
                        state["run_id"],
                    )
                ],
            )
            return result
        finally:
            self._agent._busy = False
            heart.cancel()
            await asyncio.gather(heart, return_exceptions=True)
            await self.backend.release(self._session_id, lease)

    async def execute(self, message, *, request_id=None) -> RunResult:
        return await (
            await self.submit(message, request_id=request_id, stream=False)
        ).result()

    async def export_incident(
        self, run_id: str, *, destination: str | Path,
        policy: ExportPolicy | None = None, limits: BundleLimits | None = None,
    ) -> Path:
        """Export a bound session's recording; payloads are omitted by default.

        Use ExportPolicy(include_payloads=True) only for an authorized destination.
        The artifact policy is separate. This method does not execute the incident.
        """
        if not self._session_id:
            raise RuntimeError("Call start() or resume() before exporting")
        if not getattr(self.backend, "supports_recording", False):
            raise ValueError("This backend does not support flight-recorder export")
        from .recorder import export_incident

        return await export_incident(
            self.backend, self._session_id, run_id, destination,
            policy=policy, limits=limits,
        )

    def execute_stream(self, message, *, request_id=None):
        async def run(emit):
            handle = await self.submit(message, request_id=request_id, stream=True)
            async for event in handle.events():
                await emit(event)
            return await handle.result()

        return RunStream(run)

    async def resume(self, session_id):
        if self._task and not self._task.done():
            raise SessionBusyError("Runtime already executing")
        await self.backend.initialize()
        session = await self.backend.get_session(session_id)
        if session["agent"] != wire(self._ref):
            raise ValueError("Agent definition version does not match saved session")
        saved_recording = session.get("recording", False)
        if self._recording_requested is not None and self._recording_requested != saved_recording:
            raise ValueError("Recording configuration differs from persisted session")
        if saved_recording and not getattr(self.backend, "supports_recording", False):
            raise ValueError("This backend cannot resume recorded execution")
        self.recording = saved_recording
        if getattr(self.backend, "remote", False):
            self._session_id = session_id
            self._started_at = time.time()
            self._state = RuntimeState.RUNNING
            return await self.backend.resume(session_id)
        self._config = AgentConfig(**session["config"])
        lease = await self.backend.claim(session_id)
        previous = self._agent
        candidate = None
        try:
            candidate = await self._make_agent()
            if wire(candidate.config) != wire(self._config):
                raise ValueError("Agent configuration differs from persisted session")
            candidate._session_id = session_id
            candidate._artifact_store = self.backend
            state = await self.backend.latest_run(session_id)
            if state:
                await restore(candidate, state)
            if previous is not None and previous is not candidate:
                await previous.aclose()
            self._agent = candidate
            self._session_id = session_id
            self._started_at = time.time()
            self._state = RuntimeState.RUNNING
            await self.backend.set_control(session_id, "")
            if state:
                if state["status"] not in ("completed", "cancelled"):
                    previous_status = state["status"]
                    state.update(status="running", error=None)
                    if state.get("recording"):
                        await self.backend.save_run(state, [], lease, records=[journal_record(
                            "run.resumed", {"previous_status": previous_status}, state,
                        )])
                    else:
                        await self.backend.save_run(state, [], lease)
                    self._active_run_id = state["run_id"]
                    self._task = asyncio.create_task(self._drive(state, lease))
                    return RunHandle(
                        self.backend, session_id, state["run_id"], self._task
                    )
            await self.backend.release(session_id, lease)
            return None
        except BaseException:
            await self.backend.release(session_id, lease)
            if candidate is not None and candidate is not previous and candidate is not self._provided_agent:
                await candidate.aclose()
            self._agent = previous
            raise

    async def _resolve(
        self, execution_key, *, allow=None, result=None, retry=False, abort=False
    ):
        if getattr(self.backend, "remote", False):
            return await self.backend.resolve(
                self._session_id,
                execution_key,
                allow=allow,
                result=wire(result),
                retry=retry,
                abort=abort,
            )
        lease = await self.backend.claim(self._session_id)
        try:
            state = await self.backend.latest_run(self._session_id)
            if not state or state["status"] not in (
                "awaiting_input",
                "paused",
                "failed",
                "cancelled",
            ):
                raise RuntimeError("Run is not awaiting a resolution")
            entry = next(
                t for t in state["tools"] if t["execution_key"] == execution_key
            )
            resolve_entry(
                state, entry, allow=allow, result=wire(result), retry=retry, abort=abort
            )
            if state.get("recording"):
                await self.backend.save_run(state, [], lease, records=[journal_record(
                    "tool.resolved", {
                        "allow": allow, "result": wire(result), "retry": retry,
                        "abort": abort, "status": entry["status"],
                    }, state, entry,
                )])
            else:
                await self.backend.save_run(state, [], lease)
        finally:
            await self.backend.release(self._session_id, lease)

    async def approve(self, execution_key, allow=True):
        await self._resolve(execution_key, allow=allow)

    async def resolve_tool(
        self, execution_key, *, result=None, retry=False, abort=False
    ):
        await self._resolve(execution_key, result=result, retry=retry, abort=abort)

    async def pause(self):
        await self.backend.set_control(self._session_id, "paused")
        if self._task:
            await asyncio.shield(self._task)
        self._state = RuntimeState.PAUSED
        return self._session_id

    async def cancel(self):
        await self.backend.set_control(self._session_id, "cancelled")
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        elif not getattr(self.backend, "remote", False):
            lease = await self.backend.claim(self._session_id)
            try:
                state = await self.backend.latest_run(self._session_id)
                if state and state["status"] not in ("completed", "cancelled"):
                    state = cancel_state(state)
                    from .engine import complete_turn
                    await restore(self._agent, state)
                    await complete_turn(self._agent, state)
                    await self.backend.save_run(
                        state,
                        [
                            RunEvent(
                                RunEventType.RUN_RESULT,
                                result_from_state(state),
                                self._session_id,
                                state["run_id"],
                            )
                        ],
                        lease,
                    )
            finally:
                await self.backend.release(self._session_id, lease)

    async def checkpoint(self):
        if self._task and not self._task.done():
            raise SessionBusyError("Execution already checkpoints each boundary")
        return await self.backend.latest_run(self._session_id)

    async def get_status(self):
        if getattr(self.backend, "remote", False):
            status = await self.backend.get_status(self._session_id)
            status["state"] = self._state.value
            return status
        state = await self.backend.latest_run(self._session_id)
        return {
            "session_id": self._session_id,
            "state": self._state.value,
            "run": result_from_state(state) if state else None,
        }

    async def stop(self):
        if self._state == RuntimeState.STOPPED:
            return self._last_status
        try:
            if self._session_id:
                if getattr(self.backend, "remote", False):
                    await self.backend.set_control(self._session_id, "paused")
                if self._task and not self._task.done():
                    await self.pause()
                self._last_status = await self.get_status()
        finally:
            self._state = RuntimeState.STOPPED
            self._last_status = {**(self._last_status or {"session_id": self._session_id, "run": None}), "state": "stopped"}
            try:
                if self._agent:
                    await self._agent.aclose()
            finally:
                if self._owns_backend:
                    await self.backend.aclose()
        return self._last_status

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()

    @property
    def agent(self):
        return self._agent

    @property
    def session_id(self):
        return self._session_id

    @property
    def state(self):
        return self._state


def resolve_entry(state, entry, *, allow=None, result=None, retry=False, abort=False):
    from .types import ToolResult

    if allow is not None:
        if entry["status"] != "approval":
            raise ValueError("Tool is not awaiting approval")
        if allow:
            entry.update(status="prepared", approved=True)
        else:
            entry.update(
                status="completed",
                result=wire(ToolResult(entry["call"]["id"], "Permission denied", True)),
            )
    else:
        if sum([result is not None, retry, abort]) != 1:
            raise ValueError("Choose exactly one of result, retry, abort")
        if entry["status"] not in ("uncertain", "started"):
            raise ValueError("Tool has no uncertain outcome")
        if abort:
            cancel_state(state)
        elif retry:
            if state["status"] == "cancelled":
                raise ValueError(
                    "Cancelled runs cannot be retried; supply a verified outcome or start a new turn"
                )
            entry["status"] = "prepared"
        else:
            if result["tool_call_id"] != entry["call"]["id"]:
                raise ValueError("Resolution must match tool call ID")
            entry.update(status="raw_completed", raw_result=result)
