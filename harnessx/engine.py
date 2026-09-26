"""Command implementations and the local driver for the shared state machine."""

from __future__ import annotations

import asyncio
import copy
import uuid
from dataclasses import asdict

from .execution import (
    RunEvent,
    RunEventType as E,
    ToolExecutionContext,
    _tool_context,
    ToolApprovalRequired,
    cancel_state,
    next_command,
    result_from_state,
    transition,
    wire,
)
from .extensions.base import ExtensionContext, complete_extensions
from .hooks import HookContext, HookEvent, Middleware
from .types import PermissionLevel, ProviderResponse, TokenUsage, ToolCall, ToolResult
from ._journal import RecordingError, record as journal_record


async def emit_hook(agent, event, **data):
    await agent.hooks.emit(event, HookContext(event=event, agent=agent, data=data))


async def complete_turn(agent, state):
    """Run terminal observers on local drivers and remote activities alike."""
    result = result_from_state(state)
    if result.status in ("completed", "failed", "cancelled"):
        await complete_extensions(agent, result)
        await emit_hook(agent, HookEvent.AGENT_END, result=result)


async def snapshot(agent) -> dict:
    extensions = {}
    for ext in agent.extensions:
        extensions[ext.name] = await ext.on_save_session(ExtensionContext(agent, ext.name))
    from .artifacts import capture

    return await capture(
        agent,
        wire(
            {
                "messages": agent.memory.get_messages(),
                "metadata": agent.session_metadata,
                "extensions": extensions,
                "total_usage": agent.guardrails.total_usage,
                "lifetime_iterations": agent.guardrails.lifetime_iterations,
            }
        ),
    )


async def restore(agent, state):
    from .artifacts import materialize

    state = await materialize(agent, state)
    agent._session_id = state["session_id"]
    agent.memory.set_messages(copy.deepcopy(state.get("messages", [])))
    agent.session_metadata = copy.deepcopy(state.get("metadata", {}))
    agent.guardrails._total_usage = TokenUsage(**state.get("total_usage", {}))
    agent.guardrails._iteration_count = state.get("iterations", 0)
    agent.guardrails._lifetime_iterations = state.get("lifetime_iterations", 0)
    for ext in agent.extensions:
        if ext.name in state.get("extensions", {}):
            await ext.on_load_session(
                ExtensionContext(agent, ext.name), state["extensions"][ext.name]
            )


def new_state(agent, message, *, run_id=None, durable=False):
    return {
        "version": 1,
        "session_id": agent.session_id,
        "run_id": run_id or str(uuid.uuid4()),
        "message": message,
        "messages": copy.deepcopy(agent.memory.get_messages()),
        "metadata": copy.deepcopy(agent.session_metadata),
        "phase": "start",
        "status": "running",
        "durable": durable,
        "iterations": 0,
        "usage": asdict(TokenUsage()),
        "total_usage": asdict(agent.guardrails.total_usage),
        "lifetime_iterations": agent.guardrails.lifetime_iterations,
        "tools": [],
        "attempt": 0,
        "usage_incomplete": False,
    }


def response_payload(response):
    """Provider-neutral fields only; never serialize SDK transports or credentials."""
    return wire({
        "text": response.text,
        "thinking": response.thinking,
        "tool_calls": response.tool_calls,
        "stop_reason": response.stop_reason,
        "usage": response.usage,
    })


async def command(agent, state, name, emit, *, record=None):
    """Perform one command; returning its snapshot does not commit it."""

    async def event(kind, data=None):
        await emit(
            RunEvent(
                kind,
                data,
                state["session_id"],
                state["run_id"],
                str(state.get("iterations", 0)),
                str(state.get("attempt", 0)),
            )
        )

    if name == "complete":
        await complete_turn(agent, state)
        return {}

    if name == "start":
        agent.guardrails.reset_turn()
        agent.memory.add_user_message(state["message"])
        for ext in agent.extensions:
            await ext.on_turn_start(ExtensionContext(agent, ext.name), state["message"])
        await emit_hook(agent, HookEvent.AGENT_START, message=state["message"])
        return {**await snapshot(agent), "phase": "prepare_model"}

    if name == "prepare_model":
        agent.guardrails.check_iteration_limit()
        agent.guardrails.check_cost_limit()
        agent.guardrails.record_iteration()
        await emit_hook(
            agent,
            HookEvent.LOOP_ITERATION_START,
            iteration=agent.guardrails.iteration_count,
        )
        tools = agent.tools.get_tool_params()
        system = agent._build_system_prompt()
        await agent.memory.trim_if_needed(
            agent.provider, agent.config.model, system, tools,
            max_context_tokens=agent.config.max_context_tokens,
        )
        messages, tools = await agent.middleware.process_llm_request(
            agent.memory.get_messages(), tools
        )
        return {
            **await snapshot(agent),
            "iterations": agent.guardrails.iteration_count,
            "request": wire(
                dict(
                    model=agent.config.model,
                    messages=messages,
                    system=system or None,
                    tools=tools,
                    max_tokens=agent.config.max_tokens,
                    temperature=agent.config.temperature,
                )
            ),
            "attempt": 0,
        }

    if name == "model":
        from .artifacts import materialize

        request = (await materialize(agent, state))["request"]
        if record:
            await record("model.started", state["request"])
        await emit_hook(
            agent,
            HookEvent.LLM_REQUEST,
            message_count=len(request["messages"]),
            tool_count=len(request["tools"]),
        )
        buffered = any(
            type(m).after_llm_call is not Middleware.after_llm_call
            for m in agent.middleware._middleware
        )
        response = None
        async with asyncio.timeout(agent.config.model_timeout_seconds):
            if state.get("stream", False):
                async for chunk in agent.provider.stream(**request):
                    if chunk.kind == "response":
                        response = chunk.data
                    elif not buffered and chunk.kind in (
                        "text_delta",
                        "thinking_delta",
                    ):
                        await event(E(chunk.kind), chunk.data)
                if response is None:
                    raise RuntimeError(
                        "Provider stream ended without a completed response"
                    )
            else:
                response = await agent.provider.create(**request)
        if record:
            # Copy before middleware can mutate the same ProviderResponse in place.
            await record("model.response", response_payload(response))
        response = await agent.middleware.process_llm_response(response)
        # Canonical fields after middleware are the single source for memory and output.
        canonical = ProviderResponse(
            text=getattr(response, "text", ""),
            tool_calls=getattr(response, "tool_calls", []),
            content=getattr(response, "content", [])
            if not hasattr(response, "text")
            else [],
            thinking=getattr(response, "thinking", None),
            stop_reason=response.stop_reason,
            usage=response.usage,
            _content_metadata=getattr(response, "_content_metadata", {}),
            _content_dicts=getattr(response, "_content_dicts", False),
        )
        if record:
            await record("model.processed", response_payload(canonical))
        agent.guardrails.track_usage(canonical.usage)
        agent.memory.add_assistant_message(canonical.content)
        await emit_hook(
            agent,
            HookEvent.LLM_RESPONSE,
            response=canonical,
            stop_reason=canonical.stop_reason,
        )
        if buffered and canonical.text:
            await event(E.TEXT_DELTA, canonical.text)
        if canonical.text:
            await event(E.TEXT_COMPLETE, canonical.text)
        usage = {
            k: state["usage"].get(k, 0) + v for k, v in asdict(canonical.usage).items()
        }
        return {
            **await snapshot(agent),
            "usage": usage,
            "response": wire(
                {
                    "text": canonical.text,
                    "thinking": canonical.thinking,
                    "tool_calls": canonical.tool_calls,
                    "stop_reason": canonical.stop_reason,
                    "usage": canonical.usage,
                }
            ),
        }

    if name == "prepare_tools":
        entries = []
        for raw in state["response"]["tool_calls"]:
            call = await agent.middleware.process_tool_call(ToolCall(**raw))
            if call.id != raw["id"]:
                raise ValueError("Tool middleware must preserve call identity")
            entry = {
                "call": wire(call),
                "status": "prepared",
                "attempt": 0,
                "execution_key": f"{state['run_id']}:{state['iterations']}:{call.id}",
                "policy": "manual",
                "concurrent": False,
                "timeout": 300.0,
            }
            try:
                definition = agent.tools.get_tool(call.name)
                from jsonschema import validate

                validate(call.input, definition.input_schema)
                entry.update(
                    policy=definition.replay_policy,
                    concurrent=definition.concurrent,
                    timeout=definition.timeout_seconds,
                )
                level = agent.permissions.get_effective_permission(
                    call.name, definition
                )
                if level == PermissionLevel.DENY:
                    allowed = False
                elif state["durable"] and level == PermissionLevel.ASK:
                    entry["status"] = "approval"
                    entries.append(entry)
                    continue
                else:
                    allowed = await agent.permissions.check_permission(call, definition)
                if not allowed:
                    entry.update(
                        status="completed",
                        result=wire(ToolResult(call.id, "Permission denied", True)),
                    )
            except Exception as exc:
                # Invalid/unknown tools become model-visible outcomes, not dispatches.
                entry.update(
                    status="completed", result=wire(ToolResult(call.id, str(exc), True))
                )
            entries.append(entry)
            if entry["status"] == "completed":
                await event(E.TOOL_CALL_START, call)
                await event(E.TOOL_CALL_COMPLETE, call)
                await event(E.TOOL_RESULT, ToolResult(**entry["result"]))
        if len({t["execution_key"] for t in entries}) != len(entries):
            raise ValueError("Duplicate tool call IDs in a model response")
        return {**await snapshot(agent), "tools": wire(entries)}

    if name == "collect_tools":
        results = [ToolResult(**t["result"]) for t in state["tools"]]
        agent.memory.add_tool_results(results)
        await emit_hook(
            agent, HookEvent.LOOP_ITERATION_END, iteration=state["iterations"]
        )
        return {**await snapshot(agent), "tools": []}

    if name == "finish":
        for ext in agent.extensions:
            await ext.on_turn_end(ExtensionContext(agent, ext.name), state["output"])
        await emit_hook(
            agent, HookEvent.LOOP_ITERATION_END, iteration=state["iterations"]
        )
        return await snapshot(agent)
    raise ValueError(f"Unknown engine command: {name}")


async def execute_tool(agent, state, entry, emit, *, record=None):
    from .artifacts import materialize

    local = await materialize(
        agent, {"session_id": state["session_id"], "call": entry["call"]}
    )
    call = ToolCall(**copy.deepcopy(local["call"]))
    definition = agent.tools.get_tool(call.name)
    # Re-check current policy immediately before dispatch, including after resume.
    if (
        agent.permissions.get_effective_permission(call.name, definition)
        == PermissionLevel.DENY
    ):
        return wire(ToolResult(call.id, "Permission denied", True))
    level = agent.permissions.get_effective_permission(call.name, definition)
    if state["durable"] and level == PermissionLevel.ASK and not entry.get("approved"):
        raise ToolApprovalRequired("Current policy requires approval")
    from jsonschema import validate

    validate(call.input, definition.input_schema)
    context = ToolExecutionContext(
        state["session_id"], state["run_id"], entry["execution_key"], entry["attempt"]
    )
    token = _tool_context.set(context)
    try:
        if record:
            await record("tool.dispatched", {"call": entry["call"]}, entry)
        await emit_hook(agent, HookEvent.TOOL_CALL_START, tool_call=copy.deepcopy(call))
        async with asyncio.timeout(entry["timeout"]):
            result = await agent.tools._dispatch(call, definition)
        # Commit the raw result before fallible middleware/extension work.
        return wire(result)
    finally:
        _tool_context.reset(token)


async def finish_tool(agent, entry):
    from .artifacts import materialize

    local = await materialize(agent, {"raw_result": entry["raw_result"]})
    result = await agent.middleware.process_tool_result(
        ToolResult(**local["raw_result"])
    )
    if not isinstance(result, ToolResult) or result.tool_call_id != entry["call"]["id"]:
        raise ValueError("Tool result middleware must preserve result identity")
    await emit_hook(
        agent,
        HookEvent.TOOL_CALL_END,
        tool_call=ToolCall(**entry["call"]),
        result=result,
    )
    return wire(result)


async def drive(agent, state, emit, *, commit=None, control=None):
    """Commit each transition and each concurrent tool outcome before advancement."""
    lock = asyncio.Lock()
    pending_records = []

    def records(kind, payload, entry=None):
        if not state.get("recording"):
            return []
        from .artifacts import replace_paths

        payload = replace_paths(wire(payload), getattr(agent, "_artifact_uris", {}))
        return [journal_record(kind, payload, state, entry)]

    async def persist(events=(), *, journal=()):
        # Retain IDs across an ambiguous commit failure; SQL insertion is idempotent.
        pending_records.extend(journal)
        try:
            if commit:
                if pending_records:
                    await commit(state, events, records=list(pending_records))
                else:
                    await commit(state, events)
            elif state.get("recording"):
                raise RecordingError("Flight recording requires a durable runtime")
        except Exception as exc:
            if state.get("recording") and not isinstance(exc, RecordingError):
                raise RecordingError("Required flight-recording commit failed") from exc
            raise
        pending_records.clear()
        for ev in events:
            await emit(ev)

    async def record(kind, payload, entry=None):
        if state.get("recording"):
            async with lock:
                await persist(journal=records(kind, payload, entry))

    recorder = record if state.get("recording") else None

    def ev(kind, data=None, entry=None):
        return RunEvent(
            kind,
            copy.deepcopy(data),
            state["session_id"],
            state["run_id"],
            str(state["iterations"]),
            str(entry["attempt"] if entry else state["attempt"]),
        )

    async def run_tool(entry):
        async with lock:
            if entry["status"] == "started":
                if entry["policy"] == "manual":
                    entry["status"] = "uncertain"
                    await persist([ev(E.RECOVERY_REQUIRED, entry)])
                    return
            if entry["status"] in ("prepared", "started"):
                entry["status"] = "started"
                entry["attempt"] += 1
                await persist(
                    [ev(E.TOOL_CALL_START, ToolCall(**entry["call"]), entry)],
                    journal=records("tool.started", {
                        "call": entry["call"], "replay_policy": entry["policy"],
                    }, entry),
                )
        if entry["status"] == "started":
            while True:
                try:
                    raw = await execute_tool(agent, state, entry, emit, record=recorder)
                    break
                except ToolApprovalRequired:
                    async with lock:
                        entry["status"] = "approval"
                        await persist([ev(E.APPROVAL_REQUIRED, entry)])
                    return
                except asyncio.CancelledError:
                    async with lock:
                        entry["status"] = (
                            "uncertain" if entry["policy"] == "manual" else "started"
                        )
                        await persist(journal=records("tool.interrupted", {
                            "outcome": "unknown", "status": entry["status"],
                        }, entry))
                    raise
                except Exception as exc:
                    async with lock:
                        failure = records("tool.failed", {
                            "type": type(exc).__name__, "message": str(exc),
                            "outcome": "unknown",
                        }, entry)
                        if entry["policy"] == "manual":
                            entry["status"] = "uncertain"
                            await persist([ev(E.RECOVERY_REQUIRED, entry)], journal=failure)
                            return
                        if not retryable(exc) or entry["attempt"] >= 3:
                            await persist(journal=failure)
                            raise
                        await persist(journal=failure)
                        entry["attempt"] += 1
                        await persist(journal=records("tool.started", {
                            "call": entry["call"], "replay_policy": entry["policy"],
                        }, entry))
                    await asyncio.sleep(min(entry["attempt"] - 1, 2))
            async with lock:
                if state.get("recording"):
                    from .artifacts import capture

                    raw = (await capture(agent, {"result": raw}))["result"]
                entry.update(status="raw_completed", raw_result=raw)
                await persist(journal=records("tool.returned", raw, entry))
        if entry["status"] == "raw_completed":
            processed = await finish_tool(agent, entry)
            async with lock:
                entry.update(status="completed", result=processed)
                state.update(await snapshot(agent))
                await persist(
                    [
                        ev(E.TOOL_CALL_COMPLETE, ToolCall(**entry["call"]), entry),
                        ev(E.TOOL_RESULT, ToolResult(**processed), entry),
                    ],
                    journal=records("tool.processed", processed, entry),
                )

    await restore(agent, state)
    try:
        while next_command(state) != "result":
            if control:
                requested = await control()
                if requested in ("paused", "cancelled"):
                    state["status"] = requested
                    break
            name = next_command(state)
            if name == "tools":
                pending = [
                    t
                    for t in state["tools"]
                    if t["status"] in ("approval", "uncertain")
                ]
                if pending:
                    state["status"] = "awaiting_input"
                    await persist(
                        [
                            ev(
                                E.APPROVAL_REQUIRED
                                if t["status"] == "approval"
                                else E.RECOVERY_REQUIRED,
                                t,
                            )
                            for t in pending
                        ]
                    )
                    break
                runnable = [t for t in state["tools"] if t["status"] != "completed"]
                if runnable and all(t["concurrent"] for t in runnable):
                    tasks = [asyncio.create_task(run_tool(t)) for t in runnable]
                    try:
                        await asyncio.gather(*tasks)
                    except BaseException:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise
                else:
                    for entry in runnable:
                        await run_tool(entry)
                if any(
                    t["status"] in ("approval", "uncertain") for t in state["tools"]
                ):
                    state["status"] = "awaiting_input"
                    break
                name = "collect_tools"
            if name == "model":
                if state["attempt"]:
                    state["usage_incomplete"] = True
                    await persist([ev(E.ATTEMPT_RESET, {"attempt": state["attempt"]})])
                state["attempt"] += 1
                await persist()
            buffered_events = []

            async def command_emit(event):
                if event.type in (E.TEXT_DELTA, E.THINKING_DELTA):
                    await persist([event])
                else:
                    buffered_events.append(event)

            from .artifacts import capture

            while True:
                try:
                    outcome = await command(agent, state, name, command_emit, record=recorder)
                    break
                except Exception as exc:
                    if name == "model":
                        await record("model.failed", {
                            "type": type(exc).__name__, "message": str(exc),
                        })
                    if name != "model" or not retryable(exc) or state["attempt"] >= 3:
                        raise
                    state["usage_incomplete"] = True
                    buffered_events.clear()
                    await persist([ev(E.ATTEMPT_RESET, {"attempt": state["attempt"]})])
                    state["attempt"] += 1
                    await persist()
                    await asyncio.sleep(min(state["attempt"] - 1, 2))
            outcome = await capture(agent, outcome)
            state = transition(state, name, outcome)
            await persist(buffered_events, journal=(
                records("model.completed", outcome["response"]) if name == "model" else []
            ))
    except asyncio.CancelledError:
        if state["phase"] == "model":
            await record("model.interrupted", {"outcome": "incomplete"})
        state = cancel_state(state)
        agent.memory.set_messages(copy.deepcopy(state.get("messages", [])))
        await complete_turn(agent, state)
        await persist([ev(E.RUN_RESULT, result_from_state(state))])
        raise
    except Exception as exc:
        if state["phase"] == "model":
            state["usage_incomplete"] = True
        state.update(
            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
        )
        await emit_hook(agent, HookEvent.ERROR, error=str(exc))
        await persist([ev(E.ERROR, str(exc))], journal=records("command.failed", {
            "phase": state["phase"], "type": type(exc).__name__, "message": str(exc),
        }))
    if state["status"] == "cancelled":
        state = cancel_state(state)
        agent.memory.set_messages(copy.deepcopy(state.get("messages", [])))
    result = result_from_state(state)
    await complete_turn(agent, state)
    events = [ev(E.RUN_RESULT, result)]
    if result.status == "completed":
        events.insert(0, ev(E.TURN_COMPLETE, result.stop_reason))
    await persist(events)
    return result


def retryable(exc):
    """Only transport failures, throttling and server failures are auto-retried."""
    status = getattr(exc, "status_code", None)
    return (
        isinstance(exc, (TimeoutError, ConnectionError, OSError))
        or status == 429
        or (isinstance(status, int) and status >= 500)
    )
