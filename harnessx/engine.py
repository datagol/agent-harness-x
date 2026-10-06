"""Command implementations and the local driver for the shared state machine."""

from __future__ import annotations

import asyncio
import copy
import logging
import uuid
from dataclasses import asdict, replace


from .execution import (
    RunEvent,
    RunEventType as E,
    ToolExecutionContext,
    _agent_context,
    _tool_context,
    ToolApprovalRequired,
    cancel_state,
    close_open_tool_calls,
    next_command,
    result_from_state,
    transition,
    wire,
)
from .errors import TransientToolError
from .permissions import MaxIterationsError
from .extensions.base import ExtensionContext, complete_extensions
from .hooks import HookContext, HookEvent, Middleware
from .models import (
    CONTEXT_SAFETY_TOKENS,
    context_window_from_error,
    default_reply_budget,
    learn_model_limits,
    provider_reply_default,
    reply_limit_from_error,
    resolve_model_limits,
)
from .providers.base import INVALID_ARGUMENTS, native_continuation
from .providers.retry import is_context_overflow, is_transient, retry_after_seconds
from .prompt_cache import accepts_cache, build_hint, hint_from_wire
from .tools import retry_wanted
from .types import DEFAULT_TIMEOUT_SECONDS, PermissionLevel, ProviderResponse, TokenUsage, ToolCall, ToolResult, ToolRetry
from ._journal import RecordingError, record as journal_record

logger = logging.getLogger(__name__)

# How many times one turn may condense its history. A summary that fails to
# shrink the history enough would otherwise re-trigger every iteration, which is
# the compaction loop OpenCode has open as issue 15533.
MAX_CONDENSATIONS = 3

# How many times one run may condense its history because the provider refused
# a request as too long. One condensation usually makes room; a second covers a
# count that was badly off. Past that the request cannot be made to fit.
MAX_OVERFLOW_RECOVERIES = 2

# How many times one run nudges a model that replied with nothing at all.
MAX_EMPTY_NUDGES = 2

# How many times one run resumes a turn the server paused (``pause_turn``).
MAX_PAUSE_RESUMES = 5

# The reply budget for a history summary: enough for a dense record, never the
# whole reply budget of a model that can write 128K tokens.
SUMMARY_MAX_TOKENS = 8_000

# What the model reads when the loop recovers instead of stopping. Each names
# what happened and what to do, the way a tool error does.
TRUNCATED_CALL_NOTICE = (
    "Not executed: your reply was cut off at the output token limit while writing "
    "this call, so its arguments are incomplete. The limit has been raised; call "
    "the tool again with complete arguments. If the arguments are very large, "
    "split the work into several smaller calls."
)
CONTINUE_NOTICE = (
    "Your reply was cut off at the output token limit. Continue exactly where it "
    "stopped, without repeating anything already written."
)
FINAL_ANSWER_NOTICE = (
    "You have reached the step limit for this task. Do not call any more tools. "
    "Reply now with what you have done, what remains unfinished, and your best "
    "answer from what you have so far."
)
EMPTY_NOTICE = (
    "Your last reply was empty. Continue with the task, or give your final answer."
)

# Asking for the summary. Deliberately told to preserve decisions and open work
# rather than narrate, because the model reads this instead of the transcript.
SUMMARY_SYSTEM = (
    "You condense an agent's conversation so it can keep working with less context. "
    "Write a dense factual record, not a narrative."
)
SUMMARY_INSTRUCTION = (
    "Condense the conversation above into a record the agent can continue from.\n\n"
    "Preserve, in this order and only where present:\n"
    "- The task as it currently stands, including any revisions to it.\n"
    "- Decisions made and the reasons, especially ones that constrain what comes next.\n"
    "- Facts established: file paths, identifiers, values, command results.\n"
    "- Work completed, and work still outstanding.\n"
    "- Anything that failed, and what was learned from it.\n\n"
    "Omit pleasantries and restatements. Do not invent anything not present above. "
    "Do not address the reader or offer help."
)


async def await_with_notices(awaitable, policy, event, *, on, name=None):
    """Await something slow, saying so rather than going quiet.

    The wait runs in the calling task: a side task emitting into the same run
    stream would interleave with the driver's own writes, and the durable
    backends persist events in order. Cancellation joins the owned operation
    before returning, so the next attempt cannot overlap it.
    """
    task = asyncio.ensure_future(awaitable)
    try:
        if policy is None or not policy.enabled:
            return await task
        waited = 0.0
        delay = policy.first_after_seconds
        while True:
            done, _ = await asyncio.wait({task}, timeout=delay)
            if done:
                return await task
            waited += delay
            await event(E.WAITING, {"on": on, "seconds": round(waited, 1), **({"name": name} if name else {})})
            delay = policy.repeat_every_seconds
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def stream_with_notices(iterator, policy, event, *, on, idle=None):
    """The same, for a stream: the gap before the first chunk is the slow part.

    Between chunks the wait restarts, so a provider that stalls mid-answer is
    reported too. With ``idle`` seconds set, a gap that long abandons the
    stream with ``IncompleteStreamError``, which the driver retries: the
    whole-call timeout alone let a stalled stream hold a run for many minutes.
    """
    from .errors import IncompleteStreamError

    iterator = iterator.__aiter__()
    notices = policy is not None and policy.enabled
    try:
        while True:
            step = iterator.__anext__()
            if idle:
                step = asyncio.wait_for(step, idle)
            try:
                chunk = await (await_with_notices(step, policy, event, on=on) if notices else step)
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                if not idle:
                    raise
                raise IncompleteStreamError(f"The stream stalled: no data for {idle:g}s") from exc
            yield chunk
    finally:
        closer = getattr(iterator, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception:
                pass


async def model_limits(agent):
    """The agent's model limits, resolved through ``harnessx.models``."""
    return await resolve_model_limits(agent.provider, agent.config.model)


def context_window(agent, limits) -> int:
    """The window condensing works against: the configured one, else the model's."""
    return agent.config.limits.max_context_tokens or limits.context_window


def reply_budget(agent, state, limits) -> int:
    """The reply budget for the next call.

    A budget raised in this run -- after a truncation, or lowered to a limit the
    provider named -- wins for the rest of the run. Otherwise the configured
    ``max_tokens``, else the model's own limit capped at ``DEFAULT_OUTPUT_CAP``.
    """
    raised = state.get("reply_budget")
    if raised:
        return min(int(raised), limits.max_output)
    if agent.config.max_tokens:
        return min(int(agent.config.max_tokens), limits.max_output)
    return provider_reply_default(agent.provider, agent.config.model, limits)


def escalated_budget(current: int, limits) -> int:
    """Double the budget for the next attempt, up to what the model can write."""
    return max(current, min(current * 2, limits.max_output))


async def emit_hook(agent, event, **data):
    await agent.hooks.emit(event, HookContext(event=event, agent=agent, data=data))


async def complete_turn(agent, state):
    """Run terminal observers on local drivers and remote activities alike."""
    result = result_from_state(state)
    if result.status in ("completed", "failed", "cancelled"):
        await complete_extensions(agent, result)
        await emit_hook(agent, HookEvent.AGENT_END, result=result)


def provider_state(agent):
    export = getattr(agent.provider, "export_state", None)
    return export() if callable(export) else {"limits": getattr(agent.provider, "_harness_model_limits", {})}


def restore_provider_state(agent, saved):
    if saved is None:
        return
    loader = getattr(agent.provider, "restore_state", None)
    if callable(loader):
        loader(saved)
    else:
        agent.provider._harness_model_limits = copy.deepcopy(saved.get("limits", {}))


def request_target(agent, request):
    target = getattr(agent.provider, "request_target", None)
    if callable(target):
        provider, model = target(agent.config.model)
        return provider, model, agent.provider._last_request.get("max_tokens", request.get("max_tokens", 0))
    return agent.provider, agent.config.model, request.get("max_tokens", 0)


def set_reply_budget(agent, budget):
    setter = getattr(agent.provider, "set_reply_budget", None)
    if callable(setter):
        setter(budget)
        return {}
    return {"reply_budget": budget}


def sync_accounting(agent, state):
    """Known usage survives failed commands, child cancellation and retries."""
    state["provider_state"] = wire(provider_state(agent))
    state["total_usage"] = asdict(agent.guardrails.total_usage)
    state["estimated_cost"] = agent.guardrails.estimated_cost
    baseline = state.get("usage_base")
    if baseline is not None:
        state["usage"] = {k: v - baseline.get(k, 0) for k, v in state["total_usage"].items()}
    state["usage_incomplete"] = state.get("usage_incomplete", False) or agent.guardrails.usage_incomplete


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
                "estimated_cost": agent.guardrails.estimated_cost,
                "provider_state": provider_state(agent),
                "lifetime_iterations": agent.guardrails.lifetime_iterations,
                "todos": list(getattr(agent, "todos", []) or []),
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
    agent.guardrails._estimated_cost = state.get("estimated_cost", agent.guardrails.cost_of(agent.guardrails.total_usage))
    agent.guardrails.usage_incomplete = state.get("usage_incomplete", False)
    restore_provider_state(agent, state.get("provider_state"))
    agent.guardrails._iteration_count = state.get("iterations", 0)
    agent.guardrails._lifetime_iterations = state.get("lifetime_iterations", 0)
    agent.todos = copy.deepcopy(state.get("todos", []) or [])
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
        "usage_base": asdict(agent.guardrails.total_usage),
        "estimated_cost": agent.guardrails.estimated_cost,
        "provider_state": provider_state(agent),
        "total_usage": asdict(agent.guardrails.total_usage),
        "lifetime_iterations": agent.guardrails.lifetime_iterations,
        "tools": [],
        "attempt": 0,
        "usage_incomplete": False,
        "call_history": [],
        # Carried between runs in a session, the way messages are: a task list
        # that reset every turn would be no better than not having one.
        "todos": copy.deepcopy(getattr(agent, "todos", []) or []),
        # The list as last announced with TODOS_UPDATED, so a list carried over
        # from the previous run is not announced again as if it were new.
        "todos_announced": copy.deepcopy(getattr(agent, "todos", []) or []),
    }


def response_payload(response):
    """Provider-neutral fields only; never serialize SDK transports or credentials."""
    return wire({
        "text": response.text,
        "thinking": response.thinking,
        "tool_calls": response.tool_calls,
        "stop_reason": response.stop_reason,
        "usage": response.usage,
        **({"content": [ProviderResponse._block_dict(b) for b in response.content]}
           if getattr(response, "_content_blocks", None) is not None else {}),
    })


def protected_content(response):
    """Native payloads must survive middleware verbatim, including empty thinking."""
    blocks = [ProviderResponse._block_dict(b) for b in getattr(response, "content", [])]
    return [b for b in blocks if b.get("type") == "provider"
            or (b.get("type") == "thinking" and "signature" in b)]


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
        reset_cache = getattr(agent.tools, "reset_call_cache", None)
        if reset_cache is not None:
            reset_cache()  # deduped tool results are per run, never per session
        agent.memory.add_user_message(state["message"])
        for ext in agent.extensions:
            await ext.on_turn_start(ExtensionContext(agent, ext.name), state["message"])
        await emit_hook(agent, HookEvent.AGENT_START, message=state["message"])
        return {**await snapshot(agent), "phase": "prepare_model"}

    if name == "prepare_model":
        final_answer = False
        try:
            agent.guardrails.check_iteration_limit()
        except MaxIterationsError:
            # One last call for an answer instead of a failure with nothing to
            # show (Hermes and OpenCode both do this). Once only.
            if not agent.config.limits.final_answer_on_limit or state.get("provider_continuation"):
                raise
            final_answer = True
            if not state.get("final_answer"):  # not already asked (condensing can intervene)
                agent.memory.add_user_message(FINAL_ANSWER_NOTICE)
                await emit_hook(
                    agent, HookEvent.RECOVERY, reason="iteration_limit", attempt=1,
                    reply_budget=None,
                )
        agent.guardrails.check_cost_limit()
        agent.guardrails.record_iteration()
        await emit_hook(
            agent,
            HookEvent.LOOP_ITERATION_START,
            iteration=agent.guardrails.iteration_count,
        )
        tools = agent.tools.get_tool_params()
        system = agent._build_system_prompt()
        limits = await model_limits(agent)
        reply_tokens = reply_budget(agent, state, limits)
        if hasattr(agent.memory, "last_prompt_tokens"):
            agent.memory.last_prompt_tokens = None  # counted afresh below, or not at all
        # Condensing costs a model call, so it is its own phase rather than a
        # side effect here: journaled, retried and observable like any other.
        # `condensations` bounds it, because a summary that does not shrink the
        # history enough would otherwise re-trigger on the very next iteration.
        if not state.get("provider_continuation") and state.get("condensations", 0) < MAX_CONDENSATIONS and await agent.memory.needs_condensing(
            agent.provider, agent.config.model, system, tools,
            max_context_tokens=context_window(agent, limits),
            reply_tokens=reply_tokens,
        ):
            # attempt resets here as it does for the model call: the count left
            # over from the previous model call would otherwise make the driver
            # treat condensing as a retry -- a spurious ATTEMPT_RESET, usage
            # marked incomplete, and the summary's retry budget already spent.
            return {
                **await snapshot(agent), "phase": "compact",
                "iterations": agent.guardrails.iteration_count, "attempt": 0,
                **({"final_answer": True} if final_answer else {}),
            }
        # Fit the reply into what is left of the model's window, so a long
        # history does not turn a generous budget into an overflow (Pi does the
        # same). Only when the history was counted; a short one always fits.
        counted = getattr(agent.memory, "last_prompt_tokens", None)
        if type(counted) is int and counted > 0:
            room = limits.context_window - counted - CONTEXT_SAFETY_TOKENS
            reply_tokens = max(min(reply_tokens, room), min(reply_tokens, CONTEXT_SAFETY_TOKENS))
        messages, tools = await agent.middleware.process_llm_request(
            agent.memory.get_messages(), tools
        )
        if state.get("provider_continuation") and messages != agent.memory.get_messages():
            raise ValueError("Request middleware cannot change a pending provider continuation")
        messages = (messages if state.get("provider_continuation")
                    else sanitize_history(with_task_reminder(agent, messages)))
        return {
            **await snapshot(agent),
            **({"final_answer": True} if final_answer else {}),
            "iterations": agent.guardrails.iteration_count,
            "request": wire(
                dict(
                    model=agent.config.model,
                    messages=messages,
                    system=system or None,
                    tools=tools,
                    max_tokens=reply_tokens,
                    temperature=agent.config.temperature,
                    cache=build_hint(
                        agent.config.prompt_cache, agent.config.model, system or None, tools, messages
                    ),
                )
            ),
            "attempt": 0,
            "model_call_budget": getattr(agent.provider, "call_budget", lambda budget: budget)(reply_tokens),
        }

    if name == "compact":
        before_size = len(str(agent.memory.get_messages()))
        # Clearing old tool output first: it is usually most of the context and
        # the part needed least, and the conversation itself survives intact.
        # Only when that is not enough is the history summarized. A request the
        # provider already refused as too long is summarized regardless; the
        # token count that let it through cannot be trusted to judge the cut.
        prune = getattr(agent.memory, "prune_tool_results", None)
        if callable(prune) and not state.get("condense_without_model") and prune():
            limits = await model_limits(agent)
            if not state.get("compact_forced") and not await agent.memory.needs_condensing(
                agent.provider, agent.config.model, agent._build_system_prompt(),
                agent.tools.get_tool_params(),
                max_context_tokens=context_window(agent, limits),
                reply_tokens=reply_budget(agent, state, limits),
            ):
                await emit_hook(
                    agent, HookEvent.CONTEXT_CONDENSED,
                    messages_dropped=0, summary_chars=0, summarized=False,
                )
                return {
                    **await snapshot(agent), "phase": "prepare_model",
                    "condensations": state.get("condensations", 0) + 1, "compact_forced": False,
                }
        dropped, split_point = agent.memory.pending_condensation()
        summary, usage = "", state["usage"]
        if split_point:
            if not state.get("condense_without_model"):
                # A failure here propagates, so the driver retries it under
                # RetryPolicy like any other model call. Catching it inside the
                # command is what defeated that retry entirely; the lossy
                # fallback runs only once the attempts are spent.
                summary, summary_usage = await summarize_history(agent, dropped)
                usage = {k: usage.get(k, 0) + v for k, v in asdict(summary_usage).items()}
            if summary:
                agent.memory.apply_condensation(summary, split_point)
            else:
                limits = await model_limits(agent)
                await agent.memory.trim_if_needed(
                    agent.provider, agent.config.model, agent._build_system_prompt(),
                    agent.tools.get_tool_params(),
                    max_context_tokens=context_window(agent, limits),
                    reply_tokens=reply_budget(agent, state, limits),
                    **({"force": True} if state.get("compact_forced") else {}),
                )
        if state.get("compact_forced") and len(str(agent.memory.get_messages())) >= before_size:
            raise ValueError("Context overflow: no safe compaction reduced the request")
        await emit_hook(
            agent, HookEvent.CONTEXT_CONDENSED,
            messages_dropped=len(dropped), summary_chars=len(summary),
            summarized=bool(summary),
        )
        return {
            **await snapshot(agent),
            "phase": "prepare_model",
            "usage": usage,
            "condensations": state.get("condensations", 0) + 1,
            "condense_without_model": False,
            "compact_forced": False,
        }

    if name == "model":
        from .artifacts import materialize

        request = dict((await materialize(agent, state))["request"])
        # A previous failed attempt may already have learned a smaller cap.
        request["max_tokens"] = min(request["max_tokens"], (await model_limits(agent)).max_output)
        # The hint is persisted with the request; a state saved before hints
        # existed simply has none. Providers that predate the contract are
        # called without it.
        cache_hint = hint_from_wire(request.pop("cache", None))
        streaming = state.get("stream", False)
        provider_call = agent.provider.stream if streaming else agent.provider.create
        if cache_hint is not None and accepts_cache(provider_call):
            request["cache"] = cache_hint
        if record:
            await record("model.started", state["request"])
        await emit_hook(
            agent,
            HookEvent.LLM_REQUEST,
            system=request.get("system"),
            model=request["model"],
            max_tokens=request.get("max_tokens"),
            temperature=request.get("temperature"),
            stream=streaming,
            message_count=len(request["messages"]),
            tool_count=len(request["tools"]),
            prefix_key=cache_hint.prefix_key if cache_hint is not None and cache_hint.enabled else None,
        )
        buffered = any(
            type(m).after_llm_call is not Middleware.after_llm_call
            for m in agent.middleware._middleware
        )
        # Retries for transient failures live in the driver (one policy,
        # journaled per attempt). This command makes one provider call, and a
        # second only when the first was refused for asking more than a limit
        # allows -- a reply budget the model cannot give, which no retry of the
        # same request would cure.
        progress = getattr(agent.config, "progress", None)
        delivered = False

        async def call_provider():
            nonlocal delivered
            response = None
            budget_for_timeout = getattr(agent.provider, "call_budget", lambda budget: budget)(request.get("max_tokens"))
            async with asyncio.timeout(agent.config.retry.effective_call_timeout(budget_for_timeout)):
                if streaming:
                    async for chunk in stream_with_notices(
                        agent.provider.stream(**request), progress, event, on="model",
                        idle=getattr(agent.config.retry, "stream_idle_timeout_seconds", None),
                    ):
                        if chunk.kind == "response":
                            response = chunk.data
                        elif not buffered and chunk.kind in (
                            "text_delta",
                            "thinking_delta",
                        ):
                            delivered = True
                            await event(E(chunk.kind), chunk.data)
                    if response is None:
                        raise RuntimeError(
                            "Provider stream ended without a completed response"
                        )
                else:
                    response = await await_with_notices(
                        agent.provider.create(**request), progress, event, on="model"
                    )
            return response

        healed = None
        try:
            response = await call_provider()
        except Exception as exc:
            # Both refusals arrive before any output; once text has reached the
            # caller, re-asking would show it twice, so the driver decides.
            if delivered:
                raise
            target_provider, target_model, asked = request_target(agent, request)
            asked = int(asked or 0)
            named = reply_limit_from_error(exc)
            if named is not None and 0 < named[0] < asked:
                limit, is_model_limit = named
                if is_model_limit:
                    learn_model_limits(target_model, provider=target_provider, max_output=limit)
                logger.info("reply budget %d refused; retrying within %d", asked, limit)
                request["max_tokens"] = healed = limit
                set_reply_budget(agent, limit)
                await emit_hook(agent, HookEvent.RECOVERY, reason="reply_limit", attempt=1, reply_budget=limit)
                response = await call_provider()
            elif is_context_overflow(exc) and overflow_recoverable(agent, state):
                window = context_window_from_error(exc)
                if window:
                    learn_model_limits(target_model, provider=target_provider, context_window=window)
                attempt = state.get("overflow_recoveries", 0) + 1
                logger.info("request overflowed the context window; condensing (recovery %d)", attempt)
                await emit_hook(
                    agent, HookEvent.RECOVERY, reason="context_overflow", attempt=attempt,
                    reply_budget=request.get("max_tokens"),
                )
                # No model reply to record: condense, then prepare the call again.
                # attempt resets because nothing was delivered and condensing is
                # not a retry of this call.
                return {
                    **await snapshot(agent), "recovery": "context_overflow",
                    "overflow_recoveries": attempt, "attempt": 0, "compact_forced": True,
                }
            else:
                raise
        agent.guardrails.track_usage(response.usage)
        if record:
            # Copy before middleware can mutate the same ProviderResponse in place.
            await record("model.response", response_payload(response))
        protected = protected_content(response)
        response = await agent.middleware.process_llm_response(response)
        if protected_content(response) != protected:
            raise ValueError("Response middleware cannot change signed or opaque provider content")
        # Canonical fields after middleware are the single source for memory and output.
        canonical = ProviderResponse(
            text=getattr(response, "text", ""),
            tool_calls=getattr(response, "tool_calls", []),
            content=getattr(response, "content", [])
            if not hasattr(response, "text") or getattr(response, "_content_blocks", None) is not None
            else [],
            thinking=getattr(response, "thinking", None),
            stop_reason=response.stop_reason,
            usage=response.usage,
            _content_metadata=getattr(response, "_content_metadata", {}),
            _content_dicts=getattr(response, "_content_dicts", False),
        )
        if record:
            await record("model.processed", response_payload(canonical))
        # An empty reply is not stored (see add_assistant_message): sent back,
        # it would be rejected and break the session.
        agent.memory.add_assistant_message(canonical.content)
        recovery = await plan_recovery(agent, state, canonical, int(request_target(agent, request)[2] or 0))
        await emit_hook(
            agent,
            HookEvent.LLM_RESPONSE,
            response=canonical,
            stop_reason=canonical.stop_reason,
            provider=served_by(agent),
        )
        if buffered and canonical.text:
            await event(E.TEXT_DELTA, canonical.text)
        if canonical.text:
            await event(E.TEXT_COMPLETE, canonical.text)
        usage = {
            k: state["usage"].get(k, 0) + v for k, v in asdict(canonical.usage).items()
        }
        if healed and "reply_budget" not in recovery:
            recovery.update(set_reply_budget(agent, healed))
        return {
            **await snapshot(agent),
            **recovery,
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
            "provider_continuation": native_continuation(canonical),
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
                "timeout": DEFAULT_TIMEOUT_SECONDS,
                "retry": ToolRetry().to_dict(),
            }
            try:
                unreadable = call.input.get(INVALID_ARGUMENTS) if isinstance(call.input, dict) else None
                if unreadable is not None or not isinstance(call.input, dict):
                    raise ValueError(
                        f"Invalid arguments for {call.name!r}: "
                        f"{unreadable or 'the arguments are not a JSON object'}. "
                        "Call the tool again with a complete JSON object."
                    )
                definition, corrected = agent.tools.resolve_tool(call.name)
                if corrected != call.name:
                    # A case-only miss is a typo, not a different intent.
                    logger.info("repaired tool name %r -> %r", call.name, corrected)
                    call = replace(call, name=corrected)
                    entry["call"] = wire(call)
                from jsonschema import ValidationError, validate

                try:
                    validate(call.input, definition.input_schema)
                except ValidationError as invalid:
                    # Hand the model one actionable line, the way a tool error
                    # reads, so it can reissue the call rather than stall.
                    location = ".".join(str(part) for part in invalid.absolute_path)
                    where = f" at {location!r}" if location else ""
                    raise ValueError(
                        f"Invalid arguments for {call.name!r}{where}: {invalid.message}"
                    ) from None
                entry.update(
                    policy=definition.replay_policy,
                    concurrent=definition.concurrent,
                    timeout=definition.timeout_seconds,
                    retry=(definition.retry or ToolRetry()).to_dict(),
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
        history, notice = watch_for_repetition(agent, state)
        if notice:
            # Annotate the result the model is about to read. Telling the model
            # it is repeating usually stops it; ending the run is what
            # max_iterations is for.
            results = [replace(results[0], content=f"{notice}\n\n{results[0].content}"), *results[1:]]
            await emit_hook(
                agent, HookEvent.REPETITION,
                period=notice_period(notice), laps=agent.config.limits.loop_guard.threshold,
                tool=state["tools"][0]["call"]["name"],
            )
        agent.memory.add_tool_results(results)
        announced = await announce_todos(agent, state, event)
        await emit_hook(
            agent, HookEvent.LOOP_ITERATION_END, iteration=state["iterations"]
        )
        return {**await snapshot(agent), "tools": [], "call_history": history, **announced}

    if name == "finish":
        for ext in agent.extensions:
            await ext.on_turn_end(ExtensionContext(agent, ext.name), state["output"])
        await emit_hook(
            agent, HookEvent.LOOP_ITERATION_END, iteration=state["iterations"]
        )
        return await snapshot(agent)
    raise ValueError(f"Unknown engine command: {name}")


async def announce_todos(agent, state, event) -> dict:
    """Tell the run stream and hooks when the task list changed in this batch of tools.

    Checked once per batch, in a command both runtimes run, so a durable or
    Temporal run reports the plan exactly as a direct one does.
    """
    todos = copy.deepcopy(list(getattr(agent, "todos", []) or []))
    if todos == (state.get("todos_announced") or []):
        return {}
    active = next((item["content"] for item in todos if item.get("status") == "in_progress"), None)
    payload = {
        "todos": todos,
        "completed": sum(1 for item in todos if item.get("status") == "completed"),
        "total": len(todos),
        "in_progress": active,
    }
    await event(E.TODOS_UPDATED, payload)
    await emit_hook(agent, HookEvent.TODOS_UPDATED, **copy.deepcopy(payload))
    return {"todos_announced": todos}


def overflow_recoverable(agent, state) -> bool:
    """Whether condensing could make an overflowing request fit."""
    if state.get("provider_continuation") or state.get("overflow_recoveries", 0) >= MAX_OVERFLOW_RECOVERIES:
        return False
    pending = getattr(agent.memory, "pending_condensation", None)
    if not callable(pending):
        return False
    _, split_point = pending()
    return split_point > 0


def unfinished_plan(agent) -> list[dict]:
    """Items of the agent's task list not yet completed, when it keeps one."""
    if not agent.tools.has_tool("write_todos"):
        return []
    return [item for item in (getattr(agent, "todos", None) or []) if item.get("status") != "completed"]


def plan_reminder(agent) -> str:
    from .builtin.planning import render

    return (
        "Before you finish: your task list still has unfinished items.\n"
        f"{render(getattr(agent, 'todos', []) or [])}\n"
        "If they are done, mark them completed with write_todos. If work remains, "
        "continue it, or say plainly what is left. Do not repeat your previous reply."
    )


async def plan_recovery(agent, state, response, budget: int) -> dict:
    """Decide whether a reply that hit a limit ends the run or the loop goes on.

    Returns the state fields to merge; ``recovery`` names what happened and
    ``transition`` turns it into the next phase. Memory is updated here, so the
    snapshot the command returns already carries what the model will read next.

    * Cut off while calling a tool: the arguments may be incomplete even when
      they parse, so the calls are refused -- each answered with an error that
      says so -- and the model asks again with double the budget. Ending the run
      here, as before, left the model told to "call again" and no turn to do it.
    * Cut off mid-text: the model is asked to continue, with double the budget.
      The fragments are joined into the run's output.
    * Said nothing at all: nudged to go on, a bounded number of times.
    """
    stop = getattr(response.stop_reason, "value", response.stop_reason)
    limits = agent.config.limits
    if state.get("final_answer"):
        return {}  # the last call of a run at its limit: whatever it says ends the run
    if stop == "pause_turn":
        # The server paused a long turn (server tools); sending the turn back as
        # it stands is how it is resumed. No message is added.
        done = state.get("pauses", 0)
        if done >= MAX_PAUSE_RESUMES:
            return recovery_exhausted("pause_turn")
        await emit_hook(agent, HookEvent.RECOVERY, reason="paused_turn", attempt=done + 1, reply_budget=budget or None)
        return {
            "recovery": "paused_turn", "pauses": done + 1,
            "partial_output": state.get("partial_output", "") + (response.text or ""),
        }
    if stop == "max_tokens":
        done = state.get("truncations", 0)
        if done >= limits.max_truncation_recoveries:
            return {}
        target_provider, target_model, _ = request_target(agent, {})
        raised = escalated_budget(budget, await resolve_model_limits(target_provider, target_model)) if budget else None
        if response.tool_calls:
            reason = "truncated_tool_call"
            agent.memory.add_tool_results([
                ToolResult(getattr(call, "id", None) or call["id"], TRUNCATED_CALL_NOTICE, True)
                for call in response.tool_calls
            ])
            fields = {}
        else:
            reason = "truncated_text"
            agent.memory.add_user_message(CONTINUE_NOTICE)
            fields = {"partial_output": state.get("partial_output", "") + (response.text or "")}
        logger.info("reply cut off at %d tokens (%s); continuing with %s", budget, reason, raised)
        await emit_hook(agent, HookEvent.RECOVERY, reason=reason, attempt=done + 1, reply_budget=raised)
        return {
            **fields, "recovery": reason, "truncations": done + 1,
            **(set_reply_budget(agent, raised) if raised else {}),
        }
    if (
        stop in ("end_turn", "stop_sequence") and not response.tool_calls
        and not state.get("plan_reminded") and unfinished_plan(agent)
    ):
        # The run is ending with its own task list still open: either the work
        # is done and the list is stale -- every viewer of the plan is told
        # "0/1 done" about finished work -- or something was dropped. Once per
        # run, ask the model to settle it. Its reply so far is kept.
        agent.memory.add_user_message(plan_reminder(agent))
        await emit_hook(agent, HookEvent.RECOVERY, reason="unfinished_plan", attempt=1, reply_budget=budget or None)
        text = response.text or ""
        return {
            "recovery": "unfinished_plan", "plan_reminded": True,
            "partial_output": state.get("partial_output", "") + (text + "\n\n" if text.strip() else ""),
        }
    if (
        stop not in ("safety", "refusal") and not response.tool_calls
        and not (response.text or "").strip() and not state.get("partial_output", "").strip()
    ):
        # (With output already gathered, an empty last reply is a fine ending.)
        done = state.get("empty_replies", 0)
        if done >= MAX_EMPTY_NUDGES:
            return recovery_exhausted("empty_reply")
        agent.memory.add_user_message(EMPTY_NOTICE)
        await emit_hook(agent, HookEvent.RECOVERY, reason="empty_reply", attempt=done + 1, reply_budget=budget or None)
        return {"recovery": "empty_reply", "empty_replies": done + 1}
    return {}


def recovery_exhausted(reason):
    return {"terminal_failure": {"type": "RecoveryExhaustedError", "message": f"Recovery exhausted: {reason}"},
            "stop_reason": reason}


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
    agent_token = _agent_context.set(agent)
    try:
        if record:
            await record("tool.dispatched", {"call": entry["call"]}, entry)
        await emit_hook(agent, HookEvent.TOOL_CALL_START, tool_call=copy.deepcopy(call))
        try:
            async with asyncio.timeout(entry["timeout"]):
                result = await agent.tools._dispatch(call, definition)
        except TimeoutError:
            if state["durable"]:
                # A durable run can stop and ask whether the call took effect.
                raise
            # A direct run cannot: there is no one to ask and no way to resume,
            # and stopping here left an unanswered tool call that failed every
            # later request in the session. Tell the model instead, the way the
            # registry's own timeout does.
            result = ToolResult(
                call.id,
                f"Tool '{call.name}' timed out after {entry['timeout']}s. It may have "
                "partly run; check its effects before calling it again.",
                True,
            )
        policy = tool_retry_from_wire(entry)
        if (
            entry.get("policy", "manual") != "manual"
            and int(entry.get("attempt", 0)) < policy.attempts
            and retry_wanted(definition, policy, result)
        ):
            raise TransientToolError(str(result.content))
        # Commit the raw result before fallible middleware/extension work.
        return wire(result)
    finally:
        _tool_context.reset(token)
        _agent_context.reset(agent_token)


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
        sync_accounting(agent, state)
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
                        declared = isinstance(exc, TransientToolError)
                        failure = records("tool.failed", {
                            "type": type(exc).__name__, "message": str(exc),
                            "outcome": "failed" if declared else "unknown",
                        }, entry)
                        policy = tool_retry_from_wire(entry)
                        exhausted = entry["attempt"] >= policy.attempts
                        if declared and (entry["policy"] == "manual" or exhausted):
                            # The handler said the call did not take effect: no recovery
                            # stop, and no more attempts; the model sees the failure.
                            await persist(journal=failure)
                            raw = wire(ToolResult(
                                tool_call_id=entry["call"]["id"],
                                content=f"Tool execution error: {exc}",
                                is_error=True,
                            ))
                            break
                        if entry["policy"] == "manual":
                            entry["status"] = "uncertain"
                            await persist([ev(E.RECOVERY_REQUIRED, entry)], journal=failure)
                            return
                        if not (declared or retryable(agent, exc)) or exhausted:
                            await persist(journal=failure)
                            raise
                        await persist(journal=failure)
                        wait = policy.wait_for(entry["attempt"])
                        await emit_hook(
                            agent, HookEvent.RETRY, kind="tool", name=entry["call"]["name"],
                            attempt=entry["attempt"], next_attempt=entry["attempt"] + 1,
                            wait_seconds=wait, error=str(exc), provider=None,
                        )
                        entry["attempt"] += 1
                        await persist(journal=records("tool.started", {
                            "call": entry["call"], "replay_policy": entry["policy"],
                        }, entry))
                    if wait:
                        await asyncio.sleep(wait)
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
                        # Every call finishes and records its outcome before a
                        # failure in one of them is raised: cancelling the rest
                        # threw away work that had already happened.
                        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
                    except BaseException:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise
                    for failure in outcomes:
                        if isinstance(failure, BaseException):
                            raise failure
                else:
                    for entry in runnable:
                        await run_tool(entry)
                if any(
                    t["status"] in ("approval", "uncertain") for t in state["tools"]
                ):
                    state["status"] = "awaiting_input"
                    break
                name = "collect_tools"
            if name in ("model", "compact"):
                if state["attempt"]:
                    state["usage_incomplete"] = True
                    await persist([ev(E.ATTEMPT_RESET, {"attempt": state["attempt"]})])
                state["attempt"] += 1
                await persist()
            buffered_events = []

            async def command_emit(event):
                # Deltas and "still working" notices go out as they happen; a
                # notice held until the call finishes says nothing.
                if event.type in (E.TEXT_DELTA, E.THINKING_DELTA, E.WAITING):
                    await persist([event])
                else:
                    buffered_events.append(event)

            from .artifacts import capture

            while True:
                try:
                    outcome = await command(agent, state, name, command_emit, record=recorder)
                    break
                except Exception as exc:
                    if name in ("model", "compact"):
                        await record("model.failed", {
                            "type": type(exc).__name__, "message": str(exc),
                        })
                    retry = agent.config.retry
                    spent = not retryable(agent, exc) or state["attempt"] >= retry.attempts
                    if name == "compact" and spent and not state.get("condense_without_model"):
                        # A failed summary must not fail the run. Condense at the
                        # character level instead: lossy, but it always works.
                        logger.warning(
                            "history summarization failed after %d attempt(s); "
                            "condensing without a model", state["attempt"], exc_info=True,
                        )
                        # Set before retrying, and checked above, so a fallback
                        # that itself fails raises rather than looping here.
                        state["usage_incomplete"] = True
                        state["condense_without_model"] = True
                        agent.memory.set_messages(copy.deepcopy(state.get("messages", [])))
                        buffered_events.clear()
                        continue
                    if name not in ("model", "compact") or spent:
                        raise
                    wait = retry.wait_for(state["attempt"], retry_after(agent, exc))
                    await emit_hook(
                        agent, HookEvent.RETRY, kind="model", name=agent.config.model,
                        attempt=state["attempt"], next_attempt=state["attempt"] + 1,
                        wait_seconds=wait, error=str(exc), provider=served_by(agent),
                    )
                    state["usage_incomplete"] = True
                    buffered_events.clear()
                    # The failed attempt may have written to memory before it
                    # raised (its assistant turn, say); retrying on top of that
                    # would store the turn twice. The state holds what was
                    # committed, so put memory back to it.
                    agent.memory.set_messages(copy.deepcopy(state.get("messages", [])))
                    await persist([ev(E.ATTEMPT_RESET, {"attempt": state["attempt"]})])
                    state["attempt"] += 1
                    await persist()
                    if wait:
                        await asyncio.sleep(wait)
            outcome = await capture(agent, outcome)
            before = state.get("messages")
            state = transition(state, name, outcome)
            if state.get("messages") is not before:
                # `transition` is pure and cannot reach the agent, so a repair it
                # makes -- closing the tool calls of a reply cut off at the token
                # budget -- lives only in the state dict. The next command
                # snapshots `agent.memory`, and that snapshot would put the
                # unanswered `tool_use` straight back. Mirror it across now.
                agent.memory.set_messages(copy.deepcopy(state["messages"]))
            await persist(buffered_events, journal=(
                records("model.completed", outcome["response"])
                if name == "model" and "response" in outcome else []
            ))
    except asyncio.CancelledError:
        if state["phase"] == "model":
            await record("model.interrupted", {"outcome": "incomplete"})
        sync_accounting(agent, state)
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
        # A failure after the assistant turn was saved leaves its tool calls
        # unanswered, and the session cannot be resumed from that transcript.
        state = close_open_tool_calls(
            state, f"Not executed: the run failed first ({type(exc).__name__})."
        )
        agent.memory.set_messages(copy.deepcopy(state.get("messages", [])))
        await emit_hook(agent, HookEvent.ERROR, error=str(exc))
        await persist([ev(E.ERROR, str(exc))], journal=records("command.failed", {
            "phase": state["phase"], "type": type(exc).__name__, "message": str(exc),
        }))
    if state["status"] == "cancelled":
        sync_accounting(agent, state)
        state = cancel_state(state)
        agent.memory.set_messages(copy.deepcopy(state.get("messages", [])))
    sync_accounting(agent, state)
    result = result_from_state(state)
    await complete_turn(agent, state)
    events = [ev(E.RUN_RESULT, result)]
    if result.status == "completed":
        events.insert(0, ev(E.TURN_COMPLETE, result.stop_reason))
    await persist(events)
    return result


async def summarize_history(agent, dropped: list[dict]) -> tuple[str, TokenUsage]:
    """Summarize the messages a condensation is about to drop, using the agent's model.

    Returns the summary and what it cost. The summary is a real model call, so
    its tokens go through `track_usage` and into `RunResult.usage` like any
    other; leaving them out hid spend from `Limits.max_cost_dollars`.
    """
    transcript = render_for_summary(dropped)
    if not transcript.strip():
        return "", TokenUsage()
    budget = min(SUMMARY_MAX_TOKENS, agent.config.max_tokens or default_reply_budget(await model_limits(agent)))
    async with asyncio.timeout(agent.config.retry.effective_call_timeout(budget)):
        response = await agent.provider.create(
            model=agent.config.model,
            messages=[{"role": "user", "content": f"{transcript}\n\n{SUMMARY_INSTRUCTION}"}],
            system=SUMMARY_SYSTEM,
            tools=[],
            max_tokens=budget,
            temperature=agent.config.temperature,
        )
    usage = getattr(response, "usage", None) or TokenUsage()
    agent.guardrails.track_usage(usage)
    return (getattr(response, "text", "") or "").strip(), usage


def render_for_summary(messages: list[dict], max_block_chars: int = 2_000) -> str:
    """Flatten messages into labelled lines for the summarizer.

    Tool output is capped per block: the summary needs to know a tool ran and
    roughly what it said, not to carry its payload into another model call.
    """
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if isinstance(content, str):
            if content.strip():
                lines.append(f"[{role}] {content[:max_block_chars]}")
            continue
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and block.get("text", "").strip():
                lines.append(f"[{role}] {block['text'][:max_block_chars]}")
            elif kind == "tool_use":
                lines.append(f"[{role} tool call] {block.get('name', '?')}({_preview(block.get('input'))})")
            elif kind == "tool_result":
                body = str(block.get("content", ""))[:max_block_chars]
                label = "tool error" if block.get("is_error") else "tool result"
                lines.append(f"[{label}] {body}")
    return "\n".join(lines)


def _preview(value, limit: int = 200) -> str:
    text = str(value if value is not None else "")
    return text if len(text) <= limit else text[:limit] + "…"


UNANSWERED_CALL_NOTICE = (
    "No result was recorded for this call: the run stopped before it finished. "
    "Call the tool again if the result is still needed."
)


def sanitize_history(messages: list[dict]) -> list[dict]:
    """Make the outbound transcript one every provider accepts, whatever is stored.

    Run on every request, after anything that edits it (Hermes does the same
    before every call). The stored transcript is left alone; only the request
    changes, and deterministically, so the prompt-cache prefix stays stable.

    * An assistant turn with nothing in it is dropped. Sessions saved before
      empty replies stopped being stored still carry them.
    * Turns of one role in a row are joined (see ``coalesce_same_role``).
    * A tool result answers a call made in the assistant turn just before it;
      one that answers nothing is dropped, and of two answers to one call the
      later is kept -- a resumed run re-answers a call that a failure had
      closed with a placeholder.
    * A call with no answer gets one saying so. An unanswered ``tool_use`` is a
      request Anthropic rejects outright, wedging the session.
    """
    kept = [m for m in messages if not _empty_assistant(m)]
    joined = coalesce_same_role(kept)
    repaired: list[dict] = []
    for index, message in enumerate(joined):
        previous = repaired[-1] if repaired else None
        if (
            message.get("role") == "user" and isinstance(message.get("content"), str)
            and previous is not None and previous.get("role") == "assistant" and _call_ids(previous)
        ):
            message = {**message, "content": _as_blocks(message["content"])}
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            repaired.append(message)
            if message.get("role") == "assistant":
                calls = _call_ids(message)
                follower = joined[index + 1] if index + 1 < len(joined) else None
                if calls and (follower is None or follower.get("role") != "user"):
                    # Nothing answers this turn at all: answer it here.
                    repaired.append({"role": "user", "content": [_stub(c) for c in calls]})
            continue
        calls = _call_ids(previous) if previous and previous.get("role") == "assistant" else []
        results: dict[str, dict] = {}
        others: list[dict] = []
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                if block.get("tool_use_id") in calls:
                    results[block["tool_use_id"]] = block  # a later answer replaces an earlier one
            else:
                others.append(block)
        answered = [results.get(c) or _stub(c) for c in calls]
        content = answered + others
        if content:
            repaired.append({**message, "content": content})
    return coalesce_same_role(repaired)


def _empty_assistant(message: dict) -> bool:
    if message.get("role") != "assistant":
        return False
    content = message.get("content")
    if isinstance(content, str):
        return not content.strip()
    if not isinstance(content, list) or not content:
        return True
    return all(
        isinstance(b, dict) and b.get("type") == "text" and not str(b.get("text", "")).strip()
        for b in content
    )


def _call_ids(message: dict | None) -> list[str]:
    content = (message or {}).get("content")
    if not isinstance(content, list):
        return []
    return [b["id"] for b in content if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")]


def _stub(call_id: str) -> dict:
    return {"type": "tool_result", "tool_use_id": call_id, "content": UNANSWERED_CALL_NOTICE, "is_error": True}


def coalesce_same_role(messages: list[dict]) -> list[dict]:
    """Join neighbouring messages that share a role, on the outbound request only.

    Several ordinary shapes produce two user turns in a row. A tool-result
    message is user-role, so a turn that ends right after one -- a reply cut off
    at the token budget, whose refused calls we answer ourselves -- is followed
    directly by the next user message. Anthropic merges these silently; other
    providers reject them, and a configured fallback can be any of them.

    The stored transcript keeps both messages, which is the honest record of what
    happened. Only the request the provider sees is joined.
    """
    joined: list[dict] = []
    for message in messages:
        if not joined or joined[-1].get("role") != message.get("role"):
            joined.append(message)
            continue
        previous, content = joined[-1], message.get("content")
        merged = _as_blocks(previous.get("content")) + _as_blocks(content)
        joined[-1] = {**previous, "content": merged}
    return joined


def _as_blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    return [b for b in content or [] if isinstance(b, dict)]


def with_task_reminder(agent, messages: list[dict]) -> list[dict]:
    """Append the current task list, when there is one.

    At the tail rather than in the system prompt: the system prompt heads the
    cacheable prefix, so rewriting it every time a task is ticked off would
    invalidate the prompt cache for the whole conversation. Appending leaves the
    prefix untouched.

    Rebuilt from the agent each turn, so the model always reads the current list
    rather than whichever copy survived condensing.
    """
    todos = getattr(agent, "todos", None)
    if not todos:
        return messages
    from .builtin.planning import render

    reminder = render(todos)
    if not reminder:
        return messages
    # Merge into the last message when it is already user-role, rather than
    # appending a second one. A tool-result message is user-role too, so
    # appending produced two consecutive user turns on every agentic turn.
    # Anthropic tolerates that; other providers are stricter, and a declared
    # fallback can be any of them.
    if messages and messages[-1].get("role") == "user":
        last = messages[-1]
        content = last.get("content")
        if isinstance(content, str):
            merged = {**last, "content": f"{content}\n\n{reminder}"}
        elif isinstance(content, list):
            merged = {**last, "content": [*content, {"type": "text", "text": reminder}]}
        else:
            return [*messages, {"role": "user", "content": reminder}]
        return [*messages[:-1], merged]
    return [*messages, {"role": "user", "content": reminder}]


def watch_for_repetition(agent, state) -> tuple[list[str], str | None]:
    """Track completed calls and report a cycle that has stopped producing news.

    Returns the history to persist and, when a cycle is detected, the notice to
    put in front of the model's next read.
    """
    from .loopguard import call_signature, detect_cycle, repetition_notice

    guard = agent.config.limits.loop_guard
    history = list(state.get("call_history") or [])
    if not guard.applies():
        return history, None

    # Every completed call is evidence, whatever its replay policy and whether
    # it succeeded. Progress is already encoded in the signature, which covers
    # the result as well as the arguments: an edit that changes a file, or a
    # test whose output changes, produces a different signature and so breaks
    # the cycle on its own. Clearing the history on success instead -- which is
    # what Hermes does, because its signatures carry no result -- switched the
    # guard off for every tool not declared ``safe``, which is the default.
    for entry in state["tools"]:
        result = entry.get("result") or {}
        call = entry["call"]
        history.append(call_signature(call["name"], call.get("input"), str(result.get("content", ""))))

    if len(history) > guard.history:
        history = history[-guard.history:]

    period = detect_cycle(history, max_period=guard.max_period, threshold=guard.threshold)
    if period is None:
        return history, None
    # Start again, so one cycle is reported once rather than on every call after.
    return [], repetition_notice(period, guard.threshold)


def notice_period(notice: str) -> int:
    return 1 if "This exact call" in notice else int(notice.split("cycle of ")[1].split()[0])


def retryable(agent, exc):
    """One classifier for model and tool attempts, asked of the agent's provider.

    Providers override ``is_transient`` for vendor-specific errors; duck-typed
    providers without it get the shared classifier (429, 5xx, timeouts,
    connection loss).
    """
    classify = getattr(agent.provider, "is_transient", None)
    if callable(classify):
        try:
            return bool(classify(exc))
        except Exception:
            pass
    return is_transient(exc)


def retry_after(agent, exc):
    """The server's Retry-After for a model failure, via the provider when it has a reader."""
    reader = getattr(agent.provider, "retry_after", None)
    if callable(reader):
        try:
            value = reader(exc)
            return None if value is None else max(0.0, float(value))
        except Exception:
            pass
    return retry_after_seconds(exc)


def served_by(agent) -> str:
    """The name of the provider that answered: a FallbackProvider reports the member it used."""
    provider = agent.provider
    return str(
        getattr(provider, "last_served", None) or getattr(provider, "name", "") or type(provider).__name__
    )


def tool_retry_from_wire(entry) -> ToolRetry:
    """The retry policy carried on a prepared tool entry; older run states get the default."""
    try:
        return ToolRetry.from_dict(entry.get("retry"))
    except Exception:
        return ToolRetry()
