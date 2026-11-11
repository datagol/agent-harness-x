"""LensExtension — DataGOL Lens tracing for harnessx agents.

Sends one trace per agent turn to a DataGOL Lens project. The trace holds:
  - an ``llm`` span for each model call (request, reply, model, tokens, cost)
  - a ``tool`` span for each tool call (arguments, result, error status)
  - ``event`` spans for retries and context condensation
  - the spans of sub-agents run from a tool, in the same trace

Lens takes the whole trace in one request when the turn ends. That request
runs in the background with a time limit, and a failure is logged and
dropped, so tracing never slows or breaks a run.

Usage:
    from harnessx import Agent, AgentConfig
    from harnessx.extensions import LensExtension

    agent = Agent(
        config=AgentConfig(model="claude-sonnet-4-6"),
        extensions=[LensExtension(run_name="Support Agent", session_id="thread-42")],
    )
    async with agent:  # closing the agent waits for traces still being sent
        await agent.run("What is 17 + 25?")

Configuration falls back to the environment variables that other DataGOL Lens
clients use: ``DATAGOL_LENS_ENABLED``, ``DATAGOL_LENS_API_URL``,
``DATAGOL_LENS_API_KEY``, ``DATAGOL_LENS_PROJECT_ID`` and
``DATAGOL_LENS_MAX_FIELD_CHARS``. Tracing is off unless it is enabled and has
an API key and a project id.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from ..hooks import HookContext, HookEvent, Middleware
from .base import Extension, ExtensionContext
from .langsmith import _serving_provider

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8000"
# Same default as the Node SDK: agent tool results are often tens of KB, and an
# unlimited default would send multi-MB bodies. 0 turns the limit off.
DEFAULT_MAX_FIELD_CHARS = 8000
_TRUTHY = {"1", "true", "yes", "on", "enabled"}

# Lens status for each terminal RunResult status.
_STATUS = {"completed": "success", "failed": "error", "cancelled": "stopped"}


# ── Trace state ──────────────────────────────────────────────────────────────


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class _OpenSpan:
    name: str
    span_type: str
    input: Any
    id: str = field(default_factory=_new_id)
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    closed: bool = False
    trace: _Trace | None = None
    input_ready: bool = False  # already serialized, when the input could change before the span closes


@dataclass
class _Trace:
    name: str
    input: Any
    session_id: str | None
    user_id: str | None
    tags: list[str]
    metadata: dict[str, Any]
    model: str | None = None
    started_at: float = field(default_factory=time.time)
    spans: list[dict[str, Any]] = field(default_factory=list)
    output: Any = None
    output_set: bool = False
    error: str | None = None
    status: str | None = None
    sent: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    id: str = field(default_factory=_new_id)


@dataclass
class _Run:
    """One agent's part of a trace: the root agent, or a sub-agent run from a tool."""

    trace: _Trace
    root: bool
    agent_name: str
    parent_tool: str | None = None
    parent_id: str | None = None  # the tool span a sub-agent runs under; its spans nest there
    llm: _OpenSpan | None = None
    tools: dict[str, _OpenSpan] = field(default_factory=dict)


# The tool span whose handler is running. A sub-agent that starts inside the
# handler sees it and joins the parent's trace instead of opening its own.
_current_tool: contextvars.ContextVar[_OpenSpan | None] = contextvars.ContextVar(
    "lens_current_tool", default=None
)


# ── Serialization ────────────────────────────────────────────────────────────


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _jsonable(value: Any, limit: int) -> Any:
    """A JSON-safe copy with long strings cut and binary attachments left out."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if limit <= 0 or len(value) <= limit else f"{value[:limit]}…[truncated]"
    if isinstance(value, dict):
        if value.get("type") in ("image", "document") and "source" in value:
            return {"type": value["type"], "source": "[omitted]"}
        return {str(k): _jsonable(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, limit) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {name: _jsonable(getattr(value, name, None), limit) for name in value.__dataclass_fields__
                if not name.startswith("_")}
    return _jsonable(str(value), limit)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return max(int(raw), 0) if raw else default
    except ValueError:
        return default


# ── Lens Extension ───────────────────────────────────────────────────────────


class LensExtension(Extension):
    """Extension that sends each agent turn to DataGOL Lens as one trace."""

    name = "lens"

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        project_id: str | None = None,
        run_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_llm: bool = True,
        trace_tools: bool = True,
        enabled: bool | None = None,
        auto_flush: bool = True,
        flush_timeout_seconds: float = 5.0,
        max_field_chars: int | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_url = (api_url or os.getenv("DATAGOL_LENS_API_URL") or DEFAULT_API_URL).rstrip("/")
        self._api_key = api_key or os.getenv("DATAGOL_LENS_API_KEY") or ""
        self.project_id = project_id or os.getenv("DATAGOL_LENS_PROJECT_ID") or ""
        self.run_name = run_name
        self.session_id = session_id
        self.user_id = user_id
        self.tags = list(tags) if tags else []
        self.metadata = dict(metadata) if metadata else {}
        self.trace_llm = trace_llm
        self.trace_tools = trace_tools
        self.enabled = _env_flag("DATAGOL_LENS_ENABLED") if enabled is None else enabled
        self.auto_flush = auto_flush
        self.flush_timeout_seconds = flush_timeout_seconds
        self.max_field_chars = (
            _env_int("DATAGOL_LENS_MAX_FIELD_CHARS", DEFAULT_MAX_FIELD_CHARS)
            if max_field_chars is None else max(max_field_chars, 0)
        )
        self._http = http_client
        self._owns_http = http_client is None
        self._runs: dict[int, _Run] = {}
        self._trace: _Trace | None = None  # the latest root trace this extension opened
        self._pending: set[asyncio.Task[None]] = set()

    @property
    def configured(self) -> bool:
        """True when tracing is enabled and has an API key and a project id."""
        return bool(self.enabled and self._api_key and self.project_id)

    def install(self, agent_or_ctx: Any) -> None:
        if not self.configured:
            if self.enabled:
                missing = [n for n, v in (("api_key", self._api_key), ("project_id", self.project_id)) if not v]
                logger.warning("LensExtension is enabled but has no %s; tracing is off", " or ".join(missing))
            return
        ctx = agent_or_ctx if isinstance(agent_or_ctx, ExtensionContext) else ExtensionContext(agent_or_ctx, self.name)
        ctx.add_middleware(_LensMiddleware(self, ctx.agent))
        ctx.on_hook(HookEvent.AGENT_START, self._on_agent_start)
        ctx.on_hook(HookEvent.LLM_REQUEST, self._on_llm_request)
        ctx.on_hook(HookEvent.LLM_RESPONSE, self._on_llm_response)
        ctx.on_hook(HookEvent.TOOL_CALL_START, self._on_tool_start)
        ctx.on_hook(HookEvent.TOOL_CALL_END, self._on_tool_end)
        ctx.on_hook(HookEvent.RETRY, self._on_retry)
        ctx.on_hook(HookEvent.CONTEXT_CONDENSED, self._on_condensed)
        ctx.on_hook(HookEvent.ERROR, self._on_error)
        ctx.on_hook(HookEvent.AGENT_END, self._on_agent_end)

    # ── Public API ────────────────────────────────────────────────────

    def set_output(self, output: Any) -> None:
        """Replace the output of the current or latest trace before it is sent.

        Use it with ``auto_flush=False`` when the application changes the answer
        after the run, for example to clean up citations.
        """
        if self._trace is not None and not self._trace.sent:
            self._trace.output, self._trace.output_set = output, True

    def add_metadata(self, **metadata: Any) -> None:
        """Add metadata to the current or latest trace before it is sent."""
        if self._trace is not None and not self._trace.sent:
            self._trace.metadata.update(metadata)

    @property
    def trace_id(self) -> str | None:
        """The id of the current or latest trace, as Lens stores it; log it to find the trace."""
        return self._trace.id if self._trace is not None else None

    async def flush(self) -> None:
        """Send a trace held by ``auto_flush=False``, and wait for every send in progress."""
        trace = self._trace
        if trace is not None and not trace.sent and trace.status is not None:
            self._send_later(trace)
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    async def teardown(self) -> None:
        """Send what is left, as ``stopped`` if the turn never ended, then close the HTTP client."""
        trace = self._trace
        if trace is not None and not trace.sent:
            if trace.status is None:
                for run in [r for r in self._runs.values() if r.trace is trace]:
                    self._close_open_spans(run, "agent closed before the turn ended")
                trace.status = "stopped"
            self._send_later(trace)
        self._runs.clear()
        await self.flush()
        if self._owns_http and self._http is not None:
            try:
                await self._http.aclose()
            except Exception:
                pass
            self._http = None

    # ── Lifecycle hooks ───────────────────────────────────────────────

    async def _on_agent_start(self, ctx: HookContext) -> None:
        agent = ctx.agent
        config = getattr(agent, "config", None)
        model = getattr(config, "model", None)
        agent_name = self.run_name or (f"{model} Agent" if model else "Agent Run")

        parent = _current_tool.get()
        if parent is not None and not parent.closed and parent.trace is not None and not parent.trace.sent:
            # A sub-agent inside a tool: its spans go into the parent's trace.
            self._runs[id(agent)] = _Run(
                parent.trace, root=False, agent_name=agent_name, parent_tool=parent.name, parent_id=parent.id
            )
            return

        if self._trace is not None and not self._trace.sent and self._trace.status is None:
            # The previous turn paused or never ended; send it before starting this one.
            self._trace.status = "stopped"
            self._send_later(self._trace)

        trace = _Trace(
            name=agent_name,
            input={"query": ctx.data.get("message", "")},
            session_id=self.session_id or getattr(agent, "_session_id", None) or None,
            user_id=self.user_id,
            tags=list(self.tags),
            metadata={
                "model": model,
                "provider": getattr(config, "provider", None),
                **self.metadata,
            },
            model=model,
        )
        self._trace = trace
        self._runs[id(agent)] = _Run(trace, root=True, agent_name=agent_name)

    async def _on_llm_request(self, ctx: HookContext) -> None:
        run = self._runs.get(id(ctx.agent))
        if run is None or run.llm is None:
            return
        data = ctx.data
        if isinstance(run.llm.input, dict):
            run.llm.input = {"system": _jsonable(data.get("system"), self.max_field_chars), **run.llm.input}
        run.llm.model = data.get("model") or run.llm.model
        run.llm.metadata.update(
            {key: data[key] for key in ("max_tokens", "temperature", "stream") if key in data}
        )

    async def _on_llm_response(self, ctx: HookContext) -> None:
        run = self._runs.get(id(ctx.agent))
        if run is None or run.llm is None:
            return
        span, run.llm = run.llm, None
        response = ctx.data.get("response")
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "input_tokens", 0) or 0)
        completion = int(getattr(usage, "output_tokens", 0) or 0)
        cost = 0.0
        if usage is not None:
            try:
                cost = float(ctx.agent.guardrails.cost_of(usage))
            except Exception:
                cost = 0.0
        tool_calls = [
            {"id": call.id, "name": call.name, "input": call.input}
            for call in getattr(response, "tool_calls", None) or []
        ]
        output: dict[str, Any] = {"content": getattr(response, "text", "")}
        if tool_calls:
            output["tool_calls"] = tool_calls
        stop_reason = ctx.data.get("stop_reason")
        metadata: dict[str, Any] = {
            "provider": _serving_provider(ctx.agent) or ctx.data.get("provider"),
            "stop_reason": getattr(stop_reason, "value", stop_reason),
        }
        self._close_span(
            run, span, output=output, prompt_tokens=prompt, completion_tokens=completion, cost=cost,
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            reasoning_tokens=int(getattr(usage, "thinking_tokens", 0) or 0),
            metadata=metadata,
        )

    async def _on_tool_start(self, ctx: HookContext) -> None:
        run = self._runs.get(id(ctx.agent))
        call = ctx.data.get("tool_call")
        if run is None or call is None or not self.trace_tools:
            return
        stale = run.tools.pop(call.id, None)
        if stale is not None:  # an earlier attempt of the same call that is being retried
            self._close_span(run, stale, output=None, metadata={"status": "error"}, error="retried")
        span = _OpenSpan(
            name=call.name,
            span_type="tool",
            input=call.input if isinstance(call.input, dict) else {"input": call.input},
            trace=run.trace,
        )
        run.tools[call.id] = span
        # Hooks run in the task that runs the tool, so the handler sees this.
        _current_tool.set(span)

    async def _on_tool_end(self, ctx: HookContext) -> None:
        run = self._runs.get(id(ctx.agent))
        call, result = ctx.data.get("tool_call"), ctx.data.get("result")
        if run is None or call is None or result is None or not self.trace_tools:
            return
        span = run.tools.pop(call.id, None)
        if span is None:  # refused before it ran: a zero-length span still records the call
            span = _OpenSpan(name=call.name, span_type="tool", input=call.input, trace=run.trace)
        self._close_span(run, span, output={"result": result.content},
                         metadata={"status": "error"} if result.is_error else None)

    async def _on_retry(self, ctx: HookContext) -> None:
        run = self._runs.get(id(ctx.agent))
        if run is None:
            return
        data = ctx.data
        if data.get("kind") == "model" and run.llm is not None:
            span, run.llm = run.llm, None
            self._close_span(run, span, output={"error": data.get("error")},
                             metadata={"status": "error"}, error=str(data.get("error") or "retried"))
        self._add_event(run, "retry", dict(data))

    async def _on_condensed(self, ctx: HookContext) -> None:
        run = self._runs.get(id(ctx.agent))
        if run is not None:
            self._add_event(run, "compaction", dict(ctx.data))

    async def _on_error(self, ctx: HookContext) -> None:
        run = self._runs.get(id(ctx.agent))
        if run is None:
            return
        error = str(ctx.data.get("error") or "agent error")
        self._close_open_spans(run, error)
        if run.root:
            run.trace.error = error

    async def _on_agent_end(self, ctx: HookContext) -> None:
        run = self._runs.pop(id(ctx.agent), None)
        if run is None or not run.root:
            return
        trace = run.trace
        result = ctx.data.get("result")
        status = str(getattr(getattr(result, "status", None), "value", getattr(result, "status", "")))
        self._close_open_spans(run, f"run {status or 'ended'} with the span open")
        trace.status = _STATUS.get(status, "success")
        failure = getattr(result, "error", None)
        if failure and not trace.error:
            trace.error = str(failure.get("message") if isinstance(failure, dict) else failure)
        if not trace.output_set:
            trace.output = {"content": getattr(result, "output", result) if result is not None else ""}
        stop_reason = getattr(result, "stop_reason", None)
        if stop_reason:
            trace.metadata.setdefault("stop_reason", stop_reason)
        if self.auto_flush:
            self._send_later(trace)

    # ── Span bookkeeping ──────────────────────────────────────────────

    def _close_span(
        self,
        run: _Run,
        span: _OpenSpan,
        *,
        output: Any,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost: float = 0.0,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if span.closed:
            return
        span.closed = True
        trace = run.trace
        if trace.sent:
            return
        ended = time.time()
        meta = {**span.metadata, **(metadata or {})}
        if not run.root:
            meta.update(agent=run.agent_name, parent_tool=run.parent_tool)
        if error and output is None:
            output = {"error": error}
        failed = bool(error) or meta.get("status") == "error"
        if failed and not error and isinstance(output, dict):
            error = str(output.get("result") or output.get("error") or "") or None
        limit = self.max_field_chars
        trace.spans.append({
            "id": span.id,
            "name": span.name,
            "span_type": span.span_type,
            "input": span.input if span.input_ready else _jsonable(span.input, limit),
            "output": _jsonable(output, limit),
            "model": span.model,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "reasoning_tokens": reasoning_tokens,
            # harnessx's input_tokens already include the cached ones.
            "cache_tokens_in_prompt": True,
            "cost": cost,
            "status": "error" if failed else "success",
            "error": _jsonable(error, limit) if failed else None,
            "latency_ms": int((ended - span.started_at) * 1000),
            "start_time": _utc(span.started_at),
            "end_time": _utc(ended),
            "metadata": _jsonable(meta, limit),
            "parent_id": run.parent_id,
            "redacted": False,
            "redacted_fields": 0,
        })
        trace.prompt_tokens += prompt_tokens
        trace.completion_tokens += completion_tokens
        trace.cache_read_tokens += cache_read_tokens
        trace.cache_write_tokens += cache_write_tokens
        trace.reasoning_tokens += reasoning_tokens
        trace.cost += cost
        if span.model:
            trace.model = span.model

    def _close_open_spans(self, run: _Run, error: str) -> None:
        if run.llm is not None:
            span, run.llm = run.llm, None
            self._close_span(run, span, output={"error": error}, metadata={"status": "error"}, error=error)
        for span in list(run.tools.values()):
            self._close_span(run, span, output={"error": error}, metadata={"status": "error"}, error=error)
        run.tools.clear()

    def _add_event(self, run: _Run, name: str, metadata: dict[str, Any]) -> None:
        span = _OpenSpan(name=name, span_type="event", input={}, metadata=metadata, trace=run.trace)
        span.started_at = time.time()
        self._close_span(run, span, output=None)

    # ── Sending ───────────────────────────────────────────────────────

    def _payload(self, trace: _Trace) -> dict[str, Any]:
        ended = time.time()
        limit = self.max_field_chars
        return {
            "id": trace.id,
            "name": trace.name,
            "agent_name": trace.name,
            "input": _jsonable(trace.input, limit),
            "output": _jsonable(trace.output, limit),
            "error": trace.error,
            "start_time": _utc(trace.started_at),
            "end_time": _utc(ended),
            "latency_ms": int((ended - trace.started_at) * 1000),
            "total_tokens": trace.prompt_tokens + trace.completion_tokens,
            "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": trace.completion_tokens,
            "cache_read_tokens": trace.cache_read_tokens,
            "cache_write_tokens": trace.cache_write_tokens,
            "reasoning_tokens": trace.reasoning_tokens,
            "cache_tokens_in_prompt": True,
            "total_cost": trace.cost,
            "model": trace.model,
            "status": trace.status or "success",
            "tags": list(trace.tags),
            "metadata": _jsonable(trace.metadata, limit),
            "external_user_id": trace.user_id,
            "session_id": trace.session_id,
            "virtual_key_id": None,
            "redacted": False,
            "redacted_fields": 0,
            "blocked": False,
            "guardrail_violations": [],
            "spans": list(trace.spans),
        }

    def _send_later(self, trace: _Trace) -> None:
        """Send the trace in the background, once; the run never waits for Lens."""
        if trace.sent:
            return
        trace.sent = True
        payload = self._payload(trace)
        try:
            task = asyncio.get_running_loop().create_task(self._post(payload))
        except RuntimeError:
            logger.warning("Lens trace %r dropped: no running event loop to send it", trace.name)
            return
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _post(self, payload: dict[str, Any]) -> None:
        url = f"{self.api_url}/api/v1/projects/{self.project_id}/traces"
        try:
            if self._http is None:
                self._http = httpx.AsyncClient(timeout=self.flush_timeout_seconds)
            response = await asyncio.wait_for(
                self._http.post(url, json=payload, headers={"Authorization": f"Bearer {self._api_key}"}),
                timeout=self.flush_timeout_seconds,
            )
            if response.status_code >= 400:
                logger.warning("Lens rejected trace %r: HTTP %s %s",
                               payload.get("name"), response.status_code, response.text[:300])
        except Exception as exc:
            logger.warning("Failed to send Lens trace %r to %s: %s", payload.get("name"), self.api_url,
                           exc or type(exc).__name__)


# ── Middleware: the request a model call is about to send ──────────────────


class _LensMiddleware(Middleware):
    """Opens the llm span with the messages and tools of each model request.

    Only ``before_llm_call`` is overridden. Overriding ``after_llm_call`` makes
    the engine buffer every streamed reply, so the span is closed by the
    LLM_RESPONSE hook instead.
    """

    def __init__(self, ext: LensExtension, agent: Any) -> None:
        self._ext = ext
        self._agent = agent

    async def before_llm_call(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        run = self._ext._runs.get(id(self._agent))
        if run is None or not self._ext.trace_llm:
            return messages, tools
        if run.llm is not None:  # the request was built again before the last one answered
            span, run.llm = run.llm, None
            self._ext._close_span(run, span, output=None, metadata={"status": "error"}, error="request rebuilt")
        model = getattr(getattr(self._agent, "config", None), "model", None)
        # Serialized now: the history is appended to before the reply arrives.
        run.llm = _OpenSpan(
            name="llm_call",
            span_type="llm",
            input=_jsonable(
                {"messages": messages, "tools": [t.get("name") for t in tools if isinstance(t, dict)]},
                self._ext.max_field_chars,
            ),
            model=model,
            trace=run.trace,
            input_ready=True,
        )
        return messages, tools
