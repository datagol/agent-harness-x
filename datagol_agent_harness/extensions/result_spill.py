"""ResultSpillExtension — keep huge tool results out of the LLM context.

Faithful port of pi-dca's `server/src/pi/extensions/result-spill.ts`.

When a tool returns more than `threshold_bytes` of text, the extension:
  1. Writes the full payload to a per-session file under /tmp.
  2. Replaces the tool result the LLM sees with a small envelope JSON
     ({stored: true, result_id, head|peek, size_bytes, ...}).
  3. Registers a `read_result` tool the model can call to slice / aggregate
     the spilled payload without re-running the original query:
       - tabular (JSONL): head, tail, count, count_by, sum, avg, min, max
       - non-tabular (text/json): peek, search

On agent.aclose(), the per-session spill dir is removed.

Differences from the pi-dca TS implementation:
  - Our ToolResult.content is a single string (not a Pi block array), so the
    "split text vs image blocks" logic collapses to "measure the string".
  - We hook in via Middleware.after_tool_execution rather than Pi's pi.on().
    Mutation is supported via the middleware's return-value contract.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any

from ..hooks import Middleware
from ..types import PermissionLevel, ToolResult
from .base import Extension


# ── Public extension ────────────────────────────────────────────────────────


class ResultSpillExtension(Extension):
    """Spill large tool results to disk and expose a `read_result` tool."""

    name = "result_spill"

    def __init__(
        self,
        *,
        threshold_bytes: int = 32_000,
        max_spill_bytes: int = 50_000_000,
        spill_root: str | None = None,
        inline_head_rows: int = 10,
        inline_peek_chars: int = 1000,
    ) -> None:
        self.threshold_bytes = threshold_bytes
        self.max_spill_bytes = max_spill_bytes
        self.inline_head_rows = inline_head_rows
        self.inline_peek_chars = inline_peek_chars

        self._session_id = str(uuid.uuid4())
        root = spill_root or os.path.join(tempfile.gettempdir(), "agent-harness-spill")
        self._spill_dir = os.path.join(root, self._session_id)
        self._dir_ready = False
        self._results: dict[str, _SpillMeta] = {}

    # ── Extension interface ────────────────────────────────────────────

    def install(self, agent: Any) -> None:
        agent.middleware.add(_SpillMiddleware(self))
        agent.tools.register_with_schema(
            name="read_result",
            description=(
                "Slice or aggregate a previously-stored tool result. Required "
                "when a prior tool response came back with `stored: true` and "
                "a `result_id`. Tabular ops: head, tail, count, count_by, sum, "
                "avg, min, max. Non-tabular ops: peek, search. Always prefer "
                "pushing aggregation into SQL (GROUP BY, etc.) when feasible."
            ),
            input_schema=_READ_RESULT_SCHEMA,
            handler=self._handle_read_result,
            permission=PermissionLevel.ALLOW,
        )

    async def teardown(self) -> None:
        self._results.clear()
        if self._dir_ready and os.path.isdir(self._spill_dir):
            shutil.rmtree(self._spill_dir, ignore_errors=True)
            self._dir_ready = False

    # ── Internal: spill on outsize tool results ────────────────────────

    def _maybe_spill(self, result: ToolResult) -> ToolResult:
        """If the result is over threshold, write it to disk and return a
        small envelope ToolResult in its place. Otherwise pass through."""
        if result.is_error:
            return result
        content = result.content or ""
        size = len(content)
        if size < self.threshold_bytes:
            return result

        if size > self.max_spill_bytes:
            envelope = {
                "stored": False,
                "truncated": True,
                "size_bytes": size,
                "max_spill_bytes": self.max_spill_bytes,
                "hint": (
                    "Result exceeds the spill cap. Refine your SQL with "
                    "GROUP BY / aggregations / a smaller LIMIT."
                ),
            }
            return ToolResult(
                tool_use_id=result.tool_use_id,
                content=json.dumps(envelope),
                is_error=False,
            )

        # Tabular path: parse as JSON, look for `rows`.
        parsed: Any | None = None
        tabular: _ParsedRows | None = None
        try:
            parsed = json.loads(content)
            tabular = _find_rows(parsed)
        except (json.JSONDecodeError, ValueError):
            parsed = None
            tabular = None

        self._ensure_dir()
        result_id = f"r_{uuid.uuid4()}"

        if tabular is not None:
            path = os.path.join(self._spill_dir, f"{result_id}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for row in tabular.rows:
                    f.write(json.dumps(row, default=str))
                    f.write("\n")
            meta = _SpillMeta(
                result_id=result_id,
                path=path,
                fmt="jsonl",
                row_count=len(tabular.rows),
                columns=tabular.columns,
                parent_keys=tabular.parent_keys,
                size_bytes=size,
            )
            passthrough: dict[str, Any] = {}
            if isinstance(parsed, dict):
                if isinstance(parsed.get("executedQuery"), str):
                    passthrough["executedQuery"] = parsed["executedQuery"]
                if isinstance(parsed.get("recoveryNote"), str):
                    passthrough["recoveryNote"] = parsed["recoveryNote"]

            envelope = {
                "stored": True,
                "format": "jsonl",
                "result_id": result_id,
                "row_count": len(tabular.rows),
                "columns": tabular.columns,
                "parent_keys": tabular.parent_keys,
                "head": [_truncate_row_inline(r) for r in tabular.rows[: self.inline_head_rows]],
                "size_bytes": size,
                **passthrough,
                "note": (
                    "Tabular result stored. Use read_result with head/tail/"
                    "count/count_by/sum/avg/min/max. Do NOT re-issue the same SQL."
                ),
            }
        else:
            is_json = parsed is not None
            fmt: str = "json" if is_json else "text"
            ext_suffix = "json" if is_json else "txt"
            path = os.path.join(self._spill_dir, f"{result_id}.{ext_suffix}")
            body = json.dumps(parsed, indent=2, default=str) if is_json else content
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
            meta = _SpillMeta(
                result_id=result_id,
                path=path,
                fmt=fmt,
                row_count=None,
                columns=None,
                parent_keys=None,
                size_bytes=size,
            )
            envelope = {
                "stored": True,
                "format": fmt,
                "result_id": result_id,
                "size_bytes": size,
                "peek": content[: self.inline_peek_chars],
                "peek_chars": min(self.inline_peek_chars, len(content)),
                "note": (
                    "Non-tabular result stored. Use read_result with peek({bytes}) "
                    "or search({needle}) to inspect specific regions. Row-oriented "
                    "ops (head/count_by/etc.) will be refused."
                ),
            }

        self._results[result_id] = meta
        return ToolResult(
            tool_use_id=result.tool_use_id,
            content=json.dumps(envelope, default=str),
            is_error=False,
        )

    def _ensure_dir(self) -> None:
        if self._dir_ready:
            return
        os.makedirs(self._spill_dir, exist_ok=True)
        self._dir_ready = True

    # ── Internal: read_result tool handler ─────────────────────────────

    def _handle_read_result(self, result_id: str, op: dict[str, Any]) -> str:
        meta = self._results.get(result_id)
        if meta is None:
            return json.dumps(
                {"error": f'result_id "{result_id}" not found in this session.'}
            )

        kind = op.get("kind")
        row_ops = {"head", "tail", "count", "count_by", "sum", "avg", "min", "max"}
        nontab_ops = {"peek", "search"}
        is_tabular = meta.fmt == "jsonl"

        if kind in row_ops and not is_tabular:
            return json.dumps(
                {
                    "error": (
                        f'op "{kind}" requires a tabular result (JSONL). '
                        f'This result_id is stored as "{meta.fmt}". '
                        f'Use op: {{ kind: "peek" }} or {{ kind: "search", needle }}.'
                    )
                }
            )
        if kind in nontab_ops and is_tabular:
            return json.dumps(
                {
                    "error": (
                        f'op "{kind}" is for non-tabular results. This result_id '
                        f"has rows — use head/tail/count/count_by/sum/avg/min/max."
                    )
                }
            )

        try:
            if kind == "head":
                n = int(op.get("n", 10))
                return json.dumps(
                    {
                        "result_id": result_id,
                        "op": "head",
                        "n": n,
                        "rows": list(_iter_rows(meta.path, limit=n)),
                    },
                    default=str,
                )
            if kind == "tail":
                n = int(op.get("n", 10))
                all_rows = list(_iter_rows(meta.path))
                return json.dumps(
                    {
                        "result_id": result_id,
                        "op": "tail",
                        "n": n,
                        "rows": all_rows[-n:],
                    },
                    default=str,
                )
            if kind == "count":
                return json.dumps(
                    {"result_id": result_id, "op": "count", "count": meta.row_count or 0}
                )
            if kind == "count_by":
                column = op["column"]
                top = int(op.get("top", 50))
                counts: dict[str, int] = {}
                for row in _iter_rows(meta.path):
                    v = row.get(column)
                    key = "<missing>" if v is None and column not in row else (
                        "<null>" if v is None else str(v)
                    )
                    counts[key] = counts.get(key, 0) + 1
                histogram = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
                return json.dumps(
                    {
                        "result_id": result_id,
                        "op": "count_by",
                        "column": column,
                        "top": top,
                        "distinct": len(counts),
                        "histogram": [{"value": v, "count": c} for v, c in histogram],
                    },
                    default=str,
                )
            if kind in ("sum", "avg", "min", "max"):
                column = op["column"]
                total = 0.0
                n = 0
                mn = float("inf")
                mx = float("-inf")
                for row in _iter_rows(meta.path):
                    v = _to_number(row.get(column))
                    if v is None:
                        continue
                    total += v
                    n += 1
                    if v < mn:
                        mn = v
                    if v > mx:
                        mx = v
                if kind == "sum":
                    value: Any = total
                elif kind == "avg":
                    value = None if n == 0 else total / n
                elif kind == "min":
                    value = None if n == 0 else mn
                else:
                    value = None if n == 0 else mx
                return json.dumps(
                    {
                        "result_id": result_id,
                        "op": kind,
                        "column": column,
                        "numeric_rows": n,
                        "value": value,
                    },
                    default=str,
                )
            if kind == "peek":
                bytes_ = int(op.get("bytes", 2000))
                with open(meta.path, "r", encoding="utf-8") as f:
                    buf = f.read()
                return json.dumps(
                    {
                        "result_id": result_id,
                        "op": "peek",
                        "bytes": bytes_,
                        "total_chars": len(buf),
                        "text": buf[:bytes_],
                    }
                )
            if kind == "search":
                needle = op["needle"]
                around = int(op.get("around", 80))
                max_hits = int(op.get("max_hits", 10))
                with open(meta.path, "r", encoding="utf-8") as f:
                    buf = f.read()
                hits: list[dict[str, Any]] = []
                start = 0
                while len(hits) < max_hits:
                    at = buf.find(needle, start)
                    if at == -1:
                        break
                    s = max(0, at - around)
                    e = min(len(buf), at + len(needle) + around)
                    hits.append({"offset": at, "context": buf[s:e]})
                    start = at + len(needle)
                return json.dumps(
                    {
                        "result_id": result_id,
                        "op": "search",
                        "needle": needle,
                        "around": around,
                        "hits": hits,
                        "truncated": len(hits) == max_hits,
                    }
                )
            return json.dumps({"error": f'unknown op kind: {kind!r}'})
        except KeyError as e:
            return json.dumps({"error": f"missing required field: {e.args[0]}"})
        except Exception as e:
            return json.dumps({"error": f"read_result failed: {e}"})


# ── Internal helpers ────────────────────────────────────────────────────────


@dataclass
class _SpillMeta:
    result_id: str
    path: str
    fmt: str  # 'jsonl' | 'json' | 'text'
    row_count: int | None
    columns: Any
    parent_keys: list[str] | None
    size_bytes: int


@dataclass
class _ParsedRows:
    rows: list[Any]
    columns: Any
    parent_keys: list[str]


def _find_rows(parsed: Any) -> _ParsedRows | None:
    """Look for a `rows` array at top level or under `sample`."""
    if not isinstance(parsed, dict):
        return None
    if isinstance(parsed.get("rows"), list):
        return _ParsedRows(
            rows=parsed["rows"],
            columns=parsed.get("columns"),
            parent_keys=[k for k in parsed.keys() if k != "rows"],
        )
    sample = parsed.get("sample")
    if isinstance(sample, dict) and isinstance(sample.get("rows"), list):
        return _ParsedRows(
            rows=sample["rows"],
            columns=sample.get("columns"),
            parent_keys=["sample"] + [k for k in parsed.keys() if k != "sample"],
        )
    return None


_MAX_INLINE_CHARS = 200


def _truncate_for_inline(v: Any) -> Any:
    if isinstance(v, str) and len(v) > _MAX_INLINE_CHARS:
        return v[:_MAX_INLINE_CHARS] + f"…[+{len(v) - _MAX_INLINE_CHARS}]"
    return v


def _truncate_row_inline(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    return {k: _truncate_for_inline(v) for k, v in row.items()}


def _iter_rows(path: str, limit: int | None = None) -> Any:
    """Yield parsed JSON rows from a JSONL file. Skips malformed lines."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield row
            n += 1
            if limit is not None and n >= limit:
                return


def _to_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None  # don't sum booleans
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            return None
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


# JSON Schema for read_result. Union of nine op shapes — kept inline so the
# extension is a single file.
_READ_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "result_id": {
            "type": "string",
            "description": "The result_id returned in a prior spilled tool envelope.",
        },
        "op": {
            "type": "object",
            "description": (
                "One of: {kind: 'head', n?}, {kind: 'tail', n?}, {kind: 'count'}, "
                "{kind: 'count_by', column, top?}, "
                "{kind: 'sum'|'avg'|'min'|'max', column}, "
                "{kind: 'peek', bytes?}, {kind: 'search', needle, around?, max_hits?}."
            ),
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "head", "tail", "count", "count_by",
                        "sum", "avg", "min", "max",
                        "peek", "search",
                    ],
                },
                "n": {"type": "integer", "minimum": 1, "maximum": 200},
                "column": {"type": "string"},
                "top": {"type": "integer", "minimum": 1, "maximum": 200},
                "bytes": {"type": "integer", "minimum": 1, "maximum": 32000},
                "needle": {"type": "string", "minLength": 1},
                "around": {"type": "integer", "minimum": 0, "maximum": 500},
                "max_hits": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["kind"],
        },
    },
    "required": ["result_id", "op"],
}


class _SpillMiddleware(Middleware):
    """Middleware that runs after every tool execution; defers to the
    parent extension for the actual spill decision."""

    def __init__(self, ext: ResultSpillExtension) -> None:
        self._ext = ext

    async def after_tool_execution(self, result: Any) -> Any:
        if isinstance(result, ToolResult):
            return self._ext._maybe_spill(result)
        return result
