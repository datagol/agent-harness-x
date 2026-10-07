"""Gemini provider — translates between Google's google-genai wire format
and the harness's canonical (Anthropic-shaped) format. Supports both create
and stream.

Gemini 3.x specifics handled here:

* ``thought_signature`` round-tripping. Gemini 3.x attaches an opaque
  signature to every ``function_call`` part it returns, and REQUIRES that
  signature to be echoed back verbatim on the same part in conversation
  history — otherwise the next request fails with a 400 ("Function call is
  missing a thought_signature"). Since ``ConversationMemory`` stores
  canonical Anthropic-shaped dicts (not genai ``Content`` objects), this
  provider stashes the signature (base64-encoded) on the canonical
  ``tool_use`` block under the namespaced key ``"_gemini_thought_signature"``
  and restores it onto the reconstructed ``function_call`` part when
  translating history back to Gemini contents. Canonical content blocks are
  emitted as plain dicts because ``ConversationMemory.add_assistant_message``
  round-trips dict blocks verbatim (attribute-style blocks are rebuilt and
  would drop the extra key).

* ``automatic_function_calling`` is disabled — the harness loop owns tool
  execution.

* Thinking level is configurable (constructor kwarg ``thinking_level`` or env
  ``GEMINI_THINKING_LEVEL``) rather than hardcoded, because valid levels vary
  by model (e.g. gemini-3.8-flash rejects "minimal" while 3.6 accepts it).
  When unset, no thinking_config is sent and the model uses its default.

* Explicit prompt caching is opt-in (constructor kwarg ``prompt_cache_ttl``
  or env ``GEMINI_PROMPT_CACHE_TTL``, seconds; None/0 = disabled). When
  enabled, the static prefix (system_instruction + tools) is uploaded once
  via ``caches.create`` and referenced per call as ``cached_content``; Gemini
  requires exclusivity, so the per-call config then omits system/tools.
  Caching is an optimization, never a correctness dependency: any
  ``caches.create`` failure logs a warning and the call runs uncached, and a
  call failing on a stale/evicted cache name drops the registry entry and
  retries once uncached. Note Gemini enforces a minimum cacheable size
  (4,096 tokens on 3.x flash models); smaller prefixes fail ``caches.create``
  and simply run uncached via the same fail-open path.

Authentication: the genai client reads ``GEMINI_API_KEY`` (or
``GOOGLE_API_KEY``) from the environment; an explicit ``api_key`` kwarg wins.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator

try:
    from google import genai
except ImportError as e:
    raise ImportError(
        "GeminiProvider requires the google-genai package. "
        "Install with: pip install 'harnessx[gemini]'"
    ) from e

from ..types import ProviderResponse, StopReason, StreamChunk, TokenUsage, ToolCall
from ..types import PromptCacheHint
from ..errors import IncompleteStreamError
from ..types import ToolChoice
from .base import normalize_tool_choice, closing_stream, LLMProvider

THOUGHT_SIGNATURE_KEY = "_gemini_thought_signature"

logger = logging.getLogger(__name__)

_CACHE_TTL_MARGIN_S = 60  # recreate slightly before Gemini expires it
_CACHE_RETRY_HOLD_S = 600  # after a failed create, do not retry that prefix for a while


def _is_stale_cache_error(exc: Exception) -> bool:
    """True for a generate call that failed because its cached_content is
    gone (evicted/expired server-side) — the 'drop and retry uncached' case."""
    msg = str(exc).lower()
    return "cachedcontent" in msg.replace("_", "") or "not found" in msg


class GeminiProvider(LLMProvider):
    """Talks to Google's Gemini API (google-genai SDK) and normalizes
    responses to ProviderResponse."""

    name = "gemini"

    def __init__(
        self,
        client: Any | None = None,
        api_key: str | None = None,
        thinking_level: str | None = None,
        prompt_cache_ttl: int | None = None,
        tool_choice: "str | ToolChoice | None" = None,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
            self.client = genai.Client(api_key=key) if key else genai.Client()
        self.thinking_level = thinking_level or os.environ.get(
            "GEMINI_THINKING_LEVEL"
        ) or None
        if prompt_cache_ttl is None:
            prompt_cache_ttl = int(os.environ.get("GEMINI_PROMPT_CACHE_TTL") or 0)
        self.prompt_cache_ttl = prompt_cache_ttl or None  # None/0 = disabled
        # "auto" (Gemini's default) lets the model answer with prose instead of
        # calling a tool, which is right for a chat agent and wrong for one
        # whose every turn is defined as a tool call: the model takes the prose
        # door a few percent of the time and writes the call out as text.
        # "any" removes that door. "none" forbids tools for a turn.
        self.tool_choice = normalize_tool_choice(
            tool_choice, env_var="GEMINI_TOOL_CHOICE"
        )
        # key -> (cache_name, expires_at monotonic)
        self._prompt_caches: dict[str, tuple[str, float]] = {}
        # key -> monotonic time before which caches.create is not retried
        self._cache_failures: dict[str, float] = {}

    # ── request building ────────────────────────────────────────────────

    def _build_config(
        self,
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "automatic_function_calling": {"disable": True},
        }
        if temperature is not None:
            config["temperature"] = temperature
        if system:
            config["system_instruction"] = system
        if tools:
            config["tools"] = [{"function_declarations": _to_gemini_tools(tools)}]
            if self.tool_choice:
                config["tool_config"] = {
                    "function_calling_config": {"mode": self.tool_choice.upper()}
                }
        if self.thinking_level:
            config["thinking_config"] = {"thinking_level": self.thinking_level}
        return config

    # ── explicit prompt caching (opt-in) ────────────────────────────────

    def _cache_key(
        self, model: str, system: str | None, tools: list[dict[str, Any]]
    ) -> str:
        payload = json.dumps(
            {"m": model, "p": system or "", "t": _to_gemini_tools(tools)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_settings(
        self,
        model: str,
        system: str | None,
        tools: list[dict[str, Any]],
        cache: PromptCacheHint | None,
    ) -> tuple[str, int] | None:
        """(key, ttl) when an explicit cache should be used, else None.

        A hint with ``enabled=False`` switches explicit caching off. A hint
        without a TTL, or no hint at all, falls back to the constructor/env
        TTL; when that is unset too, only Gemini's implicit caching applies.
        """
        if cache is not None and not cache.enabled:
            return None
        ttl = (cache.ttl_seconds if cache is not None else None) or self.prompt_cache_ttl
        if not ttl or not (system or tools):
            return None
        key = self._cache_key(model, system, tools)
        if cache is not None and cache.prefix_key:
            key = hashlib.sha256(f"{key}:{cache.prefix_key}".encode()).hexdigest()
        return key, int(ttl)

    async def _cached_config(
        self,
        settings: tuple[str, int] | None,
        model: str,
        system: str | None,
        tools: list[dict[str, Any]],
        base: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the generation config to send, using an explicit prompt
        cache when enabled.

        The cache holds system_instruction + tools (created once per
        (model, system, tools) and reused until TTL); the per-call config
        references it via ``cached_content`` and must then omit system/tools
        (Gemini requires exclusivity). Fails open to ``base`` on any create
        error — including prefixes below Gemini's minimum cacheable size
        (4,096 tokens on 3.x flash models) — and holds off retrying that
        prefix for a while so a small prompt does not fail on every call."""
        if settings is None:
            return base
        key, ttl = settings
        now = time.monotonic()
        cached = self._prompt_caches.get(key)
        if cached is None or cached[1] <= now:
            if self._cache_failures.get(key, 0.0) > now:
                return base
            cache_config: dict[str, Any] = {"ttl": f"{ttl}s"}
            if system:
                cache_config["system_instruction"] = system
            if tools:
                cache_config["tools"] = base["tools"]
                # Gemini refuses a request that carries cached_content and
                # tool_config together: "CachedContent can not be used with
                # GenerateContent request setting system_instruction, tools or
                # tool_config." It travels with the tools it constrains.
                if "tool_config" in base:
                    cache_config["tool_config"] = base["tool_config"]
            try:
                created = await self.client.aio.caches.create(
                    model=model, config=cache_config
                )
            except Exception as exc:
                self._cache_failures[key] = now + _CACHE_RETRY_HOLD_S
                logger.warning(
                    "Gemini prompt cache create failed; running uncached: %s", exc
                )
                return base
            self._cache_failures.pop(key, None)
            self._prompt_caches[key] = (
                created.name,
                now + max(60, ttl - _CACHE_TTL_MARGIN_S),
            )
        config = {
            k: v
            for k, v in base.items()
            if k not in ("system_instruction", "tools", "tool_config")
        }
        config["cached_content"] = self._prompt_caches[key][0]
        return config

    def _drop_cache(self, settings: tuple[str, int] | None) -> None:
        if settings is not None:
            self._prompt_caches.pop(settings[0], None)

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        cache: PromptCacheHint | None = None,
    ) -> ProviderResponse:
        base = self._build_config(system, tools, max_tokens, temperature)
        settings = self._cache_settings(model, system, tools, cache)
        config = await self._cached_config(settings, model, system, tools, base)
        contents = _to_gemini_contents(messages)
        try:
            resp = await self.client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:
            if "cached_content" not in config or not _is_stale_cache_error(exc):
                raise
            # Stale/evicted cache name: drop it and retry this call uncached.
            self._drop_cache(settings)
            resp = await self.client.aio.models.generate_content(
                model=model, contents=contents, config=base
            )
        candidates = getattr(resp, "candidates", None) or []
        parts: list[Any] = []
        finish_reason = None
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = list(getattr(content, "parts", None) or [])
            finish_reason = getattr(candidates[0], "finish_reason", None)
        return _from_gemini_parts(
            parts, finish_reason, getattr(resp, "usage_metadata", None), raw=resp
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        cache: PromptCacheHint | None = None,
    ) -> AsyncIterator[StreamChunk]:
        base = self._build_config(system, tools, max_tokens, temperature)
        settings = self._cache_settings(model, system, tools, cache)
        config = await self._cached_config(settings, model, system, tools, base)
        contents = _to_gemini_contents(messages)
        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=model, contents=contents, config=config
            )
        except Exception as exc:
            if "cached_content" not in config or not _is_stale_cache_error(exc):
                raise
            # Stale/evicted cache name: drop it and retry this call uncached.
            self._drop_cache(settings)
            stream = await self.client.aio.models.generate_content_stream(
                model=model, contents=contents, config=base
            )

        all_parts: list[Any] = []
        finish_reason = None
        usage_metadata = None
        blocked = False
        async with closing_stream(stream):
            async for chunk in stream:
                yield StreamChunk(kind="progress")  # alive, whether or not it carries text
                if getattr(chunk, "usage_metadata", None) is not None:
                    usage_metadata = chunk.usage_metadata
                feedback = getattr(chunk, "prompt_feedback", None)
                block_reason = str(getattr(feedback, "block_reason", "") or "").split(".")[-1]
                blocked = blocked or block_reason not in ("", "0", "BLOCK_REASON_UNSPECIFIED")
                candidates = getattr(chunk, "candidates", None) or []
                if not candidates:
                    continue
                if getattr(candidates[0], "finish_reason", None):
                    finish_reason = candidates[0].finish_reason
                content = getattr(candidates[0], "content", None)
                for part in list(getattr(content, "parts", None) or []):
                    all_parts.append(part)
                    text = getattr(part, "text", None)
                    if text and getattr(part, "thought", False):
                        # Thinking summaries are not user-visible text.
                        yield StreamChunk(kind="thinking_delta", data=text)
                    elif text:
                        yield StreamChunk(kind="text_delta", data=text)

        if blocked:
            finish_reason = "SAFETY"
        elif str(finish_reason or "").split(".")[-1] in ("", "0", "FINISH_REASON_UNSPECIFIED"):
            raise IncompleteStreamError("Gemini stream ended without a finish reason")

        yield StreamChunk(
            kind="response",
            data=_from_gemini_parts(all_parts, finish_reason, usage_metadata),
        )

    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
    ) -> int:
        try:
            resp = await self.client.aio.models.count_tokens(
                model=model,
                contents=_to_gemini_contents(messages),
            )
            total = int(getattr(resp, "total_tokens", 0) or 0)
            # count_tokens has no config parameter; approximate the system
            # prompt and tool declarations on top of the message count.
            return total + (len(system or "") + len(str(tools or []))) // 4
        except Exception:
            return (len(str(messages)) + len(system or "") + len(str(tools))) // 3


# ── Translation helpers ─────────────────────────────────────────────────────


# JSON Schema keywords Gemini's Schema type does not define. Passing one
# through is tolerated by generateContent and rejected outright by
# cachedContents, so a tool carrying it silently costs the caller prompt
# caching: "Unknown name \"additional_properties\" at
# cached_content.tools[0].function_declarations[8].parameters".
#
# Declaration 8 there is this package's own read_tool_result, whose schema is
# generated from its signature and includes additionalProperties, so every
# Gemini agent built on 0.4 lost explicit caching without any sign of it
# beyond one warning line.
_UNSUPPORTED_SCHEMA_KEYS = ("additionalProperties", "$schema", "additional_properties")


def _strip_unsupported(schema: Any) -> Any:
    """A schema Gemini accepts, with unknown keywords removed at every depth."""
    if isinstance(schema, dict):
        return {
            key: _strip_unsupported(value)
            for key, value in schema.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_strip_unsupported(item) for item in schema]
    return schema


def _to_gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic tool params -> Gemini functionDeclarations."""
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": _strip_unsupported(t.get("input_schema", {"type": "object"})),
        }
        for t in tools or []
    ]


def _to_gemini_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate canonical (Anthropic-shaped) messages into Gemini contents.

    Restores stashed thought signatures onto function_call parts, and maps
    tool_result blocks (addressed by tool_use_id) back to function_response
    parts (addressed by function name) via an id->name map built from prior
    assistant tool_use blocks.
    """
    contents: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}  # tool_use id -> function name

    for m in messages:
        role = m["role"]
        content = m.get("content")
        gemini_role = "model" if role == "assistant" else "user"

        if isinstance(content, str):
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
            continue

        parts: list[dict[str, Any]] = []
        for block in content or []:
            btype = _block_type(block)
            if btype == "text":
                text = _block_attr(block, "text") or ""
                if text:
                    parts.append({"text": text})
            elif btype == "tool_use":
                tc_id = _block_attr(block, "id") or ""
                name = _block_attr(block, "name") or ""
                if tc_id:
                    tool_names[tc_id] = name
                part: dict[str, Any] = {
                    "function_call": {
                        "name": name,
                        "args": _block_attr(block, "input") or {},
                    }
                }
                sig = _block_attr(block, THOUGHT_SIGNATURE_KEY)
                if sig:
                    # Gemini 3.x hard requirement: echo the signature back
                    # verbatim or the request 400s.
                    part["thought_signature"] = base64.b64decode(sig)
                parts.append(part)
            elif btype in ("image", "document", "audio"):
                # Gemini takes any of the three the same way, as inline bytes
                # with their media type. The SDK accepts raw bytes here, and
                # decoding once is cheaper than shipping base64 over the wire.
                source = _block_attr(block, "source") or {}
                media_type = source.get("media_type") or ""
                data = source.get("data") or ""
                parts.append(
                    {"inline_data": {"mime_type": media_type, "data": base64.b64decode(data)}}
                )
            elif btype == "tool_result":
                tc_id = (
                    _block_attr(block, "tool_use_id")
                    or _block_attr(block, "tool_call_id")
                    or ""
                )
                tr_content = _block_attr(block, "content")
                if isinstance(tr_content, list):
                    tr_content = "\n".join(
                        _block_attr(b, "text") or ""
                        for b in tr_content
                        if _block_type(b) == "text"
                    )
                response: dict[str, Any] = {"output": str(tr_content or "")}
                if _block_attr(block, "is_error"):
                    response["is_error"] = True
                parts.append(
                    {
                        "function_response": {
                            "name": tool_names.get(tc_id, tc_id),
                            "response": response,
                        }
                    }
                )
        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    return contents


def _from_gemini_parts(
    parts: list[Any],
    finish_reason: Any,
    usage_metadata: Any,
    raw: Any = None,
) -> ProviderResponse:
    """Normalize Gemini response parts into canonical ProviderResponse.

    Content blocks are emitted as plain dicts so ConversationMemory stores
    them verbatim, preserving the stashed thought signature across turns.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    blocks: list[dict[str, Any]] = []

    for part in parts:
        text = getattr(part, "text", None)
        if getattr(part, "thought", False):
            if text:
                thinking_parts.append(text)
            continue
        if text:
            text_parts.append(text)
            blocks.append({"type": "text", "text": text})
        call = getattr(part, "function_call", None)
        if call is not None:
            tc = ToolCall(
                id=getattr(call, "id", None) or f"call_{uuid.uuid4().hex[:12]}",
                name=getattr(call, "name", "") or "",
                input=dict(getattr(call, "args", None) or {}),
            )
            tool_calls.append(tc)
            block: dict[str, Any] = {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input,
            }
            sig = getattr(part, "thought_signature", None)
            if sig:
                if isinstance(sig, str):
                    sig = sig.encode()
                block[THOUGHT_SIGNATURE_KEY] = base64.b64encode(sig).decode()
            blocks.append(block)

    fr = str(finish_reason or "").split(".")[-1].upper()
    if fr == "MAX_TOKENS":
        # Even with tool calls: their arguments may be the part that was cut.
        stop_reason: StopReason = StopReason.MAX_TOKENS
    elif fr in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION", "IMAGE_SAFETY"):
        stop_reason = StopReason.SAFETY
    elif tool_calls:
        stop_reason = StopReason.TOOL_USE
    elif fr in ("STOP", "FINISH_REASON_UNSPECIFIED", ""):
        stop_reason = StopReason.END_TURN
    else:
        # MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL, OTHER, ...: not a
        # finished answer. The loop treats an empty one as a reply to nudge.
        stop_reason = StopReason.OTHER

    usage = TokenUsage(
        input_tokens=int(getattr(usage_metadata, "prompt_token_count", 0) or 0),
        # Every billed output token, thoughts included, as Anthropic and OpenAI
        # report it: Gemini counts thoughts apart from candidates, and cost
        # estimated from output_tokens left out the most expensive part.
        output_tokens=int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
        + int(getattr(usage_metadata, "thoughts_token_count", 0) or 0),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=int(
            getattr(usage_metadata, "cached_content_token_count", 0) or 0
        ),
        thinking_tokens=int(
            getattr(usage_metadata, "thoughts_token_count", 0) or 0
        ),
    )

    return ProviderResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        thinking="".join(thinking_parts) or None,
        stop_reason=stop_reason,
        usage=usage,
        raw=raw,
        content=blocks,
    )


def _block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_attr(block: Any, name: str) -> Any:
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)
