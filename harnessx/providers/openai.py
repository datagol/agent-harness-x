"""OpenAI provider — translates between OpenAI's chat-completions format
and the harness's canonical format. Supports both create and stream.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "OpenAIProvider requires the openai package. "
        "Install with: pip install 'harnessx[openai]'"
    ) from e

from ..types import ProviderResponse, StopReason, StreamChunk, TokenUsage, ToolCall
from ..types import PromptCacheHint
from ..errors import IncompleteStreamError
from .base import normalize_tool_choice, closing_stream, LLMProvider, parse_tool_arguments


class OpenAIProvider(LLMProvider):
    """Talks to OpenAI's Chat Completions API and normalizes responses to ProviderResponse."""

    name = "openai"
    # OpenAI caches prompt prefixes automatically; prompt_cache_key only steers
    # routing so identical prefixes land on the same cache. Gateways that reject
    # unknown parameters set this to False.
    supports_prompt_cache_key = True

    def __init__(self, client: AsyncOpenAI | None = None, *,
                 allow_missing_finish_reason_for_text: bool = False,
                 tool_choice: str | None = None) -> None:
        # OpenAI spells "any" as "required"; see base.TOOL_CHOICES.
        self.tool_choice = normalize_tool_choice(tool_choice, env_var="OPENAI_TOOL_CHOICE")
        # The engine owns retries (RetryPolicy); the SDK's own would compound them invisibly.
        self.client = client or AsyncOpenAI(max_retries=0)
        self.allow_missing_finish_reason_for_text = allow_missing_finish_reason_for_text

    def _apply_prompt_cache(self, kwargs: dict[str, Any], cache: PromptCacheHint | None) -> None:
        if cache is None or not cache.enabled or not self.supports_prompt_cache_key:
            return
        extra = dict(kwargs.get("extra_body") or {})
        extra["prompt_cache_key"] = cache.prefix_key
        kwargs["extra_body"] = extra

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
        openai_messages = _to_openai_messages(messages, system)
        openai_tools = _to_openai_tools(tools)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if openai_tools:
            kwargs["tools"] = openai_tools
            if self.tool_choice:
                kwargs["tool_choice"] = (
                    "required" if self.tool_choice == "any" else self.tool_choice
                )
        self._apply_prompt_cache(kwargs, cache)

        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "temperature" in str(exc) and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                resp = await self.client.chat.completions.create(**kwargs)
            else:
                raise
        return _from_openai_response(resp)

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
        openai_messages = _to_openai_messages(messages, system)
        openai_tools = _to_openai_tools(tools)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if openai_tools:
            kwargs["tools"] = openai_tools
            if self.tool_choice:
                kwargs["tool_choice"] = (
                    "required" if self.tool_choice == "any" else self.tool_choice
                )
        self._apply_prompt_cache(kwargs, cache)

        collected_text = ""
        collected_refusal = ""
        collected_reasoning = ""
        collected_usage = TokenUsage()
        finish_reason = None
        tool_call_chunks: dict[int, dict[str, Any]] = {}

        try:
            stream_resp = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            retry = False
            exc_str = str(exc).lower()
            if "temperature" in exc_str and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                retry = True
            if "stream_options" in exc_str and "stream_options" in kwargs:
                kwargs.pop("stream_options", None)
                retry = True
            if retry:
                stream_resp = await self.client.chat.completions.create(**kwargs)
            else:
                raise
        async with closing_stream(stream_resp):
            async for chunk in stream_resp:
                if getattr(chunk, "usage", None):
                    collected_usage = _extract_token_usage(chunk.usage)

                if not chunk.choices:
                    continue
                # A tool call's arguments arrive as deltas with no text; say the
                # stream is alive so a long argument is not taken for a stall.
                yield StreamChunk(kind="progress")
                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                reasoning = _extract_reasoning(delta)
                if reasoning:
                    collected_reasoning += reasoning
                    yield StreamChunk(kind="thinking_delta", data=reasoning)

                if getattr(delta, "content", None):
                    text = delta.content
                    collected_text += text
                    yield StreamChunk(kind="text_delta", data=text)
                if isinstance(getattr(delta, "refusal", None), str) and delta.refusal:
                    collected_refusal += delta.refusal

                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_chunks:
                            tool_call_chunks[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name if tc.function and tc.function.name else "",
                                "arguments": tc.function.arguments if tc.function and tc.function.arguments else "",
                            }
                        else:
                            if tc.id:
                                tool_call_chunks[idx]["id"] += tc.id
                            if tc.function and tc.function.name:
                                tool_call_chunks[idx]["name"] += tc.function.name
                            if tc.function and tc.function.arguments:
                                tool_call_chunks[idx]["arguments"] += tc.function.arguments

        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_call_chunks.keys()):
            tc_data = tool_call_chunks[idx]
            tool_calls.append(
                ToolCall(
                    id=tc_data["id"] or _synthetic_call_id(),
                    name=tc_data["name"],
                    input=parse_tool_arguments(tc_data["arguments"]),
                )
            )

        if not finish_reason and (
            tool_call_chunks or not (collected_text or collected_refusal).strip()
            or not self.allow_missing_finish_reason_for_text
        ):
            # Missing termination is an interrupted response. Compatibility
            # mode permits nonempty text only; it never permits tool calls.
            raise IncompleteStreamError("Stream ended without a finish reason")
        stop_reason = _stop_reason(finish_reason, collected_refusal)

        final_response = ProviderResponse(
            text=collected_text or collected_refusal,
            tool_calls=tool_calls,
            thinking=collected_reasoning or None,
            stop_reason=stop_reason,
            usage=collected_usage,
        )
        yield StreamChunk(kind="response", data=final_response)

    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
    ) -> int:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            return (len(str(messages)) + len(system or "") + len(str(tools))) // 3

        text = system or ""
        for m in messages:
            text += "\n" + json.dumps(m, default=str)
        for t in tools or []:
            text += "\n" + json.dumps(t, default=str)
        return len(enc.encode(text))


# ── Translation helpers ─────────────────────────────────────────────────────


#: Chat Completions wants an audio container name where the canonical block
#: carries a media type.
_OPENAI_AUDIO_FORMATS: dict[str, str] = {
    "audio/wav": "wav",
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
}


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object"}),
            },
        }
        for t in tools or []
    ]


def _to_openai_messages(
    messages: list[dict[str, Any]], system: str | None
) -> list[dict[str, Any]]:
    """Translate canonical messages into OpenAI's format."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for m in messages:
        role = m["role"]
        content = m.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content or []:
                btype = _block_type(block)
                if btype == "text":
                    text_parts.append(_block_attr(block, "text") or "")
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": _block_attr(block, "id"),
                        "type": "function",
                        "function": {
                            "name": _block_attr(block, "name"),
                            "arguments": json.dumps(_block_attr(block, "input") or {}),
                        },
                    })

            msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                msg["content"] = "\n".join(text_parts)
            else:
                msg["content"] = None
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if text_parts or tool_calls:
                out.append(msg)
            continue

        text_parts = []
        media_parts: list[dict[str, Any]] = []
        for block in content or []:
            btype = _block_type(block)
            if btype in ("image", "document", "audio"):
                source = _block_attr(block, "source") or {}
                media_type = source.get("media_type") or ""
                data = source.get("data") or ""
                if btype == "image":
                    media_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    })
                elif btype == "audio":
                    # Chat Completions names the container, not the media type.
                    fmt = _OPENAI_AUDIO_FORMATS.get(media_type)
                    if fmt is None:
                        raise ValueError(
                            f"OpenAI does not accept {media_type!r} audio; "
                            f"send one of {', '.join(sorted(set(_OPENAI_AUDIO_FORMATS.values())))}"
                        )
                    media_parts.append({
                        "type": "input_audio",
                        "input_audio": {"data": data, "format": fmt},
                    })
                else:
                    media_parts.append({
                        "type": "file",
                        "file": {
                            "filename": "document.pdf",
                            "file_data": f"data:{media_type};base64,{data}",
                        },
                    })
            elif btype == "tool_result":
                tc_id = _block_attr(block, "tool_use_id") or _block_attr(block, "tool_call_id")
                tr_content = _block_attr(block, "content")
                if isinstance(tr_content, list):
                    tr_content = "\n".join(
                        _block_attr(b, "text") or "" for b in tr_content
                        if _block_type(b) == "text"
                    )
                out.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": str(tr_content or ""),
                })
            elif btype == "text":
                text_parts.append(_block_attr(block, "text") or "")
        if media_parts:
            # Mixed content has to go as a parts list; text alone stays a plain
            # string so nothing changes for the overwhelming majority of calls.
            joined = "\n".join(t for t in text_parts if t)
            parts: list[dict[str, Any]] = []
            if joined:
                parts.append({"type": "text", "text": joined})
            parts.extend(media_parts)
            out.append({"role": "user", "content": parts})
        elif text_parts:
            out.append({"role": "user", "content": "\n".join(text_parts)})

    return out


def _extract_reasoning(obj: Any) -> str | None:
    """Extract thinking/reasoning text from a delta or message object."""
    if obj is None:
        return None
    val = getattr(obj, "reasoning", None) or getattr(obj, "reasoning_content", None)
    if val:
        return str(val)
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        val = model_extra.get("reasoning") or model_extra.get("reasoning_content")
        if val:
            return str(val)
    if isinstance(obj, dict):
        val = obj.get("reasoning") or obj.get("reasoning_content")
        if val:
            return str(val)
    return None


def _extract_token_usage(raw_usage: Any) -> TokenUsage:
    """Normalize raw usage object or dict into canonical TokenUsage."""
    if not raw_usage:
        return TokenUsage()
    if isinstance(raw_usage, dict):
        p_details = raw_usage.get("prompt_tokens_details") or {}
        cached = (
            p_details.get("cached_tokens", 0)
            if isinstance(p_details, dict)
            else getattr(p_details, "cached_tokens", 0) or 0
        )
        c_details = raw_usage.get("completion_tokens_details") or {}
        return TokenUsage(
            input_tokens=raw_usage.get("prompt_tokens", 0) or 0,
            output_tokens=raw_usage.get("completion_tokens", 0) or 0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached,
            thinking_tokens=(
                c_details.get("reasoning_tokens", 0) or 0
                if isinstance(c_details, dict)
                else getattr(c_details, "reasoning_tokens", 0) or 0
            ),
        )
    return TokenUsage(
        input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
        cache_creation_input_tokens=0,
        thinking_tokens=getattr(
            getattr(raw_usage, "completion_tokens_details", None), "reasoning_tokens", 0
        ) or 0,
        cache_read_input_tokens=getattr(
            getattr(raw_usage, "prompt_tokens_details", None), "cached_tokens", 0
        ) or 0,
    )


_FINISH_REASONS = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.SAFETY,
}


def _stop_reason(finish_reason: Any, refusal: str = "") -> StopReason:
    """A declined request is a refusal, whatever finish reason accompanies it."""
    if refusal:
        return StopReason.REFUSAL
    return _FINISH_REASONS.get(finish_reason, StopReason.OTHER if finish_reason else StopReason.END_TURN)


def _synthetic_call_id() -> str:
    """An id for a tool call that arrived without one. Some OpenAI-compatible
    servers omit it; an empty id would fail the whole turn instead of one call."""
    import uuid

    return f"call_{uuid.uuid4().hex[:24]}"


def _from_openai_response(resp: Any) -> ProviderResponse:
    """Wrap an OpenAI ChatCompletion in canonical ProviderResponse."""
    choice = resp.choices[0]
    msg = choice.message

    refusal = getattr(msg, "refusal", None)
    refusal = refusal if isinstance(refusal, str) else ""
    text = msg.content or refusal
    reasoning = _extract_reasoning(msg)
    tool_calls: list[ToolCall] = []
    for tc in (msg.tool_calls or []):
        tool_calls.append(
            ToolCall(
                id=tc.id or _synthetic_call_id(),
                name=tc.function.name,
                input=parse_tool_arguments(tc.function.arguments),
            )
        )

    stop_reason = _stop_reason(choice.finish_reason, refusal)

    usage = _extract_token_usage(getattr(resp, "usage", None))

    return ProviderResponse(
        text=text,
        tool_calls=tool_calls,
        thinking=reasoning or None,
        stop_reason=stop_reason,
        usage=usage,
        raw=resp,
    )


def _block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_attr(block: Any, name: str) -> Any:
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)
