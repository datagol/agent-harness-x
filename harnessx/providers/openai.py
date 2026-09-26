"""OpenAI provider — translates between OpenAI's chat-completions format
and the harness's canonical format. Supports both create and stream.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, AsyncIterator

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "OpenAIProvider requires the openai package. "
        "Install with: pip install 'harnessx[openai]'"
    ) from e

from ..types import ProviderResponse, StopReason, StreamChunk, TokenUsage, ToolCall
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """Talks to OpenAI's Chat Completions API and normalizes responses to ProviderResponse."""

    name = "openai"

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self.client = client or AsyncOpenAI()

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
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

        collected_text = ""
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
        async for chunk in stream_resp:
            if getattr(chunk, "usage", None):
                collected_usage = _extract_token_usage(chunk.usage)

            if not chunk.choices:
                continue
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
            try:
                args = json.loads(tc_data["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc_data["arguments"]}
            tool_calls.append(
                ToolCall(
                    id=tc_data["id"],
                    name=tc_data["name"],
                    input=args,
                )
            )

        finish_map = {
            "stop": StopReason.END_TURN,
            "tool_calls": StopReason.TOOL_USE,
            "length": StopReason.MAX_TOKENS,
            "content_filter": StopReason.SAFETY,
        }
        stop_reason = finish_map.get(
            finish_reason, StopReason.OTHER if finish_reason else StopReason.END_TURN
        )

        final_response = ProviderResponse(
            text=collected_text,
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
        system: str,
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
            out.append(msg)
            continue

        text_parts = []
        for block in content or []:
            btype = _block_type(block)
            if btype == "tool_result":
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
        if text_parts:
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
        return TokenUsage(
            input_tokens=raw_usage.get("prompt_tokens", 0) or 0,
            output_tokens=raw_usage.get("completion_tokens", 0) or 0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached,
        )
    return TokenUsage(
        input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=getattr(
            getattr(raw_usage, "prompt_tokens_details", None), "cached_tokens", 0
        ) or 0,
    )


def _from_openai_response(resp: Any) -> ProviderResponse:
    """Wrap an OpenAI ChatCompletion in canonical ProviderResponse."""
    choice = resp.choices[0]
    msg = choice.message

    text = msg.content or ""
    reasoning = _extract_reasoning(msg)
    tool_calls: list[ToolCall] = []
    for tc in (msg.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw": tc.function.arguments}
        tool_calls.append(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=args,
            )
        )

    finish_map = {
        "stop": StopReason.END_TURN,
        "tool_calls": StopReason.TOOL_USE,
        "length": StopReason.MAX_TOKENS,
        "content_filter": StopReason.SAFETY,
    }
    stop_reason = finish_map.get(
        choice.finish_reason, StopReason.OTHER if choice.finish_reason else StopReason.END_TURN
    )

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
