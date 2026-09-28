"""Anthropic provider — thin wrapper around AsyncAnthropic."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from ..types import ProviderResponse, StopReason, StreamChunk, TokenUsage, ToolCall
from ..types import PromptCacheHint
from .base import LLMProvider


def _filter_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs based on the callable's signature if it doesn't accept arbitrary **kwargs."""
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return dict(kwargs)


def _messages_for_request(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove SDK-only text metadata, including from previously saved history."""
    prepared = deepcopy(messages)

    def clean_content(content: Any) -> None:
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                block.pop("parsed_output", None)
            elif block.get("type") == "tool_result":
                clean_content(block.get("content"))

    for message in prepared:
        clean_content(message.get("content"))
    return prepared


def _cache_control(cache: PromptCacheHint) -> dict[str, Any]:
    control: dict[str, Any] = {"type": "ephemeral"}
    if cache.ttl_seconds is not None and cache.ttl_seconds >= 3600:
        control["ttl"] = "1h"
    return control


def _apply_prompt_cache(kwargs: dict[str, Any], cache: PromptCacheHint | None) -> dict[str, Any]:
    """Place cache_control markers at the hint's breakpoints.

    Anthropic caches the prefix up to each marker. The system prompt, the tool
    list, and the last message each get one, which stays under the API's
    four-marker limit. Blocks that cannot carry a marker are left alone.
    """
    if cache is None or not cache.enabled or not cache.breakpoints:
        return kwargs
    out = dict(kwargs)
    control = _cache_control(cache)
    for breakpoint in cache.breakpoints:
        if breakpoint == "system" and out.get("system"):
            system = out["system"]
            if isinstance(system, str):
                out["system"] = [{"type": "text", "text": system, "cache_control": dict(control)}]
            elif isinstance(system, list) and system and isinstance(system[-1], dict):
                system = deepcopy(system)
                system[-1]["cache_control"] = dict(control)
                out["system"] = system
        elif breakpoint == "tools" and out.get("tools"):
            tools = deepcopy(out["tools"])
            if isinstance(tools[-1], dict):
                tools[-1]["cache_control"] = dict(control)
                out["tools"] = tools
        elif breakpoint.startswith("message:"):
            try:
                index = int(breakpoint.split(":", 1)[1])
            except ValueError:
                continue
            messages = out.get("messages") or []
            if not 0 <= index < len(messages):
                continue
            messages = deepcopy(messages)
            message = messages[index]
            content = message.get("content")
            if isinstance(content, str) and content:
                message["content"] = [{"type": "text", "text": content, "cache_control": dict(control)}]
            elif isinstance(content, list) and content and isinstance(content[-1], dict):
                if content[-1].get("type") not in ("thinking", "redacted_thinking"):
                    content[-1]["cache_control"] = dict(control)
            out["messages"] = messages
    return out


def _rejected_cache_control(exc: BaseException) -> bool:
    return "cache_control" in str(exc)


class AnthropicProvider(LLMProvider):
    """Talks to Anthropic's Messages API and normalizes responses to ProviderResponse."""

    name = "anthropic"

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        self.client = client or AsyncAnthropic()

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
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _messages_for_request(messages),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        uncached = _filter_kwargs(self.client.messages.create, kwargs)
        filtered_kwargs = _filter_kwargs(self.client.messages.create, _apply_prompt_cache(kwargs, cache))
        try:
            resp = await self.client.messages.create(**filtered_kwargs)
        except TypeError as exc:
            if "temperature" in str(exc) and "temperature" in filtered_kwargs:
                filtered_kwargs.pop("temperature", None)
                resp = await self.client.messages.create(**filtered_kwargs)
            else:
                raise
        except Exception as exc:
            # Caching is an optimization: a rejected marker runs the call uncached.
            if filtered_kwargs is not uncached and filtered_kwargs != uncached and _rejected_cache_control(exc):
                resp = await self.client.messages.create(**uncached)
            else:
                raise
        return _from_anthropic_response(resp)

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
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _messages_for_request(messages),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        uncached = _filter_kwargs(self.client.messages.stream, kwargs)
        filtered_kwargs = _filter_kwargs(self.client.messages.stream, _apply_prompt_cache(kwargs, cache))
        try:
            stream_ctx = self.client.messages.stream(**filtered_kwargs)
        except TypeError as exc:
            if "temperature" in str(exc) and "temperature" in filtered_kwargs:
                filtered_kwargs.pop("temperature", None)
                stream_ctx = self.client.messages.stream(**filtered_kwargs)
            else:
                raise

        try:
            async with stream_ctx as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(kind="text_delta", data=text)
                final_resp = await stream.get_final_message()
        except Exception as exc:
            # Only before any output: a rejected marker fails at stream open.
            if filtered_kwargs != uncached and _rejected_cache_control(exc):
                async with self.client.messages.stream(**uncached) as stream:
                    async for text in stream.text_stream:
                        yield StreamChunk(kind="text_delta", data=text)
                    final_resp = await stream.get_final_message()
            else:
                raise
        yield StreamChunk(kind="response", data=_from_anthropic_response(final_resp))

    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
    ) -> int:
        try:
            resp = await self.client.messages.count_tokens(
                model=model,
                tools=tools or [],
                messages=_messages_for_request(messages),
                **({"system": system} if system else {}),
            )
            return int(resp.input_tokens)
        except Exception:
            return len(str(messages)) // 3


def _from_anthropic_response(resp: Any) -> ProviderResponse:
    """Normalize raw Anthropic Messages API response into canonical ProviderResponse."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    thinking: str | None = None

    raw_content = getattr(resp, "content", []) or []
    for block in raw_content:
        b_type = getattr(block, "type", None)
        if b_type == "text":
            text_parts.append(getattr(block, "text", ""))
        elif b_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=getattr(block, "id", ""),
                    name=getattr(block, "name", ""),
                    input=getattr(block, "input", {}) or {},
                )
            )
        elif b_type == "thinking":
            thinking = getattr(block, "thinking", "")

    raw_stop = getattr(resp, "stop_reason", None)
    stop_map = {
        "end_turn": StopReason.END_TURN,
        "tool_use": StopReason.TOOL_USE,
        "max_tokens": StopReason.MAX_TOKENS,
        "stop_sequence": StopReason.STOP_SEQUENCE,
    }
    stop_reason = stop_map.get(raw_stop, StopReason.OTHER if raw_stop else StopReason.END_TURN)

    raw_usage = getattr(resp, "usage", None)
    cache_creation = int(getattr(raw_usage, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(getattr(raw_usage, "cache_read_input_tokens", 0) or 0)
    usage = TokenUsage(
        # Anthropic reports input_tokens net of cache activity; TokenUsage counts
        # every prompt token, with the cache counters as sub-counts, like the
        # OpenAI and Gemini providers do.
        input_tokens=int(getattr(raw_usage, "input_tokens", 0) or 0) + cache_creation + cache_read,
        output_tokens=int(getattr(raw_usage, "output_tokens", 0) or 0),
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )

    return ProviderResponse(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        thinking=thinking,
        stop_reason=stop_reason,
        usage=usage,
        raw=resp,
        content=list(raw_content),
    )
