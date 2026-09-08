"""Anthropic provider — thin wrapper around AsyncAnthropic."""

from __future__ import annotations

from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from ..types import ProviderResponse, StopReason, StreamChunk, TokenUsage, ToolCall
from .base import LLMProvider


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
        temperature: float,
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        resp = await self.client.messages.create(**kwargs)
        return _from_anthropic_response(resp)

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[StreamChunk]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(kind="text_delta", data=text)
            final_resp = await stream.get_final_message()
            yield StreamChunk(kind="response", data=_from_anthropic_response(final_resp))

    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
    ) -> int:
        try:
            resp = await self.client.messages.count_tokens(
                model=model,
                system=system,
                tools=tools or [],
                messages=messages,
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
    usage = TokenUsage(
        input_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
        output_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
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
