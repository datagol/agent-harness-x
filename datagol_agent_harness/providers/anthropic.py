"""Anthropic provider — thin wrapper around AsyncAnthropic."""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from .base import LLMProvider, ProviderResponse


class AnthropicProvider(LLMProvider):
    """Talks to Anthropic's Messages API. The canonical response shape is
    Anthropic's native shape, so this provider passes responses through
    unchanged."""

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
        return await self.client.messages.create(**kwargs)

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
