"""Anthropic provider — thin wrapper around AsyncAnthropic."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from ..types import ProviderResponse, StopReason, StreamChunk, TokenUsage, ToolCall
from ..types import PromptCacheHint
from ..errors import IncompleteStreamError
from ..models import ModelLimits
from ..types import ToolChoice
from .base import normalize_tool_choice, LLMProvider


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
    """Make a stored history one the Messages API accepts, whoever wrote it.

    Removes SDK-only text metadata, and what other vendors leave behind when a
    FallbackProvider moves a conversation here: thinking blocks without an
    Anthropic signature (OpenAI and Gemini reasoning, which Anthropic rejects as
    unsigned) and private bookkeeping keys such as Gemini's thought signature.
    """
    prepared = deepcopy(messages)

    def clean_content(content: Any) -> Any:
        if not isinstance(content, list):
            return content
        kept = []
        for block in content:
            if not isinstance(block, dict):
                kept.append(block)
                continue
            kind = block.get("type")
            if kind == "provider":
                if block.get("provider") == "anthropic":
                    kept.append(deepcopy(block["data"]))
                continue
            if kind == "audio":
                # Claude has no audio input. Dropping the block would send the
                # prompt without the clip it refers to, and the model would
                # answer about nothing; say so instead.
                raise ValueError(
                    "Anthropic models do not accept audio input; transcribe it first "
                    "or send this turn to a provider that does"
                )
            if kind == "thinking" and not block.get("signature"):
                continue
            for key in [k for k in block if k.startswith("_")]:
                block.pop(key)
            if kind == "text":
                block.pop("parsed_output", None)
            elif kind == "tool_result":
                block["content"] = clean_content(block.get("content"))
            kept.append(block)
        return kept

    for message in prepared:
        message["content"] = clean_content(message.get("content"))
    # A turn that held only foreign reasoning is now empty, which the API rejects.
    return [m for m in prepared if m.get("content") != []]


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


def _eager(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Our tools, marked to stream their arguments as the model writes them.

    Without it the API holds a tool call back until its arguments are complete:
    a model writing a whole report into ``write_file`` sends nothing for
    minutes, which the stall check takes for a dead stream and retries from
    scratch. Server tools (``web_search`` and the like) have no ``input_schema``
    and are left as they are; a tool that already says either way keeps its
    choice. Sent on every request, streamed or not, so the tool definitions --
    part of the cached prefix -- are the same for both.
    """
    return [
        {**tool, "eager_input_streaming": True}
        if isinstance(tool, dict) and "input_schema" in tool
        and tool.get("type", "custom") == "custom" and "eager_input_streaming" not in tool
        else tool
        for tool in tools
    ]


def _rejected_eager_streaming(exc: BaseException) -> bool:
    return "eager_input_streaming" in str(exc)


class AnthropicProvider(LLMProvider):
    """Talks to Anthropic's Messages API and normalizes responses to ProviderResponse."""

    name = "anthropic"
    first_event_promptly = True  # message_start arrives before any thinking or text

    def __init__(self, client: AsyncAnthropic | None = None, *,
                 tool_choice: "str | ToolChoice | None" = None,
                 eager_tool_streaming: bool = True) -> None:
        # Anthropic spells it {"type": "any"}; see base.TOOL_CHOICES.
        self.tool_choice = normalize_tool_choice(
            tool_choice, env_var="ANTHROPIC_TOOL_CHOICE"
        )
        # See _eager. An Anthropic-compatible endpoint that refuses the field
        # turns it off for this provider on the first refusal.
        self.eager_tool_streaming = eager_tool_streaming
        # The engine owns retries (RetryPolicy); the SDK's own would compound them invisibly.
        self.client = client or AsyncAnthropic(max_retries=0)
        self._limits: dict[str, ModelLimits | None] = {}

    async def model_limits(self, model: str) -> ModelLimits | None:
        """Ask the Models API: it reports ``max_input_tokens`` and ``max_tokens``
        for every model it serves, including ones released after this package.

        Asked once per model. Any failure -- an older API, a proxy that does not
        serve ``/v1/models``, a client double in tests -- is remembered as "no
        answer", and the built-in table is used instead.
        """
        if model in self._limits:
            return self._limits[model]
        found: ModelLimits | None = None
        try:
            import asyncio

            info = await asyncio.wait_for(self.client.models.retrieve(model), timeout=10)
            window, output = getattr(info, "max_input_tokens", None), getattr(info, "max_tokens", None)
            if type(window) is int and type(output) is int and window > 0 and output > 0:
                found = ModelLimits(window, output)
        except Exception:
            found = None
        self._limits[model] = found
        return found

    async def _create_via_stream(self, kwargs: dict[str, Any]) -> Any:
        """The SDK refuses a non-streaming call that could run past ten minutes;
        stream it and return the assembled message instead."""
        stream_kwargs = _filter_kwargs(self.client.messages.stream, kwargs)
        async with self.client.messages.stream(**stream_kwargs) as stream:
            response = await stream.get_final_message()
        if not getattr(response, "stop_reason", None):
            raise IncompleteStreamError("Anthropic stream ended before the message was complete")
        return response

    async def _create(self, kwargs: dict[str, Any]) -> Any:
        try:
            return await self.client.messages.create(**kwargs)
        except ValueError as exc:
            if "Streaming is required" in str(exc):
                return await self._create_via_stream(kwargs)
            raise

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
            kwargs["tools"] = _eager(tools) if self.eager_tool_streaming else tools
            if self.tool_choice:
                kwargs["tool_choice"] = {"type": self.tool_choice}

        uncached = _filter_kwargs(self.client.messages.create, kwargs)
        filtered_kwargs = _filter_kwargs(self.client.messages.create, _apply_prompt_cache(kwargs, cache))
        try:
            resp = await self._create(filtered_kwargs)
        except TypeError as exc:
            if "temperature" in str(exc) and "temperature" in filtered_kwargs:
                filtered_kwargs.pop("temperature", None)
                resp = await self._create(filtered_kwargs)
            else:
                raise
        except Exception as exc:
            if self.eager_tool_streaming and tools and _rejected_eager_streaming(exc):
                self.eager_tool_streaming = False
                return await self.create(
                    model=model, messages=messages, system=system, tools=tools,
                    max_tokens=max_tokens, temperature=temperature, cache=cache,
                )
            # Caching is an optimization: a rejected marker runs the call uncached.
            if filtered_kwargs is not uncached and filtered_kwargs != uncached and _rejected_cache_control(exc):
                resp = await self._create(uncached)
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
            kwargs["tools"] = _eager(tools) if self.eager_tool_streaming else tools
            if self.tool_choice:
                kwargs["tool_choice"] = {"type": self.tool_choice}

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

        delivered = False
        try:
            async with stream_ctx as stream:
                async for chunk in _chunks(stream):
                    delivered = delivered or chunk.kind == "text_delta"
                    yield chunk
                final_resp = await stream.get_final_message()
        except Exception as exc:
            if not delivered and self.eager_tool_streaming and tools and _rejected_eager_streaming(exc):
                self.eager_tool_streaming = False
                async for chunk in self.stream(
                    model=model, messages=messages, system=system, tools=tools,
                    max_tokens=max_tokens, temperature=temperature, cache=cache,
                ):
                    yield chunk
                return
            # Only before any output: a rejected marker fails at stream open, and
            # restarting a stream that has delivered text would repeat it.
            if not delivered and filtered_kwargs != uncached and _rejected_cache_control(exc):
                async with self.client.messages.stream(**uncached) as stream:
                    async for chunk in _chunks(stream):
                        yield chunk
                    final_resp = await stream.get_final_message()
            else:
                raise
        if not getattr(final_resp, "stop_reason", None):
            # The stream closed before message_delta carried a stop reason: what
            # was assembled is a fragment, and may hold a half-written tool call.
            raise IncompleteStreamError("Anthropic stream ended before the message was complete")
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


async def _chunks(stream: Any) -> AsyncIterator[StreamChunk]:
    """Every event the stream delivers, as text, thinking, or a sign of progress.

    Reading only ``text_stream`` hid everything else: a model writing a long
    tool argument -- a whole report into a file -- streams for minutes without
    a single text event, which looked exactly like a stalled connection.
    """
    if not hasattr(stream, "__aiter__"):  # a client without event iteration
        async for text in stream.text_stream:
            yield StreamChunk(kind="text_delta", data=text)
        return
    async for event in stream:
        kind = getattr(event, "type", None)
        if kind == "text" and getattr(event, "text", ""):
            yield StreamChunk(kind="text_delta", data=event.text)
        elif kind == "thinking" and getattr(event, "thinking", ""):
            yield StreamChunk(kind="thinking_delta", data=event.thinking)
        else:
            yield StreamChunk(kind="progress")


def _from_anthropic_response(resp: Any) -> ProviderResponse:
    """Normalize raw Anthropic Messages API response into canonical ProviderResponse."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

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

    raw_stop = getattr(resp, "stop_reason", None)
    stop_map = {
        "end_turn": StopReason.END_TURN,
        "tool_use": StopReason.TOOL_USE,
        "max_tokens": StopReason.MAX_TOKENS,
        # The reply filled what was left of the context window: cut off, like
        # max_tokens, and recovered the same way.
        "model_context_window_exceeded": StopReason.MAX_TOKENS,
        "stop_sequence": StopReason.STOP_SEQUENCE,
        "refusal": StopReason.REFUSAL,
        "pause_turn": StopReason.PAUSE_TURN,
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
        thinking=None,  # derived from all ordered thinking blocks below
        stop_reason=stop_reason,
        usage=usage,
        raw=resp,
        # Preserve opaque native blocks in order; only tool_use dispatches locally.
        content=[
            b if getattr(b, "type", None) in _CANONICAL_BLOCKS else
            {"type": "provider", "provider": "anthropic", "data": ProviderResponse._block_dict(b)}
            for b in raw_content
        ],
    )


_CANONICAL_BLOCKS = ("text", "thinking", "tool_use")
