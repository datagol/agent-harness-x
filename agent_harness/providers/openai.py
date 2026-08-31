"""OpenAI provider — translates between OpenAI's chat-completions format
and the harness's canonical (Anthropic-shape) format.

Mapping (Anthropic ↔ OpenAI):
    user/assistant message with text          ↔ {role, content: <string>}
    assistant message with tool_use blocks    ↔ {role: assistant, tool_calls: [...]}
    user message with tool_result block       ↔ {role: tool, tool_call_id, content}
    tool schema {name, description,           ↔ {type: function, function:
                  input_schema}                   {name, description, parameters}}
    stop_reason 'end_turn' / 'tool_use' /     ↔ finish_reason 'stop' / 'tool_calls' /
                'max_tokens'                      'length'

Streaming is NOT supported by this provider (the StreamingAgent class is
Anthropic-specific). Non-streaming Agent works.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "OpenAIProvider requires the openai package. "
        "Install with: pip install 'agent-harness[openai]'"
    ) from e

from .base import LLMProvider, ProviderResponse


class OpenAIProvider(LLMProvider):
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
        temperature: float,
    ) -> ProviderResponse:
        openai_messages = _to_openai_messages(messages, system)
        openai_tools = _to_openai_tools(tools)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        resp = await self.client.chat.completions.create(**kwargs)
        return _from_openai_response(resp)

    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
    ) -> int:
        # OpenAI has no count_tokens endpoint. Rough estimate: ~3 chars / token.
        # Use tiktoken if installed for a tighter number.
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
    """Translate canonical (Anthropic-shape) messages into OpenAI's format.

    Anthropic represents tool calls as content blocks inside assistant messages,
    and tool results as content blocks inside user messages. OpenAI lifts both
    into separate top-level messages.
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for m in messages:
        role = m["role"]
        content = m.get("content")

        # Simple string content: pass through.
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        # Block-list content.
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
                # 'thinking' blocks are intentionally dropped for OpenAI.

            msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                msg["content"] = "\n".join(text_parts)
            else:
                msg["content"] = None
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
            continue

        # role == "user" — tool_result blocks become separate "tool" messages.
        text_parts = []
        for block in content or []:
            btype = _block_type(block)
            if btype == "tool_result":
                tc_id = _block_attr(block, "tool_use_id")
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


def _from_openai_response(resp: Any) -> ProviderResponse:
    """Wrap an OpenAI ChatCompletion in canonical (Anthropic-shape) form."""
    choice = resp.choices[0]
    msg = choice.message

    content: list[Any] = []
    if msg.content:
        content.append(SimpleNamespace(type="text", text=msg.content))
    for tc in (msg.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw": tc.function.arguments}
        content.append(SimpleNamespace(
            type="tool_use",
            id=tc.id,
            name=tc.function.name,
            input=args,
        ))

    stop_reason = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
    }.get(choice.finish_reason, "end_turn")

    usage = SimpleNamespace(
        input_tokens=getattr(resp.usage, "prompt_tokens", 0),
        output_tokens=getattr(resp.usage, "completion_tokens", 0),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=getattr(
            getattr(resp.usage, "prompt_tokens_details", None), "cached_tokens", 0
        ) or 0,
    )

    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)


def _block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_attr(block: Any, name: str) -> Any:
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)
