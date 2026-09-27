"""Validated conversation messages shared by every provider adapter."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Literal, TypedDict


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ThinkingBlock(TypedDict):
    type: Literal["thinking"]
    thinking: str


class ToolUseBlock(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(TypedDict, total=False):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[dict[str, Any]]
    is_error: bool


ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True)
class Message(Mapping[str, Any]):
    """Canonical SDK message. Dict views are detached compatibility representations.

    Tool-result blocks share a user-role message for backward compatibility;
    provider adapters remain responsible for vendor-specific wire conversion.
    """

    role: Literal["user", "assistant", "system"]
    content: str | list[ContentBlock]

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid message role: {self.role!r}")
        if not isinstance(self.content, (str, list)):
            raise TypeError("Message content must be text or a list of content blocks")
        if isinstance(self.content, list):
            for raw_block in self.content:
                if not isinstance(raw_block, dict):
                    raise TypeError("Content blocks must be dictionaries")
                block: Mapping[str, Any] = raw_block
                kind = block.get("type")
                if kind in ("text", "thinking"):
                    if not isinstance(block.get(kind), str):
                        raise TypeError(f"{kind} content must be a string")
                elif kind == "tool_use":
                    if self.role != "assistant" or not all(isinstance(block.get(key), str) and block[key] for key in ("id", "name")) or not isinstance(block.get("input"), dict):
                        raise ValueError("Invalid tool-use block")
                elif kind == "tool_result":
                    if self.role != "user" or not isinstance(block.get("tool_use_id"), str) or not block["tool_use_id"] or not isinstance(block.get("content"), (str, list)):
                        raise ValueError("Invalid tool-result block")
                    if "is_error" in block and type(block["is_error"]) is not bool:
                        raise TypeError("Tool result is_error must be a bool")
                else:
                    raise ValueError(f"Unsupported content block: {kind!r}")
        json.dumps(self.content, allow_nan=False)
        object.__setattr__(self, "content", deepcopy(self.content))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Message:
        if set(value) != {"role", "content"}:
            raise ValueError("Message requires exactly role and content")
        return cls(role=value["role"], content=value["content"])

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": deepcopy(self.content)}

    def __getitem__(self, key: str) -> Any:
        if key == "role":
            return self.role
        if key == "content":
            return self.content
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("role", "content"))

    def __len__(self) -> int:
        return 2
