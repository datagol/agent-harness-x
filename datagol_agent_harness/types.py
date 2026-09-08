"""Shared type definitions for the agent harness. Pure dataclasses, zero logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable


class PermissionLevel(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    TOOL_CALLS = "tool_use"  # alias
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    SAFETY = "safety"
    OTHER = "other"


@dataclass
class ToolDefinition:
    """A registered tool: its schema (for the LLM) + its implementation (for the harness)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    permission_level: PermissionLevel = PermissionLevel.ASK
    concurrent: bool = True


@dataclass
class ToolCall:
    """What the LLM asked us to do."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """What happened when we ran it."""

    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False
    tool_use_id: str = ""  # back-compat alias

    def __post_init__(self) -> None:
        if not self.tool_call_id and self.tool_use_id:
            self.tool_call_id = self.tool_use_id
        elif self.tool_call_id and not self.tool_use_id:
            self.tool_use_id = self.tool_call_id


@dataclass
class TokenUsage:
    """Accumulated token counts for cost tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class ProviderResponse:
    """Normalized, vendor-agnostic LLM response."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None
    stop_reason: StopReason | str = StopReason.END_TURN
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: Any = None
    content: list[Any] = field(default_factory=list)  # back-compat content block representation

    def __post_init__(self) -> None:
        if not self.content:
            blocks: list[Any] = []
            if self.thinking:
                blocks.append(SimpleNamespace(type="thinking", thinking=self.thinking))
            if self.text:
                blocks.append(SimpleNamespace(type="text", text=self.text))
            for tc in self.tool_calls:
                blocks.append(SimpleNamespace(type="tool_use", id=tc.id, name=tc.name, input=tc.input))
            self.content = blocks
        elif not self.text and not self.tool_calls:
            text_parts: list[str] = []
            calls: list[ToolCall] = []
            for b in self.content:
                b_type = getattr(b, "type", None) if not isinstance(b, dict) else b.get("type")
                if b_type == "text":
                    t = getattr(b, "text", "") if not isinstance(b, dict) else b.get("text", "")
                    text_parts.append(t)
                elif b_type == "tool_use":
                    tc_id = getattr(b, "id", "") if not isinstance(b, dict) else b.get("id", "")
                    tc_name = getattr(b, "name", "") if not isinstance(b, dict) else b.get("name", "")
                    tc_input = getattr(b, "input", {}) if not isinstance(b, dict) else b.get("input", {})
                    calls.append(ToolCall(id=tc_id, name=tc_name, input=tc_input))
                elif b_type == "thinking":
                    self.thinking = getattr(b, "thinking", "") if not isinstance(b, dict) else b.get("thinking", "")
            if text_parts:
                self.text = "\n".join(text_parts)
            if calls:
                self.tool_calls = calls


@dataclass
class StreamChunk:
    """A streaming chunk emitted by an LLMProvider."""

    kind: str  # "text_delta", "thinking_delta", "response"
    data: Any = None


@dataclass
class AgentConfig:
    """All configuration for an Agent, with sensible defaults."""

    model: str = "claude-sonnet-4-6"
    provider: str = "anthropic"  # 'anthropic' or 'openai'
    max_tokens: int = 8192
    max_iterations: int = 50
    system_prompt: str = "You are a helpful assistant."
    temperature: float | None = None
    max_result_chars: int = 12_000  # eviction threshold for tool results (~3K tokens)


@dataclass
class SessionState:
    """Serializable session state for save/resume."""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxConfig:
    """Configuration for sandboxed code execution."""

    timeout_seconds: int = 30
    max_memory_mb: int = 512
    max_cpu_seconds: int = 60
    max_file_size_mb: int = 100
    allowed_paths: list[str] = field(default_factory=list)
    network_enabled: bool = False
    tier: str = "process"  # "process", "docker", or "seatbelt"


@dataclass
class SandboxResult:
    """Result from sandboxed code execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    memory_exceeded: bool = False
    execution_time_ms: float = 0.0
    files_created: list[str] = field(default_factory=list)


class RuntimeState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class RuntimeConfig:
    """Configuration for the agent runtime."""

    storage_dir: str = ".agent_sessions"
    checkpoint_interval: int = 5  # checkpoint every N iterations
    max_checkpoints: int = 10
    max_session_duration_seconds: float | None = None


@dataclass
class CheckpointData:
    """Serializable checkpoint for pause/resume."""

    session_state: SessionState
    agent_config_dict: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    iteration_count: int = 0


@dataclass
class RuntimeStatus:
    """Current status of an agent runtime."""

    session_id: str
    state: RuntimeState = RuntimeState.INITIALIZING
    started_at: str = ""
    uptime_seconds: float = 0.0
    iterations_completed: int = 0
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    checkpoints: list[str] = field(default_factory=list)
