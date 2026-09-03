"""Shared type definitions for the agent harness. Pure dataclasses, zero logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class PermissionLevel(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class ToolDefinition:
    """A registered tool: its schema (for the LLM) + its implementation (for the harness)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    permission_level: PermissionLevel = PermissionLevel.ASK


@dataclass
class ToolCall:
    """What the LLM asked us to do."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """What happened when we ran it."""

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class TokenUsage:
    """Accumulated token counts for cost tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class AgentConfig:
    """All configuration for an Agent, with sensible defaults."""

    model: str = "claude-sonnet-4-6"
    provider: str = "anthropic"  # 'anthropic' or 'openai'
    max_tokens: int = 8192
    max_iterations: int = 50
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.0
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
