"""Shared type definitions for the agent harness. Pure dataclasses, zero logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import warnings
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
    permission_level: PermissionLevel | None = None
    concurrent: bool = True
    replay_policy: str = "manual"
    timeout_seconds: float = 300.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or any(c.isspace() for c in self.name):
            raise ValueError("Tool name must be nonempty and contain no whitespace")
        if not callable(self.handler):
            raise TypeError("Tool handler must be callable")
        if not isinstance(self.description, str) or not isinstance(self.input_schema, dict):
            raise TypeError("Tool description must be a string and input_schema must be a dictionary")
        if self.permission_level is not None and not isinstance(self.permission_level, PermissionLevel):
            raise TypeError("Tool permission must be a PermissionLevel")
        if not isinstance(self.concurrent, bool):
            raise TypeError("Tool concurrent must be a bool")
        if self.replay_policy not in ("safe", "idempotent", "manual"):
            raise ValueError("Invalid tool replay policy")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Tool timeout must be finite and positive")


@dataclass
class ToolCall:
    """What the LLM asked us to do."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(init=False)
class ToolResult:
    """Canonical result with one call ID; tool_use_id is a derived legacy alias."""

    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False

    def __init__(self, tool_call_id: str = "", content: str = "", is_error: bool = False, tool_use_id: str = "") -> None:
        if tool_call_id and tool_use_id and tool_call_id != tool_use_id:
            raise ValueError("Conflicting tool result call IDs")
        if not isinstance(content, str) or type(is_error) is not bool:
            raise TypeError("Tool result content must be text and is_error must be bool")
        self.tool_call_id = tool_call_id or tool_use_id
        self.content = content
        self.is_error = is_error

    @property
    def tool_use_id(self) -> str:
        return self.tool_call_id

    @tool_use_id.setter
    def tool_use_id(self, value: str) -> None:
        self.tool_call_id = value


@dataclass
class TokenUsage:
    """Accumulated token counts for cost tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    # Reasoning/thinking tokens, where the provider reports them separately
    # (Gemini's thoughts_token_count, OpenAI's reasoning_tokens). They are
    # billed at the OUTPUT rate, so anything estimating cost from
    # input+output alone under-reports by the most expensive component.
    # 0 when the provider does not report them.
    thinking_tokens: int = 0


@dataclass(init=False)
class ProviderResponse:
    """Canonical fields are authoritative; legacy content is a derived view."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None
    stop_reason: StopReason | str = StopReason.END_TURN
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: Any = None
    _content_metadata: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _content_dicts: bool = field(default=False, repr=False)

    def __init__(
        self, text: str = "", tool_calls: list[ToolCall] | None = None,
        thinking: str | None = None, stop_reason: StopReason | str = StopReason.END_TURN,
        usage: TokenUsage | None = None, raw: Any = None, content: list[Any] | None = None,
        _content_metadata: dict[str, dict[str, Any]] | None = None,
        _content_dicts: bool = False,
    ) -> None:
        self.text, self.tool_calls, self.thinking = text, list(tool_calls or []), thinking
        self.stop_reason, self.usage, self.raw = stop_reason, usage or TokenUsage(), raw
        self._content_metadata = dict(_content_metadata or {})
        self._content_dicts = _content_dicts
        if content:
            parsed_text, parsed_calls, parsed_thinking, metadata = self._parse_content(content)
            joined = "".join(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "") for block in content)
            if text and text not in (parsed_text, joined):
                raise ValueError("ProviderResponse text and content disagree")
            if tool_calls and tool_calls != parsed_calls:
                raise ValueError("ProviderResponse tool_calls and content disagree")
            if thinking is not None and parsed_thinking is not None and thinking != parsed_thinking:
                raise ValueError("ProviderResponse thinking and content disagree")
            self.text, self.tool_calls = text or parsed_text, parsed_calls
            self.thinking = parsed_thinking if parsed_thinking is not None else thinking
            self._content_metadata = metadata
            self._content_dicts = all(isinstance(block, dict) for block in content)

    @staticmethod
    def _parse_content(content):
        from copy import deepcopy
        text_parts, calls, thinking, metadata = [], [], None, {}
        for block in content:
            if isinstance(block, dict):
                data = block
            elif callable(model_dump := getattr(block, "model_dump", None)):
                # SDK blocks contain nested models (e.g. Anthropic callers and
                # citations). A shallow vars() leaves these non-JSON objects in
                # conversation history and persisted execution state. Honor SDK
                # API exclusions so local helpers (e.g. parsed_output) are not
                # persisted as content to send back on the next model request.
                data = model_dump(mode="json", exclude=getattr(block, "__api_exclude__", None))
            else:
                data = vars(block)
            kind = data.get("type")
            if kind == "text":
                text_parts.append(data["text"])
            elif kind == "thinking":
                thinking = data["thinking"]
            elif kind == "tool_use":
                calls.append(ToolCall(data["id"], data["name"], deepcopy(data["input"])))
            else:
                raise ValueError(f"Unsupported provider content block: {kind!r}")
            key = data["id"] if kind == "tool_use" else kind
            extras = {k: deepcopy(v) for k, v in data.items() if k not in ("type", "text", "thinking", "id", "name", "input")}
            if extras:
                metadata[key] = extras
        return "\n".join(text_parts), calls, thinking, metadata

    @property
    def content(self) -> list[Any]:
        from copy import deepcopy
        blocks = []
        def add(key, **fields):
            extras = self._content_metadata.get(key)
            blocks.append({**deepcopy(extras or {}), **fields} if extras or self._content_dicts else SimpleNamespace(**fields))
        if self.thinking:
            add("thinking", type="thinking", thinking=self.thinking)
        if self.text:
            add("text", type="text", text=self.text)
        for call in self.tool_calls:
            add(call.id, type="tool_use", id=call.id, name=call.name, input=deepcopy(call.input))
        return blocks

    @content.setter
    def content(self, blocks: list[Any]) -> None:
        self.text, self.tool_calls, self.thinking, self._content_metadata = self._parse_content(blocks)
        self._content_dicts = all(isinstance(block, dict) for block in blocks)


@dataclass
class StreamChunk:
    """A streaming chunk emitted by an LLMProvider."""

    kind: str  # "text_delta", "thinking_delta", "response"
    data: Any = None


@dataclass
class AgentConfig:
    """All configuration for an Agent, with sensible defaults."""

    model: str = "claude-sonnet-4-6"
    provider: str = "anthropic"  # 'anthropic', 'openai', 'gemini', 'openrouter', or 'azure'
    max_tokens: int = 8192
    max_iterations: int = 50
    system_prompt: str = "You are a helpful assistant."
    temperature: float | None = None
    model_timeout_seconds: float = 300.0
    # Retry transient provider failures (429, 5xx, timeouts). 1 disables it.
    # A streaming call is only retried before its first chunk, since after
    # that a retry would duplicate output the caller has already seen.
    llm_max_attempts: int = 2
    llm_retry_backoff_seconds: float = 0.5
    max_result_chars: int = 12_000  # eviction threshold for tool results (~3K tokens)
    max_context_tokens: int = 150_000
    max_cost_dollars: float | None = None
    input_cost_per_m: float | None = None
    output_cost_per_m: float | None = None

    def __post_init__(self) -> None:
        if self.provider not in ("anthropic", "openai", "gemini", "openrouter", "azure"):
            raise ValueError(f"Unknown provider: {self.provider!r}")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty string")
        if not isinstance(self.system_prompt, str):
            raise TypeError("system_prompt must be a string")
        for name in ("max_tokens", "max_result_chars", "max_context_tokens"):
            _positive(getattr(self, name), name, integer=True)
        if type(self.max_iterations) is not int or self.max_iterations < 0:
            raise ValueError("max_iterations must be nonnegative (0 means unlimited)")
        _positive(self.model_timeout_seconds, "model_timeout_seconds")
        if type(self.llm_max_attempts) is not int or self.llm_max_attempts < 1:
            raise ValueError("llm_max_attempts must be a positive integer (1 disables retry)")
        if isinstance(self.llm_retry_backoff_seconds, bool) or not math.isfinite(self.llm_retry_backoff_seconds) or self.llm_retry_backoff_seconds < 0:
            raise ValueError("llm_retry_backoff_seconds must be finite and nonnegative")
        for name in ("temperature", "max_cost_dollars", "input_cost_per_m", "output_cost_per_m"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass
class SessionState:
    """Serializable session state for save/resume."""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)
    lifetime_iterations: int = 0
    version: int = 2


@dataclass
class SandboxConfig:
    """Configuration for sandboxed code execution."""

    timeout_seconds: int = 30
    max_memory_mb: int = 512
    max_cpu_seconds: int = 60
    max_file_size_mb: int = 100
    allowed_paths: list[str] = field(default_factory=list)
    network_enabled: bool | None = None  # None: host access for process; disabled for isolated tiers
    tier: str = "process"  # "process", "docker", or "seatbelt"

    def __post_init__(self) -> None:
        if self.tier not in ("process", "docker", "seatbelt"):
            raise ValueError(f"Unknown sandbox tier: {self.tier!r}")
        for name in ("timeout_seconds", "max_memory_mb", "max_cpu_seconds", "max_file_size_mb"):
            _positive(getattr(self, name), name, integer=True)
        if self.network_enabled is not None and type(self.network_enabled) is not bool:
            raise TypeError("network_enabled must be bool or None")
        if self.tier == "process" and (self.network_enabled is False or self.allowed_paths):
            raise ValueError("Process execution cannot restrict network or filesystem access; choose docker or seatbelt")


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
    checkpoint_interval: int | None = None  # deprecated: execution checkpoints every boundary
    max_checkpoints: int | None = None  # deprecated: retention belongs to the backend
    max_session_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.storage_dir, str) or not self.storage_dir:
            raise ValueError("storage_dir must be nonempty")
        if self.max_session_duration_seconds is not None:
            _positive(self.max_session_duration_seconds, "max_session_duration_seconds")
        if self.checkpoint_interval is not None or self.max_checkpoints is not None:
            warnings.warn("checkpoint_interval/max_checkpoints are deprecated and ignored; durable execution checkpoints every boundary", DeprecationWarning, stacklevel=2)


def _positive(value: Any, name: str, *, integer: bool = False) -> None:
    if (integer and type(value) is not int) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive {'integer' if integer else 'number'}")


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
