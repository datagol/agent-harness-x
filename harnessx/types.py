"""Shared type definitions for the agent harness. Pure dataclasses, zero logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import math
import warnings
from types import SimpleNamespace
from typing import Any, Callable, Mapping, TypeVar

from .errors import ConfigurationError

DEFAULT_TIMEOUT_SECONDS = 300.0  # model calls, tool calls, and sub-agent delegation


class PermissionLevel(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    SAFETY = "safety"
    OTHER = "other"


class ReplayPolicy(str, Enum):
    """What durable execution may do with a tool whose outcome was lost."""

    SAFE = "safe"  # repeating the call is harmless
    IDEMPOTENT = "idempotent"  # the integration itself deduplicates repeats
    MANUAL = "manual"  # stop and ask; the default


@dataclass
class ToolDefinition:
    """A registered tool: its schema (for the LLM) + its implementation (for the harness)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    permission_level: PermissionLevel | None = None
    concurrent: bool = True
    replay_policy: ReplayPolicy | str = "manual"
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or any(c.isspace() for c in self.name):
            raise ConfigurationError("Tool name must be nonempty and contain no whitespace")
        if not callable(self.handler):
            raise TypeError("Tool handler must be callable")
        if not isinstance(self.description, str) or not isinstance(self.input_schema, dict):
            raise TypeError("Tool description must be a string and input_schema must be a dictionary")
        if self.permission_level is not None and not isinstance(self.permission_level, PermissionLevel):
            raise TypeError("Tool permission must be a PermissionLevel")
        if not isinstance(self.concurrent, bool):
            raise TypeError("Tool concurrent must be a bool")
        try:
            self.replay_policy = ReplayPolicy(self.replay_policy).value  # member or string, stored as str
        except ValueError:
            raise ConfigurationError("Invalid tool replay policy") from None
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ConfigurationError("Tool timeout must be finite and positive")


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
    """Accumulated token counts for cost tracking.

    ``input_tokens`` counts every prompt token, cached or not. The two cache
    counters are the portions of it served from, or written to, the vendor's
    cache; providers that report input net of cache activity are normalized.
    """

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


@dataclass(frozen=True)
class PromptCachePolicy:
    """How the harness asks providers to cache the stable prompt prefix.

    The engine turns this into a per-request PromptCacheHint; each provider
    translates the hint into its vendor's mechanism and fails open when the
    vendor rejects it. ``AgentConfig(prompt_cache=None)`` disables caching.
    """

    ttl_seconds: int | None = None  # None means the vendor's default lifetime
    cache_history: bool = True  # also mark the last complete turn as cacheable
    key_salt: str = ""  # separate caches for otherwise identical prompts

    def __post_init__(self) -> None:
        if self.ttl_seconds is not None and (
            isinstance(self.ttl_seconds, bool) or type(self.ttl_seconds) is not int or self.ttl_seconds <= 0
        ):
            raise ConfigurationError("ttl_seconds must be a positive integer or None")
        if not isinstance(self.cache_history, bool):
            raise TypeError("cache_history must be a bool")
        if not isinstance(self.key_salt, str):
            raise TypeError("key_salt must be a string")


@dataclass(frozen=True)
class PromptCacheHint:
    """Per-request caching instruction the engine hands to a provider.

    ``prefix_key`` is a stable hash of model, system prompt, tools, and salt.
    ``breakpoints`` name where the reusable prefix ends: ``"system"``,
    ``"tools"``, or ``"message:<index>"``. ``enabled=False`` tells a provider
    that caching is switched off for this agent, which differs from ``None``
    (a caller that predates the hint).
    """

    prefix_key: str
    breakpoints: tuple[str, ...] = ()
    ttl_seconds: int | None = None
    enabled: bool = True


def _nonnegative_or_none(value: Any, name: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
    ):
        raise ConfigurationError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class Limits:
    """Budgets the harness enforces: iterations, context size, memory thresholds, and cost."""

    max_iterations: int = 50  # 0 means unlimited
    max_context_tokens: int = 150_000  # conversation trim threshold
    max_result_chars: int = 12_000  # tool results above this are spilled to disk (~3K tokens)
    max_cost_dollars: float | None = None
    input_cost_per_m: float | None = None
    output_cost_per_m: float | None = None

    def __post_init__(self) -> None:
        if type(self.max_iterations) is not int or self.max_iterations < 0:
            raise ConfigurationError("max_iterations must be nonnegative (0 means unlimited)")
        _positive(self.max_context_tokens, "max_context_tokens", integer=True)
        _positive(self.max_result_chars, "max_result_chars", integer=True)
        for name in ("max_cost_dollars", "input_cost_per_m", "output_cost_per_m"):
            _nonnegative_or_none(getattr(self, name), name)


@dataclass(frozen=True)
class RetryPolicy:
    """Transient-failure handling for one model call (429, 5xx, timeouts, connection loss).

    ``attempts`` is the total number of model calls for one step, so ``1``
    disables retry. A retried streaming call re-sends text the caller may
    already have shown; the run stream signals that with ``ATTEMPT_RESET``.
    """

    attempts: int = 2
    backoff_seconds: float = 0.5  # base delay, doubled per attempt
    # Wall clock around one attempt. None derives it from the reply budget, so a
    # long reply is not cut off by a timeout sized for short ones.
    call_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if type(self.attempts) is not int or self.attempts < 1:
            raise ConfigurationError("attempts must be a positive integer (1 disables retry)")
        _nonnegative_or_none(self.backoff_seconds, "backoff_seconds")
        if self.call_timeout_seconds is not None:
            _positive(self.call_timeout_seconds, "call_timeout_seconds")

    def effective_call_timeout(self, max_tokens: int | None) -> float:
        """The timeout for one model call: the explicit value, or one sized for ``max_tokens``."""
        if self.call_timeout_seconds is not None:
            return float(self.call_timeout_seconds)
        return call_timeout_for(max_tokens)


def call_timeout_for(max_tokens: int | None) -> float:
    """A timeout that fits a reply of ``max_tokens``: never below the default,
    and at the Anthropic SDK's planning rate (128K tokens per hour) plus a minute."""
    if not max_tokens:
        return DEFAULT_TIMEOUT_SECONDS
    return max(DEFAULT_TIMEOUT_SECONDS, max_tokens * 3600 / 128_000 + 60)


@dataclass(frozen=True)
class ToolPolicy:
    """Registry-wide tool options that an Agent applies to the registry it adopts."""

    default_timeout_seconds: float | None = None  # None: the registry's own default, else DEFAULT_TIMEOUT_SECONDS
    dedupe_calls: bool = False  # identical repeated calls within one run return the first result

    def __post_init__(self) -> None:
        if self.default_timeout_seconds is not None:
            _positive(self.default_timeout_seconds, "default_timeout_seconds")
        if type(self.dedupe_calls) is not bool:
            raise TypeError("dedupe_calls must be a bool")


_LEGACY_FIELDS: dict[str, tuple[str, str]] = {
    # 0.3 flat name -> (sub-policy field, attribute). The aliases are removed in 0.5.
    "max_iterations": ("limits", "max_iterations"),
    "max_context_tokens": ("limits", "max_context_tokens"),
    "max_result_chars": ("limits", "max_result_chars"),
    "max_cost_dollars": ("limits", "max_cost_dollars"),
    "input_cost_per_m": ("limits", "input_cost_per_m"),
    "output_cost_per_m": ("limits", "output_cost_per_m"),
    "llm_max_attempts": ("retry", "attempts"),
    "llm_retry_backoff_seconds": ("retry", "backoff_seconds"),
    "model_timeout_seconds": ("retry", "call_timeout_seconds"),
}
_OMITTED: Any = object()  # "argument not given", where None is itself a legal value

_P = TypeVar("_P", Limits, RetryPolicy, ToolPolicy, PromptCachePolicy)


def _coerce(kind: type[_P], value: Any, name: str) -> _P:
    if value is None:
        return kind()
    if isinstance(value, dict):  # restored from a snapshot, session, or run state
        return kind(**value)
    if not isinstance(value, kind):
        raise TypeError(f"{name} must be a {kind.__name__}")
    return value


def _warn_flat(name: str) -> None:
    group, attr = _LEGACY_FIELDS[name]
    warnings.warn(
        f"AgentConfig.{name} is deprecated and will be removed in harnessx 0.5; "
        f"use AgentConfig.{group}.{attr}",
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass(init=False)
class AgentConfig:
    """All configuration for an Agent, with sensible defaults.

    Budgets live on ``limits``, transient-failure handling on ``retry``, prompt
    caching on ``prompt_cache``, and registry-wide tool options on ``tools``.
    The flat 0.3 names (``max_iterations``, ``llm_max_attempts``, ...) still
    work as keyword arguments and attributes but warn; they go away in 0.5.
    """

    model: str = "claude-sonnet-4-6"
    provider: str = "anthropic"  # a built-in name or one passed to register_provider()
    max_tokens: int | None = None  # reply token budget; None lets the provider choose for the model
    system_prompt: str = "You are a helpful assistant."
    temperature: float | None = None
    limits: Limits = field(default_factory=Limits)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    prompt_cache: PromptCachePolicy | None = field(default_factory=PromptCachePolicy)  # None disables caching
    tools: ToolPolicy = field(default_factory=ToolPolicy)

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        provider: str = "anthropic",
        max_tokens: int | None = None,
        *,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float | None = None,
        limits: Limits | dict[str, Any] | None = None,
        retry: RetryPolicy | dict[str, Any] | None = None,
        prompt_cache: PromptCachePolicy | dict[str, Any] | None = _OMITTED,
        tools: ToolPolicy | dict[str, Any] | None = None,
        # Deprecated 0.3 flat names. Each overrides the matching sub-policy attribute.
        max_iterations: int | None = None,
        max_context_tokens: int | None = None,
        max_result_chars: int | None = None,
        max_cost_dollars: float | None = None,
        input_cost_per_m: float | None = None,
        output_cost_per_m: float | None = None,
        llm_max_attempts: int | None = None,
        llm_retry_backoff_seconds: float | None = None,
        model_timeout_seconds: float | None = None,
    ) -> None:
        self.model, self.provider, self.max_tokens = model, provider, max_tokens
        self.system_prompt, self.temperature = system_prompt, temperature
        self.limits = _coerce(Limits, limits, "limits")
        self.retry = _coerce(RetryPolicy, retry, "retry")
        self.tools = _coerce(ToolPolicy, tools, "tools")
        if prompt_cache is _OMITTED:
            self.prompt_cache = PromptCachePolicy()
        elif prompt_cache is None:
            self.prompt_cache = None
        else:
            self.prompt_cache = _coerce(PromptCachePolicy, prompt_cache, "prompt_cache")
        legacy = {
            name: value
            for name, value in (
                ("max_iterations", max_iterations),
                ("max_context_tokens", max_context_tokens),
                ("max_result_chars", max_result_chars),
                ("max_cost_dollars", max_cost_dollars),
                ("input_cost_per_m", input_cost_per_m),
                ("output_cost_per_m", output_cost_per_m),
                ("llm_max_attempts", llm_max_attempts),
                ("llm_retry_backoff_seconds", llm_retry_backoff_seconds),
                ("model_timeout_seconds", model_timeout_seconds),
            )
            if value is not None
        }
        if legacy:
            warnings.warn(
                f"AgentConfig flat fields {sorted(legacy)} are deprecated and will be removed in "
                "harnessx 0.5; pass limits=Limits(...) and retry=RetryPolicy(...) instead",
                DeprecationWarning,
                stacklevel=2,
            )
            for name, value in legacy.items():
                group, attr = _LEGACY_FIELDS[name]
                setattr(self, group, replace(getattr(self, group), **{attr: value}))
        self.__post_init__()  # init=False: dataclasses will not call it for us

    def __post_init__(self) -> None:
        from .providers.registry import is_known_provider

        if not isinstance(self.provider, str) or not is_known_provider(self.provider):
            raise ConfigurationError(
                f"Unknown provider: {self.provider!r}; expected a built-in name or one passed to register_provider()"
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigurationError("model must be a nonempty string")
        if not isinstance(self.system_prompt, str):
            raise TypeError("system_prompt must be a string")
        if self.max_tokens is not None:
            _positive(self.max_tokens, "max_tokens", integer=True)
        _nonnegative_or_none(self.temperature, "temperature")
        for name, kind in (("limits", Limits), ("retry", RetryPolicy), ("tools", ToolPolicy)):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"{name} must be a {kind.__name__}")
        if self.prompt_cache is not None and not isinstance(self.prompt_cache, PromptCachePolicy):
            raise TypeError("prompt_cache must be a PromptCachePolicy or None")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentConfig:
        """Rebuild from a persisted dict, 0.3 flat or 0.4 nested, without deprecation warnings.

        Flat keys are lifted into their sub-policy and override nested values,
        matching the constructor's precedence.
        """
        payload = dict(data)
        lifted: dict[str, dict[str, Any]] = {}
        for name, (group, attr) in _LEGACY_FIELDS.items():
            if name in payload:
                lifted.setdefault(group, {})[attr] = payload.pop(name)
        for group, values in lifted.items():
            current = payload.get(group)
            if isinstance(current, dict):
                base = dict(current)
            elif current is not None:
                base = asdict(current)
            else:
                base = {}
            payload[group] = {**base, **values}
        return cls(**payload)

    # ── Deprecated 0.3 flat aliases; removed in 0.5 ─────────────────────────
    @property
    def max_iterations(self) -> int:
        _warn_flat("max_iterations")
        return self.limits.max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        _warn_flat("max_iterations")
        self.limits = replace(self.limits, max_iterations=value)

    @property
    def max_context_tokens(self) -> int:
        _warn_flat("max_context_tokens")
        return self.limits.max_context_tokens

    @max_context_tokens.setter
    def max_context_tokens(self, value: int) -> None:
        _warn_flat("max_context_tokens")
        self.limits = replace(self.limits, max_context_tokens=value)

    @property
    def max_result_chars(self) -> int:
        _warn_flat("max_result_chars")
        return self.limits.max_result_chars

    @max_result_chars.setter
    def max_result_chars(self, value: int) -> None:
        _warn_flat("max_result_chars")
        self.limits = replace(self.limits, max_result_chars=value)

    @property
    def max_cost_dollars(self) -> float | None:
        _warn_flat("max_cost_dollars")
        return self.limits.max_cost_dollars

    @max_cost_dollars.setter
    def max_cost_dollars(self, value: float | None) -> None:
        _warn_flat("max_cost_dollars")
        self.limits = replace(self.limits, max_cost_dollars=value)

    @property
    def input_cost_per_m(self) -> float | None:
        _warn_flat("input_cost_per_m")
        return self.limits.input_cost_per_m

    @input_cost_per_m.setter
    def input_cost_per_m(self, value: float | None) -> None:
        _warn_flat("input_cost_per_m")
        self.limits = replace(self.limits, input_cost_per_m=value)

    @property
    def output_cost_per_m(self) -> float | None:
        _warn_flat("output_cost_per_m")
        return self.limits.output_cost_per_m

    @output_cost_per_m.setter
    def output_cost_per_m(self, value: float | None) -> None:
        _warn_flat("output_cost_per_m")
        self.limits = replace(self.limits, output_cost_per_m=value)

    @property
    def llm_max_attempts(self) -> int:
        _warn_flat("llm_max_attempts")
        return self.retry.attempts

    @llm_max_attempts.setter
    def llm_max_attempts(self, value: int) -> None:
        _warn_flat("llm_max_attempts")
        self.retry = replace(self.retry, attempts=value)

    @property
    def llm_retry_backoff_seconds(self) -> float:
        _warn_flat("llm_retry_backoff_seconds")
        return self.retry.backoff_seconds

    @llm_retry_backoff_seconds.setter
    def llm_retry_backoff_seconds(self, value: float) -> None:
        _warn_flat("llm_retry_backoff_seconds")
        self.retry = replace(self.retry, backoff_seconds=value)

    @property
    def model_timeout_seconds(self) -> float | None:
        _warn_flat("model_timeout_seconds")
        return self.retry.call_timeout_seconds

    @model_timeout_seconds.setter
    def model_timeout_seconds(self, value: float | None) -> None:
        _warn_flat("model_timeout_seconds")
        self.retry = replace(self.retry, call_timeout_seconds=value)


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
            raise ConfigurationError(f"Unknown sandbox tier: {self.tier!r}")
        for name in ("timeout_seconds", "max_memory_mb", "max_cpu_seconds", "max_file_size_mb"):
            _positive(getattr(self, name), name, integer=True)
        if self.network_enabled is not None and type(self.network_enabled) is not bool:
            raise TypeError("network_enabled must be bool or None")
        if self.tier == "process" and (self.network_enabled is False or self.allowed_paths):
            raise ConfigurationError("Process execution cannot restrict network or filesystem access; choose docker or seatbelt")


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
    max_session_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.storage_dir, str) or not self.storage_dir:
            raise ConfigurationError("storage_dir must be nonempty")
        if self.max_session_duration_seconds is not None:
            _positive(self.max_session_duration_seconds, "max_session_duration_seconds")


def _positive(value: Any, name: str, *, integer: bool = False) -> None:
    if (integer and type(value) is not int) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} must be a finite positive {'integer' if integer else 'number'}")


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
