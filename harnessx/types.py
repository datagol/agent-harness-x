"""Shared type definitions for the agent harness. Pure dataclasses, zero logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, TypeVar

from .errors import ConfigurationError

if TYPE_CHECKING:  # type-only: providers import this module at runtime
    from .providers.base import LLMProvider

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
    SAFETY = "safety"  # a content filter stopped or blanked the reply
    REFUSAL = "refusal"  # the model declined (Anthropic's refusal, OpenAI's message.refusal)
    PAUSE_TURN = "pause_turn"  # the server paused a long turn; send it back to continue
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
    # Automatic re-execution after a transient failure; None means the registry
    # default, else ToolRetry(). Applies only to safe and idempotent tools.
    retry: ToolRetry | None = None
    # Application check on a successful result: True asks for a retry (the call
    # failed inside a 200). Never persisted; lives on the definition only.
    retry_if_result: Callable[[ToolResult], bool] | None = None

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
        if isinstance(self.retry, dict):
            self.retry = ToolRetry.from_dict(self.retry)
        if self.retry is not None and not isinstance(self.retry, ToolRetry):
            raise TypeError("Tool retry must be a ToolRetry")
        if self.retry_if_result is not None and not callable(self.retry_if_result):
            raise TypeError("Tool retry_if_result must be callable")


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
    # (Gemini's thoughts_token_count, OpenAI's reasoning_tokens): a breakdown
    # of output_tokens, which already includes them for every built-in
    # provider. 0 when the provider does not report them.
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
    _content_blocks: list[dict[str, Any]] | None = field(default=None, repr=False)

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
        self._content_blocks = None
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
            self._content_blocks = [self._block_dict(block) for block in content]

    @staticmethod
    def _block_dict(block):
        from copy import deepcopy
        if isinstance(block, dict):
            return deepcopy(block)
        if callable(dump := getattr(block, "model_dump", None)):
            return dump(mode="json", exclude=getattr(block, "__api_exclude__", None))
        return deepcopy(vars(block))

    @staticmethod
    def _parse_content(content):
        from copy import deepcopy
        text_parts, calls, thoughts, metadata = [], [], [], {}
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
                thoughts.append(data["thinking"])
            elif kind == "tool_use":
                calls.append(ToolCall(data["id"], data["name"], deepcopy(data["input"])))
            elif kind == "provider":
                continue
            else:
                raise ValueError(f"Unsupported provider content block: {kind!r}")
            key = data["id"] if kind == "tool_use" else kind
            extras = {k: deepcopy(v) for k, v in data.items() if k not in ("type", "text", "thinking", "id", "name", "input")}
            if extras:
                metadata[key] = extras
        return "\n".join(text_parts), calls, "\n".join(thoughts) if thoughts else None, metadata

    @property
    def content(self) -> list[Any]:
        from copy import deepcopy
        if self._content_blocks is not None:
            return self._ordered_content()
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
        self._content_blocks = [self._block_dict(block) for block in blocks]

    def _ordered_content(self) -> list[Any]:
        """Project editable canonical fields onto an immutable native block layout."""
        from copy import deepcopy
        original = self._content_blocks or []
        text, _, thinking, _ = self._parse_content(original)
        unchanged_text = self.text in (text, "".join(b.get("text", "") for b in original))
        changed_thinking = self.thinking != thinking
        if changed_thinking and any(b.get("signature") or b.get("type") == "provider" for b in original):
            raise ValueError("Middleware cannot modify protected provider thinking")
        calls = {call.id: call for call in self.tool_calls}
        blocks = []
        wrote_text = wrote_thinking = False
        for source in original:
            block = deepcopy(source)
            kind = block["type"]
            if kind == "text":
                if not unchanged_text:
                    if wrote_text or not self.text:
                        continue
                    block["text"] = self.text
                wrote_text = True
            elif kind == "thinking":
                if changed_thinking:
                    if wrote_thinking or self.thinking is None:
                        continue
                    block["thinking"] = self.thinking
                wrote_thinking = True
            elif kind == "tool_use":
                call = calls.pop(block["id"], None)
                if call is None:
                    continue
                block.update(name=call.name, input=deepcopy(call.input))
            blocks.append(block)
        if self.thinking is not None and not wrote_thinking:
            blocks.insert(0, {"type": "thinking", "thinking": self.thinking})
        if self.text and not wrote_text:
            blocks.append({"type": "text", "text": self.text})
        blocks.extend({"type": "tool_use", "id": c.id, "name": c.name, "input": deepcopy(c.input)} for c in calls.values())
        if self._content_dicts:
            return blocks
        plain = {"type", "text", "thinking", "id", "name", "input"}
        return [SimpleNamespace(**b) if set(b) <= plain else b for b in blocks]


@dataclass
class StreamChunk:
    """A streaming chunk emitted by an LLMProvider."""

    # "text_delta", "thinking_delta", "response", or "progress": the provider
    # received something that is not visible text -- a streamed tool argument, a
    # signature, a block boundary. It carries no data; it tells the engine the
    # stream is alive, so a long tool call is not mistaken for a stall.
    kind: str
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
    # The context window condensing works against. None uses the model's own
    # window (see harnessx.models); set it to condense earlier than that.
    max_context_tokens: int | None = None
    max_result_chars: int = 12_000  # tool results above this are spilled to disk (~3K tokens)
    max_cost_dollars: float | None = None
    input_cost_per_m: float | None = None
    output_cost_per_m: float | None = None
    loop_guard: LoopGuard = field(default_factory=lambda: LoopGuard())
    # How many times one run may recover from a reply cut off at its token
    # budget -- re-asking for a truncated tool call, or continuing truncated
    # text -- each time with double the budget. 0 ends the run at the first cut.
    max_truncation_recoveries: int = 3
    # On reaching max_iterations, ask the model once more -- told to use no
    # tools and say what it did, what is left, and its best answer -- rather
    # than failing with nothing to show. The run then completes with
    # stop_reason "max_iterations" (not ok). False fails it as before.
    final_answer_on_limit: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.loop_guard, dict):  # restored from a snapshot
            object.__setattr__(self, "loop_guard", LoopGuard(**self.loop_guard))
        if not isinstance(self.loop_guard, LoopGuard):
            raise TypeError("loop_guard must be a LoopGuard")
        if type(self.max_iterations) is not int or self.max_iterations < 0:
            raise ConfigurationError("max_iterations must be nonnegative (0 means unlimited)")
        if self.max_context_tokens is not None:
            _positive(self.max_context_tokens, "max_context_tokens", integer=True)
        if type(self.final_answer_on_limit) is not bool:
            raise TypeError("final_answer_on_limit must be a bool")
        if type(self.max_truncation_recoveries) is not int or self.max_truncation_recoveries < 0:
            raise ConfigurationError("max_truncation_recoveries must be a nonnegative integer")
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

    # Four calls in all: three retries ride out a typical overload or rate-limit
    # spell (OpenCode makes five, Pi and Hermes four) without hiding a real outage.
    attempts: int = 4
    backoff_seconds: float = 0.5  # base delay, doubled per attempt
    # Wall clock around one attempt. None derives it from the reply budget, so a
    # long reply is not cut off by a timeout sized for short ones.
    call_timeout_seconds: float | None = None
    # Cap on one wait, whether from backoff or the server's Retry-After.
    max_backoff_seconds: float = 60.0
    # Longest silence tolerated between two chunks of a streamed reply. A
    # stream that stalls past it is abandoned and the call retried; the whole-
    # call timeout alone let a stalled stream hold a run for many minutes.
    # None turns the check off.
    stream_idle_timeout_seconds: float | None = 180.0
    # Failover, when AgentConfig.fallbacks names other providers. Transient
    # failures one provider may collect before the chain moves to the next, and
    # how long a provider stays out of rotation after being passed over.
    switch_after: int = 1
    cooldown_seconds: float = 0.0
    # Proportional randomness on each wait. Without it, concurrent agents that
    # hit one rate limit retry in lockstep and hit it again together.
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if type(self.attempts) is not int or self.attempts < 1:
            raise ConfigurationError("attempts must be a positive integer (1 disables retry)")
        _nonnegative_or_none(self.backoff_seconds, "backoff_seconds")
        if self.call_timeout_seconds is not None:
            _positive(self.call_timeout_seconds, "call_timeout_seconds")
        _positive(self.max_backoff_seconds, "max_backoff_seconds")
        if self.stream_idle_timeout_seconds is not None:
            _positive(self.stream_idle_timeout_seconds, "stream_idle_timeout_seconds")
        if isinstance(self.jitter, bool) or not 0 <= self.jitter <= 1:
            raise ConfigurationError("jitter must be between 0 and 1")
        if type(self.switch_after) is not int or self.switch_after < 1:
            raise ConfigurationError("switch_after must be a positive integer")
        _nonnegative_or_none(self.cooldown_seconds, "cooldown_seconds")

    def effective_call_timeout(self, max_tokens: int | None) -> float:
        """The timeout for one model call: the explicit value, or one sized for ``max_tokens``."""
        if self.call_timeout_seconds is not None:
            return float(self.call_timeout_seconds)
        return call_timeout_for(max_tokens)

    def wait_for(self, attempt: int, retry_after: float | None = None, *, rand: Any = None) -> float:
        """Seconds to wait after ``attempt`` (1-based) failed.

        Exponential backoff from ``backoff_seconds``, raised to the server's
        ``retry_after`` when it sent one, spread by ``jitter``, and capped at
        ``max_backoff_seconds``.
        """
        wait = _backoff(self.backoff_seconds, self.max_backoff_seconds, attempt, retry_after)
        if not self.jitter or wait <= 0:
            return wait
        import random

        draw = (rand or random.random)()
        # Spread downward only: never wait longer than the cap or the server asked.
        return max(0.0, wait * (1 - self.jitter * draw))


def _backoff(base: float | None, cap: float, attempt: int, retry_after: float | None = None) -> float:
    wait = float(base or 0.0) * (2 ** max(0, attempt - 1))
    if retry_after is not None:
        wait = max(wait, float(retry_after))
    return min(wait, cap)


@dataclass(frozen=True)
class ToolRetry:
    """Automatic re-execution of one tool call after a transient failure.

    ``attempts`` is the total number of tries, so ``1`` disables retry; the
    default matches what the engine always did for a failed safe tool. Retry
    applies only to tools whose replay policy is ``safe`` or ``idempotent``: a
    ``manual`` tool stops for recovery on an unknown outcome instead. A handler
    asks for a retry by raising ``TransientToolError``; with
    ``retry_error_results`` an error result whose text reads as throttling or
    overload is retried too.
    """

    attempts: int = 3
    backoff_seconds: float = 0.5  # base delay, doubled per attempt
    max_backoff_seconds: float = 30.0
    retry_error_results: bool = False

    def __post_init__(self) -> None:
        if type(self.attempts) is not int or self.attempts < 1:
            raise ConfigurationError("attempts must be a positive integer (1 disables retry)")
        _nonnegative_or_none(self.backoff_seconds, "backoff_seconds")
        _positive(self.max_backoff_seconds, "max_backoff_seconds")
        if type(self.retry_error_results) is not bool:
            raise TypeError("retry_error_results must be a bool")

    def wait_for(self, attempt: int) -> float:
        """Seconds to wait after ``attempt`` (1-based) failed."""
        return _backoff(self.backoff_seconds, self.max_backoff_seconds, attempt)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ToolRetry:
        """Rebuild from a run-state entry; a missing or empty mapping is the default policy."""
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def call_timeout_for(max_tokens: int | None) -> float:
    """A timeout that fits a reply of ``max_tokens``: never below the default,
    and at the Anthropic SDK's planning rate (128K tokens per hour) plus a minute."""
    if not max_tokens:
        return DEFAULT_TIMEOUT_SECONDS
    return max(DEFAULT_TIMEOUT_SECONDS, max_tokens * 3600 / 128_000 + 60)


@dataclass(frozen=True)
class LoopGuard:
    """Noticing when an agent has stopped making progress.

    Watches for a repeating cycle of tool calls, up to ``max_period`` long,
    repeated ``threshold`` times with the same results. A trip annotates the
    result the model reads; it never fails the run, which is what
    ``Limits.max_iterations`` is for.

    On by default. A trip needs identical arguments *and* identical results for
    ``threshold`` laps running, which is a genuine loop rather than a slow
    agent, and the only consequence is a note on the tool result the model
    reads. ``LoopGuard(enabled=False)`` opts out.
    """

    enabled: bool = True
    max_period: int = 4
    threshold: int = 3
    history: int = 64

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool")
        _positive(self.max_period, "max_period", integer=True)
        _positive(self.history, "history", integer=True)
        if type(self.threshold) is not int or self.threshold < 2:
            raise ConfigurationError("threshold must be an integer of 2 or more")

    def applies(self) -> bool:
        return self.enabled


@dataclass(frozen=True)
class Fallback:
    """One provider to try when the ones before it fail, named in ``AgentConfig.fallbacks``.

    ``provider`` is a built-in name or one passed to ``register_provider()``.
    ``AgentConfig.fallbacks`` accepts names only, because the config is persisted
    and compared; a live provider is allowed here for the object API, where you
    pass the chain to ``FallbackProvider`` yourself.

    ``model`` is this vendor's id for the model, since the same model is spelled
    differently on different vendors; ``None`` reuses ``AgentConfig.model``.
    ``max_tokens`` likewise overrides the reply budget for this provider only.
    """

    provider: str | LLMProvider
    model: str | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        from .providers.registry import is_known_provider

        if isinstance(self.provider, str):
            if not is_known_provider(self.provider):
                raise ConfigurationError(
                    f"Unknown fallback provider: {self.provider!r}; expected a built-in name "
                    "or one passed to register_provider()"
                )
        elif not hasattr(self.provider, "create"):
            raise ConfigurationError(
                "Fallback provider must be a provider name or an LLMProvider"
            )
        if self.model is not None and (not isinstance(self.model, str) or not self.model):
            raise ConfigurationError("Fallback model must be a nonempty string")
        if self.max_tokens is not None:
            _positive(self.max_tokens, "max_tokens", integer=True)


@dataclass(frozen=True)
class ProgressPolicy:
    """How long a call may run silently before the run stream says it is still working.

    A provider that is slow rather than broken produces nothing at all: no
    deltas, no error, no retry. From the caller's side that is
    indistinguishable from a hung agent, so every consumer ends up writing the
    same timer, or -- more often -- ships an interface that looks frozen.
    ``WAITING`` events say which side is slow, the model or a named tool, and
    how long it has been.

    They are notices, not deadlines: nothing is cancelled, and the call is
    still bounded by ``RetryPolicy.effective_call_timeout`` and the tool's own
    timeout. ``first_after_seconds=None`` turns them off.
    """

    first_after_seconds: float | None = 10.0
    repeat_every_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.first_after_seconds is not None:
            _positive(self.first_after_seconds, "first_after_seconds")
        _positive(self.repeat_every_seconds, "repeat_every_seconds")

    @property
    def enabled(self) -> bool:
        return self.first_after_seconds is not None


@dataclass(frozen=True)
class ToolPolicy:
    """Registry-wide tool options that an Agent applies to the registry it adopts."""

    default_timeout_seconds: float | None = None  # None: the registry's own default, else DEFAULT_TIMEOUT_SECONDS
    dedupe_calls: bool = False  # identical repeated calls within one run return the first result
    retry: ToolRetry | None = None  # default for tools registered without a retry of their own

    def __post_init__(self) -> None:
        if self.default_timeout_seconds is not None:
            _positive(self.default_timeout_seconds, "default_timeout_seconds")
        if type(self.dedupe_calls) is not bool:
            raise TypeError("dedupe_calls must be a bool")
        if isinstance(self.retry, dict):  # restored from a snapshot
            object.__setattr__(self, "retry", ToolRetry.from_dict(self.retry))
        if self.retry is not None and not isinstance(self.retry, ToolRetry):
            raise TypeError("retry must be a ToolRetry")


# 0.3 flat name -> (sub-policy field, attribute). The constructor keywords and
# attributes are gone as of 0.5; this mapping survives only so ``from_dict`` can
# still load sessions and snapshots persisted by 0.3.
_FLAT_0_3_FIELDS: dict[str, tuple[str, str]] = {
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

_P = TypeVar("_P", Limits, RetryPolicy, ToolPolicy, ProgressPolicy, PromptCachePolicy)


def _coerce(kind: type[_P], value: Any, name: str) -> _P:
    if value is None:
        return kind()
    if isinstance(value, dict):  # restored from a snapshot, session, or run state
        return kind(**value)
    if not isinstance(value, kind):
        raise TypeError(f"{name} must be a {kind.__name__}")
    return value


@dataclass(init=False)
class AgentConfig:
    """All configuration for an Agent, with sensible defaults.

    Budgets live on ``limits``, transient-failure handling on ``retry``, prompt
    caching on ``prompt_cache``, and registry-wide tool options on ``tools``.
    The flat 0.3 names (``max_iterations``, ``llm_max_attempts``, ...) were
    removed in 0.5: pass ``limits=Limits(...)`` and ``retry=RetryPolicy(...)``
    instead. ``from_dict`` still reads the flat shape so sessions saved by 0.3
    keep loading.
    """

    model: str = "claude-sonnet-4-6"
    provider: str = "anthropic"  # a built-in name or one passed to register_provider()
    # Give the agent a task list it writes and the harness keeps showing it.
    # Set False for an agent whose work is never multi-step.
    planning: bool = True
    # Providers to try, in order, when the primary fails transiently. Empty means
    # no failover. RetryPolicy.switch_after and .cooldown_seconds govern the chain.
    fallbacks: tuple[Fallback, ...] = ()
    max_tokens: int | None = None  # reply token budget; None resolves it from the model's limits
    system_prompt: str = "You are a helpful assistant."
    temperature: float | None = None
    limits: Limits = field(default_factory=Limits)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    prompt_cache: PromptCachePolicy | None = field(default_factory=PromptCachePolicy)  # None disables caching
    tools: ToolPolicy = field(default_factory=ToolPolicy)
    progress: ProgressPolicy = field(default_factory=ProgressPolicy)

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        provider: str = "anthropic",
        max_tokens: int | None = None,
        *,
        planning: bool = True,
        fallbacks: Sequence[Fallback | Mapping[str, Any]] = (),
        system_prompt: str = "You are a helpful assistant.",
        temperature: float | None = None,
        limits: Limits | dict[str, Any] | None = None,
        retry: RetryPolicy | dict[str, Any] | None = None,
        prompt_cache: PromptCachePolicy | dict[str, Any] | None = _OMITTED,
        tools: ToolPolicy | dict[str, Any] | None = None,
        progress: ProgressPolicy | dict[str, Any] | None = None,
    ) -> None:
        self.model, self.provider, self.max_tokens = model, provider, max_tokens
        self.planning = planning
        self.fallbacks = tuple(
            item if isinstance(item, Fallback) else Fallback(**dict(item)) for item in fallbacks
        )
        self.system_prompt, self.temperature = system_prompt, temperature
        self.limits = _coerce(Limits, limits, "limits")
        self.retry = _coerce(RetryPolicy, retry, "retry")
        self.tools = _coerce(ToolPolicy, tools, "tools")
        self.progress = _coerce(ProgressPolicy, progress, "progress")
        if prompt_cache is _OMITTED:
            self.prompt_cache = PromptCachePolicy()
        elif prompt_cache is None:
            self.prompt_cache = None
        else:
            self.prompt_cache = _coerce(PromptCachePolicy, prompt_cache, "prompt_cache")
        self.__post_init__()  # init=False: dataclasses will not call it for us

    def __post_init__(self) -> None:
        from .providers.registry import is_known_provider

        if type(self.planning) is not bool:
            raise TypeError("planning must be a bool")
        for item in self.fallbacks:
            if not isinstance(item, Fallback):
                raise ConfigurationError("AgentConfig.fallbacks must contain Fallback entries")
            if not isinstance(item.provider, str):
                raise ConfigurationError(
                    "AgentConfig.fallbacks entries must name a provider: the config is "
                    "persisted and compared, and a live provider cannot be. Build the chain "
                    "with FallbackProvider and pass it as Agent(provider=...) instead"
                )
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
        """Rebuild from a persisted dict, 0.3 flat or 0.4 nested.

        The flat 0.3 keys are no longer accepted as constructor keywords, but a
        session file written by 0.3 is still a session file: they are lifted
        into their sub-policy here, silently, and override nested values.
        """
        payload = dict(data)
        lifted: dict[str, dict[str, Any]] = {}
        for name, (group, attr) in _FLAT_0_3_FIELDS.items():
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
    estimated_cost: float | None = None
    provider_state: dict[str, Any] = field(default_factory=dict)
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
    # What the sandbox's network policy refused during the command (remote
    # backends that enforce one, such as OpenShell); empty otherwise.
    denials: list[str] = field(default_factory=list)


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
