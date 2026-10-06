"""What a model can take in and give back: its context window and reply limit.

The reply budget used to be guessed from the model name -- a regex that knew
Claude 4 and nothing after it, and a flat 8192 for everything else. A guess that
is too small cuts replies off mid tool call; one that is too large is rejected.
So limits are resolved, in this order:

1. ``register_model_limits`` -- what the application says. Always wins.
2. What a provider told us by refusing a request ("max_tokens: 64000 > 32000"
   or "prompt is too long: 250000 tokens > 200000 maximum"). Learned once, then
   scoped to that provider instance and exact deployment/model id.
3. The provider's own lookup, where it has one (Anthropic's Models API reports
   ``max_input_tokens`` and ``max_tokens`` for every model it serves).
4. The built-in table below, matched by longest prefix.
5. ``UNKNOWN_MODEL`` -- generous, because a model the table has not heard of
   is far more likely to be newer than older, and a too-large request heals
   itself through step 2 while a too-small one silently truncates.
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The reply budget when AgentConfig.max_tokens is unset: the model's own limit,
# capped here. A reply that needs more is escalated after it is cut off, so the
# cap costs nothing but keeps call timeouts and rate-limit reservations sane.
DEFAULT_OUTPUT_CAP = 32_000

# Reply budget kept free when clamping to what is left of the context window.
CONTEXT_SAFETY_TOKENS = 1_024


@dataclass(frozen=True)
class ModelLimits:
    """The context window (input + output) and the largest reply one call may ask for."""

    context_window: int
    max_output: int

    def __post_init__(self) -> None:
        for name in ("context_window", "max_output"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


UNKNOWN_MODEL = ModelLimits(context_window=128_000, max_output=DEFAULT_OUTPUT_CAP)

# Normalized id prefix -> limits. Longest prefix wins, so a specific entry
# ("claude-3-5-haiku") overrides a family one ("claude-3"). Sources: Anthropic's
# models overview, OpenAI's and Google's model pages. The table is a fallback:
# Anthropic models are looked up live first, and any limit a provider reports
# in an error replaces the entry for the rest of the process.
_BUILTIN: dict[str, ModelLimits] = {
    # Anthropic
    "claude-fable-5": ModelLimits(1_000_000, 128_000),
    "claude-mythos-5": ModelLimits(1_000_000, 128_000),
    "claude-opus-5": ModelLimits(1_000_000, 128_000),
    "claude-sonnet-5": ModelLimits(1_000_000, 128_000),
    "claude-opus-4-8": ModelLimits(1_000_000, 128_000),
    "claude-opus-4-7": ModelLimits(1_000_000, 128_000),
    "claude-opus-4-6": ModelLimits(1_000_000, 128_000),
    "claude-sonnet-4-6": ModelLimits(1_000_000, 128_000),
    "claude-opus-4-5": ModelLimits(200_000, 64_000),
    "claude-sonnet-4-5": ModelLimits(200_000, 64_000),
    "claude-haiku-4-5": ModelLimits(200_000, 64_000),
    "claude-opus-4": ModelLimits(200_000, 32_000),  # 4.0 and 4.1
    "claude-sonnet-4": ModelLimits(200_000, 64_000),
    "claude-3-7-sonnet": ModelLimits(200_000, 64_000),
    "claude-3-5": ModelLimits(200_000, 8_192),
    "claude-3": ModelLimits(200_000, 4_096),
    # OpenAI
    "gpt-5": ModelLimits(400_000, 128_000),
    "gpt-4.1": ModelLimits(1_047_576, 32_768),
    "gpt-4o": ModelLimits(128_000, 16_384),
    "gpt-4-turbo": ModelLimits(128_000, 4_096),
    "gpt-4": ModelLimits(8_192, 4_096),
    "o1": ModelLimits(200_000, 100_000),
    "o3": ModelLimits(200_000, 100_000),
    "o4-mini": ModelLimits(200_000, 100_000),
    # Google
    "gemini-2.5": ModelLimits(1_048_576, 65_536),
    "gemini-2.0": ModelLimits(1_048_576, 8_192),
    "gemini-1.5": ModelLimits(1_048_576, 8_192),
}

_registered: dict[str, ModelLimits] = {}
_learned: dict[str, dict[str, int]] = {}  # exact normalized id -> fields a provider reported


def normalize_model_id(model: str) -> str:
    """Reduce a deployment-specific id to the model name the tables use.

    Drops router and cloud prefixes (``anthropic/``, ``us.anthropic.``,
    ``openai/``), Vertex's ``@date`` suffix, Bedrock's ``-v1:0`` version, and a
    trailing release date (``-20250929``).
    """
    name = (model or "").strip().lower()
    name = name.rsplit("/", 1)[-1]  # openrouter "anthropic/claude-...", "models/gemini-..."
    name = re.sub(r"^(?:[a-z]{2,4}\.)?anthropic\.", "", name)  # bedrock "us.anthropic.claude-..."
    name = name.split("@", 1)[0]  # vertex "claude-opus-4-5@20251101"
    name = re.sub(r"-v\d+(?::\d+)?$", "", name)  # bedrock "...-v1:0"
    name = re.sub(r"-\d{8}$", "", name)  # "...-20250929"
    name = re.sub(r"-latest$", "", name)
    return name


def register_model_limits(prefix: str, *, context_window: int, max_output: int) -> None:
    """Declare the limits for every model whose normalized id starts with ``prefix``.

    For models the built-in table does not know, or knows wrongly. Registered
    limits beat everything else, including the provider's own lookup.
    """
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a nonempty string")
    _registered[normalize_model_id(prefix)] = ModelLimits(context_window, max_output)


def _longest_prefix(table: dict[str, ModelLimits], name: str) -> ModelLimits | None:
    best = None
    for prefix, limits in table.items():
        if name.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, limits)
    return best[1] if best else None


def lookup_model_limits(model: str) -> ModelLimits | None:
    """Registered, then built-in limits for ``model``; None when neither knows it."""
    name = normalize_model_id(model)
    return _longest_prefix(_registered, name) or _longest_prefix(_BUILTIN, name)


def learn_model_limits(model: str, *, provider: Any = None, context_window: int | None = None, max_output: int | None = None) -> None:
    """Record a limit a provider reported, so no later request repeats the mistake."""
    if provider is None:
        entry = _learned.setdefault(normalize_model_id(model), {})
    else:
        cache = getattr(provider, "_harness_model_limits", None)
        if cache is None:
            cache = {}
            try:
                provider._harness_model_limits = cache
            except (AttributeError, TypeError):
                return
        entry = cache.setdefault(model, {})
    if context_window and context_window > 0:
        entry["context_window"] = int(context_window)
    if max_output and max_output > 0:
        entry["max_output"] = int(max_output)


def _apply_learned(model: str, limits: ModelLimits, provider: Any = None) -> ModelLimits:
    learned = (getattr(provider, "_harness_model_limits", {}).get(model)
               or _learned.get(normalize_model_id(model)))
    if not learned:
        return limits
    return ModelLimits(
        learned.get("context_window", limits.context_window),
        learned.get("max_output", limits.max_output),
    )


async def resolve_model_limits(provider: Any, model: str) -> ModelLimits:
    """The best-known limits for ``model`` served by ``provider``. Never raises."""
    name = normalize_model_id(model)
    registered = _longest_prefix(_registered, name)
    if registered is not None and not getattr(provider, "_resolves_member_limits", False):
        return registered
    found: ModelLimits | None = None
    hook = getattr(provider, "model_limits", None)
    if callable(hook):
        try:
            value = hook(model)
            if inspect.isawaitable(value):
                value = await value
            if isinstance(value, ModelLimits):
                found = value
        except Exception:
            logger.debug("model_limits lookup failed for %r", model, exc_info=True)
    if found is None:
        found = _longest_prefix(_BUILTIN, name) or UNKNOWN_MODEL
    return _apply_learned(model, found, provider)


def provider_reply_default(provider: Any, model: str, limits: ModelLimits) -> int:
    """Honor the original customization hook without treating it as a model ceiling."""
    from .providers.base import LLMProvider
    chooser = getattr(provider, "default_max_tokens", None)
    if callable(chooser) and getattr(type(provider), "default_max_tokens", None) is not LLMProvider.default_max_tokens:
        try:
            chosen = chooser(model)
            if type(chosen) is int and chosen > 0:
                return min(chosen, limits.max_output)
        except Exception:
            logger.debug("default_max_tokens lookup failed", exc_info=True)
    return default_reply_budget(limits)


def default_reply_budget(limits: ModelLimits) -> int:
    """The reply budget when the application did not set one."""
    return min(limits.max_output, DEFAULT_OUTPUT_CAP)


# ── reading limits out of provider errors ────────────────────────────────────

_OUTPUT_LIMIT_PATTERNS = (
    # Anthropic: "max_tokens: 64000 > 32000, which is the maximum allowed number of output tokens for ..."
    re.compile(r"max_tokens:\s*\d+\s*>\s*(\d+)"),
    # OpenAI: "max_tokens is too large: 50000. This model supports at most 16384 completion tokens"
    re.compile(r"supports at most\s+(\d+)\s+(?:completion|output)\s+tokens"),
    # Generic: "max_output_tokens must be less than or equal to 65536", "... must be <= 8192"
    re.compile(r"max_(?:output_|completion_)?tokens[^.\d]{0,80}?(?:<=|less than or equal to|at most|maximum(?: value)? (?:is|of))\s*(\d+)"),
)
# Anthropic: "input length and `max_tokens` exceed context limit: 190000 + 32000 > 200000"
_CONTEXT_SUM_PATTERN = re.compile(r"(\d+)\s*\+\s*\d+\s*>\s*(\d+)")
_CONTEXT_WINDOW_PATTERNS = (
    # Anthropic: "prompt is too long: 250000 tokens > 200000 maximum"
    re.compile(r"too long:\s*\d+\s*tokens?\s*>\s*(\d+)"),
    # OpenAI: "This model's maximum context length is 128000 tokens"
    re.compile(r"maximum context length is\s+(\d+)"),
)


def reply_limit_from_error(exc: BaseException) -> tuple[int, bool] | None:
    """The largest reply the request could have asked for, read from a refusal.

    Returns ``(limit, is_model_limit)``: True when the number is the model's own
    output cap (worth remembering), False when it is only what this request's
    input left room for. None when the error says nothing about the budget.
    """
    text = str(exc)
    lowered = text.lower()
    if "max_tokens" not in lowered and "max_output_tokens" not in lowered and "max_completion_tokens" not in lowered:
        return None
    if "context" in lowered:
        found = _CONTEXT_SUM_PATTERN.search(text)
        if found:
            room = int(found.group(2)) - int(found.group(1)) - CONTEXT_SAFETY_TOKENS
            return (room, False) if room > 0 else None
    for pattern in _OUTPUT_LIMIT_PATTERNS:
        found = pattern.search(lowered)
        if found and int(found.group(1)) > 0:
            return int(found.group(1)), True
    return None


def context_window_from_error(exc: BaseException) -> int | None:
    """The context window a provider named when it refused a request, if it did."""
    lowered = str(exc).lower()
    for pattern in _CONTEXT_WINDOW_PATTERNS:
        found = pattern.search(lowered)
        if found and int(found.group(1)) > 0:
            return int(found.group(1))
    return None
