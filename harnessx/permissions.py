"""Guardrails and safety: permission management and resource limits."""

from __future__ import annotations

from .errors import HarnessError

import asyncio
import copy
import inspect
from typing import Any, Awaitable, Callable

from .types import PermissionLevel, TokenUsage, ToolCall, ToolDefinition


class MaxIterationsError(HarnessError):
    pass


class CostLimitError(HarnessError):
    pass


class PermissionManager:
    """Controls tool execution; tools without an explicit policy default to ALLOW."""

    def __init__(
        self, default_level: PermissionLevel = PermissionLevel.ALLOW, *,
        approval_callback: Callable[[ToolCall, ToolDefinition], bool | Awaitable[bool]] | None = None,
    ) -> None:
        if not isinstance(default_level, PermissionLevel):
            raise TypeError("default_level must be a PermissionLevel")
        self._default_level = default_level
        self.approval_callback = approval_callback
        self._overrides: dict[str, PermissionLevel] = {}
        self._session_grants: set[str] = set()

    def set_permission(self, tool_name: str, level: PermissionLevel) -> None:
        """Override permission for a specific tool."""
        if not isinstance(level, PermissionLevel):
            raise TypeError("level must be a PermissionLevel")
        self._overrides[tool_name] = level

    def get_effective_permission(self, tool_name: str, tool_def: ToolDefinition) -> PermissionLevel:
        """Resolve effective permission considering overrides."""
        if tool_name in self._overrides:
            return self._overrides[tool_name]
        return tool_def.permission_level if tool_def.permission_level is not None else self._default_level

    async def check_permission(self, tool_call: ToolCall, tool_def: ToolDefinition) -> bool:
        """Check if a tool call is allowed to proceed.

        ASK uses the configured callback and fails closed when none is supplied.
        """
        level = self.get_effective_permission(tool_call.name, tool_def)
        if level == PermissionLevel.DENY:
            return False
        if tool_call.name in self._session_grants:
            return True

        if level == PermissionLevel.ALLOW:
            return True
        if self.approval_callback is None:
            return False
        answer = self.approval_callback(copy.deepcopy(tool_call), copy.copy(tool_def))
        if inspect.isawaitable(answer):
            answer = await answer
        if type(answer) is not bool:
            raise TypeError("Approval callbacks must return bool")
        return answer

    def grant_session(self, tool_name: str) -> None:
        """Grant a tool for the rest of this session."""
        self._session_grants.add(tool_name)

    def reset_session_grants(self) -> None:
        self._session_grants.clear()


class CliPermissionManager(PermissionManager):
    """Opt-in terminal approval policy, defaulting to ASK for unspecified tools."""

    def __init__(self, default_level: PermissionLevel = PermissionLevel.ASK) -> None:
        super().__init__(default_level, approval_callback=self._prompt)

    async def _prompt(self, tool_call: ToolCall, tool_def: ToolDefinition) -> bool:
        prompt_text = (
            f"\n  Tool: {tool_call.name}\n"
            f"  Input: {_truncate(str(tool_call.input), 200)}\n"
            f"  Allow? [y]es / [n]o / [a]lways: "
        )

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: input(prompt_text).strip().lower())

        if response in ("y", "yes"):
            return True
        if response in ("a", "always"):
            self._session_grants.add(tool_call.name)
            return True
        return False

class GuardrailsEngine:
    """Enforces safety limits on the agentic loop."""

    # Approximate default pricing per million tokens
    INPUT_COST_PER_M = 3.0
    OUTPUT_COST_PER_M = 15.0

    def __init__(
        self,
        max_iterations: int = 50,
        max_cost_dollars: float | None = None,
        input_cost_per_m: float | None = None,
        output_cost_per_m: float | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_cost_dollars = max_cost_dollars
        self.input_cost_per_m = input_cost_per_m if input_cost_per_m is not None else self.INPUT_COST_PER_M
        self.output_cost_per_m = output_cost_per_m if output_cost_per_m is not None else self.OUTPUT_COST_PER_M
        self._iteration_count = 0
        self._lifetime_iterations = 0
        self._total_usage = TokenUsage()

    def reset_turn(self) -> None:
        """Reset turn iteration counter at the start of a user turn."""
        self._iteration_count = 0

    def record_iteration(self) -> None:
        self._iteration_count += 1
        self._lifetime_iterations += 1

    def check_iteration_limit(self) -> None:
        if self.max_iterations and self._iteration_count >= self.max_iterations:
            raise MaxIterationsError(
                f"Agent reached maximum iterations ({self.max_iterations}) for this turn. "
                f"Total usage: {self.usage_summary}"
            )

    def track_usage(self, usage: Any) -> None:
        """Accumulate token usage from an API response."""
        self._total_usage.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self._total_usage.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self._total_usage.cache_creation_input_tokens += getattr(
            usage, "cache_creation_input_tokens", 0
        ) or 0
        self._total_usage.cache_read_input_tokens += getattr(
            usage, "cache_read_input_tokens", 0
        ) or 0

    def check_cost_limit(self) -> None:
        if self.max_cost_dollars is None:
            return
        cost = self.estimated_cost
        if cost >= self.max_cost_dollars:
            raise CostLimitError(
                f"Estimated cost ${cost:.4f} exceeds limit ${self.max_cost_dollars:.4f}. "
                f"Usage: {self.usage_summary}"
            )

    @property
    def estimated_cost(self) -> float:
        u = self._total_usage
        return (u.input_tokens * self.input_cost_per_m + u.output_tokens * self.output_cost_per_m) / 1_000_000

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    @property
    def lifetime_iterations(self) -> int:
        return self._lifetime_iterations

    @property
    def total_usage(self) -> TokenUsage:
        return self._total_usage

    @property
    def usage_summary(self) -> dict[str, Any]:
        return {
            "iterations": self._iteration_count,
            "lifetime_iterations": self._lifetime_iterations,
            "input_tokens": self._total_usage.input_tokens,
            "output_tokens": self._total_usage.output_tokens,
            "cache_creation_input_tokens": self._total_usage.cache_creation_input_tokens,
            "cache_read_input_tokens": self._total_usage.cache_read_input_tokens,
            "thinking_tokens": self._total_usage.thinking_tokens,
            "estimated_cost": f"${self.estimated_cost:.4f}",
        }

    def reset(self) -> None:
        self._iteration_count = 0
        self._lifetime_iterations = 0
        self._total_usage = TokenUsage()


def _truncate(s: str, max_len: int) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s
