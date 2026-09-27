"""Evaluators for HarnessX."""

from __future__ import annotations

from typing import Any, Callable
from langsmith.schemas import Example, Run

from .correctness import (
    contains_evaluator,
    exact_match_evaluator,
    json_valid_evaluator,
    regex_evaluator,
)
from .delegation import subagent_delegated_evaluator
from .llm_judge import LLMJudgeEvaluator
from .skills import skill_invoked_evaluator
from .tools import (
    no_tool_errors_evaluator,
    tool_args_evaluator,
    tool_call_count_evaluator,
    tool_selection_evaluator,
)
from .trajectory import (
    iteration_budget_evaluator,
    no_agent_errors_evaluator,
    step_sequence_evaluator,
)


def default_evaluators(
    *,
    include_llm_judge: bool = False,
    judge_model: str = "claude-sonnet-4-6",
) -> list[Callable[[Run, Example], Any]]:
    """Returns the standard recommended suite of deterministic evaluators."""
    evals: list[Callable[[Run, Example], Any]] = [
        tool_selection_evaluator,
        tool_args_evaluator,
        no_tool_errors_evaluator,
        contains_evaluator,
        exact_match_evaluator,
        regex_evaluator,
        json_valid_evaluator,
        iteration_budget_evaluator,
        no_agent_errors_evaluator,
        skill_invoked_evaluator,
        subagent_delegated_evaluator,
    ]

    if include_llm_judge:
        evals.append(LLMJudgeEvaluator(model=judge_model))

    return evals


_EVALUATOR_MAP: dict[str, Callable[[Run, Example], Any]] = {
    "tool_selection": tool_selection_evaluator,
    "tool_selection_evaluator": tool_selection_evaluator,
    "tools": tool_selection_evaluator,
    "tool_args": tool_args_evaluator,
    "tool_args_evaluator": tool_args_evaluator,
    "args": tool_args_evaluator,
    "no_tool_errors": no_tool_errors_evaluator,
    "no_tool_errors_evaluator": no_tool_errors_evaluator,
    "tool_call_count": tool_call_count_evaluator,
    "tool_call_count_evaluator": tool_call_count_evaluator,
    "count": tool_call_count_evaluator,
    "contains": contains_evaluator,
    "contains_evaluator": contains_evaluator,
    "exact_match": exact_match_evaluator,
    "exact_match_evaluator": exact_match_evaluator,
    "exact": exact_match_evaluator,
    "regex": regex_evaluator,
    "regex_evaluator": regex_evaluator,
    "json_valid": json_valid_evaluator,
    "json_valid_evaluator": json_valid_evaluator,
    "json": json_valid_evaluator,
    "iteration_budget": iteration_budget_evaluator,
    "iteration_budget_evaluator": iteration_budget_evaluator,
    "iterations": iteration_budget_evaluator,
    "step_sequence": step_sequence_evaluator,
    "step_sequence_evaluator": step_sequence_evaluator,
    "sequence": step_sequence_evaluator,
    "no_agent_errors": no_agent_errors_evaluator,
    "no_agent_errors_evaluator": no_agent_errors_evaluator,
    "skill_invoked": skill_invoked_evaluator,
    "skill_invoked_evaluator": skill_invoked_evaluator,
    "skills": skill_invoked_evaluator,
    "subagent_delegated": subagent_delegated_evaluator,
    "subagent_delegated_evaluator": subagent_delegated_evaluator,
    "delegation": subagent_delegated_evaluator,
}


def list_evaluator_names() -> list[str]:
    """Returns canonical evaluator names available for selection."""
    return [
        "tool_selection",
        "tool_args",
        "no_tool_errors",
        "tool_call_count",
        "contains",
        "exact_match",
        "regex",
        "json_valid",
        "iteration_budget",
        "step_sequence",
        "no_agent_errors",
        "skill_invoked",
        "subagent_delegated",
        "llm_judge",
    ]


def get_evaluator(
    name: str,
    *,
    judge_model: str = "claude-sonnet-4-6",
) -> Callable[[Run, Example], Any]:
    """Resolve an evaluator by name or alias."""
    key = name.lower().strip()
    if key in ("llm_judge", "judge"):
        return LLMJudgeEvaluator(model=judge_model)
    if key in _EVALUATOR_MAP:
        return _EVALUATOR_MAP[key]
    raise ValueError(
        f"Unknown evaluator '{name}'. Available evaluators: {list_evaluator_names()}"
    )


def resolve_evaluators(
    names: list[str] | tuple[str, ...],
    *,
    judge_model: str = "claude-sonnet-4-6",
) -> list[Callable[[Run, Example], Any]]:
    """Resolves a list of evaluator names or comma-separated strings."""
    resolved: list[Callable[[Run, Example], Any]] = []
    for item in names:
        for part in item.split(","):
            part = part.strip()
            if part:
                resolved.append(get_evaluator(part, judge_model=judge_model))
    return resolved


__all__ = [
    # Tools
    "tool_selection_evaluator",
    "tool_args_evaluator",
    "no_tool_errors_evaluator",
    "tool_call_count_evaluator",
    # Correctness
    "contains_evaluator",
    "exact_match_evaluator",
    "regex_evaluator",
    "json_valid_evaluator",
    # Trajectory
    "iteration_budget_evaluator",
    "step_sequence_evaluator",
    "no_agent_errors_evaluator",
    # Skills & Multi-Agent
    "skill_invoked_evaluator",
    "subagent_delegated_evaluator",
    # LLM Judge
    "LLMJudgeEvaluator",
    # Default suite & resolution helpers
    "default_evaluators",
    "list_evaluator_names",
    "get_evaluator",
    "resolve_evaluators",
]
