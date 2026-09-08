"""Evaluators for DataGOL Agent Harness."""

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
    # Default suite
    "default_evaluators",
]
