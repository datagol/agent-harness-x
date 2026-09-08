"""DataGOL Agent Harness Evaluations — built on LangSmith."""

from __future__ import annotations

try:
    import langsmith
except ImportError:
    raise ImportError(
        "The 'langsmith' package is required to use evaluations. "
        "Install it with: pip install 'datagol-agent-harness[langsmith]' or pip install langsmith"
    ) from None

from .datasets import (
    build_example,
    get_dataset,
    get_guardrails_dataset,
    get_memory_dataset,
    get_multi_agent_dataset,
    get_skills_dataset,
    get_tool_calling_dataset,
    list_datasets,
    sync_dataset_to_langsmith,
)
from .evaluators import (
    LLMJudgeEvaluator,
    contains_evaluator,
    default_evaluators,
    exact_match_evaluator,
    get_evaluator,
    iteration_budget_evaluator,
    json_valid_evaluator,
    list_evaluator_names,
    no_agent_errors_evaluator,
    no_tool_errors_evaluator,
    regex_evaluator,
    resolve_evaluators,
    skill_invoked_evaluator,
    step_sequence_evaluator,
    subagent_delegated_evaluator,
    tool_args_evaluator,
    tool_call_count_evaluator,
    tool_selection_evaluator,
)
from .runner import EvaluationSummary, evaluate_agent
from .target import AgentTarget, create_agent_target

__all__ = [
    # Runner
    "evaluate_agent",
    "EvaluationSummary",
    # Target
    "AgentTarget",
    "create_agent_target",
    # Evaluators
    "default_evaluators",
    "get_evaluator",
    "list_evaluator_names",
    "resolve_evaluators",
    "tool_selection_evaluator",
    "tool_args_evaluator",
    "no_tool_errors_evaluator",
    "tool_call_count_evaluator",
    "contains_evaluator",
    "exact_match_evaluator",
    "regex_evaluator",
    "json_valid_evaluator",
    "iteration_budget_evaluator",
    "step_sequence_evaluator",
    "no_agent_errors_evaluator",
    "skill_invoked_evaluator",
    "subagent_delegated_evaluator",
    "LLMJudgeEvaluator",
    # Datasets
    "build_example",
    "sync_dataset_to_langsmith",
    "get_dataset",
    "list_datasets",
    "get_tool_calling_dataset",
    "get_skills_dataset",
    "get_multi_agent_dataset",
    "get_guardrails_dataset",
    "get_memory_dataset",
]
