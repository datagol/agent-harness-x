"""Trajectory and efficiency evaluators for DataGOL Agent Harness."""

from __future__ import annotations

from typing import Any
from langsmith.schemas import Example, Run


def iteration_budget_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates whether the agent completed within the allotted iteration budget.

    Looks for `max_allowed_iterations` in example.outputs or example.metadata.
    """
    outputs = run.outputs or {}
    iterations = outputs.get("iterations", 1)

    example_outputs = example.outputs or {}
    max_budget = (
        example_outputs.get("max_allowed_iterations")
        or example.metadata.get("max_allowed_iterations")
        or 20
    )

    if iterations <= max_budget:
        return {
            "key": "iteration_budget",
            "score": 1.0,
            "comment": f"Completed in {iterations} iterations (budget: {max_budget})",
        }

    return {
        "key": "iteration_budget",
        "score": 0.0,
        "comment": f"Exceeded iteration budget: took {iterations} > max {max_budget}",
    }


def step_sequence_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates whether tools were invoked in a specific sequential order.

    Looks for `expected_tool_sequence` in example.outputs (e.g. ['read_file', 'write_file']).
    """
    example_outputs = example.outputs or {}
    expected_sequence: list[str] = (
        example_outputs.get("expected_tool_sequence")
        or example.metadata.get("expected_tool_sequence")
        or []
    )

    if not expected_sequence:
        return {
            "key": "step_sequence",
            "score": 1.0,
            "comment": "No sequence requirement specified",
        }

    outputs = run.outputs or {}
    tool_calls = outputs.get("tool_calls", [])
    actual_names = [tc.get("name") for tc in tool_calls]

    # Check subsequence
    it = iter(actual_names)
    matches = all(item in it for item in expected_sequence)

    if matches:
        return {
            "key": "step_sequence",
            "score": 1.0,
            "comment": f"Executed required sequence: {' -> '.join(expected_sequence)}",
        }

    return {
        "key": "step_sequence",
        "score": 0.0,
        "comment": f"Failed sequence constraint. Expected: {expected_sequence}, got: {actual_names}",
    }


def no_agent_errors_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates that the agent run threw no unhandled runtime exceptions."""
    outputs = run.outputs or {}
    error = outputs.get("error")

    example_outputs = example.outputs or {}
    expect_error = example_outputs.get("expect_error", False)

    if expect_error:
        # Negative test expecting failure
        return {
            "key": "no_agent_errors",
            "score": 1.0 if error else 0.0,
            "comment": "Expected error occurred" if error else "Expected error, but agent succeeded",
        }

    if not error:
        return {
            "key": "no_agent_errors",
            "score": 1.0,
            "comment": "Agent completed without runtime exceptions",
        }

    return {
        "key": "no_agent_errors",
        "score": 0.0,
        "comment": f"Agent raised exception: {error}",
    }
