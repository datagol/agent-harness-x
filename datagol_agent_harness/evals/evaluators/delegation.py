"""Multi-agent delegation evaluators for DataGOL Agent Harness."""

from __future__ import annotations

from typing import Any
from langsmith.schemas import Example, Run


def subagent_delegated_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates whether an orchestrator delegated to the expected sub-agent tool.

    Looks for `expected_delegation` in example.outputs or example.metadata.
    """
    example_outputs = example.outputs or {}
    expected_delegation = (
        example_outputs.get("expected_delegation")
        or example.metadata.get("expected_delegation")
    )

    if not expected_delegation:
        return {
            "key": "subagent_delegation",
            "score": 1.0,
            "comment": "No delegation requirement specified",
        }

    outputs = run.outputs or {}
    tools_used = outputs.get("tools_used", [])

    delegated = any(expected_delegation.lower() in t.lower() for t in tools_used)
    if delegated:
        return {
            "key": "subagent_delegation",
            "score": 1.0,
            "comment": f"Correctly delegated via tool '{expected_delegation}'",
        }

    return {
        "key": "subagent_delegation",
        "score": 0.0,
        "comment": f"Failed to delegate to '{expected_delegation}'. Tools called: {tools_used}",
    }
