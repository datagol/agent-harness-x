"""Skills invocation evaluators for HarnessX."""

from __future__ import annotations

from typing import Any
from langsmith.schemas import Example, Run


def skill_invoked_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates whether the agent lazily invoked the expected skill via the Skill tool.

    Looks for `expected_skill` in example.outputs or example.metadata.
    """
    example_outputs = example.outputs or {}
    expected_skill = (
        example_outputs.get("expected_skill")
        or example.metadata.get("expected_skill")
    )

    outputs = run.outputs or {}
    skills_invoked = outputs.get("skills_invoked", [])

    # Also inspect tool_calls for name == 'Skill'
    if not skills_invoked:
        for tc in outputs.get("tool_calls", []):
            if tc.get("name") in ("Skill", "skill"):
                skill_arg = tc.get("input", {}).get("skill") or tc.get("input", {}).get("name")
                if skill_arg and skill_arg not in skills_invoked:
                    skills_invoked.append(skill_arg)

    if not expected_skill:
        # Negative test: ensure NO skill was invoked
        if not skills_invoked:
            return {
                "key": "skill_routing",
                "score": 1.0,
                "comment": "Correctly did not invoke any skill for non-skill prompt",
            }
        return {
            "key": "skill_routing",
            "score": 0.0,
            "comment": f"Expected no skill, but invoked: {skills_invoked}",
        }

    # Positive test: ensure expected_skill was invoked
    invoked = any(expected_skill.lower() in s.lower() for s in skills_invoked)
    if invoked:
        return {
            "key": "skill_routing",
            "score": 1.0,
            "comment": f"Correctly invoked skill '{expected_skill}'",
        }

    return {
        "key": "skill_routing",
        "score": 0.0,
        "comment": f"Failed to invoke expected skill '{expected_skill}'. Invoked: {skills_invoked}",
    }
