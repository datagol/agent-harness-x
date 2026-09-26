"""Skills lazy routing and adherence benchmark dataset."""

from __future__ import annotations

from langsmith.schemas import Example
from .loader import build_example


def get_skills_dataset() -> list[Example]:
    """Returns benchmark test cases for lazy skill invocation and instruction following."""
    return [
        build_example(
            inputs={
                "prompt": (
                    "Write a conventional commit message for: refactored auth middleware "
                    "to extract token parsing into a separate function"
                )
            },
            outputs={
                "expected_skill": "commit-message",
                "regex_pattern": r"^(refactor|feat|fix|chore)(\(.*\))?: .+$",
                "max_allowed_iterations": 4,
            },
            metadata={"category": "skill_invocation", "skill": "commit-message"},
        ),
        build_example(
            inputs={
                "prompt": (
                    "Triage this bug: users get 500 when applying two coupons; "
                    "happens for me in prod but not staging"
                )
            },
            outputs={
                "expected_skill": "bug-triage",
                "contains_all": ["Severity"],
                "max_allowed_iterations": 4,
            },
            metadata={"category": "skill_invocation", "skill": "bug-triage"},
        ),
        build_example(
            inputs={
                "prompt": (
                    "Explain this SQL: SELECT * FROM orders WHERE LOWER(email) = 'a@b.com'"
                )
            },
            outputs={
                "expected_skill": "sql-explain",
                "contains_all": ["index"],
                "max_allowed_iterations": 4,
            },
            metadata={"category": "skill_invocation", "skill": "sql-explain"},
        ),
        build_example(
            inputs={
                "prompt": "Write a 3-line haiku about a snowy winter morning."
            },
            outputs={
                "expected_skill": None,  # No skill should be invoked
                "max_allowed_iterations": 2,
            },
            metadata={"category": "skill_avoidance_negative"},
        ),
    ]
