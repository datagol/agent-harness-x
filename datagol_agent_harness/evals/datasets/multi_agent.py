"""Multi-agent orchestrator delegation benchmark dataset."""

from __future__ import annotations

from langsmith.schemas import Example
from .loader import build_example


def get_multi_agent_dataset() -> list[Example]:
    """Returns benchmark test cases for orchestrator delegation and sub-agent synthesis."""
    return [
        build_example(
            inputs={
                "prompt": (
                    "Please research the architectural differences between DuckDB and SQLite for OLAP."
                )
            },
            outputs={
                "expected_delegation": "delegate_research",
                "contains_all": ["DuckDB", "SQLite"],
                "max_allowed_iterations": 6,
            },
            metadata={"category": "specialist_delegation", "subagent": "research"},
        ),
        build_example(
            inputs={
                "prompt": (
                    "Review this Python function for security flaws:\n"
                    "def execute_query(db, user_input):\n"
                    "    return db.cursor().execute(f'SELECT * FROM users WHERE id = {user_input}')\n"
                )
            },
            outputs={
                "expected_delegation": "delegate_code_review",
                "contains_all": ["injection"],
                "max_allowed_iterations": 6,
            },
            metadata={"category": "specialist_delegation", "subagent": "code_review"},
        ),
    ]
