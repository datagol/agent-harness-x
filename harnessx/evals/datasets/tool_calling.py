"""Tool calling benchmark dataset for HarnessX."""

from __future__ import annotations

from langsmith.schemas import Example
from .loader import build_example


def get_tool_calling_dataset() -> list[Example]:
    """Returns benchmark test cases for tool selection, argument accuracy, and error handling."""
    return [
        build_example(
            inputs={"prompt": "What is 144 / 12?"},
            outputs={
                "expected_tools": ["calculate"],
                "expected_output": "12",
                "max_allowed_iterations": 3,
            },
            metadata={"category": "single_tool_math", "difficulty": "easy"},
        ),
        build_example(
            inputs={"prompt": "Calculate the result of (25 * 4) + 50."},
            outputs={
                "expected_tools": ["calculate"],
                "expected_output": "150",
                "max_allowed_iterations": 3,
            },
            metadata={"category": "single_tool_math", "difficulty": "easy"},
        ),
        build_example(
            inputs={"prompt": "What is the capital city of France?"},
            outputs={
                "forbidden_tools": ["calculate", "read_file", "run_bash"],
                "contains_all": ["Paris"],
                "max_allowed_iterations": 2,
            },
            metadata={"category": "tool_avoidance_negative", "difficulty": "easy"},
        ),
        build_example(
            inputs={"prompt": "Add 35 and 65 together using your calculator tool."},
            outputs={
                "expected_tools": ["calculate"],
                "expected_output": "100",
                "max_allowed_iterations": 3,
            },
            metadata={"category": "parameter_accuracy", "difficulty": "medium"},
        ),
        build_example(
            inputs={"prompt": "Read the contents of non_existent_file_9876.txt. If it fails, report that the file does not exist."},
            outputs={
                "expected_tools": ["read_file"],
                "contains_all": ["exist"],
                "max_allowed_iterations": 4,
            },
            metadata={"category": "error_recovery", "difficulty": "medium"},
        ),
    ]
