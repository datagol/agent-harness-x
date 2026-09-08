"""Evaluation example: Benchmarking agents using the LangSmith eval framework.

Demonstrates:
  - Defining benchmark test cases with LangSmith Examples
  - Running deterministic and model-graded evaluators
  - Capturing tool accuracy, iteration counts, token usage, and cost
  - Inspecting evaluation results locally and in the LangSmith Web UI

Prerequisites for LangSmith upload:
  export ANTHROPIC_API_KEY=sk-ant-...
  export LANGSMITH_API_KEY=lsv2_pt_...
  export LANGSMITH_PROJECT="datagol-agent-evals"

Run offline (no keys required, zero cost):
  python -m examples.run_evals --offline

Run live against Anthropic and upload to LangSmith:
  python -m examples.run_evals
"""

from __future__ import annotations

import argparse
import os
import sys

from datagol_agent_harness import Agent, AgentConfig, PermissionLevel
from datagol_agent_harness.evals import (
    build_example,
    contains_evaluator,
    default_evaluators,
    evaluate_agent,
    no_tool_errors_evaluator,
    tool_args_evaluator,
    tool_selection_evaluator,
)


def build_math_agent(inputs: dict) -> Agent:
    """Agent factory providing arithmetic tools."""
    agent = Agent(
        config=AgentConfig(
            model=os.getenv("AGENT_MODEL", "claude-sonnet-4-6"),
            system_prompt=(
                "You are an expert math assistant. Always use the calculate tool "
                "to evaluate mathematical expressions accurately."
            ),
            max_iterations=5,
        )
    )

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def calculate(expression: str) -> str:
        """Safely compute simple arithmetic expressions.

        Args:
            expression: Arithmetic expression like '144 / 12'.
        """
        try:
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return "Error: only basic arithmetic characters allowed"
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Error: {e}"

    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent evaluations with LangSmith")
    parser.add_argument("--offline", action="store_true", help="Run locally without LangSmith upload")
    parser.add_argument("--suite", default="tool_calling", help="Benchmark dataset name or 'custom'")
    args = parser.parse_args()

    has_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    offline = args.offline or not has_key

    print("=" * 65)
    print("DataGOL Agent Harness — LangSmith Evaluation Demo")
    print(f"Mode: {'OFFLINE (Local)' if offline else 'LIVE (Uploading to LangSmith)'}")
    print("=" * 65)

    # 1. Define custom evaluation dataset or load from registry
    if args.suite == "custom":
        dataset = [
            build_example(
                inputs={"prompt": "What is 48 * 2?"},
                outputs={
                    "expected_tools": ["calculate"],
                    "contains_all": ["96"],
                    "expected_args": {"calculate": {"expression": "48 * 2"}},
                    "max_allowed_iterations": 3,
                },
                metadata={"difficulty": "easy"},
            ),
            build_example(
                inputs={"prompt": "Can you compute (15 + 35) / 2?"},
                outputs={
                    "expected_tools": ["calculate"],
                    "contains_all": ["25"],
                    "max_allowed_iterations": 3,
                },
                metadata={"difficulty": "easy"},
            ),
            build_example(
                inputs={"prompt": "What is the capital of Japan?"},
                outputs={
                    "forbidden_tools": ["calculate"],
                    "contains_all": ["Tokyo"],
                    "max_allowed_iterations": 2,
                },
                metadata={"category": "negative_tool_avoidance"},
            ),
        ]
    else:
        dataset = args.suite

    # 2. Select evaluators
    evaluators = [
        tool_selection_evaluator,
        tool_args_evaluator,
        no_tool_errors_evaluator,
        contains_evaluator,
    ]

    # 3. Run evaluation
    summary = evaluate_agent(
        agent=build_math_agent,
        dataset=dataset,
        evaluators=evaluators,
        experiment_prefix="demo-math-agent",
        offline=offline,
        print_summary=True,
    )

    if summary.url:
        print(f"\nExplore interactive traces and comparison in LangSmith:\n{summary.url}\n")


if __name__ == "__main__":
    main()
