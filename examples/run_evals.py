"""Evaluation example: Benchmarking agents using the LangSmith eval framework.

Demonstrates:
  - Defining benchmark test cases with LangSmith Examples
  - Running deterministic tool-selection and output evaluators
  - Capturing tool accuracy and iteration counts
  - Inspecting evaluation results locally and in the LangSmith Web UI

Prerequisites:
  pip install -e '.[langsmith]'

For live models and LangSmith upload:
  export ANTHROPIC_API_KEY=sk-ant-...
  export LANGSMITH_API_KEY=lsv2_pt_...
  export LANGSMITH_PROJECT="harnessx-evals"

Run the scripted fixture (no keys, model calls, or uploads):
  python -m examples.run_evals --offline

Run live against Anthropic and upload to LangSmith:
  python -m examples.run_evals
"""

from __future__ import annotations

import argparse
import os
from functools import partial

from harnessx import Agent, AgentConfig, Limits, PermissionLevel, ProviderResponse, ToolCall
from examples._calculator import calculate
from examples._fixtures import ScriptedProvider
from harnessx.evals import (
    build_example,
    contains_evaluator,
    evaluate_agent,
    no_tool_errors_evaluator,
    tool_args_evaluator,
    tool_selection_evaluator,
)


def build_math_agent(inputs: dict, *, fixture: bool = False) -> Agent:
    """Fresh agent per row; the fixture verifies integration, not model quality."""
    provider = None
    if fixture:
        expressions = {
            "What is 48 * 2?": "48 * 2",
            "Can you compute (15 + 35) / 2?": "(15 + 35) / 2",
        }
        prompt = inputs["prompt"]
        if prompt in expressions:
            expression = expressions[prompt]
            responses = [
                ProviderResponse(
                    tool_calls=[
                        ToolCall("calculate", "calculate", {"expression": expression})
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(text=calculate(expression)),
            ]
        elif prompt == "What is the capital of Japan?":
            responses = [ProviderResponse(text="Tokyo")]
        else:
            raise ValueError("This prompt has no scripted fixture")
        provider = ScriptedProvider(responses)
    agent = Agent(
        provider=provider,
        config=AgentConfig(
            model=os.getenv("AGENT_MODEL", "claude-sonnet-4-6"),
            system_prompt=(
                "Use the calculate tool for arithmetic. Answer non-arithmetic "
                "questions directly without invoking it."
            ),
            limits=Limits(max_iterations=5),
        ),
    )

    agent.tools.register_tool(
        calculate, permission=PermissionLevel.ALLOW, replay_policy="safe"
    )

    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent evaluations with LangSmith")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the scripted custom fixture; no model calls or uploads",
    )
    parser.add_argument(
        "--suite",
        default="custom",
        choices=["custom"],
        help="The three-case arithmetic example; use harnessx.evals.cli for broader suites",
    )
    args = parser.parse_args()

    has_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    upload = not args.offline and has_key

    print("=" * 65)
    print("HarnessX — LangSmith Evaluation Demo")
    print(
        "Mode:",
        "SCRIPTED FIXTURE (no network)"
        if args.offline
        else "LIVE MODEL (API usage applies)",
    )
    print("LangSmith upload:", "enabled" if upload else "disabled")
    print("=" * 65)

    # 1. Define this example's three evaluation cases.
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

    # 2. Select evaluators
    evaluators = [
        tool_selection_evaluator,
        tool_args_evaluator,
        no_tool_errors_evaluator,
        contains_evaluator,
    ]

    # 3. Run evaluation
    summary = evaluate_agent(
        agent=partial(build_math_agent, fixture=args.offline),
        dataset=dataset,
        evaluators=evaluators,
        experiment_prefix="demo-math-agent",
        offline=not upload,  # Runner offline=True disables upload, not model calls.
        print_summary=True,
    )

    if summary.url:
        print(
            f"\nExplore interactive traces and comparison in LangSmith:\n{summary.url}\n"
        )


if __name__ == "__main__":
    main()
