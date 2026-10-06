"""Measure an agent against a small dataset: did it pick the right tool, pass the right arguments, answer correctly?

``build_example`` describes each case (the prompt, the tools it should or must not call, the words the answer
must contain). ``evaluate_agent_async`` runs a fresh agent per case and scores it with deterministic evaluators,
then prints a summary. With LangSmith keys set, a live run also uploads the experiment so you can compare runs.

Run:
    python examples/06-quality/evaluating_with_datasets.py --offline   # scripted model, no network
    python examples/06-quality/evaluating_with_datasets.py             # live model; uploads when LangSmith is set
Needs: pip install "harnessx[langsmith]". --offline needs nothing else. Live: ANTHROPIC_API_KEY, and optionally
LANGSMITH_API_KEY (and LANGSMITH_PROJECT) to upload.
"""

import argparse
import ast
import asyncio
import operator
import os
from functools import partial

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, Limits, PermissionLevel, ProviderResponse, ToolCall
from harnessx.evals import (
    build_example,
    contains_evaluator,
    evaluate_agent_async,
    no_tool_errors_evaluator,
    tool_args_evaluator,
    tool_selection_evaluator,
)
from harnessx.providers import LLMProvider

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculate(expression: str) -> str:
    """Evaluate arithmetic with numbers, parentheses and + - * /.

    Args:
        expression: Arithmetic expression, such as '(25 * 4) + 50'.
    """
    # Walk the syntax tree instead of eval(), so the model can never run Python.
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("Only numbers, parentheses and + - * / are allowed")

    if len(expression) > 200:
        raise ValueError("Expression must be at most 200 characters")
    result = evaluate(ast.parse(expression, mode="eval").body)
    return str(int(result) if float(result).is_integer() else result)


class ScriptedProvider(LLMProvider):
    """Fixed replies, so the example runs the real engine without a model."""

    def __init__(self, replies):
        self.replies = list(replies)

    async def create(self, **kwargs):
        return self.replies.pop(0)

    async def count_tokens(self, **kwargs):
        return 0


def scripted_replies(prompt: str) -> list[ProviderResponse]:
    """What a well-behaved model would do; the fixture checks the wiring, not model quality."""
    expressions = {"What is 48 * 2?": "48 * 2", "Can you compute (15 + 35) / 2?": "(15 + 35) / 2"}
    if prompt in expressions:
        expression = expressions[prompt]
        return [
            ProviderResponse(tool_calls=[ToolCall("calc", "calculate", {"expression": expression})],
                             stop_reason="tool_use"),
            ProviderResponse(text=calculate(expression)),
        ]
    if prompt == "What is the capital of Japan?":
        return [ProviderResponse(text="Tokyo")]
    raise ValueError("This prompt has no scripted fixture")


def build_math_agent(inputs: dict, *, fixture: bool = False) -> Agent:
    """A fresh agent per dataset row, so one case cannot leak history into the next."""
    agent = Agent(
        provider=ScriptedProvider(scripted_replies(inputs["prompt"])) if fixture else None,
        config=AgentConfig(
            model=os.getenv("AGENT_MODEL", "claude-sonnet-4-6"),
            system_prompt="Use the calculate tool for arithmetic. Answer other questions directly without it.",
            limits=Limits(max_iterations=5),
        ),
    )
    agent.tools.register_tool(calculate, permission=PermissionLevel.ALLOW, replay_policy="safe")
    return agent


DATASET = [
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
        outputs={"expected_tools": ["calculate"], "contains_all": ["25"], "max_allowed_iterations": 3},
        metadata={"difficulty": "easy"},
    ),
    build_example(
        inputs={"prompt": "What is the capital of Japan?"},
        outputs={"forbidden_tools": ["calculate"], "contains_all": ["Tokyo"], "max_allowed_iterations": 2},
        metadata={"category": "negative_tool_avoidance"},
    ),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent evaluations with LangSmith")
    parser.add_argument("--offline", action="store_true", help="Use the scripted fixture; no model calls or uploads")
    parser.add_argument(
        "--suite", default="custom", choices=["custom"],
        help="The three-case arithmetic example; use harnessx.evals.cli for broader suites",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    upload = not args.offline and bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    print("Mode:", "SCRIPTED FIXTURE (no network)" if args.offline else "LIVE MODEL (API usage applies)")
    print("LangSmith upload:", "enabled" if upload else "disabled")

    summary = await evaluate_agent_async(
        partial(build_math_agent, fixture=args.offline),
        DATASET,
        evaluators=[tool_selection_evaluator, tool_args_evaluator, no_tool_errors_evaluator, contains_evaluator],
        experiment_prefix="demo-math-agent",
        offline=not upload,  # offline=True only turns off the upload; a live model is still called
        print_summary=True,
    )
    if summary.url:
        print(f"\nCompare this experiment in LangSmith: {summary.url}")


if __name__ == "__main__":
    asyncio.run(main())
