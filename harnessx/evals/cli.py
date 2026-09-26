"""CLI entrypoint for HarnessX evaluations.

Usage:
    # Run tool calling eval suite offline
    python -m harnessx.evals.cli --suite tool_calling --offline

    # Run skills benchmark against Anthropic and upload to LangSmith
    python -m harnessx.evals.cli --suite skills --model claude-sonnet-4-6 --upload

    # Run all benchmarks with concurrency
    python -m harnessx.evals.cli --suite all --concurrency 2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()
if not os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]

from ..core import Agent
from ..subagents import SubAgent
from ..permissions import PermissionLevel
from ..skills import SkillManager
from ..types import AgentConfig
from .datasets.registry import list_datasets
from .evaluators import default_evaluators, list_evaluator_names, resolve_evaluators
from .runner import evaluate_agent


def _build_benchmark_agent_factory(
    model: str,
    provider: str,
    suite: str,
) -> Any:
    """Creates an agent factory configured for the selected benchmark suite."""
    skills_dir = Path(__file__).resolve().parent.parent.parent / "examples" / "skills"
    sample_skills = []
    if skills_dir.exists():
        sample_skills = [
            str(skills_dir / "commit-message"),
            str(skills_dir / "code-review"),
            str(skills_dir / "bug-triage"),
            str(skills_dir / "sql-explain.md"),
        ]

    def factory(inputs: dict[str, Any]) -> Agent:
        system_prompt = (
            "You are a helpful and precise assistant. Use tools when helpful to "
            "perform calculations or inspect files. If a user request matches a skill, "
            "invoke the Skill tool. For specialized research, delegate to delegate_research."
        )

        agent = Agent(
            config=AgentConfig(
                model=model,
                provider=provider,
                system_prompt=system_prompt,
                max_iterations=10,
            ),
            skills=sample_skills if sample_skills and suite in ("skills", "all") else None,
            subagents=[
                SubAgent(
                    name="research",
                    description="Delegate a research question to a specialist.",
                    config=AgentConfig(
                        model=model, provider=provider,
                        system_prompt="You are a research specialist. Provide key comparison points.",
                        max_iterations=5,
                    ),
                ),
                SubAgent(
                    name="code_review",
                    description="Delegate code review to a specialist.",
                    config=AgentConfig(
                        model=model, provider=provider,
                        system_prompt="You are a code review specialist. Highlight any security flaws.",
                        max_iterations=5,
                    ),
                ),
            ],
        )

        # Register standard benchmark tools
        @agent.tools.register(permission=PermissionLevel.ALLOW)
        def calculate(expression: str) -> str:
            """Safely evaluate arithmetic expressions.

            Args:
                expression: Arithmetic expression like '144 / 12'.
            """
            try:
                allowed_chars = set("0123456789+-*/(). ")
                if not all(c in allowed_chars for c in expression):
                    return "Error: only basic arithmetic characters allowed"
                return str(eval(expression, {"__builtins__": None}, {}))
            except Exception as e:
                return f"Error: {e}"

        @agent.tools.register(permission=PermissionLevel.ALLOW)
        def read_file(path: str) -> str:
            """Read file contents.

            Args:
                path: File path to read.
            """
            if "non_existent" in path or not os.path.exists(path):
                return f"Error: File '{path}' does not exist"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"Error: {e}"

        return agent

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HarnessX Evaluation Suite (LangSmith)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        "-s",
        choices=list_datasets() + ["all"],
        default="tool_calling",
        help="Benchmark suite to evaluate",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=os.getenv("AGENT_MODEL", "claude-sonnet-4-6"),
        help="Model ID to evaluate",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["anthropic", "openai"],
        default=os.getenv("AGENT_PROVIDER", "anthropic"),
        help="LLM provider name",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Execute locally without uploading traces to LangSmith",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload evaluation results to LangSmith",
    )
    parser.add_argument(
        "--evaluator",
        "-e",
        action="append",
        dest="evaluators",
        help=(
            f"Specific evaluator(s) to run (can be used multiple times or comma-separated). "
            f"Options: {', '.join(list_evaluator_names())}. Defaults to all standard evaluators."
        ),
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=1,
        help="Maximum concurrent evaluation runs",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Include model-graded LLM judge in evaluators",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Custom experiment prefix name",
    )

    args = parser.parse_args()

    # Determine upload mode
    offline_mode = True
    has_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    if args.upload:
        offline_mode = False
    elif not args.offline and has_key:
        offline_mode = False

    prefix = args.prefix or f"eval-{args.suite}-{args.model}"

    agent_factory = _build_benchmark_agent_factory(
        model=args.model,
        provider=args.provider,
        suite=args.suite,
    )

    if args.evaluators:
        eval_suite = resolve_evaluators(
            args.evaluators,
            judge_model=args.model,
        )
    else:
        eval_suite = default_evaluators(
            include_llm_judge=args.judge,
            judge_model=args.model,
        )

    try:
        summary = evaluate_agent(
            agent=agent_factory,
            dataset=args.suite,
            evaluators=eval_suite,
            experiment_prefix=prefix,
            max_concurrency=args.concurrency,
            offline=offline_mode,
            print_summary=True,
        )
        if summary.pass_rate < 0.5:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nEvaluation canceled.")
        sys.exit(130)


if __name__ == "__main__":
    main()
