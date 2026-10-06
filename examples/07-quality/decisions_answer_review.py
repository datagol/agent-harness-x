"""Grade a finished answer with a decision model: how complete is it, and is it supported by the evidence?

After an agent run completes, ``review_answer`` sends the query, the answer and the evidence to a
decision provider (Jev) with two questions: a ``ScoreQuestion`` that rates coverage on the four
``COVERAGE_LEVELS``, and a ``BooleanQuestion`` that asks whether every claim follows from the evidence.
The review is a separate assessment: it never changes the run's status, output or usage, and runs
that did not complete are not reviewed.

Run:
    python examples/07-quality/decisions_answer_review.py
    python examples/07-quality/decisions_answer_review.py --live   # Jev reviews the fixed answer
Needs: Nothing by default: fixed decisions and a scripted agent, no network.
--live needs pip install "harnessx[jev]" and TYPESAFE_API_KEY; the agent stays scripted.
"""

import argparse
import asyncio
import json
from collections.abc import Mapping

from dotenv import load_dotenv

from harnessx import Agent, ProviderResponse, RunResult
from harnessx.decisions import (
    Answer, BooleanAnswer, BooleanCriteria, BooleanQuestion, DecisionBatch, DecisionProvider,
    DecisionRequest, JevDecisionProvider, ScoreAnswer, ScoreQuestion,
)
from harnessx.providers import LLMProvider

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

COVERAGE_LEVELS = (
    "Does not answer the user's request.",
    "Addresses a small part of the request, with major omissions.",
    "Addresses most of the request, with minor omissions.",
    "Fully addresses every part of the request.",
)


class ScriptedProvider(LLMProvider):
    """Fixed replies, so the example runs the real engine without a model."""

    def __init__(self, replies):
        self.replies = list(replies)

    async def create(self, **kwargs):
        return self.replies.pop(0)

    async def count_tokens(self, **kwargs):
        return 0


class FixedDecisionProvider(DecisionProvider):
    """Fixed answers keyed by question name, so the offline run performs no inference."""

    def __init__(self, answers: Mapping[str, Answer]) -> None:
        self.answers = answers

    async def evaluate(
        self, request: DecisionRequest, *, timeout_seconds: float | None = None, max_attempts: int | None = None,
    ) -> DecisionBatch:
        if not set(request.questions) <= set(self.answers):
            raise ValueError("fixture does not match the example's questions")
        return DecisionBatch(
            "completed", request.check_id, request.version,
            answers={name: self.answers[name] for name in request.questions},
            requested_model="offline-fixture", resolved_model="offline-fixture",
        )


async def review_answer(
    result: RunResult, *, query: str, evidence: str, decisions: DecisionProvider,
) -> DecisionBatch | None:
    if result.status != "completed":
        return None
    # This separate assessment never changes the run's status, output, or model usage.
    return await decisions.evaluate(DecisionRequest(
        "answer_review", "v1", {"query": query, "answer": result.output, "evidence": evidence},
        {
            "coverage": ScoreQuestion("How completely does the answer address the user's request?", COVERAGE_LEVELS),
            "supported": BooleanQuestion(
                "Is every factual claim in the answer supported by the supplied evidence?",
                BooleanCriteria(
                    true="Every factual claim follows from the supplied evidence.",
                    false="At least one factual claim is unsupported or contradicts the evidence.",
                ),
            ),
        },
    ))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Call Jev to review the fixture answer.")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    query = "What were sales in January and February?"
    evidence = "Sales report: January $100; February $120."
    agent_reply = ProviderResponse(text="Sales were $100 in January and $120 in February.")
    async with Agent(provider=ScriptedProvider([agent_reply])) as agent:
        result = await agent.run(query)
    judge = JevDecisionProvider() if args.live else FixedDecisionProvider({
        "coverage": ScoreAnswer(2.85, (0, 0, .15, .85), COVERAGE_LEVELS, .88),
        "supported": BooleanAnswer(.96),
    })
    async with judge:
        review = await review_answer(result, query=query, evidence=evidence, decisions=judge)
    print(json.dumps({
        "mode": "live Jev review of fixture answer" if args.live else "fixed offline fixtures",
        "run_status": result.status.value, "answer": result.output,
        "review": review.to_dict() if review is not None else None,
    }, indent=2))
    return 0 if review is not None and review.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
