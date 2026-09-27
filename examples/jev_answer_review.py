"""Review a completed answer with Score (coverage) and Noul (evidence support).

Run: python -m examples.jev_answer_review
Add --live to send the fixed answer and evidence to Jev. The agent stays scripted.
"""

import argparse
import asyncio
import json

from harnessx import Agent, ProviderResponse, RunResult
from harnessx.decisions import (
    BooleanAnswer, BooleanCriteria, BooleanQuestion, DecisionBatch, DecisionProvider,
    DecisionRequest, JevDecisionProvider, ScoreAnswer, ScoreQuestion,
)

from examples._decision_fixtures import FixedDecisionProvider
from examples._fixtures import ScriptedProvider

COVERAGE_LEVELS = (
    "Does not answer the user's request.",
    "Addresses a small part of the request, with major omissions.",
    "Addresses most of the request, with minor omissions.",
    "Fully addresses every part of the request.",
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


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call Jev to review the fixture answer.")
    args = parser.parse_args(argv)
    query = "What were sales in January and February?"
    evidence = "Sales report: January $100; February $120."
    async with Agent(provider=ScriptedProvider([
        ProviderResponse(text="Sales were $100 in January and $120 in February."),
    ])) as agent:
        result = await agent.run(query)
    judge = (JevDecisionProvider() if args.live else FixedDecisionProvider({
        "coverage": ScoreAnswer(2.85, (0, 0, .15, .85), COVERAGE_LEVELS, .88),
        "supported": BooleanAnswer(.96),
    }))
    async with judge:
        review = await review_answer(result, query=query, evidence=evidence, decisions=judge)
    print(json.dumps({"mode": "live Jev review of fixture answer" if args.live else "fixed offline fixtures",
                      "run_status": result.status.value, "answer": result.output,
                      "review": review.to_dict() if review is not None else None}, indent=2))
    return 0 if review is not None and review.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
