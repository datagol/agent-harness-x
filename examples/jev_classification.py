"""Classify document text with Choice; defaults to a fixed W2 fixture.

Run: python -m examples.jev_classification
Live: python -m examples.jev_classification --live --text 'Document text...'
"""

import argparse
import asyncio
import json

from harnessx.decisions import (
    ChoiceAnswer, ChoiceQuestion, DecisionBatch, DecisionProvider, DecisionRequest, JevDecisionProvider,
)

from examples._decision_fixtures import FixedDecisionProvider

DOCUMENT_TYPES = {
    "W2": "An annual W-2 wage and tax statement from an employer.",
    "Deposit": "A bank deposit slip or deposit confirmation.",
    "Insurance": "An insurance policy, certificate, or coverage document.",
    "Payroll": "A pay stub or payroll record for a pay period, excluding annual W-2 forms.",
    "unknown": "No clear match, unreadable text, or insufficient information.",
}


async def classify_document(text: str, decisions: DecisionProvider) -> DecisionBatch:
    return await decisions.evaluate(DecisionRequest(
        "document_classification", "v1", {"document_text": text},
        {"document_type": ChoiceQuestion("Classify the document. Use unknown when evidence is insufficient.", DOCUMENT_TYPES)},
    ))


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call Jev using TYPESAFE_API_KEY.")
    parser.add_argument("--text", default="Form W-2 Wage and Tax Statement. Wages: $50,000. Federal tax withheld: $5,000.")
    args = parser.parse_args(argv)
    judge = (JevDecisionProvider() if args.live else FixedDecisionProvider({
        "document_type": ChoiceAnswer("W2", {"W2": .92, "Deposit": .01, "Insurance": .01, "Payroll": .04, "unknown": .02}, .9),
    }))
    async with judge:
        result = await classify_document(args.text, judge)
    print(json.dumps({"mode": "live" if args.live else "fixed offline fixtures", "decision": result.to_dict()}, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
