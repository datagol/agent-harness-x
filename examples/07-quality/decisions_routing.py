"""Let a decision model pick the path: classify a document, then route a query to one of three agents.

A ``ChoiceQuestion`` asks a decision provider (Jev) to choose one option from a fixed set and returns a
probability per option and a confidence. ``classify_document`` labels document text as W2, Deposit,
Insurance, Payroll or unknown. ``choose_route`` picks the sql, research or general agent for a query
and falls back to general unless the answer is a known route with at least ``--min-confidence``;
``run_routed_query`` then runs only that application-defined agent. The threshold here is
illustrative; calibrate your own.

Run:
    python examples/07-quality/decisions_routing.py --min-confidence 0.8
    python examples/07-quality/decisions_routing.py --min-confidence 0.8 --live --provider anthropic --model MODEL_ID \
        --query "Top customers by revenue" --text "W-2 Wage and Tax Statement"
Needs: Nothing by default: fixed decisions and scripted agents, no network.
--live needs pip install "harnessx[jev]", TYPESAFE_API_KEY, and the chosen provider's API key.
"""

import argparse
import asyncio
import json
import math
from collections.abc import Callable, Mapping

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, ProviderResponse, RunResult
from harnessx.decisions import (
    Answer, ChoiceAnswer, ChoiceQuestion, DecisionBatch, DecisionProvider, DecisionRequest, JevDecisionProvider,
)
from harnessx.providers import LLMProvider

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

ROUTES = {
    "sql": "Questions about internal structured data or writing SQL.",
    "research": "Questions requiring external research and source synthesis.",
    "general": "General writing, explanation, and assistance.",
    "unknown": "Insufficient information, conflicting needs, or no clear route.",
}

DOCUMENT_TYPES = {
    "W2": "An annual W-2 wage and tax statement from an employer.",
    "Deposit": "A bank deposit slip or deposit confirmation.",
    "Insurance": "An insurance policy, certificate, or coverage document.",
    "Payroll": "A pay stub or payroll record for a pay period, excluding annual W-2 forms.",
    "unknown": "No clear match, unreadable text, or insufficient information.",
}


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


async def classify_document(text: str, decisions: DecisionProvider) -> DecisionBatch:
    return await decisions.evaluate(DecisionRequest(
        "document_classification", "v1", {"document_text": text},
        {"document_type": ChoiceQuestion(
            "Classify the document. Use unknown when evidence is insufficient.", DOCUMENT_TYPES,
        )},
    ))


async def choose_route(
    query: str, decisions: DecisionProvider, *, min_confidence: float,
) -> tuple[str, DecisionBatch]:
    if (isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float))
            or not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1):
        raise ValueError("min_confidence must be a finite number between zero and one")
    batch = await decisions.evaluate(DecisionRequest(
        check_id="query_routing", version="v1", state={"query": query},
        questions={"route": ChoiceQuestion("Choose the best route; use unknown when ambiguous.", ROUTES)},
    ))
    # An unavailable service, "unknown", or low confidence all fall back to the general agent.
    route = "general"
    answer = batch.choices.get("route")
    if (batch.status == "completed" and answer is not None
            and answer.choice in {"sql", "research", "general"}
            and answer.confidence is not None and answer.confidence >= min_confidence):
        route = answer.choice
    return route, batch


async def run_routed_query(
    query: str, decisions: DecisionProvider, factories: Mapping[str, Callable[[], Agent]],
    *, min_confidence: float,
) -> tuple[str, DecisionBatch, RunResult]:
    if set(factories) != {"sql", "research", "general"}:
        raise ValueError("provide fixed sql, research, and general agent factories")
    route, batch = await choose_route(query, decisions, min_confidence=min_confidence)
    # Only an application-defined factory is ever built; a failed run is returned as-is, never re-routed.
    async with factories[route]() as agent:
        result = await agent.run(query)
    return route, batch, result


def agent_factories(*, live: bool, provider: str, model: str) -> dict[str, Callable[[], Agent]]:
    def create(prompt: str, fixture: str) -> Agent:
        return Agent(
            config=AgentConfig(provider=provider, model=model, system_prompt=prompt),
            provider=None if live else ScriptedProvider([ProviderResponse(text=fixture)]),
        )

    # These minimal specialists draft answers; attach your approved data/research tools here.
    return {
        "sql": lambda: create(
            "Help draft SQL. Ask for schema details when missing; do not claim to execute queries.",
            "SELECT COUNT(*) FROM orders; (Draft SQL; no database was queried.)"),
        "research": lambda: create(
            "Help plan research. Distinguish supplied evidence from facts needing verification.",
            "Start with the source documents and verify their publication dates."),
        "general": lambda: create(
            "Help with writing and explanations. Ask for missing context when needed.",
            "Please share a little more context so I can help."),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Call Jev and the selected agent's model.")
    parser.add_argument("--query", default="Write SQL to count rows in the orders table.")
    parser.add_argument(
        "--text", help="Document text to classify.",
        default="Form W-2 Wage and Tax Statement. Wages: $50,000. Federal tax withheld: $5,000.",
    )
    parser.add_argument("--min-confidence", type=float, required=True)
    parser.add_argument("--provider", choices=("anthropic", "openai", "gemini", "openrouter"), default="anthropic")
    parser.add_argument("--model", help="Agent model ID; required with --live.")
    args = parser.parse_args(argv)
    if args.live and (args.model is None or not args.model.strip()):
        parser.error("--live requires --model and the selected provider's API key")
    return args


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    judge = JevDecisionProvider() if args.live else FixedDecisionProvider({
        "document_type": ChoiceAnswer(
            "W2", {"W2": .92, "Deposit": .01, "Insurance": .01, "Payroll": .04, "unknown": .02}, .9,
        ),
        "route": ChoiceAnswer("sql", {"sql": .9, "research": .03, "general": .05, "unknown": .02}, .9),
    })
    async with judge:
        classification = await classify_document(args.text, judge)
        route, decision, result = await run_routed_query(
            args.query, judge, agent_factories(live=args.live, provider=args.provider, model=args.model or "fixture"),
            min_confidence=args.min_confidence,
        )
    print(json.dumps({
        "mode": "live" if args.live else "fixed offline fixtures",
        "classification": classification.to_dict(),
        "route": route, "decision": decision.to_dict(),
        "run_status": result.status.value, "answer": result.output,
    }, indent=2))
    return 0 if result.status == "completed" and classification.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
