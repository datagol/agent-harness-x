"""Choose one agent using Choice. Runs with fixed offline responses by default.

Run: python -m examples.jev_routing --min-confidence 0.8
Live: add --live --provider anthropic --model YOUR_MODEL_ID
The example threshold is illustrative; applications must calibrate their own.
"""

import argparse
import asyncio
from collections.abc import Callable, Mapping
import json
import math

from harnessx import Agent, AgentConfig, ProviderResponse, RunResult
from harnessx.decisions import (
    ChoiceAnswer, ChoiceQuestion, DecisionBatch, DecisionProvider, DecisionRequest, JevDecisionProvider,
)

from examples._decision_fixtures import FixedDecisionProvider
from examples._fixtures import ScriptedProvider

ROUTES = {
    "sql": "Questions about internal structured data or writing SQL.",
    "research": "Questions requiring external research and source synthesis.",
    "general": "General writing, explanation, and assistance.",
    "unknown": "Insufficient information, conflicting needs, or no clear route.",
}


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
    # Select only an application-defined factory. A failed run is returned as-is.
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


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call Jev and the selected agent's model.")
    parser.add_argument("--query", default="Write SQL to count rows in the orders table.")
    parser.add_argument("--min-confidence", type=float, required=True)
    parser.add_argument("--provider", choices=("anthropic", "openai", "gemini", "openrouter"), default="anthropic")
    parser.add_argument("--model", help="Agent model ID; required with --live.")
    args = parser.parse_args(argv)
    if args.live and (args.model is None or not args.model.strip()):
        parser.error("--live requires --model and the selected provider's API key")
    judge = (JevDecisionProvider() if args.live else FixedDecisionProvider({
        "route": ChoiceAnswer("sql", {"sql": .9, "research": .03, "general": .05, "unknown": .02}, .9),
    }))
    async with judge:
        route, decision, result = await run_routed_query(
            args.query, judge, agent_factories(live=args.live, provider=args.provider, model=args.model or "fixture"),
            min_confidence=args.min_confidence,
        )
    print(json.dumps({"mode": "live" if args.live else "fixed offline fixtures", "route": route,
                      "decision": decision.to_dict(), "run_status": result.status.value, "answer": result.output}, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
