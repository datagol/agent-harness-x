"""Prompt caching: keep the prefix stable and watch cache reads arrive.

Every iteration of the agent loop resends the system prompt, the tool list,
and the conversation. HarnessX computes a stable prefix key for that request
and tells the provider where the reusable prefix ends; each provider translates
the hint into its vendor's caching mechanism. Caching is on by default.

This example runs a two-iteration loop (one tool call, one answer), records the
prefix key of every model request through the LLM_REQUEST hook, and prints the
cache counters from the usage summary. It is self-contained: copy this file
anywhere after `pip install harnessx`.

    python -m examples.prompt_caching            # scripted provider, no keys
    python -m examples.prompt_caching --off      # same, with caching disabled
    python -m examples.prompt_caching --live     # real model via ANTHROPIC_API_KEY

Live note: vendors only cache prefixes above a minimum size (Anthropic: about
1,024 tokens on most models), so a short system prompt shows zero cache reads
even though the request carried the markers. Cache writes cost slightly more
than plain input on Anthropic; reads cost much less.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from harnessx import (
    Agent,
    AgentConfig,
    HookContext,
    HookEvent,
    PromptCacheHint,
    PromptCachePolicy,
    ProviderResponse,
    StopReason,
    TokenUsage,
    ToolCall,
)
from harnessx.providers import LLMProvider

SYSTEM_PROMPT = (
    "You are a pricing assistant for a hardware store. Use the lookup_price tool "
    "for any price question and answer with the total. Keep this prompt stable: "
    "no timestamps or per-request details belong here, they go in the user turn."
)


def lookup_price(item: str, quantity: int) -> str:
    """Return the unit and total price for a catalog item."""
    unit = {"widget": 2.5, "bracket": 4.0}.get(item.lower(), 1.0)
    return f"{quantity} x {item} at {unit:.2f} each = {unit * quantity:.2f}"


class CachingFixtureProvider(LLMProvider):
    """A scripted provider that reports what a caching vendor would report.

    The first call writes the prefix to the cache; the second call, whose
    prefix is byte-identical, reads it back. It also records the hint the
    engine sent so the example can show what a provider receives.
    """

    name = "fixture"

    def __init__(self) -> None:
        self.hints: list[PromptCacheHint | None] = []

    async def count_tokens(self, **kwargs: Any) -> int:
        return 0

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        cache: PromptCacheHint | None = None,
    ) -> ProviderResponse:
        self.hints.append(cache)
        caching = cache is not None and cache.enabled
        if len(self.hints) == 1:
            return ProviderResponse(
                tool_calls=[ToolCall("call_1", "lookup_price", {"item": "widget", "quantity": 12})],
                stop_reason=StopReason.TOOL_USE,
                usage=TokenUsage(
                    input_tokens=1_200,
                    output_tokens=40,
                    cache_creation_input_tokens=1_150 if caching else 0,
                ),
            )
        return ProviderResponse(
            text="A dozen widgets cost 30.00.",
            usage=TokenUsage(
                input_tokens=1_260,
                output_tokens=18,
                cache_read_input_tokens=1_150 if caching else 0,
            ),
        )


async def main(live: bool = False, disabled: bool = False) -> int:
    policy = None if disabled else PromptCachePolicy(ttl_seconds=3600)
    config = AgentConfig(system_prompt=SYSTEM_PROMPT, prompt_cache=policy)
    provider = None if live else CachingFixtureProvider()
    prefix_keys: list[str | None] = []

    def watch(ctx: HookContext) -> None:
        prefix_keys.append(ctx.data.get("prefix_key"))

    agent = Agent(config=config, provider=provider, tools=[lookup_price])
    agent.hooks.on(HookEvent.LLM_REQUEST, watch)
    async with agent:
        result = await agent.run("What does a dozen widgets cost? Use lookup_price.")
        usage = agent.guardrails.total_usage

    if result.error:
        print(f"Run failed: {result.error['message']}", file=sys.stderr)
        return 1

    print(f"Answer: {result.output}")
    print(f"Caching: {'off' if disabled else 'on'}  |  model requests: {len(prefix_keys)}")
    for index, key in enumerate(prefix_keys, start=1):
        print(f"  request {index} prefix key: {key[:16] + '...' if key else None}")
    stable = len(set(prefix_keys)) == 1 and prefix_keys[0] is not None
    print(f"  prefix stable across iterations: {'yes' if stable else 'no'}")
    if isinstance(provider, CachingFixtureProvider):
        hint = provider.hints[-1]
        print(f"  hint the provider received: enabled={hint.enabled if hint else None}, "
              f"breakpoints={list(hint.breakpoints) if hint else []}")
    print("Usage:")
    print(f"  input tokens          {usage.input_tokens}")
    print(f"  cache writes (tokens) {usage.cache_creation_input_tokens}")
    print(f"  cache reads (tokens)  {usage.cache_read_input_tokens}")
    print(f"  output tokens         {usage.output_tokens}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show prompt caching across one agent run.")
    parser.add_argument("--live", action="store_true", help="Use the configured provider instead of the fixture.")
    parser.add_argument("--off", action="store_true", help="Disable prompt caching for this run.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(asyncio.run(main(live=args.live, disabled=args.off)))
