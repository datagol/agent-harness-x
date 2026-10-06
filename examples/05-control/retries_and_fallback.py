"""Retries and fallback: ride out a flaky provider, then fail over to a backup.

`AgentConfig.retry` (a `RetryPolicy`) decides how many times the engine tries one model step and how long it waits
between tries; each scheduled retry fires `HookEvent.RETRY`. A `FallbackProvider` is an ordered chain of providers:
once the primary has collected `switch_after` transient failures, the chain asks the next member within the same
call. With named vendors you would write `AgentConfig(provider="anthropic", fallbacks=[Fallback("openai", ...)])`;
the scripted providers here are live objects, so they go in a `FallbackProvider` passed as `provider=`.

The primary raises an error carrying HTTP 503, which harnessx classifies as transient (429, 5xx, timeouts and
connection loss are; other 4xx errors stop at once). The first failure is retried by the engine; the second moves
the chain to the backup, which answers. Waits are a few milliseconds so the example runs in about a second.

Run:
    python examples/05-control/retries_and_fallback.py
Needs: Nothing: a scripted model, no network.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, FallbackProvider, HookEvent, ProviderResponse, RetryPolicy
from harnessx.providers import LLMProvider

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables


class ScriptedProvider(LLMProvider):
    """Fixed replies, so the example runs the real engine without a model."""

    def __init__(self, replies):
        self.replies = list(replies)

    async def create(self, **kwargs):
        return self.replies.pop(0)

    async def count_tokens(self, **kwargs):
        return 0


class Overloaded(Exception):
    """What a vendor SDK raises for a 503; the status code is what marks it transient."""

    status_code = 503


class FlakyProvider(LLMProvider):
    """A primary that is down for this whole example."""

    name = "primary"

    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        print(f"  primary: call {self.calls} -> 503 overloaded")
        raise Overloaded("503 Service Unavailable: overloaded")

    async def count_tokens(self, **kwargs):
        return 0


async def main() -> None:
    primary = FlakyProvider()
    backup = ScriptedProvider([ProviderResponse(text="An index is a sorted lookup structure that lets the database "
                                                     "find rows without scanning the whole table.")])
    backup.name = "backup"  # the label hooks and agent.provider.last_served report

    # switch_after=2: the first 503 goes back to the engine (a RETRY), the second moves the chain to the backup.
    chain = FallbackProvider(primary, fallbacks=[backup], switch_after=2)
    config = AgentConfig(retry=RetryPolicy(attempts=3, backoff_seconds=0.05, jitter=0))

    async with chain, Agent(config=config, provider=chain) as agent:
        agent.hooks.on(HookEvent.RETRY, lambda ctx: print(
            f"RETRY: {ctx.data['kind']} attempt {ctx.data['attempt']} failed ({ctx.data['error']}); "
            f"attempt {ctx.data['next_attempt']} in {ctx.data['wait_seconds']:.2f}s"))
        agent.hooks.on(HookEvent.LLM_RESPONSE, lambda ctx: print(f"LLM_RESPONSE: served by {ctx.data['provider']}"))

        result = await agent.run("Explain a database index in one sentence.")

    if result.error:
        print(f"Error: {result.error['message']}")
        return
    print(f"\nAnswer: {result.output}")
    print(f"Primary calls: {primary.calls}, served by: {chain.last_served}, run status: {result.status.value}")


if __name__ == "__main__":
    asyncio.run(main())
