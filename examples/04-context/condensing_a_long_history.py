"""Condensing a long history: what the harness does when the conversation outgrows the window.

Before each model call the harness counts the request with the provider's `count_tokens`. Past 80% of
`Limits.max_context_tokens` (minus the reply budget) it condenses, in two steps. First the cheap one: old tool output
beyond the newest ~160K characters is replaced by a one-line note, and if that is enough the run carries on with
the conversation intact. Only when it is not does the harness ask the model to summarize the oldest messages, and
the summary replaces them. Both fire `HookEvent.CONTEXT_CONDENSED` (`messages_dropped`, `summary_chars`,
`summarized`).

A scripted model reads five large daily logs with a small window, so the fourth read triggers a prune and the
fifth a summary. Its `count_tokens` is a plain characters/4 estimate, and it answers the summary request, which the
engine sends with its own system prompt and no tools.

Run:
    python examples/04-context/condensing_a_long_history.py
Needs: Nothing: a scripted model, no network.
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, HookEvent, Limits, ProviderResponse, StopReason, ToolCall
from harnessx.providers import LLMProvider

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

LOG_CHARS = {1: 90_000, 2: 90_000, 3: 90_000, 4: 90_000, 5: 150_000}
SUMMARY = "Task: find the day the checkout errors started. Days 1-3 read: no checkout errors."


class ScriptedProvider(LLMProvider):
    """Fixed replies, so the example runs the real engine without a model."""

    def __init__(self, replies):
        self.replies = list(replies)

    async def create(self, **kwargs):
        if not kwargs["tools"]:  # the summary request: the engine's own system prompt and no tools
            print(f"  (summary requested for {len(kwargs['messages'][0]['content']):,} chars of transcript)")
            return ProviderResponse(text=SUMMARY)
        return self.replies.pop(0)

    async def count_tokens(self, *, messages, system=None, **kwargs):
        return (len(json.dumps(messages)) + len(system or "")) // 4  # about four characters per token


def fetch_log(day: int) -> str:
    """Return the full access log for one day of the incident week."""
    errors = "checkout 500s begin at 09:14" if day == 4 else "no checkout errors"
    line = f"2026-10-0{day} GET /api/orders 200 12ms\n"
    return f"day {day}: {errors}\n" + line * (LOG_CHARS[day] // len(line))


def read(day: int) -> ProviderResponse:
    return ProviderResponse(tool_calls=[ToolCall(f"log_{day}", "fetch_log", {"day": day})],
                            stop_reason=StopReason.TOOL_USE)


SCRIPT = [*(read(day) for day in LOG_CHARS), ProviderResponse(text="The checkout errors started on day 4 at 09:14.")]


def describe(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content[:90].replace("\n", " | ")
    parts = []
    for block in content:
        if block["type"] == "tool_use":
            parts.append(f"call {block['name']}({json.dumps(block['input'])})")
        elif block["type"] == "tool_result":
            parts.append(f"result {len(block['content']):,} chars: {block['content'][:60].strip()!r}")
        elif block["type"] == "text":
            parts.append(block["text"][:80])
    return "; ".join(parts)


def on_condensed(ctx) -> None:
    data = ctx.data
    print(f"CONTEXT_CONDENSED: messages_dropped={data['messages_dropped']} "
          f"summary_chars={data['summary_chars']} summarized={data['summarized']}")
    if not data["summarized"]:
        # Pruning alone made room: the messages are all still there, only old tool output is gone.
        for message in ctx.agent.memory.get_messages():
            for block in message["content"] if isinstance(message["content"], list) else []:
                if block["type"] == "tool_result" and len(block["content"]) < 200:
                    print(f"    {block['tool_use_id']} now reads: {block['content']}")


async def main() -> None:
    # Room for about 307K characters of history: 80% of (100K tokens - a 4K reply) at four characters per token.
    # max_result_chars is raised so each log stays in the conversation instead of being spilled to disk.
    config = AgentConfig(max_tokens=4_000, limits=Limits(max_context_tokens=100_000, max_result_chars=200_000))
    async with Agent(config=config, provider=ScriptedProvider(SCRIPT), tools=[fetch_log]) as agent:
        agent.hooks.on(HookEvent.TOOL_CALL_END, lambda ctx: print(
            f"  > {ctx.data['tool_call'].name}({json.dumps(ctx.data['tool_call'].input)}) "
            f"-> {len(ctx.data['result'].content):,} chars"))
        agent.hooks.on(HookEvent.CONTEXT_CONDENSED, on_condensed)

        result = await agent.run("Read the five daily access logs and tell me which day the checkout errors began.")

        print("\nWhat survives in the history:")
        for index, message in enumerate(agent.memory.get_messages()):
            print(f"  {index} {message['role']:<9} {describe(message)}")

    if result.error:
        print(f"Error: {result.error['message']}")
        return
    print(f"\nAnswer: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
