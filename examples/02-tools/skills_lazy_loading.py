"""Skills load lazily: the model sees a catalog and pulls a skill's body in with the `Skill` tool.

`Agent(skills=[...])` reads SKILL.md files from disk and puts only each skill's name and description in the
system prompt. When the model calls the real `Skill` tool, the body is loaded into context and the
SKILL_INVOKED hook fires. A scripted model plays the part a real one would: it asks for `code-review`
on its first turn and answers on the second. The hook log at the end shows the order things happen in.

Run:
    python examples/02-tools/skills_lazy_loading.py

Needs: Nothing: a scripted model, no network
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, HookContext, HookEvent, ProviderResponse, ToolCall
from harnessx.providers import LLMProvider

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "code-review"


class ScriptedProvider(LLMProvider):
    """Fixed replies, so the example runs the real engine without a model."""

    def __init__(self, replies):
        self.replies = list(replies)

    async def create(self, **kwargs):
        return self.replies.pop(0)

    async def count_tokens(self, **kwargs):
        return 0


async def main() -> None:
    provider = ScriptedProvider(
        [
            ProviderResponse(
                tool_calls=[ToolCall("load-skill", "Skill", {"skill": "code-review"})], stop_reason="tool_use"
            ),
            ProviderResponse(text="Skill loaded, ready to review."),
        ]
    )
    agent = Agent(
        config=AgentConfig(system_prompt="You are a careful reviewer."),
        provider=provider,
        skills=[str(SKILL_PATH)],
    )

    async with agent:
        # Watch the generic tool events and the dedicated skill event side by side.
        events: list[str] = []

        async def on_skill(ctx: HookContext) -> None:
            events.append(
                f"SKILL_INVOKED  skill={ctx.data['skill']}  "
                f"found={ctx.data['found']}  body_chars={ctx.data.get('body_chars')}"
            )

        async def on_tool_start(ctx: HookContext) -> None:
            call = ctx.data["tool_call"]
            events.append(f"TOOL_CALL_START  name={call.name}  input={call.input}")

        async def on_tool_end(ctx: HookContext) -> None:
            events.append(f"TOOL_CALL_END    name={ctx.data['tool_call'].name}")

        agent.hooks.on(HookEvent.SKILL_INVOKED, on_skill)
        agent.hooks.on(HookEvent.TOOL_CALL_START, on_tool_start)
        agent.hooks.on(HookEvent.TOOL_CALL_END, on_tool_end)

        print("== Available skills (what the system prompt carries) ==")
        print(agent.skills.render_catalog())
        print()

        print("== Running agent ==")
        result = await agent.run("Please review the attached diff.")
        if result.status != "completed":
            print(f"Error: {result.error['message'] if result.error else result.status.value}")
            return
        print(f"Final response: {result.output!r}")
        print()

        print("== Hook events fired (in order) ==")
        for line in events:
            print(f"  - {line}")


if __name__ == "__main__":
    asyncio.run(main())
