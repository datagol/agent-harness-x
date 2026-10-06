"""End-to-end demo: load a skill from disk, run an agent that picks it up,
observe the SKILL_INVOKED hook firing.

Run: python -m examples.skills_demo

A scripted provider runs without network or an API key. The fixture
returns tool_use(Skill, 'code-review') on the first turn, then end_turn on
the second turn — simulating what a real model would do when it decides to
load a skill.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from harnessx import (
    Agent,
    AgentConfig,
    HookContext,
    HookEvent,
    ProviderResponse,
    ToolCall,
)
from examples._fixtures import ScriptedProvider
from examples._console import completed_output


HERE = Path(__file__).parent
SKILL_PATH = HERE / "skills" / "code-review"


async def main() -> None:
    agent = Agent(
        config=AgentConfig(system_prompt="You are a careful reviewer."),
        provider=ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[
                        ToolCall("load-skill", "Skill", {"skill": "code-review"})
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(text="Skill loaded — ready to review."),
            ]
        ),
        skills=[str(SKILL_PATH)],
    )

    async with agent:
        # Observe both the generic tool event and the dedicated skill event.
        events: list[str] = []

        async def on_skill(ctx: HookContext) -> None:
            events.append(
                f"SKILL_INVOKED  skill={ctx.data['skill']}  "
                f"found={ctx.data['found']}  body_chars={ctx.data.get('body_chars')}"
            )

        async def on_tool_start(ctx: HookContext) -> None:
            tc = ctx.data["tool_call"]
            events.append(f"TOOL_CALL_START  name={tc.name}  input={tc.input}")

        async def on_tool_end(ctx: HookContext) -> None:
            events.append(f"TOOL_CALL_END    name={ctx.data['tool_call'].name}")

        agent.hooks.on(HookEvent.SKILL_INVOKED, on_skill)
        agent.hooks.on(HookEvent.TOOL_CALL_START, on_tool_start)
        agent.hooks.on(HookEvent.TOOL_CALL_END, on_tool_end)

        print("== Available skills (from system prompt) ==")
        assert agent.skills is not None
        print(agent.skills.render_catalog())
        print()

        print("== Running agent ==")
        result = completed_output(await agent.run("Please review the attached diff."))
        print(f"Final response: {result!r}")
        print()

        print("== Hook events fired (in order) ==")
        for line in events:
            print(f"  • {line}")


if __name__ == "__main__":
    asyncio.run(main())
