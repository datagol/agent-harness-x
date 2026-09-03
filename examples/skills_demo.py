"""End-to-end demo: load a skill from disk, run an agent that picks it up,
observe the SKILL_INVOKED hook firing.

The LLM is mocked so the demo runs without network or an API key. The mock
returns tool_use(Skill, 'code-review') on the first turn, then end_turn on
the second turn — simulating what a real model would do when it decides to
load a skill.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from datagol_agent_harness import Agent, AgentConfig, HookContext, HookEvent


HERE = Path(__file__).parent
SKILL_PATH = HERE / "skills" / "code-review"


class _MockMessages:
    """Stand-in for AsyncAnthropic().messages — scripted two-turn conversation."""

    def __init__(self) -> None:
        self._turn = 0

    async def create(self, **kwargs: Any) -> Any:
        self._turn += 1
        usage = SimpleNamespace(
            input_tokens=10, output_tokens=10,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )

        if self._turn == 1:
            # Turn 1: model decides to load the skill
            block = SimpleNamespace(
                type="tool_use",
                id="call_1",
                name="Skill",
                input={"skill": "code-review"},
            )
            return SimpleNamespace(
                content=[block], stop_reason="tool_use", usage=usage,
            )

        # Turn 2: model produces a final answer (after seeing skill body)
        text = SimpleNamespace(type="text", text="Skill loaded — ready to review.")
        return SimpleNamespace(
            content=[text], stop_reason="end_turn", usage=usage,
        )


class MockAnthropic:
    def __init__(self) -> None:
        self.messages = _MockMessages()


async def main() -> None:
    agent = Agent(
        config=AgentConfig(system_prompt="You are a careful reviewer."),
        client=MockAnthropic(),  # type: ignore[arg-type]
        skills=[str(SKILL_PATH)],
    )

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
    catalog_start = agent.config.system_prompt.index("<available-skills>")
    print(agent.config.system_prompt[catalog_start:])
    print()

    print("== Running agent ==")
    result = await agent.run("Please review the attached diff.")
    print(f"Final response: {result!r}")
    print()

    print("== Hook events fired (in order) ==")
    for line in events:
        print(f"  • {line}")


if __name__ == "__main__":
    asyncio.run(main())
