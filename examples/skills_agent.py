"""Interactive agent that loads a set of sample skills from disk.

Skills live in examples/skills/. The agent sees only each skill's name +
description in its system prompt. When it decides a skill applies, it calls
the `Skill` tool — the body is pulled into context lazily, and the
SKILL_INVOKED hook fires so we can show a live indicator.

Run (Anthropic, default):
    python -m examples.skills_agent

Run with OpenAI:
    AGENT_PROVIDER=openai AGENT_MODEL=gpt-5 python -m examples.skills_agent

Try prompts like:
  • "Write a commit message for: refactored auth middleware to extract token
    parsing into a separate function"
  • "Triage this bug: users get 500 when applying two coupons; happens for me
    in prod but not staging"
  • "Explain this SQL: SELECT * FROM orders WHERE LOWER(email) = 'a@b.com'"
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from datagol_agent_harness import (
    Agent,
    AgentConfig,
    HookContext,
    HookEvent,
)
from examples._console import (
    console,
    get_user_input,
    print_banner,
    print_error,
    print_status,
)


SKILLS_DIR = Path(__file__).parent / "skills"

# Every entry can be a SKILL.md file or a folder containing one.
SKILL_PATHS = [
    SKILLS_DIR / "code-review",
    SKILLS_DIR / "commit-message",
    SKILLS_DIR / "bug-triage",
    SKILLS_DIR / "sql-explain.md",
]


def _build_agent() -> Agent:
    provider = os.environ.get("AGENT_PROVIDER", "anthropic")
    default_model = "claude-sonnet-4-6" if provider == "anthropic" else "gpt-5"
    model = os.environ.get("AGENT_MODEL", default_model)

    return Agent(
        config=AgentConfig(
            model=model,
            provider=provider,
            system_prompt=(
                "You are a focused engineering assistant. When a user request "
                "matches a skill in <available-skills>, call the Skill tool "
                "first to load its instructions, then follow them. Don't load "
                "a skill unless it clearly applies."
            ),
        ),
        skills=[str(p) for p in SKILL_PATHS],
    )


async def main() -> None:
    agent = _build_agent()

    # ── Live indicator when a skill is picked up ────────────────────────────
    async def on_skill_invoked(ctx: HookContext) -> None:
        if ctx.data.get("found"):
            console.print(
                f"  [info]📚 loaded skill[/info] [tool.name]{ctx.data['skill']}[/tool.name] "
                f"[dim]({ctx.data['body_chars']} chars from "
                f"{Path(ctx.data['source_path']).name})[/dim]"
            )
        else:
            console.print(
                f"  [warning]⚠ unknown skill requested:[/warning] {ctx.data['skill']}"
            )

    agent.hooks.on(HookEvent.SKILL_INVOKED, on_skill_invoked)

    skill_lines = "\n".join(
        f"  • [tool.name]{s.name}[/tool.name] — {s.description}"
        for s in agent.skills.list()
    )
    print_banner(
        "DataGOL Agent Harness — Skills Demo",
        subtitle=f"provider={agent.config.provider}  model={agent.config.model}",
        commands={"quit": "Exit", "usage": "Token stats", "skills": "Re-list skills"},
    )
    console.print("[bold]Available skills:[/bold]")
    console.print(skill_lines)
    console.print()

    while True:
        user_input = get_user_input()
        if user_input is None or user_input.lower() == "quit":
            break
        if not user_input:
            continue
        if user_input.lower() == "usage":
            print_status({"Usage": agent.guardrails.usage_summary})
            continue
        if user_input.lower() == "skills":
            console.print(skill_lines)
            continue

        try:
            response = await agent.run(user_input)
            console.print()
            console.print(f"[agent.label]assistant[/agent.label] {response}")
            console.print()
        except Exception as e:
            print_error(e)

    console.print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
