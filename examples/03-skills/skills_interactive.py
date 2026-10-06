"""Chat with an agent that has a set of skills on disk and loads each one only when it applies.

Skills live in examples/skills/ (a SKILL.md file, or a folder holding one). The agent sees only each skill's
name and description; when a request matches, it calls the `Skill` tool, the body is pulled into context,
and the SKILL_INVOKED hook fires, which this example prints as a live indicator.

Run:
    python examples/03-skills/skills_interactive.py
    AGENT_PROVIDER=openai AGENT_MODEL=gpt-4.1 python examples/03-skills/skills_interactive.py

Needs: ANTHROPIC_API_KEY (or another provider's key, its SDK extra, and AGENT_PROVIDER + AGENT_MODEL)

Try:
  - "Write a commit message for: refactored auth middleware to extract token parsing into a separate function"
  - "Triage this bug: users get 500 when applying two coupons; happens in prod but not staging"
  - "Explain this SQL: SELECT * FROM orders WHERE LOWER(email) = 'a@b.com'"
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, HookContext, HookEvent, RunEventType, ToolCall, ToolResult

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"

# Every entry can be a SKILL.md file or a folder containing one.
SKILL_PATHS = [
    SKILLS_DIR / "code-review",
    SKILLS_DIR / "commit-message",
    SKILLS_DIR / "bug-triage",
    SKILLS_DIR / "sql-explain.md",
]


def _build_agent() -> Agent:
    provider = os.environ.get("AGENT_PROVIDER", "anthropic")
    model = os.environ.get("AGENT_MODEL")
    if not model:
        if provider != "anthropic":
            raise ValueError("Set AGENT_MODEL when selecting a non-default provider")
        model = AgentConfig().model

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


async def stream_reply(agent: Agent, text: str) -> None:
    midline = False  # streamed text has no trailing newline until the answer ends
    async with agent.run_stream(text) as stream:
        async for event in stream:
            if event.type is RunEventType.TEXT_DELTA:
                print(event.data, end="", flush=True)
                midline = True
            elif event.type is RunEventType.TOOL_CALL_START and isinstance(event.data, ToolCall):
                args = json.dumps(event.data.input, default=str)[:120]
                print(("\n" if midline else "") + f"  > {event.data.name}({args})")
                midline = False
            elif event.type is RunEventType.TOOL_RESULT and isinstance(event.data, ToolResult) and event.data.is_error:
                print(f"  ! {event.data.content[:200]}")
        result = await stream.result()
    if midline:
        print()
    if result.status != "completed":
        print(f"Error: {result.error['message'] if result.error else result.status.value}")


async def main() -> None:
    agent = _build_agent()

    async with agent:

        async def on_skill_invoked(ctx: HookContext) -> None:
            if ctx.data.get("found"):
                source = Path(ctx.data["source_path"]).name
                print(f"  [loaded skill {ctx.data['skill']}: {ctx.data['body_chars']} chars from {source}]")
            else:
                print(f"  [unknown skill requested: {ctx.data['skill']}]")

        agent.hooks.on(HookEvent.SKILL_INVOKED, on_skill_invoked)

        skill_lines = "\n".join(f"  - {s.name}: {s.description}" for s in agent.skills.list())
        print(f"Skills agent (provider={agent.config.provider}, model={agent.config.model})")
        print(f"Available skills:\n{skill_lines}")
        print("Commands: skills, usage, quit")

        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit"):
                break
            if not text:
                continue
            if text.lower() == "usage":
                print(f"Usage: {agent.guardrails.usage_summary}")
                continue
            if text.lower() == "skills":
                print(skill_lines)
                continue
            try:
                await stream_reply(agent, text)
            except Exception as exc:
                print(f"Error: {exc}")

        print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
