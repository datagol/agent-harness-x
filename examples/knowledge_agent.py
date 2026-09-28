"""Interactive agent that answers from an Open Knowledge Format (OKF) bundle.

The bundle lives in examples/knowledge/: markdown concepts with YAML
frontmatter, an index.md for progressive disclosure, and links between
concepts. The agent sees only the bundle index in its system prompt and pulls
concepts into context lazily with `search_concepts`, `read_concept`, and
`get_neighbors`. The KNOWLEDGE_ACCESSED hook fires on every call so we can
show a live indicator.

Run (Anthropic, default):
    python -m examples.knowledge_agent

Point it at another bundle — a folder or a git URL — with AGENT_KNOWLEDGE:
    AGENT_KNOWLEDGE=https://github.com/GoogleCloudPlatform/open-knowledge-format/tree/main/bundles/ga4 \\
        python -m examples.knowledge_agent

Other providers require their optional SDK extra, API key, and AGENT_MODEL.
Set AGENT_PROVIDER and AGENT_MODEL before running this module.

Try prompts like:
  • "How are active users defined, and which events count?"
  • "What table feeds the engagement metrics, and how is it partitioned?"
  • "Walk me through the weekly report. Is that guidance final?"
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from harnessx import (
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
    completed_output,
)


KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


def _build_agent() -> Agent:
    provider = os.environ.get("AGENT_PROVIDER", "anthropic")
    model = os.environ.get("AGENT_MODEL")
    if not model:
        if provider != "anthropic":
            raise ValueError("Set AGENT_MODEL when selecting a non-default provider")
        model = AgentConfig().model

    # A local folder or a git URL; several sources can be separated with commas.
    sources = [s.strip() for s in os.environ.get("AGENT_KNOWLEDGE", str(KNOWLEDGE_DIR)).split(",") if s.strip()]

    return Agent(
        config=AgentConfig(
            model=model,
            provider=provider,
            system_prompt=(
                "You are an analytics assistant. Answer only from the knowledge "
                "bundle: search for concepts, read the ones that apply, and cite "
                "each concept's path and its sources. If the bundle does not "
                "cover a question, say so instead of guessing."
            ),
        ),
        knowledge=sources,
    )


async def main() -> None:
    agent = _build_agent()

    async with agent:
        # ── Live indicator when the model touches the bundle ────────────────
        async def on_knowledge(ctx: HookContext) -> None:
            data = ctx.data
            target = data.get("query") or data.get("path") or ""
            if data.get("found", True):
                extra = f" → {data['hits']} hits" if "hits" in data else ""
                console.print(
                    f"  [info]📖 {data['tool']}[/info] [tool.name]{target}[/tool.name]"
                    f"[dim]{extra} ({data.get('bundle')})[/dim]"
                )
            else:
                console.print(f"  [warning]⚠ {data['tool']} found nothing for:[/warning] {target}")

        agent.hooks.on(HookEvent.KNOWLEDGE_ACCESSED, on_knowledge)

        assert agent.knowledge is not None
        bundle_lines = "\n".join(
            f"  • [tool.name]{b.name}[/tool.name] — {len(b.concepts)} concepts"
            + (f", okf_version {b.okf_version}" if b.okf_version else "")
            for b in agent.knowledge.list()
        )
        print_banner(
            "HarnessX — Knowledge Demo (OKF)",
            subtitle=f"provider={agent.config.provider}  model={agent.config.model}",
            commands={
                "quit": "Exit",
                "usage": "Token stats",
                "index": "Show the bundle index",
                "warnings": "Show loader warnings",
            },
        )
        console.print("[bold]Loaded bundles:[/bold]")
        console.print(bundle_lines)
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
            if user_input.lower() == "index":
                for bundle in agent.knowledge.list():
                    console.print(bundle.render_summary())
                continue
            if user_input.lower() == "warnings":
                warnings = agent.knowledge.warnings()
                console.print("\n".join(warnings) if warnings else "[dim]no warnings[/dim]")
                continue

            try:
                response = completed_output(await agent.run(user_input))
                console.print()
                console.print(f"[agent.label]assistant[/agent.label] {response}")
                console.print()
            except Exception as e:
                print_error(e)

        console.print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
