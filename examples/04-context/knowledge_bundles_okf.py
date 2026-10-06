"""Chat with an agent that answers from an Open Knowledge Format (OKF) bundle, pulling concepts in on demand.

`Agent(knowledge=[...])` loads a bundle: markdown concepts with YAML frontmatter, an index.md, and links
between concepts. Only the bundle index goes into the system prompt; the model fetches what it needs with
`search_concepts`, `read_concept`, and `get_neighbors`, and the KNOWLEDGE_ACCESSED hook fires on each call,
which this example prints as a live indicator. The sample bundle is examples/knowledge/.

Run:
    python examples/04-context/knowledge_bundles_okf.py
    AGENT_KNOWLEDGE=https://github.com/GoogleCloudPlatform/open-knowledge-format/tree/main/bundles/ga4 \\
        python examples/04-context/knowledge_bundles_okf.py

AGENT_KNOWLEDGE takes a folder or git URL (several, comma-separated). AGENT_PROVIDER and AGENT_MODEL pick
another provider.

Needs: ANTHROPIC_API_KEY (or another provider's key, its SDK extra, and AGENT_PROVIDER + AGENT_MODEL)

Try:
  - "How are active users defined, and which events count?"
  - "What table feeds the engagement metrics, and how is it partitioned?"
  - "Walk me through the weekly report. Is that guidance final?"
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, HookContext, HookEvent, RunEventType, ToolCall, ToolResult

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"


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

        async def on_knowledge(ctx: HookContext) -> None:
            data = ctx.data
            target = data.get("query") or data.get("path") or ""
            if data.get("found", True):
                hits = f", {data['hits']} hits" if "hits" in data else ""
                print(f"  [{data['tool']} {target}{hits} ({data.get('bundle')})]")
            else:
                print(f"  [{data['tool']} found nothing for: {target}]")

        agent.hooks.on(HookEvent.KNOWLEDGE_ACCESSED, on_knowledge)

        print(f"Knowledge agent (provider={agent.config.provider}, model={agent.config.model})")
        print("Loaded bundles:")
        for bundle in agent.knowledge.list():
            version = f", okf_version {bundle.okf_version}" if bundle.okf_version else ""
            print(f"  - {bundle.name}: {len(bundle.concepts)} concepts{version}")
        print("Commands: index, warnings, usage, quit")

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
            if text.lower() == "index":
                for bundle in agent.knowledge.list():
                    print(bundle.render_summary())
                continue
            if text.lower() == "warnings":
                print("\n".join(agent.knowledge.warnings()) or "no warnings")
                continue
            try:
                await stream_reply(agent, text)
            except Exception as exc:
                print(f"Error: {exc}")

        print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
