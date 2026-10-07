"""Autonomous task runner: give it a task on the command line; it researches, writes files, runs them, and exits.

Unlike the interactive examples, this takes the whole task up front and exits when the work is done, which suits
a script, a cron job, or a CI step. Hooks narrate the run as it goes (steps, tool calls, retries, loops), so a long
task is not a silent one. Everything optional degrades rather than fails: web search appears only with a Tavily
key (over MCP), tracing only with a LangSmith key. The file and shell tools are always there, which is enough to
produce a deliverable in any format.

Run:
    python examples/05-control/autonomous_task_runner.py "Chart this month's downloads as an HTML page" -o chart.html
    python examples/05-control/autonomous_task_runner.py "Write a CSV of the first 20 primes" --max-iterations 10 --max-cost 0.50
Needs: ANTHROPIC_API_KEY. Optional: TAVILY_API_KEY for web search, LANGSMITH_API_KEY for tracing
(`pip install "harnessx[langsmith]"`).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

from dotenv import load_dotenv

from harnessx import (
    Agent,
    AgentConfig,
    HookEvent,
    HookManager,
    Limits,
    MCPManager,
    MCPServerConfig,
    PermissionLevel,
)
from harnessx.builtin import edit_file, read_file, run_bash, write_file

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

SYSTEM_PROMPT = (
    "You are an autonomous engineer. You research, write code, and produce the "
    "deliverable the user asked for.\n\n"
    "Honour the requested format exactly. A Python script means an executable "
    ".py file. A report means .html or .md. Data means .json or .csv.\n\n"
    "Build a long artifact incrementally: write_file the first section, then "
    "edit_file to append each one after it. Never put a whole document in one "
    "reply. Each edit_file tells you whether it applied, so trust it rather "
    "than reading the file back.\n\n"
    "Use run_bash to execute what you write, and fix what fails before you "
    "report success.\n\n"
    "Finish with a short summary naming the files you created."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", help="What you want done.")
    parser.add_argument("-o", "--output", help="Where the deliverable should go.")
    parser.add_argument("--max-iterations", type=int, default=25, help="Loop budget (default: 25).")
    parser.add_argument("--max-cost", type=float, default=None, help="Stop above this spend, in dollars.")
    return parser.parse_args(argv)


def build_hooks() -> HookManager:
    """Narrate the run as it happens."""
    hooks = HookManager()

    def tool_start(ctx) -> None:
        call = ctx.data["tool_call"]
        arguments = ", ".join(f"{k}={v!r}" for k, v in call.input.items())
        print(f"  > {call.name}({arguments[:120]})")

    def tool_end(ctx) -> None:
        result = ctx.data["result"]
        body = str(result.content)[:100].replace("\n", " ")
        print(f"  ! {body}" if result.is_error else f"    {len(str(result.content))} chars: {body}")

    hooks.on(HookEvent.LOOP_ITERATION_START, lambda ctx: print(f"step {ctx.data['iteration']}"))
    hooks.on(HookEvent.TOOL_CALL_START, tool_start)
    hooks.on(HookEvent.TOOL_CALL_END, tool_end)
    hooks.on(HookEvent.REPETITION, lambda ctx: print(f"going in circles: cycle of {ctx.data['period']}"))
    hooks.on(HookEvent.RETRY, lambda ctx: print(f"retrying in {ctx.data['wait_seconds']:.1f}s: {ctx.data['error']}"))
    hooks.on(HookEvent.ERROR, lambda ctx: print(f"  ! {ctx.data.get('error')}"))
    return hooks


def build_extensions() -> list:
    """LangSmith tracing, when a key is configured."""
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return []
    from harnessx import LangSmithExtension

    return [LangSmithExtension(project_name=os.getenv("LANGSMITH_PROJECT", "harnessx-tasks"))]


async def connect_search(mcp: MCPManager) -> bool:
    """Tavily web search over MCP, when a key is configured."""
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return False
    await mcp.connect(
        MCPServerConfig.http(
            "research",
            url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={key}",
            permission=PermissionLevel.ALLOW,
        )
    )
    return True


async def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    task = args.task
    if args.output:
        task += f"\n\nSave the deliverable to {args.output}."
    limits = Limits(max_iterations=args.max_iterations, max_cost_dollars=args.max_cost)

    async with MCPManager() as mcp:
        searching = await connect_search(mcp)
        print(f"Task: {args.task}" + ("  (web search on)" if searching else ""))

        async with Agent(
            config=AgentConfig(system_prompt=SYSTEM_PROMPT, limits=limits),
            tools=[run_bash, write_file, edit_file, read_file],
            mcp=mcp,
            hooks=build_hooks(),
            extensions=build_extensions(),
        ) as agent:
            # Tavily ships several tools; one search is all this agent needs.
            for name in agent.mcp_tools:
                if name.startswith("research_") and not name.endswith("_search"):
                    agent.tools.unregister(name)

            started = time.perf_counter()
            result = await agent.run(task)
            elapsed = time.perf_counter() - started

    if result.error:
        print(f"Error: {result.error['message']}")
    else:
        print(f"\n{result.output or '(no output)'}")
    usage = result.usage
    print(
        f"\nStatus {result.status.value} | steps {agent.guardrails.iteration_count} | "
        f"tokens in {usage.input_tokens:,} out {usage.output_tokens:,} cache reads {usage.cache_read_input_tokens:,} | "
        f"{elapsed:.1f}s | ${agent.guardrails.estimated_cost:.4f}"
    )
    return 0 if result.status.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
