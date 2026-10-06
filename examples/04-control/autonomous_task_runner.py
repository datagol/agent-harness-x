"""One-shot CLI agent: give it a task, it researches, writes files, and runs them.

    python -m examples.task_agent "Chart this month's downloads as an HTML page" -o chart.html

Unlike the interactive examples, this one takes the whole task on the command
line and exits when the work is done. That shape suits a script, a cron job, or
a CI step.

Everything optional degrades rather than fails. Web search appears only with a
Tavily key, tracing only with a LangSmith key. The agent always has the file
and shell tools, which is enough to produce a deliverable in any format.

Requires an Anthropic key. Set TAVILY_API_KEY for web search and
LANGSMITH_API_KEY for tracing.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

from rich.panel import Panel
from rich.table import Table

from examples._console import console
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", help="What you want done.")
    parser.add_argument("-o", "--output", help="Where the deliverable should go.")
    parser.add_argument(
        "--max-iterations", type=int, default=25, help="Loop budget (default: 25)."
    )
    parser.add_argument(
        "--max-cost", type=float, default=None, help="Stop above this spend, in dollars."
    )
    return parser.parse_args()


def build_hooks() -> HookManager:
    """Narrate the run as it happens, so a long task is not a silent one."""
    hooks = HookManager()

    def preview(value: object, limit: int = 70) -> str:
        text = repr(value)
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def _iteration(ctx) -> None:
        console.print(f"[warning]step {ctx.data.get('iteration', 1)}[/warning]")

    @hooks.before_tool
    def _tool_start(ctx) -> None:
        call = ctx.data.get("tool_call")
        if call:
            arguments = ", ".join(f"{k}={preview(v)}" for k, v in call.input.items())
            console.print(f"  [tool.name]{call.name}[/tool.name]([tool.input]{arguments}[/tool.input])")

    @hooks.after_tool
    def _tool_end(ctx) -> None:
        result = ctx.data.get("result")
        body = str(getattr(result, "content", ""))[:100].replace("\n", " ")
        if getattr(result, "is_error", False):
            console.print(f"  [tool.error]failed:[/tool.error] {body}")
        else:
            console.print(f"  [info]{len(str(getattr(result, 'content', '')))} chars[/info] {body}")

    def _repetition(ctx) -> None:
        console.print(f"[warning]going in circles: cycle of {ctx.data['period']}[/warning]")

    def _retry(ctx) -> None:
        console.print(f"[warning]retrying in {ctx.data['wait_seconds']:.1f}s: {ctx.data['error']}[/warning]")

    def _error(ctx) -> None:
        console.print(f"[tool.error]{ctx.data.get('error')}[/tool.error]")

    hooks.on(HookEvent.LOOP_ITERATION_START, _iteration)
    hooks.on(HookEvent.REPETITION, _repetition)
    hooks.on(HookEvent.RETRY, _retry)
    hooks.on(HookEvent.ERROR, _error)
    return hooks


def build_extensions() -> list:
    """LangSmith tracing, when a key is configured."""
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not key:
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


def report(result, agent, elapsed: float) -> None:
    table = Table(title="Run", show_header=False)
    table.add_column("", style="dim")
    table.add_column("", justify="right")
    table.add_row("Status", str(result.status.value))
    table.add_row("Steps", str(agent.guardrails.iteration_count))
    table.add_row("Input tokens", f"{result.usage.input_tokens:,}")
    table.add_row("Output tokens", f"{result.usage.output_tokens:,}")
    table.add_row("Cache reads", f"{result.usage.cache_read_input_tokens:,}")
    table.add_row("Elapsed", f"{elapsed:.1f}s")
    table.add_row("Estimated cost", f"${agent.guardrails.estimated_cost:.4f}")
    console.print(table)


async def main() -> None:
    args = parse_args()

    task = args.task
    if args.output:
        task += f"\n\nSave the deliverable to {args.output}."

    limits = Limits(max_iterations=args.max_iterations, max_cost_dollars=args.max_cost)
    hooks = build_hooks()

    async with MCPManager() as mcp:
        searching = await connect_search(mcp)
        console.print(
            Panel(
                f"[user.label]{args.task}[/user.label]",
                title="task" + ("  ·  web search on" if searching else ""),
                border_style="cyan",
            )
        )

        async with Agent(
            config=AgentConfig(system_prompt=SYSTEM_PROMPT, limits=limits),
            tools=[run_bash, write_file, edit_file, read_file],
            mcp=mcp,
            hooks=hooks,
            extensions=build_extensions(),
        ) as agent:
            # Tavily ships several tools; one search is all this agent needs.
            for name in agent.mcp_tools:
                if name.startswith("research_") and not name.endswith("_search"):
                    agent.tools.unregister(name)

            started = time.perf_counter()
            result = await agent.run(task)
            elapsed = time.perf_counter() - started

    console.print(Panel(result.output or "(no output)", title="result", border_style="green"))
    report(result, agent, elapsed)
    result.raise_for_status()


if __name__ == "__main__":
    asyncio.run(main())
