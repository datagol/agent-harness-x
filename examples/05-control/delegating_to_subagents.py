"""Delegating to subagents: an orchestrator hands self-contained jobs to declared specialists.

Each `SubAgent` declares a name, a description, its own config, tools and permission; the SDK turns it into a
`delegate_<name>` tool on the parent and creates (and closes) a child agent with an isolated conversation for every
task it is given. The orchestrator here has three specialists (code review, web research, file analysis) and its
own filesystem tools, and runs one task: a default that uses two of them, or your own.

Run:
    python examples/05-control/delegating_to_subagents.py
    python examples/05-control/delegating_to_subagents.py "Research https://example.com and summarize it"
Needs: ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, CliPermissionManager, HookEvent, Limits, PermissionLevel, SubAgent, ToolRegistry
from harnessx.builtin.filesystem import register_filesystem_tools
from harnessx.builtin.web import fetch_url

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

DEFAULT_TASK = (
    "Ask the file_analysis specialist to list the current directory and say in two sentences what this project "
    "appears to be. Then ask the code_review specialist to review this function:\n\n"
    "def average(values):\n    return sum(values) / len(values)\n\n"
    "Combine both answers into one short report."
)


def specialist_definitions(base_path: str) -> list[SubAgent]:
    """Declare specialist capabilities; creating children is the SDK's job."""
    files = ToolRegistry()
    register_filesystem_tools(
        files, include=["read_file", "list_directory"], base_path=base_path,
        permission=PermissionLevel.ALLOW,
    )
    web = ToolRegistry()
    web.register_tool(fetch_url, permission=PermissionLevel.ALLOW)
    return [
        SubAgent(
            name="code_review",
            description="Review supplied code for correctness, quality, and security.",
            config=AgentConfig(
                system_prompt="Review the supplied code. Explain specific issues and reference lines where possible.",
                limits=Limits(max_iterations=5),
            ),
            permission=PermissionLevel.ALLOW,
        ),
        SubAgent(
            name="research",
            description="Read public URLs supplied in a research task and compare their evidence.",
            config=AgentConfig(
                system_prompt="Read the supplied URLs, answer the research question, and cite your sources.",
                limits=Limits(max_iterations=10),
            ),
            tools=web,
            permission=PermissionLevel.ASK,
        ),
        SubAgent(
            name="file_analysis",
            description="Read files in the workspace and answer questions about their contents.",
            config=AgentConfig(
                system_prompt="Read and analyze the requested workspace files. Explain your findings.",
                limits=Limits(max_iterations=10),
            ),
            tools=files,
            permission=PermissionLevel.ALLOW,
        ),
    ]


def show_tool_call(ctx) -> None:
    call = ctx.data["tool_call"]
    print(f"  > {call.name}({json.dumps(call.input)[:120]})")


async def main(task: str = DEFAULT_TASK) -> None:
    async with Agent(
        permissions=CliPermissionManager(),  # asks on the terminal before delegating to research
        config=AgentConfig(
            system_prompt=(
                "Break complex tasks into self-contained assignments for the appropriate "
                "specialists. Supply the code, URLs, or paths they need, then synthesize "
                "their results into a coherent response. You also have filesystem tools."
            ),
            limits=Limits(max_iterations=20),
        ),
        subagents=specialist_definitions(str(Path.cwd())),
    ) as orchestrator:
        register_filesystem_tools(orchestrator.tools)
        for name in ("read_file", "list_directory"):
            orchestrator.permissions.set_permission(name, PermissionLevel.ALLOW)
        orchestrator.hooks.on(HookEvent.TOOL_CALL_START, show_tool_call)

        print(f"Task: {task}\n")
        result = await orchestrator.run(task)
        if result.error:
            print(f"Error: {result.error['message']}")
        elif result.status.value != "completed":
            print(f"Error: run status {result.status.value}")
        else:
            print(f"\n{result.output}")
        print(f"\nOrchestrator-only usage: {orchestrator.guardrails.usage_summary}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one task through an orchestrator and its specialists.")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK, help="The task (default: a built-in demo task).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args().task))
