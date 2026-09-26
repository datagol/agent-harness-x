"""Constructor-defined specialists with isolated conversations per task.

Run: python -m examples.multi_agent
"""

import asyncio
from pathlib import Path

from harnessx import Agent, AgentConfig, CliPermissionManager, PermissionLevel, SubAgent, ToolRegistry
from harnessx.builtin.filesystem import register_filesystem_tools
from harnessx.builtin.web import fetch_url
from examples._console import (
    completed_output,
    get_user_input,
    print_banner,
    print_error,
    print_response,
    print_status,
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
                max_iterations=5,
            ),
            permission=PermissionLevel.ALLOW,
        ),
        SubAgent(
            name="research",
            description="Read public URLs supplied in a research task and compare their evidence.",
            config=AgentConfig(
                system_prompt="Read the supplied URLs, answer the research question, and cite your sources.",
                max_iterations=10,
            ),
            tools=web,
            permission=PermissionLevel.ASK,
        ),
        SubAgent(
            name="file_analysis",
            description="Read files in the workspace and answer questions about their contents.",
            config=AgentConfig(
                system_prompt="Read and analyze the requested workspace files. Explain your findings.",
                max_iterations=10,
            ),
            tools=files,
            permission=PermissionLevel.ALLOW,
        ),
    ]


async def main():
    async with Agent(
        permissions=CliPermissionManager(),
        config=AgentConfig(
            system_prompt=(
                "Break complex tasks into self-contained assignments for the appropriate "
                "specialists. Supply the code, URLs, or paths they need, then synthesize "
                "their results into a coherent response. You also have filesystem tools."
            ),
            max_iterations=20,
        ),
        subagents=specialist_definitions(str(Path.cwd())),
    ) as orchestrator:
        register_filesystem_tools(orchestrator.tools)
        for name in ("read_file", "list_directory"):
            orchestrator.permissions.set_permission(name, PermissionLevel.ALLOW)
        print_banner(
            "HarnessX — Multi-Agent Orchestrator",
            subtitle="Specialists: code review, research, file analysis",
            commands={"quit": "Exit"},
        )
        while True:
            user_input = get_user_input()
            if user_input is None or user_input.lower() == "quit":
                break
            if not user_input:
                continue
            try:
                print_response(completed_output(await orchestrator.run(user_input)))
            except Exception as e:
                print_error(e)
        print_status({"Orchestrator-only usage": orchestrator.guardrails.usage_summary})


if __name__ == "__main__":
    asyncio.run(main())
