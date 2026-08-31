"""Streaming coding agent with logging middleware.

Demonstrates streaming output, custom middleware for auditing,
and permission overrides for read-only operations.

Run: python -m examples.coding_agent
"""

import asyncio
import logging

from agent_harness import (
    AgentConfig,
    Middleware,
    MiddlewarePipeline,
    PermissionLevel,
    StreamingAgent,
)
from examples._console import console, get_user_input, handle_stream_event, print_banner, print_error, print_status
from agent_harness.builtin.bash import register_bash_tools
from agent_harness.builtin.filesystem import register_filesystem_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("coding_agent")


class AuditMiddleware(Middleware):
    """Logs every tool call and result to the Python logger."""

    async def before_tool_execution(self, tool_call):
        logger.info(f"AUDIT: tool={tool_call.name} input={str(tool_call.input)[:200]}")
        return tool_call

    async def after_tool_execution(self, result):
        logger.info(f"AUDIT: result error={result.is_error} content={result.content[:100]}")
        return result


async def main():
    # Set up middleware
    middleware = MiddlewarePipeline()
    middleware.add(AuditMiddleware())

    # Create streaming agent
    agent = StreamingAgent(
        config=AgentConfig(
            system_prompt=(
                "You are an expert coding assistant. You can read files, "
                "write files, and run bash commands to help the user with "
                "their coding tasks. Always read files before editing them. "
                "Think step-by-step about the best approach."
            ),
            max_iterations=30,
        ),
        middleware=middleware,
    )

    register_filesystem_tools(agent.tools)
    register_bash_tools(agent.tools)

    # Auto-allow read operations
    agent.permissions.set_permission("read_file", PermissionLevel.ALLOW)
    agent.permissions.set_permission("list_directory", PermissionLevel.ALLOW)

    print_banner(
        "DataGOL Agent Harness — Coding Agent",
        subtitle="Streaming mode | Read operations auto-allowed, writes require permission",
        commands={"quit": "Exit"},
    )

    while True:
        user_input = get_user_input()
        if user_input is None or user_input.lower() == "quit":
            break
        if not user_input:
            continue

        try:
            console.print()
            async for event in agent.run_stream(user_input):
                handle_stream_event(event)
        except Exception as e:
            print_error(e)

    print_status({"Session stats": agent.guardrails.usage_summary})


if __name__ == "__main__":
    asyncio.run(main())
