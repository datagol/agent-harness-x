"""Minimal streaming agent: interactive chat with streaming events.

Demonstrates the streaming agentic loop — text appears token-by-token,
and tool calls are shown as they happen.

Run: python -m examples.simple_chat
"""

import asyncio
from pathlib import Path

from harnessx import Agent, AgentConfig
from harnessx.builtin.filesystem import register_filesystem_tools

from examples._console import (
    console,
    get_user_input,
    handle_stream_event,
    print_banner,
    print_error,
    print_status,
)
from examples._calculator import calculate


async def main():
    agent = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are a helpful AI assistant with access to tools. "
                "Be concise and direct. Use tools when helpful."
            ),
        ),
        tools=[calculate],
    )

    async with agent:
        # Inspection only, confined to this workspace; symlinks below it are rejected.
        register_filesystem_tools(
            agent.tools,
            include=["read_file", "list_directory"],
            base_path=str(Path.cwd()),
        )

        print_banner(
            "HarnessX — Simple Chat",
            subtitle="Streaming mode",
            commands={"quit": "Exit", "usage": "Token stats"},
        )

        while True:
            user_input = get_user_input()
            if user_input is None or user_input.lower() == "quit":
                break
            if not user_input:
                continue
            if user_input.lower() == "usage":
                print_status({"Usage": agent.guardrails.usage_summary})
                continue

            try:
                console.print()
                async with agent.run_stream(user_input) as stream:
                    async for event in stream:
                        handle_stream_event(event)
            except Exception as e:
                print_error(e)

        print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
