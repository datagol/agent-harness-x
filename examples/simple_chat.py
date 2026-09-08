"""Minimal streaming agent: interactive chat with streaming events.

Demonstrates the streaming agentic loop — text appears token-by-token,
and tool calls are shown as they happen.

Run: python -m examples.simple_chat
"""

import asyncio

from datagol_agent_harness import AgentConfig, PermissionLevel, StreamingAgent
from datagol_agent_harness.builtin.filesystem import register_filesystem_tools

from examples._console import console, get_user_input, handle_stream_event, print_banner, print_error, print_status


async def main():
    agent = StreamingAgent(
        config=AgentConfig(
            system_prompt=(
                "You are a helpful AI assistant with access to tools. "
                "Be concise and direct. Use tools when helpful."
            ),
            temperature=0.0,
        ),
    )

    # Register basic tools
    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression.

        Args:
            expression: Arithmetic expression (e.g., '144 / 12' or '(25 * 4) + 50').
        """
        try:
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Error evaluating expression: {e}"

    # Also register safe filesystem inspection tools
    register_filesystem_tools(agent.tools)
    agent.permissions.set_permission("list_directory", PermissionLevel.ALLOW)
    agent.permissions.set_permission("read_file", PermissionLevel.ALLOW)

    print_banner(
        "DataGOL Agent Harness — Simple Chat",
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
            async for event in agent.run_stream(user_input):
                handle_stream_event(event)
        except Exception as e:
            print_error(e)

    print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())

