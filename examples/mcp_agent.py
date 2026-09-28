"""Agent with MCP (Model Context Protocol) server tools.

Demonstrates connecting to MCP servers and using their tools transparently
alongside native tools. The agent sees all tools (native + MCP) the same way.

Prerequisites:
    pip install 'harnessx[mcp]'

Example MCP servers to try:
    # Filesystem server (npx)
    python -m examples.mcp_agent --server filesystem --command npx --args "-y @modelcontextprotocol/server-filesystem /tmp"

    # Python-based MCP server
    python -m examples.mcp_agent --server myserver --command python --args "my_mcp_server.py"

    # Remote SSE server
    python -m examples.mcp_agent --server remote --url http://localhost:8000/sse

Run: python -m examples.mcp_agent --help
"""

import argparse
import asyncio
import sys
import shlex
from contextlib import AsyncExitStack

from harnessx import (
    Agent,
    AgentConfig,
    CliPermissionManager,
    Limits,
    MCPManager,
    MCPServerConfig,
)
from examples._console import (
    console,
    create_hooks,
    get_user_input,
    print_banner,
    print_error,
    print_response,
    print_status,
    completed_output,
)
from harnessx.builtin.filesystem import register_filesystem_tools


async def main():
    parser = argparse.ArgumentParser(description="Agent with MCP server tools")
    parser.add_argument("--server", required=True, help="Name for the MCP server")
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--command", help="Command to launch stdio MCP server")
    parser.add_argument("--args", help="Arguments for the command (space-separated)")
    transport.add_argument("--url", help="URL for SSE MCP server")
    parser.add_argument(
        "--permission",
        default="ask",
        choices=["allow", "ask", "deny"],
        help="Permission level for MCP tools",
    )
    args = parser.parse_args()

    hooks = create_hooks()

    async with AsyncExitStack() as resources:
        # The manager disconnects every server when the stack unwinds.
        mcp = await resources.enter_async_context(MCPManager())

        console.print(
            f"Connecting to MCP server [bold cyan]'{args.server}'[/bold cyan]..."
        )
        try:
            server_args = shlex.split(args.args) if args.args else []
            server = (
                MCPServerConfig.stdio(
                    args.server, args.command, args=server_args, permission=args.permission
                )
                if args.command
                else MCPServerConfig.http(args.server, args.url, permission=args.permission)
            )
            tools = await mcp.connect(server)
            console.print(f"  [info]Connected! Discovered {len(tools)} tools:[/info]")
            for t in tools:
                console.print(
                    f"    [bold cyan]-[/bold cyan] {t.tool_name}: [dim]{t.description[:80]}[/dim]"
                )
        except Exception as e:
            print_error(e)
            sys.exit(1)

        # Create agent with MCP tools
        agent = Agent(
            config=AgentConfig(
                system_prompt=(
                    "You are a helpful assistant with access to tools. "
                    "Use the available tools to help the user."
                ),
                limits=Limits(max_iterations=20),
            ),
            hooks=hooks,
            permissions=CliPermissionManager(),
            mcp=mcp,  # bridges the discovered tools into agent.tools
        )

        await resources.enter_async_context(agent)

        registered = agent.mcp_tools
        console.print(
            f"  [info]Registered {len(registered)} MCP tools into agent[/info]"
        )

        # Also register native filesystem tools
        register_filesystem_tools(agent.tools)

        print_banner(
            "Agent ready with MCP tools",
            subtitle=f"Servers: {mcp.list_servers()}",
            commands={"quit": "Exit", "tools": "List all tools"},
        )

        while True:
            user_input = get_user_input()
            if user_input is None or user_input.lower() == "quit":
                break
            if not user_input:
                continue
            if user_input.lower() == "tools":
                console.print("  [bold]Native tools:[/bold]")
                for name in agent.tools.list_tools():
                    if name not in registered:
                        console.print(f"    [cyan]-[/cyan] {name}")
                console.print(f"  [bold]MCP tools ({args.server}):[/bold]")
                for t in mcp.list_tools():
                    console.print(
                        f"    [cyan]-[/cyan] {t.tool_name}: [dim]{t.description[:60]}[/dim]"
                    )
                continue

            try:
                response = completed_output(await agent.run(user_input))
                print_response(response)
            except Exception as e:
                print_error(e)

        print_status({"Session stats": agent.guardrails.usage_summary})


if __name__ == "__main__":
    asyncio.run(main())
