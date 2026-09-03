"""Agent with MCP (Model Context Protocol) server tools.

Demonstrates connecting to MCP servers and using their tools transparently
alongside native tools. The agent sees all tools (native + MCP) the same way.

Prerequisites:
    pip install 'datagol-agent-harness[mcp]'

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

from datagol_agent_harness import (
    Agent,
    AgentConfig,
    MCPManager,
    PermissionLevel,
)
from examples._console import (
    console,
    create_hooks,
    get_user_input,
    print_banner,
    print_error,
    print_response,
    print_status,
)
from datagol_agent_harness.builtin.filesystem import register_filesystem_tools


async def main():
    parser = argparse.ArgumentParser(description="Agent with MCP server tools")
    parser.add_argument("--server", required=True, help="Name for the MCP server")
    parser.add_argument("--command", help="Command to launch stdio MCP server")
    parser.add_argument("--args", help="Arguments for the command (space-separated)")
    parser.add_argument("--url", help="URL for SSE MCP server")
    parser.add_argument("--permission", default="ask", choices=["allow", "ask", "deny"],
                        help="Permission level for MCP tools")
    args = parser.parse_args()

    if not args.command and not args.url:
        parser.error("Must specify either --command (stdio) or --url (sse)")

    hooks = create_hooks()

    # Create MCP manager and connect
    mcp = MCPManager()

    console.print(f"Connecting to MCP server [bold cyan]'{args.server}'[/bold cyan]...")
    try:
        server_args = args.args.split() if args.args else []
        tools = await mcp.connect(
            args.server,
            command=args.command,
            args=server_args,
            url=args.url,
            permission=PermissionLevel(args.permission),
        )
        console.print(f"  [info]Connected! Discovered {len(tools)} tools:[/info]")
        for t in tools:
            console.print(f"    [bold cyan]-[/bold cyan] {t.tool_name}: [dim]{t.description[:80]}[/dim]")
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
            max_iterations=20,
        ),
        hooks=hooks,
        mcp=mcp,
    )

    # Register MCP tools into agent's tool registry
    registered = mcp.register_tools(agent.tools)
    console.print(f"  [info]Registered {len(registered)} MCP tools into agent[/info]")

    # Also register native filesystem tools
    register_filesystem_tools(agent.tools)

    print_banner(
        "Agent ready with MCP tools",
        subtitle=f"Servers: {mcp.list_servers()}",
        commands={"quit": "Exit", "tools": "List all tools"},
    )

    try:
        while True:
            user_input = get_user_input()
            if user_input is None or user_input.lower() == "quit":
                break
            if not user_input:
                continue
            if user_input.lower() == "tools":
                console.print("  [bold]Native tools:[/bold]")
                for name in agent.tools.list_tools():
                    if not name.startswith(args.server + "__"):
                        console.print(f"    [cyan]-[/cyan] {name}")
                console.print(f"  [bold]MCP tools ({args.server}):[/bold]")
                for t in mcp.list_tools():
                    console.print(f"    [cyan]-[/cyan] {t.tool_name}: [dim]{t.description[:60]}[/dim]")
                continue

            try:
                response = await agent.run(user_input)
                print_response(response)
            except Exception as e:
                print_error(e)

    finally:
        console.print("\n[dim]Disconnecting MCP servers...[/dim]")
        await mcp.disconnect_all()
        print_status({"Session stats": agent.guardrails.usage_summary})


if __name__ == "__main__":
    asyncio.run(main())
