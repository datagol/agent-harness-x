"""Give an agent the tools of an MCP (Model Context Protocol) server, next to its native tools.

`MCPManager` connects to a server (a local command over stdio, or a URL over HTTP/SSE) and discovers its
tools; passing it as `Agent(mcp=...)` registers them, so the model sees MCP and native tools the same way.
`--permission` sets the policy for the MCP tools; tools on ASK are approved in the terminal with `input()`.
Type `tools` in the chat to list both kinds.

Run:
    python examples/02-tools/mcp_servers.py --server filesystem --command npx \\
        --args "-y @modelcontextprotocol/server-filesystem /tmp"
    python examples/02-tools/mcp_servers.py --server myserver --command python --args "my_mcp_server.py"
    python examples/02-tools/mcp_servers.py --server remote --url http://localhost:8000/sse

Needs: ANTHROPIC_API_KEY and pip install "harnessx[mcp]" (plus whatever the server itself needs, e.g. npx)
"""

import argparse
import asyncio
import json
import shlex
import sys
from contextlib import AsyncExitStack

from dotenv import load_dotenv

from harnessx import (
    ToolResult,
    ToolCall,
    Agent,
    AgentConfig,
    CliPermissionManager,
    Limits,
    MCPManager,
    MCPServerConfig,
    RunEventType,
)
from harnessx.builtin.filesystem import register_filesystem_tools

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent with MCP server tools")
    parser.add_argument("--server", required=True, help="Name for the MCP server")
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--command", help="Command to launch a stdio MCP server")
    parser.add_argument("--args", help="Arguments for the command, shell-quoted (e.g. \"-y 'a dir'\")")
    transport.add_argument("--url", help="URL of an HTTP/SSE MCP server")
    parser.add_argument(
        "--permission", default="ask", choices=["allow", "ask", "deny"], help="Permission level for MCP tools"
    )
    return parser.parse_args(argv)


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


async def main(argv=None) -> None:
    args = parse_args(argv)

    async with AsyncExitStack() as resources:
        # The manager disconnects every server when the stack unwinds.
        mcp = await resources.enter_async_context(MCPManager())

        print(f"Connecting to MCP server '{args.server}'...")
        try:
            if args.command:
                server_args = shlex.split(args.args) if args.args else []
                server = MCPServerConfig.stdio(args.server, args.command, args=server_args, permission=args.permission)
            else:
                server = MCPServerConfig.http(args.server, args.url, permission=args.permission)
            tools = await mcp.connect(server)
        except Exception as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        print(f"Connected. Discovered {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.tool_name}: {tool.description[:80]}")

        agent = Agent(
            config=AgentConfig(
                system_prompt=(
                    "You are a helpful assistant with access to tools. "
                    "Use the available tools to help the user."
                ),
                limits=Limits(max_iterations=20),
            ),
            permissions=CliPermissionManager(),
            mcp=mcp,  # bridges the discovered tools into agent.tools
        )
        await resources.enter_async_context(agent)
        registered = agent.mcp_tools

        # Native tools sit beside the MCP ones in the same registry.
        register_filesystem_tools(agent.tools)

        print(f"Registered {len(registered)} MCP tools. Servers: {list(mcp.list_servers())}")
        print("Type tools to list every tool, quit to exit.")
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit"):
                break
            if not text:
                continue
            if text.lower() == "tools":
                print("Native tools:")
                for name in agent.tools.list_tools():
                    if name not in registered:
                        print(f"  - {name}")
                print(f"MCP tools ({args.server}):")
                for tool in mcp.list_tools():
                    print(f"  - {tool.tool_name}: {tool.description[:60]}")
                continue
            try:
                await stream_reply(agent, text)
            except Exception as exc:
                print(f"Error: {exc}")

        print(f"Session stats: {agent.guardrails.usage_summary}")


if __name__ == "__main__":
    asyncio.run(main())
