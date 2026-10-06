"""A coding agent with the built-in filesystem and bash tools, asking before it writes or runs anything.

The agent gets HarnessX's built-in file tools (confined to the current directory) and `run_bash`.
`CliPermissionManager` makes every tool ASK by default and asks you in the terminal with `input()`
("Allow? [y]es / [n]o / [a]lways"); reads are switched to ALLOW so only writes and commands stop for you.
An audit middleware logs every tool call and result, the place to hook your own policy or logging.

Run:
    python examples/02-tools/filesystem_tools_and_permissions.py

Needs: ANTHROPIC_API_KEY
"""

import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from harnessx import (
    ToolResult,
    ToolCall,
    Agent,
    AgentConfig,
    CliPermissionManager,
    Limits,
    Middleware,
    MiddlewarePipeline,
    PermissionLevel,
    RunEventType,
)
from harnessx.builtin.bash import register_bash_tools
from harnessx.builtin.filesystem import register_filesystem_tools

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

# Only this example's audit lines; library loggers stay at WARNING.
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("coding_agent")
logger.setLevel(logging.INFO)


class AuditMiddleware(Middleware):
    """Logs every tool call and result to the Python logger."""

    async def before_tool_execution(self, tool_call):
        logger.info(f"AUDIT: tool={tool_call.name} input={str(tool_call.input)[:200]}")
        return tool_call

    async def after_tool_execution(self, result):
        logger.info(f"AUDIT: result error={result.is_error} content={result.content[:100]}")
        return result


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


async def main() -> None:
    middleware = MiddlewarePipeline()
    middleware.add(AuditMiddleware())

    agent = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are an expert coding assistant. You can read files, "
                "write files, and run bash commands to help the user with "
                "their coding tasks. Always read files before editing them. "
                "Think step-by-step about the best approach."
            ),
            limits=Limits(max_iterations=30),
        ),
        middleware=middleware,
        # Unlisted tools default to ASK; the manager prompts with input() before each one runs.
        permissions=CliPermissionManager(),
    )

    async with agent:
        register_filesystem_tools(agent.tools, base_path=str(Path.cwd()))
        register_bash_tools(agent.tools)

        # Reading is harmless, so it runs without a prompt.
        agent.permissions.set_permission("read_file", PermissionLevel.ALLOW)
        agent.permissions.set_permission("list_directory", PermissionLevel.ALLOW)

        print(f"Coding agent in {Path.cwd()}. Reads run freely; writes and commands ask first. Type quit to exit.")
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit"):
                break
            if not text:
                continue
            try:
                await stream_reply(agent, text)
            except Exception as exc:
                print(f"Error: {exc}")

        print(f"Session stats: {agent.guardrails.usage_summary}")


if __name__ == "__main__":
    asyncio.run(main())
