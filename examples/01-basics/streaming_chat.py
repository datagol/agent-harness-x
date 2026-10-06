"""Interactive chat that streams the answer and shows each tool call as it happens.

`agent.run_stream()` yields typed events while the run is in flight: text fragments to print as they arrive, and
tool calls before their results. The agent has a bounded calculator and read-only access to the current directory.

Run: python examples/01-basics/streaming_chat.py
Needs: ANTHROPIC_API_KEY
"""

import ast
import asyncio
import json
import math
import operator
from pathlib import Path

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, RunEventType
from harnessx.builtin.filesystem import register_filesystem_tools

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def calculate(expression: str) -> str:
    """Evaluate bounded arithmetic using numbers and +, -, *, /, //, %, **.

    Args:
        expression: Arithmetic expression, such as '(25 * 4) + 50'.
    """
    # Walks the parsed tree instead of calling eval(): model-written input never runs as Python.
    if len(expression) > 512:
        raise ValueError("Expression must be at most 512 characters")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Expression is too complex")

    def checked(value):
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 10**100:
            raise ValueError("Arithmetic result is outside the supported numeric range")
        return value

    def evaluate(node):
        if isinstance(node, ast.Constant):
            return checked(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return checked(value if isinstance(node.op, ast.UAdd) else -value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent magnitude must be at most 100")
            return checked(_OPERATORS[type(node.op)](left, right))
        raise ValueError("Only numeric literals and arithmetic operators are allowed")

    return str(evaluate(tree.body))


def show(event) -> None:
    if event.type == RunEventType.TEXT_DELTA:
        print(event.data, end="", flush=True)
    elif event.type == RunEventType.TEXT_COMPLETE:
        print()
    elif event.type == RunEventType.ATTEMPT_RESET:
        # A retried model call starts the answer again; the text above it is incomplete.
        print("\n[model retried: the text above is incomplete]")
    elif event.type == RunEventType.TOOL_CALL_START:
        print(f"  > {event.data.name}({json.dumps(event.data.input)[:120]})")
    elif event.type == RunEventType.TOOL_RESULT and event.data.is_error:
        print(f"  ! {event.data.content}")
    elif event.type == RunEventType.ERROR:
        print(f"Error: {event.data}")


async def main() -> None:
    agent = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are a helpful AI assistant with access to tools. Be concise and direct. Use tools when helpful."
            ),
        ),
        tools=[calculate],
    )

    async with agent:
        # Inspection only, confined to this workspace; symlinks below it are rejected.
        register_filesystem_tools(agent.tools, include=["read_file", "list_directory"], base_path=str(Path.cwd()))
        print("Streaming chat. Type 'usage' for token stats, 'quit' to exit.")

        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit"):
                break
            if not text:
                continue
            if text.lower() == "usage":
                print(f"Usage: {agent.guardrails.usage_summary}")
                continue
            try:
                async with agent.run_stream(text) as stream:
                    async for event in stream:
                        show(event)
            except Exception as exc:
                print(f"Error: {exc}")

        print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
