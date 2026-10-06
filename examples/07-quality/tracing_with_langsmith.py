"""Trace an agent in LangSmith: one run tree per turn, with a child span for each model call and tool call.

Adding ``LangSmithExtension`` to an Agent is the whole integration. Each turn becomes a LangSmith run named
``run_name``; every model call nests under it with its token usage, and every tool call with its arguments
and result. Tags and metadata are attached to the run so you can filter for them in the LangSmith UI.

Run:   python examples/07-quality/tracing_with_langsmith.py
Needs: pip install "harnessx[langsmith]", ANTHROPIC_API_KEY and LANGSMITH_API_KEY
(optionally LANGSMITH_PROJECT, default "harnessx-demo").
"""

import ast
import asyncio
import json
import operator
import os

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, LangSmithExtension, Limits, PermissionLevel, RunEventType, ToolCall, ToolResult

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables


OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculate(expression: str) -> str:
    """Evaluate arithmetic with numbers, parentheses and + - * /.

    Args:
        expression: Arithmetic expression, such as '(25 * 4) + 50'.
    """
    # Walk the syntax tree instead of eval(), so the model can never run Python.
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("Only numbers, parentheses and + - * / are allowed")

    if len(expression) > 200:
        raise ValueError("Expression must be at most 200 characters")
    return str(evaluate(ast.parse(expression, mode="eval").body))


async def main() -> None:
    project = os.getenv("LANGSMITH_PROJECT", "harnessx-demo")
    tracing = LangSmithExtension(
        project_name=project,
        run_name="Main Assistant",
        tags=["demo", "cli"],
        metadata={"environment": "development"},
    )
    agent = Agent(
        config=AgentConfig(
            model="claude-sonnet-4-6",
            system_prompt="You are a helpful assistant. Use the calculate tool for arithmetic.",
            limits=Limits(max_iterations=5),
        ),
        extensions=[tracing],
    )
    agent.tools.register_tool(calculate, permission=PermissionLevel.ALLOW, replay_policy="safe")

    # Leaving the context closes the extension, which flushes pending traces before exit.
    async with agent:
        prompt = "What is 144 / 12, and what is that times 7?"
        print(f"You: {prompt}\n")
        async with agent.run_stream(prompt) as stream:
            async for event in stream:
                if event.type == RunEventType.TEXT_DELTA:
                    print(event.data, end="", flush=True)
                elif event.type == RunEventType.TOOL_CALL_START and isinstance(event.data, ToolCall):
                    print(f"  > {event.data.name}({json.dumps(event.data.input)[:120]})")
                elif event.type == RunEventType.TOOL_RESULT and isinstance(event.data, ToolResult) and event.data.is_error:
                    print(f"  ! {event.data.content}")
            result = await stream.result()
        print()
        if result.error:
            print("Error:", result.error["message"])
        print(f"\nTraces are in LangSmith project {project!r}: https://smith.langchain.com")


if __name__ == "__main__":
    asyncio.run(main())
