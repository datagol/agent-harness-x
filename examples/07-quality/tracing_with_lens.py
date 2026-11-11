"""Trace an agent in DataGOL Lens: one trace per turn, with a span for each model call and tool call.

Adding ``LensExtension`` to an Agent is the whole integration. Each turn becomes one Lens trace named
``run_name``. The trace holds an llm span for each model call, with its tokens and cost, and a tool span
for each tool call, with its arguments and result. The session id groups the turns of one conversation
in Lens, and tags and metadata let you filter for them.

Run:   python examples/07-quality/tracing_with_lens.py
Needs: ANTHROPIC_API_KEY, DATAGOL_LENS_ENABLED=true, DATAGOL_LENS_API_KEY and DATAGOL_LENS_PROJECT_ID
(optionally DATAGOL_LENS_API_URL, default http://localhost:8000).
"""

import ast
import asyncio
import json
import operator
import uuid

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, LensExtension, Limits, PermissionLevel, RunEventType, ToolCall, ToolResult

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
    tracing = LensExtension(
        run_name="Main Assistant",
        session_id=f"demo-{uuid.uuid4().hex[:8]}",
        tags=["demo", "cli"],
        metadata={"environment": "development"},
    )
    if not tracing.configured:
        print("Lens tracing is off: set DATAGOL_LENS_ENABLED=true, DATAGOL_LENS_API_KEY and DATAGOL_LENS_PROJECT_ID.\n")
    agent = Agent(
        config=AgentConfig(
            model="claude-sonnet-4-6",
            system_prompt="You are a helpful assistant. Use the calculate tool for arithmetic.",
            limits=Limits(max_iterations=5),
        ),
        extensions=[tracing],
    )
    agent.tools.register_tool(calculate, permission=PermissionLevel.ALLOW, replay_policy="safe")

    # Leaving the context closes the extension, which waits for the trace to be sent.
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
    if tracing.configured:
        print(f"\nThe trace is in Lens project {tracing.project_id!r}, session {tracing.session_id!r}.")


if __name__ == "__main__":
    asyncio.run(main())
