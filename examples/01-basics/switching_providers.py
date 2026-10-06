"""Run the same agent and tool on any supported provider by changing two config fields.

`AgentConfig(provider=..., model=...)` picks the vendor; the tools, the loop, streaming, and the result shape stay
the same. Use `--prompt` for one turn, or chat interactively (`usage` prints token stats, `quit` exits).

Run:
    python examples/01-basics/switching_providers.py
    python examples/01-basics/switching_providers.py --provider openai --model YOUR_MODEL_ID
    python examples/01-basics/switching_providers.py --prompt "What is 40 * 10?" --no-stream
Needs: the chosen provider's API key (ANTHROPIC_API_KEY by default) and its SDK extra, e.g.
    pip install "harnessx[openai]"; see examples/README.md for each provider.
"""

import argparse
import asyncio
import json
import operator
import re

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, RunEventType, ToolCall, ToolResult

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

_NUMBER = r"\s*(-?\d+(?:\.\d+)?)\s*"
_EXPRESSION = re.compile(f"^{_NUMBER}([-+*/]){_NUMBER}$")
_OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def calculate(expression: str) -> str:
    """Apply one of +, -, *, / to two numbers.

    Args:
        expression: Two numbers and one operator, such as '40 * 10'.
    """
    # Parsed with a fixed pattern, never eval(): the model cannot reach Python through this tool.
    match = _EXPRESSION.match(expression)
    if not match:
        raise ValueError("Use exactly two numbers and one of + - * /, such as '40 * 10'")
    left, symbol, right = match.groups()
    a = float(left) if "." in left else int(left)
    b = float(right) if "." in right else int(right)
    if symbol == "/" and b == 0:
        raise ValueError("Cannot divide by zero")
    return str(_OPERATORS[symbol](a, b))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = AgentConfig()
    parser = argparse.ArgumentParser(description="Chat with the same agent on any supported provider.")
    parser.add_argument(
        "--provider",
        choices=("anthropic", "openai", "gemini", "openrouter", "azure"),
        default=defaults.provider,
        help="Model provider (default: anthropic).",
    )
    parser.add_argument("--model", help=f"Model ID; required outside Anthropic (default: {defaults.model}).")
    parser.add_argument("--prompt", help="Run one prompt and exit instead of chatting.")
    parser.add_argument("--no-stream", action="store_true", help="Use Agent.run().")
    args = parser.parse_args(argv)
    if args.model is None:
        if args.provider != defaults.provider:
            parser.error("--model is required when selecting a non-Anthropic provider")
        args.model = defaults.model
    args.model = args.model.strip()
    if not args.model:
        parser.error("--model must not be empty")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must not be empty")
    return args


def output_of(result) -> str:
    """The answer of a completed run; a failed or paused run raises instead of printing nothing."""
    if result.status != "completed":
        raise RuntimeError(result.error["message"] if result.error else f"Run status: {result.status.value}")
    return result.output


async def respond(agent: Agent, prompt: str, *, streaming: bool) -> None:
    if not streaming:
        print(output_of(await agent.run(prompt)))
        return
    async with agent.run_stream(prompt) as stream:
        async for event in stream:
            if event.type == RunEventType.TEXT_DELTA:
                print(event.data, end="", flush=True)
            elif event.type == RunEventType.TEXT_COMPLETE:
                print()
            elif event.type == RunEventType.ATTEMPT_RESET:
                print("\n[model retried: the text above is incomplete]")
            elif event.type == RunEventType.TOOL_CALL_START and isinstance(event.data, ToolCall):
                print(f"  > {event.data.name}({json.dumps(event.data.input)[:120]})")
            elif event.type == RunEventType.TOOL_RESULT and isinstance(event.data, ToolResult) and event.data.is_error:
                print(f"  ! {event.data.content}")
            # ERROR events are skipped: output_of reports a failed run once, from the result.
        output_of(await stream.result())


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    streaming = not args.no_stream
    try:
        agent = Agent(
            config=AgentConfig(
                provider=args.provider,
                model=args.model,
                system_prompt="You are a helpful assistant. Be concise and use the calculate tool for arithmetic.",
            ),
            tools=[calculate],
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    async with agent:
        print(f"{args.provider} | {args.model} | {'streaming' if streaming else 'ordinary run'}")
        if args.prompt is not None:
            try:
                await respond(agent, args.prompt, streaming=streaming)
                return 0
            except Exception as exc:
                print(f"Error: {exc}")
                return 1

        print("Type 'usage' for token stats, 'quit' to exit.")
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
                await respond(agent, text, streaming=streaming)
            except Exception as exc:
                print(f"Error: {exc}")
        print("Goodbye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
