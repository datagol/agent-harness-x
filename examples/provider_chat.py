"""Use the same calculator agent with any supported provider.

Run with the SDK default (Anthropic):
    python -m examples.provider_chat

Choose another provider and a model available to your account:
    python -m examples.provider_chat --provider openai --model YOUR_MODEL_ID

Add --prompt "What is 40 * 10?" for one turn, or --no-stream for ordinary runs.
See examples/README.md for each provider's SDK extra and API key.
"""

import argparse
import asyncio

from harnessx import Agent, AgentConfig, RunEventType

from examples._calculator import calculate
from examples._console import (
    completed_output,
    console,
    get_user_input,
    handle_stream_event,
    print_banner,
    print_error,
    print_response,
    print_status,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = AgentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("anthropic", "openai", "gemini", "openrouter"),
        default=defaults.provider,
        help="Model provider (default: anthropic).",
    )
    parser.add_argument(
        "--model",
        help=f"Model ID; required outside Anthropic (default: {defaults.model}).",
    )
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


async def respond(agent: Agent, prompt: str, *, streaming: bool) -> None:
    if streaming:
        async with agent.run_stream(prompt) as stream:
            async for event in stream:
                # completed_output below reports a failed run once.
                if event.type != RunEventType.ERROR:
                    handle_stream_event(event)
            completed_output(await stream.result())
    else:
        print_response(completed_output(await agent.run(prompt)))


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        agent = Agent(
            config=AgentConfig(
                provider=args.provider,
                model=args.model,
                system_prompt=(
                    "You are a helpful assistant. Be concise and use the "
                    "calculate tool for arithmetic."
                ),
            ),
            tools=[calculate],
        )
    except Exception as exc:
        print_error(exc)
        return 1

    async with agent:
        print_banner(
            "HarnessX — Provider Chat",
            subtitle=f"{args.provider} | {args.model} | {'Ordinary' if args.no_stream else 'Streaming'}",
            commands=None
            if args.prompt is not None
            else {"quit": "Exit", "usage": "Token stats"},
        )
        if args.prompt is not None:
            try:
                await respond(agent, args.prompt, streaming=not args.no_stream)
                return 0
            except Exception as exc:
                print_error(exc)
                return 1

        while True:
            prompt = get_user_input()
            if prompt is None or prompt.lower() == "quit":
                break
            if not prompt:
                continue
            if prompt.lower() == "usage":
                usage = agent.guardrails.total_usage
                print_status(
                    {
                        "Input tokens": usage.input_tokens,
                        "Output tokens": usage.output_tokens,
                    }
                )
                continue
            try:
                console.print()
                await respond(agent, prompt, streaming=not args.no_stream)
            except Exception as exc:
                print_error(exc)
        console.print("\nGoodbye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
