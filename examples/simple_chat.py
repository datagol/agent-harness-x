"""Minimal streaming agent: chat + filesystem tools.

Demonstrates the streaming agentic loop — text appears token-by-token,
tool calls are shown as they happen.

Run: python -m examples.simple_chat
"""

import asyncio

from agent_harness import AgentConfig, PermissionLevel, StreamingAgent

from examples._console import console, get_user_input, handle_stream_event, print_banner, print_error, print_status
from examples.grocery_prompt import SYSTEM_PROMPT
from examples.grocery_state import grocery_state


async def main():
    agent = StreamingAgent(
        config=AgentConfig(
            system_prompt=SYSTEM_PROMPT,
            temperature=1.0,
        ),
    )
    _register_grocery_tools(agent, grocery_state)

    print_banner(
        "DataGOL Agent Harness — Simple Chat",
        subtitle="Streaming mode",
        commands={"quit": "Exit", "usage": "Token stats", "state": "Grocery list + memory"},
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
        if user_input.lower() == "state":
            print_status({"Grocery state": _describe_state()})
            continue

        try:
            console.print()
            async for event in agent.run_stream(user_input):
                handle_stream_event(event)
                if event.type.value == "tool_call_complete":
                    _handle_client_action(event.data)
        except Exception as e:
            print_error(e)

    print("\nGoodbye!")


def _describe_state() -> str:
    parts = []
    lst = grocery_state.get("list")
    parts.append(f"list: {lst if lst else '(empty)'}")
    mem = grocery_state.get("memory")
    if mem and any(mem.values()):
        parts.append(f"memory: { {k: v for k, v in sorted(mem.items()) if v} }")
    else:
        parts.append("memory: (empty)")
    return " | ".join(parts)


def _handle_client_action(tool_call) -> None:
    """Act on the client_action argument of respond_text / present_choice."""
    if tool_call.name not in ("respond_text", "present_choice"):
        return
    action = tool_call.input.get("client_action")
    if not action:
        return
    if action == "show_recently_added":
        status = grocery_state.get("staples_status")
        staples = grocery_state.get("staples")
        if status == "none":
            console.print("  [dim cyan]→ client: no usuals saved yet[/dim cyan]")
        else:
            items = ", ".join(staples) if staples else "(none saved yet)"
            console.print(f"  [dim cyan]→ client: usuals screen opened — {items}[/dim cyan]")
    elif action == "focus_text_input":
        console.print("  [dim cyan]→ client: focus text input[/dim cyan]")
    elif action == "open_photo_picker":
        console.print("  [dim cyan]→ client: photo picker opened[/dim cyan]")
    else:
        console.print(f"  [dim cyan]→ client action: {action}[/dim cyan]")


def _register_grocery_tools(agent: StreamingAgent, state: dict) -> None:
    """Same six functions as the web app; kept in one place."""
    from agent_harness.grocery.tools import register_grocery_tools

    register_grocery_tools(agent, state)


if __name__ == "__main__":
    asyncio.run(main())
