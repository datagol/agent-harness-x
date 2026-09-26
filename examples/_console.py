"""Shared rich console output for all examples."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from harnessx import (
    HookContext,
    HookManager,
    RunEvent,
    RunEventType,
    RunResult,
    ToolCall,
    ToolResult,
)

theme = Theme(
    {
        "tool.name": "bold cyan",
        "tool.input": "dim",
        "tool.error": "bold red",
        "agent.label": "bold magenta",
        "user.label": "bold green",
        "info": "dim cyan",
        "warning": "bold yellow",
    }
)

console = Console(theme=theme)


def print_banner(
    title: str, subtitle: str = "", commands: dict[str, str] | None = None
) -> None:
    """Print a styled startup banner."""
    lines = [f"[bold]{title}[/bold]"]
    if subtitle:
        lines.append(f"[dim]{subtitle}[/dim]")
    if commands:
        lines.append("")
        for cmd, desc in commands.items():
            lines.append(f"  [bold cyan]{cmd}[/bold cyan]  {desc}")
    content = "\n".join(lines)
    console.print(Panel(content, border_style="bright_blue", padding=(1, 2)))


def print_tool_call(name: str, input_data: Any) -> None:
    """Print a formatted tool call."""
    text = Text()
    text.append("  > ", style="dim")
    text.append(name, style="tool.name")
    if input_data:
        try:
            if isinstance(input_data, str):
                args_str = input_data
            else:
                args_str = json.dumps(input_data, indent=None)
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
        except (TypeError, ValueError):
            args_str = str(input_data)[:200]
        text.append(f"({args_str})", style="tool.input")
    console.print(text)


def print_tool_result(result: Any, *, is_error: bool = False) -> None:
    """Print a formatted tool result."""
    if is_error:
        content = getattr(result, "content", str(result))
        console.print(f"  [tool.error]< ERROR:[/tool.error] {content[:200]}")


def print_response(text: str) -> None:
    """Print the agent's response as rendered markdown in a panel."""
    md = Markdown(text)
    console.print(
        Panel(
            md,
            title="[agent.label]Assistant[/agent.label]",
            border_style="magenta",
            padding=(0, 1),
        )
    )


def print_error(error: Exception | str) -> None:
    """Print an error message."""
    console.print(
        Panel(str(error), title="[bold red]Error[/bold red]", border_style="red")
    )


def print_status(data: dict[str, Any]) -> None:
    """Print status key-value pairs."""
    for key, value in data.items():
        console.print(f"  [info]{key}:[/info] {value}")


def completed_output(result: RunResult) -> str:
    """Surface failed or pending runs instead of displaying an empty answer."""
    if result.status != "completed":
        message = (
            result.error["message"]
            if result.error
            else f"Run status: {result.status.value}"
        )
        raise RuntimeError(message)
    return result.output


def print_delegation(from_agent: str, to_agent: str) -> None:
    """Print agent delegation arrow."""
    console.print(
        f"  [warning]{from_agent}[/warning] [dim]->[/dim] [warning]{to_agent}[/warning]"
    )


def get_user_input(prompt: str = "You") -> str | None:
    """Prompt the user for input with styling. Returns None on EOF/interrupt."""
    try:
        console.print()
        return console.input(f"[user.label]{prompt}:[/user.label] ").strip()
    except (EOFError, KeyboardInterrupt):
        return None


def handle_stream_event(event: RunEvent) -> None:
    """Handle a single streaming event with rich output."""
    if event.type == RunEventType.TEXT_DELTA:
        console.print(event.data, end="", markup=False)
    elif event.type == RunEventType.TEXT_COMPLETE:
        console.print()  # newline after streamed text
    elif event.type == RunEventType.ATTEMPT_RESET:
        console.print(
            "\n[warning]Model retry: preceding partial text is incomplete.[/warning]"
        )
    elif event.type == RunEventType.TOOL_CALL_START and isinstance(
        event.data, ToolCall
    ):
        print_tool_call(event.data.name, event.data.input)
    elif event.type == RunEventType.TOOL_RESULT and isinstance(event.data, ToolResult):
        if event.data.is_error:
            print_tool_result(event.data, is_error=True)
    elif event.type == RunEventType.ERROR:
        print_error(str(event.data))


def create_hooks() -> HookManager:
    """Create a HookManager with rich-formatted output for tool calls and results."""
    hooks = HookManager()

    @hooks.before_tool
    async def _show_tool_call(ctx: HookContext):
        tc = ctx.data.get("tool_call")
        if tc:
            print_tool_call(tc.name, tc.input)

    @hooks.after_tool
    async def _show_tool_result(ctx: HookContext):
        result = ctx.data.get("result")
        if result and result.is_error:
            print_tool_result(result, is_error=True)

    return hooks
