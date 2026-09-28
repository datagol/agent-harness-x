"""Built-in bash tool: execute shell commands."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from harnessx.types import PermissionLevel
from ._registration import select_tools

if TYPE_CHECKING:
    from harnessx.tools import ToolRegistry


async def run_bash(command: str, timeout: int = 30) -> str:
    """Execute a bash command and return its output.

    Args:
        command: The shell command to execute.
        timeout: Maximum seconds to wait (default 30).
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            output += f"\nSTDERR:\n{stderr.decode('utf-8', errors='replace')}"
        if proc.returncode and proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"
        return output.strip() or "(no output)"

    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"Command timed out after {timeout}s"


def register_bash_tools(
    registry: ToolRegistry,
    sandbox: Any | None = None,
    *,
    permission: PermissionLevel | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    replace: bool = False,
) -> list[str]:
    """Register bash execution tools.

    If a sandbox is provided, commands run inside it. Otherwise, direct subprocess.

    Args:
        registry: Target tool registry.
        sandbox: Optional sandbox execution engine.
        permission: Explicit policy; None inherits the manager default (normally ALLOW).
        include: Specific tool names to register ('run_bash').
        exclude: Tool names to omit.

    Returns:
        List of registered tool names.
    """
    if not select_tools(registry, ["run_bash"], include, exclude, replace=replace):
        return []
    if sandbox is None:
        sandbox = getattr(registry, "sandbox", None)

    if sandbox is not None:
        async def sandboxed_run_bash(command: str, timeout: int = 30) -> str:
            """Execute a bash command in sandbox and return its output.

            Args:
                command: The shell command to execute.
                timeout: Maximum seconds to wait (default 30).
            """
            result = await sandbox.execute_command(command)
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.timed_out:
                output += f"\n[timed out after {timeout}s]"
            if result.exit_code != 0:
                output += f"\n[exit code: {result.exit_code}]"
            return output.strip() or "(no output)"

        registry.register_tool(sandboxed_run_bash, name="run_bash", permission=permission, replace=replace)
    else:
        registry.register_tool(run_bash, name="run_bash", permission=permission, replace=replace)

    return ["run_bash"]
