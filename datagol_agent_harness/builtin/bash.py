"""Built-in bash tool: execute shell commands."""

from __future__ import annotations

import asyncio

from datagol_agent_harness.tools import ToolRegistry
from datagol_agent_harness.types import PermissionLevel


def register_bash_tools(registry: ToolRegistry, sandbox=None) -> None:
    """Register bash execution tools.

    If a sandbox is provided, commands run inside it. Otherwise, direct subprocess.
    """

    @registry.register(permission=PermissionLevel.ASK)
    async def run_bash(command: str, timeout: int = 30) -> str:
        """Execute a bash command and return its output.

        Args:
            command: The shell command to execute.
            timeout: Maximum seconds to wait (default 30).
        """
        if sandbox is not None:
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

        # Direct subprocess execution
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
            proc.kill()
            return f"Command timed out after {timeout}s"
