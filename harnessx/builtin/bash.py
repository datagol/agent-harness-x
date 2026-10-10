"""Built-in bash tool: execute shell commands."""

from __future__ import annotations

import asyncio
import functools
import os
import signal
from typing import TYPE_CHECKING, Any

from harnessx.types import DEFAULT_TIMEOUT_SECONDS, PermissionLevel
from ._registration import select_tools
from .filesystem import _mark_builtin

if TYPE_CHECKING:
    from harnessx.tools import ToolRegistry


async def run_bash(command: str, timeout: int = 30) -> str:
    """Execute a bash command and return its output.

    Args:
        command: The shell command to execute.
        timeout: Maximum seconds to wait (default 30).
    """
    return await _run_bash(command, timeout)


async def _run_bash(command: str, timeout: int = 30, cwd: str | None = None) -> str:
    proc = None
    spawn = asyncio.create_task(asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            start_new_session=True,
        ))
    try:
        proc = await asyncio.shield(spawn)
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
        await _kill_command(proc)
        return f"Command timed out after {timeout}s"
    except asyncio.CancelledError:
        # The harness's own deadline for the tool fired first (a model can ask
        # for a longer timeout than the tool is allowed); do not leave the
        # process running behind the cancelled call.
        if proc is None:
            # Cancellation may arrive while the subprocess transport is opening.
            try:
                proc = await spawn
            except Exception:
                pass
        await _kill_command(proc)
        raise


async def _kill_command(proc) -> None:
    if proc is None:
        return
    # The shell may already have exited while a child still holds its pipes.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await proc.communicate()


def register_bash_tools(
    registry: ToolRegistry,
    sandbox: Any | None = None,
    *,
    permission: PermissionLevel | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    replace: bool = False,
    cwd: str | None = None,
) -> list[str]:
    """Register bash execution tools.

    If a sandbox is provided, commands run inside it. Otherwise, direct subprocess.

    Args:
        registry: Target tool registry.
        sandbox: Optional sandbox execution engine.
        permission: Explicit policy; None inherits the manager default (normally ALLOW).
        include: Specific tool names to register ('run_bash').
        exclude: Tool names to omit.
        cwd: Directory commands start in, for a direct subprocess. None is the
            process's own working directory. An agent serving several users from
            one process gives each its own; a sandbox has its own and ignores it.
            Commands can still ``cd`` out: this is a starting point, not a jail.

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
            result = await sandbox.execute_command(command, timeout=timeout)
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.timed_out:
                output += f"\n[timed out after {timeout}s]"
            if result.exit_code != 0:
                output += f"\n[exit code: {result.exit_code}]"
            denials = getattr(result, "denials", None)
            if denials:
                output += (f"\n[blocked by the sandbox's network policy: {', '.join(denials)}. "
                           "This is not a network fault; ask the user to allow it.]")
            return output.strip() or "(no output)"

        _mark_builtin(sandboxed_run_bash, sandbox, functools.partial(
            register_bash_tools, include=["run_bash"], permission=permission, replace=True))
        registry.register_tool(sandboxed_run_bash, name="run_bash", permission=permission, replace=replace,
                               timeout_seconds=DEFAULT_TIMEOUT_SECONDS)
    elif cwd is not None:
        async def scoped_run_bash(command: str, timeout: int = 30) -> str:
            """Execute a bash command in the working directory and return its output.

            Args:
                command: The shell command to execute.
                timeout: Maximum seconds to wait (default 30).
            """
            return await _run_bash(command, timeout, cwd)

        _mark_builtin(scoped_run_bash, None, functools.partial(
            register_bash_tools, include=["run_bash"], permission=permission, replace=True))
        registry.register_tool(scoped_run_bash, name="run_bash", permission=permission, replace=replace,
                               timeout_seconds=DEFAULT_TIMEOUT_SECONDS)
    else:
        # The public function is shared, so the mark goes on a per-registration wrapper.
        async def host_run_bash(command: str, timeout: int = 30) -> str:
            return await run_bash(command, timeout)

        host_run_bash.__doc__ = run_bash.__doc__
        _mark_builtin(host_run_bash, None, functools.partial(
            register_bash_tools, include=["run_bash"], permission=permission, replace=True))
        registry.register_tool(host_run_bash, name="run_bash", permission=permission, replace=replace,
                               timeout_seconds=DEFAULT_TIMEOUT_SECONDS)

    return ["run_bash"]
