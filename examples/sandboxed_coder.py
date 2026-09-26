"""Coding agent with a temporary workdir and process resource limits.

The process tier retains host filesystem/network access. Use Docker or Seatbelt
when access isolation is required. Demonstrates execution with resource limits,
and the AgentRuntime for lifecycle management with checkpointing.

Run: python -m examples.sandboxed_coder
"""

import asyncio
from pathlib import Path

from harnessx import (
    AgentConfig,
    AgentRuntime,
    PermissionLevel,
    RuntimeConfig,
    Sandbox,
    SandboxConfig,
)
from examples._console import (
    console,
    create_hooks,
    get_user_input,
    print_banner,
    print_error,
    print_response,
    print_status,
    completed_output,
)
from harnessx.builtin.filesystem import register_filesystem_tools


async def main():
    # Configure sandbox
    sandbox_config = SandboxConfig(
        timeout_seconds=15,
        max_memory_mb=256,
        max_cpu_seconds=10,
        max_file_size_mb=50,
        tier="process",  # Use "docker" or "seatbelt" for stronger isolation
    )

    hooks = create_hooks()

    # Create sandbox
    async with Sandbox(sandbox_config) as sandbox:
        console.print(f"  [info]Sandbox workdir:[/info] {sandbox.workdir}")

        # Create runtime
        runtime = AgentRuntime(
            agent_config=AgentConfig(
                system_prompt=(
                    "You are a coding assistant that can write and execute Python code "
                    "in a temporary working directory. You have these tools:\n\n"
                    "- write_code: Write a Python file to the sandbox\n"
                    "- run_code: Execute a Python file with process resource limits\n"
                    "- run_python: Execute Python code directly in the sandbox\n"
                    "- read_sandbox_file: Read a file from the sandbox\n"
                    "- list_sandbox_files: List files in the sandbox\n\n"
                    "Limits: 15s timeout and 10s CPU; the 256MB address-space limit is platform-dependent.\n"
                    "If code fails, read the error and iterate. Write clean, tested code."
                ),
                max_iterations=25,
            ),
            runtime_config=RuntimeConfig(
                storage_dir=".sandbox_sessions",
            ),
            hooks=hooks,
        )

        async with runtime:
            # Start runtime
            session_id = runtime.session_id
            agent = runtime.agent
            console.print(f"  [info]Session:[/info] {session_id}")

            # Register sandbox tools
            @agent.tools.register(permission=PermissionLevel.ALLOW)
            async def write_code(filename: str, code: str) -> str:
                """Write a Python file to the sandbox.

                Args:
                    filename: Name of the file (e.g., 'solution.py').
                    code: Python code to write.
                """
                path = await sandbox.write_file(filename, code)
                return f"Written {len(code)} bytes to {filename} (sandbox path: {path})"

            @agent.tools.register(permission=PermissionLevel.ALLOW)
            async def run_code(filename: str) -> str:
                """Execute a Python file in the sandbox.

                Args:
                    filename: Name of the file to run (must be in sandbox).
                """
                code = await sandbox.read_file(filename)
                result = await sandbox.execute(code)

                output = f"Exit code: {result.exit_code}\n"
                output += f"Time: {result.execution_time_ms:.0f}ms\n"
                if result.timed_out:
                    output += "STATUS: TIMED OUT\n"
                if result.memory_exceeded:
                    output += "STATUS: MEMORY EXCEEDED\n"
                if result.stdout:
                    output += f"\nSTDOUT:\n{result.stdout}"
                if result.stderr:
                    output += f"\nSTDERR:\n{result.stderr}"
                if result.files_created:
                    output += f"\nFiles created: {result.files_created}"
                return output

            @agent.tools.register(permission=PermissionLevel.ALLOW)
            async def run_python(code: str) -> str:
                """Execute Python code directly in the sandbox.

                Args:
                    code: Python code to execute.
                """
                result = await sandbox.execute(code)

                output = f"Exit code: {result.exit_code} | Time: {result.execution_time_ms:.0f}ms"
                if result.timed_out:
                    output += " | TIMED OUT"
                if result.stdout:
                    output += f"\n{result.stdout}"
                if result.stderr:
                    output += f"\nSTDERR: {result.stderr}"
                return output

            @agent.tools.register(permission=PermissionLevel.ALLOW)
            async def read_sandbox_file(filename: str) -> str:
                """Read a file from the sandbox.

                Args:
                    filename: Name of the file to read.
                """
                return await sandbox.read_file(filename)

            @agent.tools.register(permission=PermissionLevel.ALLOW)
            async def list_sandbox_files() -> str:
                """List all files in the sandbox."""
                files = await sandbox.list_files()
                return "\n".join(files) if files else "(empty)"

            # Also register filesystem tools for reading existing code
            register_filesystem_tools(
                agent.tools,
                include=["read_file", "list_directory"],
                base_path=str(Path.cwd()),
            )

            print_banner(
                "HarnessX — Sandboxed Coder",
                subtitle="Process resource limits; host filesystem and network remain accessible",
                commands={"quit": "Exit", "status": "Runtime status"},
            )

            while True:
                user_input = get_user_input()
                if user_input is None or user_input.lower() == "quit":
                    break
                if not user_input:
                    continue
                if user_input.lower() == "status":
                    status = await runtime.get_status()
                    print_status(
                        {
                            "State": status["state"],
                            "Usage": agent.guardrails.usage_summary,
                        }
                    )
                    continue

                try:
                    response = completed_output(await runtime.execute(user_input))
                    print_response(response)
                except Exception as e:
                    print_error(e)

        print_status({"State": "stopped"})


if __name__ == "__main__":
    asyncio.run(main())
