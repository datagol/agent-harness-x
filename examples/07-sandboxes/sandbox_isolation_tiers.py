"""A coding agent that runs the Python it writes inside a `Sandbox`, and the three isolation tiers to choose from.

`SandboxConfig(tier=...)` picks how code runs:
  - "process" (default): a subprocess with a temporary workdir and CPU, memory, file-size, and time limits. It is NOT
    an isolation boundary: the code keeps the host's filesystem and network access.
  - "docker": a container with memory, CPU, and network limits; network off unless enabled.
  - "seatbelt": macOS's OS-enforced sandbox profile; network off unless enabled.
Pick docker or seatbelt whenever the code is untrusted. The agent runs inside an `AgentRuntime`, which checkpoints
each run to SQLite under `.sandbox_sessions/`. Type `status` for the runtime state, `quit` to exit.

Run:
    python examples/07-sandboxes/sandbox_isolation_tiers.py
    python examples/07-sandboxes/sandbox_isolation_tiers.py --tier docker    # or --tier seatbelt on macOS
Needs: ANTHROPIC_API_KEY (and Docker running for --tier docker).
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from harnessx import (
    AgentConfig,
    AgentRuntime,
    HookContext,
    HookManager,
    Limits,
    PermissionLevel,
    RuntimeConfig,
    Sandbox,
    SandboxConfig,
)
from harnessx.builtin.filesystem import register_filesystem_tools

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A coding agent whose code runs in a sandbox tier of your choice.")
    parser.add_argument(
        "--tier", choices=("process", "docker", "seatbelt"), default="process", help="Sandbox tier (default: process)."
    )
    return parser.parse_args(argv)


def print_tool_calls() -> HookManager:
    # runtime.run() returns only the final result, so hooks are what show the calls while they happen.
    hooks = HookManager()

    @hooks.before_tool
    async def show_call(ctx: HookContext):
        call = ctx.data.get("tool_call")
        if call:
            print(f"  > {call.name}({json.dumps(call.input)[:120]})")

    @hooks.after_tool
    async def show_error(ctx: HookContext):
        result = ctx.data.get("result")
        if result and result.is_error:
            print(f"  ! {result.content}")

    return hooks


async def main(tier: str = "process") -> None:
    sandbox_config = SandboxConfig(
        timeout_seconds=15,
        max_memory_mb=256,
        max_cpu_seconds=10,
        max_file_size_mb=50,
        tier=tier,
    )

    async with Sandbox(sandbox_config) as sandbox:
        print(f"Sandbox tier: {tier} | workdir: {sandbox.workdir}")
        if tier == "process":
            print("The process tier limits resources only; the host filesystem and network stay reachable.")

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
                limits=Limits(max_iterations=25),
            ),
            runtime_config=RuntimeConfig(storage_dir=".sandbox_sessions"),
            hooks=print_tool_calls(),
        )

        async with runtime:
            agent = runtime.agent
            print(f"Session: {runtime.session_id}")

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
                result = await sandbox.execute(await sandbox.read_file(filename))
                output = f"Exit code: {result.exit_code}\nTime: {result.execution_time_ms:.0f}ms\n"
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

            # Read-only access to the existing code in this workspace, for context.
            register_filesystem_tools(agent.tools, include=["read_file", "list_directory"], base_path=str(Path.cwd()))

            print("Type 'status' for the runtime state, 'quit' to exit.")
            while True:
                try:
                    text = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if text.lower() in ("quit", "exit"):
                    break
                if not text:
                    continue
                if text.lower() == "status":
                    status = await runtime.status()
                    print(f"State: {status['state']} | Usage: {agent.guardrails.usage_summary}")
                    continue
                try:
                    result = await runtime.run(text)
                except Exception as exc:
                    print(f"Error: {exc}")
                    continue
                if result.status != "completed":
                    print(f"Error: {result.error['message'] if result.error else f'Run status: {result.status.value}'}")
                else:
                    print(result.output)

    print("Stopped.")


if __name__ == "__main__":
    asyncio.run(main(parse_args().tier))
