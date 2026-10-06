"""Run an agent's tools inside an NVIDIA OpenShell sandbox while the agent itself stays here.

`OpenShellSandbox` is an execution backend: `run_bash` and the file tools run in an OpenShell sandbox, and
the loop, model calls and memory stay in this process. You never manage the sandbox. The agent creates it on
the first tool call that needs it, copies the project in, copies changed files back after every run, and
deletes it when the agent closes. Network access is closed unless you open it with `--allow`, and
`--secret` lets commands use an environment variable without being able to read it.

With no `--project`, this builds a small demo project with one failing test and asks the agent to fix it,
so you can watch the fix arrive back in your folder.

Run:
    python examples/08-sandboxes/openshell_backend.py
    python examples/08-sandboxes/openshell_backend.py --project ./my-repo --allow pypi.org --secret GITHUB_TOKEN
    python examples/08-sandboxes/openshell_backend.py --prompt "Run the tests and fix the failure"
Needs: pip install "harnessx[openshell]", a running OpenShell gateway (`openshell status`; on macOS use its
MicroVM driver), and ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, HookContext, HookEvent, PermissionLevel, RunEventType, ToolCall, ToolResult
from harnessx.openshell import OpenShellSandbox

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

DEMO_TASK = "Run the tests with `python3 -m unittest -q`, find why one fails, fix the code, and run them again."

# python:3.12-slim has python3; OpenShell's default image has neither python3 nor curl.
DEFAULT_IMAGE = "python:3.12-slim"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", help="Folder the agent works on (default: a demo project, created if missing)")
    parser.add_argument("--allow", action="append", default=[], metavar="HOST",
                        help="Host commands may reach, e.g. pypi.org (repeatable)")
    parser.add_argument("--secret", action="append", default=[], metavar="ENV_VAR",
                        help="Environment variable commands may use but never read, e.g. GITHUB_TOKEN (repeatable)")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help=f"OCI image for the sandbox (default: {DEFAULT_IMAGE})")
    parser.add_argument("--keep", action="store_true", help="Leave the sandbox running when the agent closes")
    parser.add_argument("--prompt", help="One task, then exit, instead of a conversation")
    return parser.parse_args(argv)


def demo_project(folder: Path) -> Path:
    """A tiny project with one bug, so there is something to fix."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "calc.py").write_text("def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n")
    (folder / "test_calc.py").write_text(
        "import unittest\n\nfrom calc import add, mul\n\n\n"
        "class CalcTest(unittest.TestCase):\n"
        "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n"
        "    def test_mul(self):\n        self.assertEqual(mul(2, 3), 6)\n"
    )
    return folder


def show_blocked(ctx: HookContext) -> None:
    for host in ctx.data.get("denials") or []:
        print(f"  ! the sandbox's network policy blocked {host}")


async def ask(agent: Agent, sandbox: OpenShellSandbox, text: str) -> None:
    midline = False
    async with agent.run_stream(text) as stream:
        async for event in stream:
            if event.type == RunEventType.TOOL_CALL_START and isinstance(event.data, ToolCall):
                print(("\n" if midline else "") + f"  > {event.data.name}({json.dumps(event.data.input)[:120]})")
                midline = False
            elif isinstance(event.data, ToolResult) and event.data.is_error:
                print(f"  ! {str(event.data.content)[:200]}")
            elif event.type == RunEventType.TEXT_DELTA:
                print(event.data, end="", flush=True)
                midline = True
        result = await stream.result()
    if midline:
        print()
    if result.status != "completed":
        print(f"Error: {result.error['message'] if result.error else result.status.value}")
    # The run has ended, so its changes are already back in the project folder.
    report = sandbox.last_sync
    if report and (report.updated or report.deleted or report.conflicts):
        print(f"Synced back: updated {report.updated}, deleted {report.deleted}, kept your version of {report.conflicts}")


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project = Path(args.project) if args.project else demo_project(Path("openshell-demo"))
    prompt = args.prompt or (None if args.project else DEMO_TASK)

    sandbox = OpenShellSandbox(
        project, image=args.image, allow=args.allow, secrets=args.secret, keep=args.keep, timeout_seconds=120,
    )
    agent = Agent(
        AgentConfig(system_prompt="You are a careful engineer. The project is your working directory."),
        tools=["filesystem", "bash"],
        sandbox=sandbox,  # the agent starts, syncs and closes it; there is nothing else to manage
    )
    # Inside the sandbox nothing can reach the host, so the commands need no approval prompts.
    agent.permissions.set_permission("run_bash", PermissionLevel.ALLOW)
    for name in ("write_file", "edit_file", "delete", "generate_file"):
        agent.permissions.set_permission(name, PermissionLevel.ALLOW)
    agent.hooks.on(HookEvent.SANDBOX_EXEC, show_blocked)

    print(f"Project: {project.resolve()}  (tools run in OpenShell; changes come back after each run)")
    async with agent:
        if prompt:
            print(f"You: {prompt}")
            await ask(agent, sandbox, prompt)
        else:
            while True:
                try:
                    text = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if text.lower() in ("quit", "exit"):
                    break
                if text:
                    await ask(agent, sandbox, text)
    print("Sandbox closed." if not args.keep else f"Sandbox kept: {sandbox.name}")


if __name__ == "__main__":
    asyncio.run(main())
