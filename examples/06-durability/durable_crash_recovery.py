"""Durable runs on PostgreSQL: a run that survives its process crashing, without repeating a tool.

AgentRuntime saves every step of a run in the database. The ``crash`` command starts a run whose
tool writes a report, then kills the process (exit code 42) right after the tool's result is
committed. ``resume``, in a new process, finishes the same run from the database: the report tool
is not called again, because its saved result is replayed. ``status`` shows what was saved without
changing it, ``check`` creates or validates the schema, and the default ``chat`` streams one live
answer through the PostgreSQL runtime.

Run:
    python examples/06-durability/durable_crash_recovery.py check  --config examples/06-durability/postgres.config.json
    python examples/06-durability/durable_crash_recovery.py crash  --config examples/06-durability/postgres.config.json
    python examples/06-durability/durable_crash_recovery.py status --config examples/06-durability/postgres.config.json
    python examples/06-durability/durable_crash_recovery.py resume --config examples/06-durability/postgres.config.json
    python examples/06-durability/durable_crash_recovery.py chat   --config examples/06-durability/postgres.config.json
Copy postgres.config.example.json beside this file to postgres.config.json and fill it in, or set DATABASE_URL.
--workspace (default .agent_sessions/postgres-demo) and --schema (default harness_x) must be the same for every
command of one experiment.

Needs: pip install "harnessx[postgres]" and a PostgreSQL database. Only ``chat`` (the default) calls a model,
so it also needs ANTHROPIC_API_KEY. Walkthrough: https://harnessx-site.vercel.app/docs/postgres-durability/
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv

from harnessx import (
    Agent,
    AgentConfig,
    AgentRuntime,
    Middleware,
    PostgresBackend,
    ProviderResponse,
    RunEventType,
    SessionBusyError,
    ToolCall,
    ToolResult,
)
from harnessx.providers import LLMProvider

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

# The local config lives beside this file (git-ignored); the template next to it shows the fields.
CONFIG_FILE = Path(__file__).resolve().with_name("postgres.config.json")
CONFIG_TEMPLATE = CONFIG_FILE.with_name("postgres.config.example.json")
REPORT = "Weekly sales report\nOrders: 12\nRevenue: $1,200\n"
ANSWER = "The weekly report is ready: 12 orders, $1,200 revenue."


def load_connection_string(config: Path | None = None) -> str:
    """Explicit JSON config overrides DATABASE_URL, without logging credentials."""
    if config is None:
        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            raise SystemExit(f"Set DATABASE_URL or pass --config {CONFIG_FILE} (copy {CONFIG_TEMPLATE.name})")
        return dsn
    try:
        settings = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SystemExit("Cannot read PostgreSQL config; check the --config path and JSON syntax") from None
    if not isinstance(settings, dict):
        raise SystemExit("PostgreSQL config must be a JSON object with connection_string")
    dsn = settings.get("connection_string")
    if not isinstance(dsn, str) or not dsn.strip():
        raise SystemExit("Set connection_string in the PostgreSQL config")
    if "{password}" in dsn:
        if not dsn.startswith(("postgresql://", "postgres://")):
            raise SystemExit("The {password} placeholder requires a PostgreSQL URL")
        password = settings.get("password")
        if not isinstance(password, str) or not password or password == "REPLACE_WITH_PASSWORD":
            raise SystemExit(
                "Set the password field in the PostgreSQL config; "
                "replace REPLACE_WITH_PASSWORD and leave {password} in connection_string"
            )
        # Raw passwords may contain @, /, :, %, or other URL delimiters.
        dsn = dsn.replace("{password}", quote(password, safe=""))
    return dsn


async def check_connection(dsn: str, schema: str) -> None:
    """Validate connectivity and provision/validate the runtime schema."""
    backend = PostgresBackend(dsn, schema=schema)
    try:
        await backend.initialize()
        print(f"Connected to PostgreSQL. HarnessX schema {schema!r} is ready.")
    finally:
        await backend.aclose()


class ReportProvider(LLMProvider):
    """Choose responses from restored messages, with no process-local counter."""

    async def create(self, **kwargs: Any) -> ProviderResponse:
        for message in kwargs["messages"]:
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if block.get("is_error"):
                        raise RuntimeError("Report tool failed: " + str(block["content"]))
                    return ProviderResponse(text=ANSWER)
        return ProviderResponse(
            tool_calls=[ToolCall("report", "write_report", {})],
            stop_reason="tool_use",
        )

    async def count_tokens(self, **kwargs: Any) -> int:
        return 0


class CrashAfterSavedResult(Middleware):
    """Demo fault injection only; never install this in an application agent."""

    async def after_tool_execution(self, result: ToolResult) -> ToolResult:
        # The SQL driver commits raw_result before invoking result middleware.
        if result.is_error:
            raise RuntimeError("Cannot demonstrate recovery: report tool failed")
        print(
            "Tool result committed. Intentionally exiting this process (code 42).",
            flush=True,
        )
        print(
            "Wait 30 seconds for the lease to expire, then run the resume command.",
            flush=True,
        )
        os._exit(42)  # Bypass cleanup, including lease release, just like a hard crash.


def report_agent(workspace: Path, *, crash: bool) -> Agent:
    agent = Agent(provider=ReportProvider())

    @agent.tools.register(replay_policy="manual")
    def write_report() -> str:
        """Write the fixed demo report and record each tool invocation."""
        with (workspace / "tool-invocations.txt").open("a") as log:
            log.write("write_report\n")
        # Exclusive creation makes an accidental second execution visible.
        with (workspace / "report.txt").open("x") as report:
            report.write(REPORT)
        print("write_report executed: report.txt created", flush=True)
        return "Report created: 12 orders, $1,200 revenue."

    if crash:
        agent.middleware.add(CrashAfterSavedResult())
    return agent


async def print_demo_status(backend, workspace: Path, session_id: str) -> None:
    state = await backend.latest_run(session_id)
    print("Session:", session_id)
    print("Run:", state["run_id"] if state else "not submitted")
    print("Status:", state["status"] if state else "idle")
    for entry in state.get("tools", []) if state else []:
        print("Tool:", entry["call"]["name"], "| Saved state:", entry["status"])
    print("Report exists:", (workspace / "report.txt").exists())
    log = workspace / "tool-invocations.txt"
    print("Tool invocations:", len(log.read_text().splitlines()) if log.exists() else 0)
    if state and state.get("output"):
        print("Answer:", state["output"])


async def recovery_demo(command: str, dsn: str, workspace: Path, schema: str) -> None:
    workspace = workspace.resolve()
    manifest = workspace / "session.json"
    if command == "crash":
        workspace.mkdir(parents=True, exist_ok=True)
        if any(
            (workspace / name).exists()
            for name in ("session.json", "report.txt", "tool-invocations.txt")
        ):
            raise SystemExit(
                "Demo files already exist. Use status/resume, or choose a new --workspace."
            )
        session_id = ""
    else:
        if not manifest.exists():
            raise SystemExit(
                "No demo session found. Run crash first with the same --workspace."
            )
        saved = json.loads(manifest.read_text())
        session_id, schema = saved["session_id"], saved["schema"]

    backend = PostgresBackend(dsn, schema=schema)
    try:
        await backend.initialize()
        if command == "status":
            await print_demo_status(backend, workspace, session_id)
            return
        # Each process supplies a fresh Agent with the same configuration/tools.
        # The Agent context also closes it if resume fails before runtime binding.
        async with report_agent(workspace, crash=command == "crash") as agent:
            runtime = AgentRuntime(agent, backend=backend)
            try:
                if command == "crash":
                    session_id = await runtime.start()
                    with manifest.open("x") as file:
                        json.dump({"session_id": session_id, "schema": schema}, file)
                    print("Session:", session_id, flush=True)
                    print("Workspace:", workspace, flush=True)
                    result = await runtime.run(
                        "Create the weekly sales report", request_id="weekly-report"
                    )
                    raise RuntimeError(
                        f"Expected an intentional process exit; run ended as {result.status.value}"
                    )
                try:
                    handle = await runtime.resume(session_id)
                except SessionBusyError:
                    raise SystemExit(
                        "Session lease is still active. Wait at least 30 seconds after the crash, "
                        "then retry resume."
                    ) from None
                if handle:
                    result = await handle.result()
                    if result.status != "completed":
                        raise RuntimeError(
                            f"Recovery ended as {result.status.value}: {result.error}"
                        )
                else:
                    print("No unfinished run; showing saved results.")
                await print_demo_status(backend, workspace, session_id)
                invocations = (workspace / "tool-invocations.txt").read_text().splitlines()
                if invocations != ["write_report"]:
                    raise RuntimeError("Expected exactly one tool invocation in this demo")
                if (workspace / "report.txt").read_text() != REPORT:
                    raise RuntimeError("The demo report is missing or changed")
            finally:
                await runtime.stop()
    finally:
        await backend.aclose()


async def live_chat(dsn: str) -> None:
    backend = PostgresBackend(dsn)
    try:
        agent = Agent(config=AgentConfig(system_prompt="You are helpful."))
        async with agent, AgentRuntime(agent, backend=backend) as runtime:
            print("Session:", runtime.session_id)
            async with runtime.run_stream(
                "Explain durable execution briefly."
            ) as stream:
                async for event in stream:
                    if event.type == RunEventType.TEXT_DELTA:
                        print(event.data, end="", flush=True)
                result = await stream.result()
            print("\nStatus:", result.status.value)
            if result.error:
                print("Error:", result.error["message"])
    finally:
        await backend.aclose()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "command", nargs="?", choices=("chat", "check", "crash", "status", "resume"), default="chat"
    )
    parser.add_argument(
        "--config", type=Path,
        help=f"JSON connection config; overrides DATABASE_URL. Template: {CONFIG_TEMPLATE}",
    )
    parser.add_argument(
        "--workspace", type=Path, default=Path(".agent_sessions/postgres-demo"),
        help="Demo files and session ID; use the same directory for crash/status/resume",
    )
    parser.add_argument(
        "--schema", default="harness_x",
        help="PostgreSQL schema for check or a new crash demo; status/resume use session.json",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dsn = load_connection_string(args.config)
    if args.command == "chat":
        await live_chat(dsn)
    elif args.command == "check":
        await check_connection(dsn, args.schema)
    else:
        await recovery_demo(args.command, dsn, args.workspace, args.schema)


if __name__ == "__main__":
    asyncio.run(main())
