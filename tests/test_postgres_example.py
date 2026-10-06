"""Exercise the demo across real process exits, with optional PostgreSQL coverage."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from example_loader import example_path, load_example
from harnessx import PostgresBackend, SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = example_path("05-durability/durable_crash_recovery.py")
DSN = os.environ.get("HARNESS_TEST_POSTGRES_DSN")


@pytest.mark.parametrize("database", [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.skipif(not DSN, reason="HARNESS_TEST_POSTGRES_DSN is not configured")),
])
def test_report_survives_process_crash_without_repeating_tool(tmp_path, database):
    schema = "harness_test_" + uuid.uuid4().hex
    workspace = tmp_path / "report-demo"
    dsn = DSN if database == "postgres" else str(tmp_path / "runtime.db")
    config = tmp_path / "postgres.config.json"
    config.touch(mode=0o600)
    config.write_text(json.dumps({"connection_string": dsn}))
    env = {**os.environ, "DATABASE_URL": "unused-config-takes-precedence", "PYTHONPATH": str(ROOT)}
    # A short real PostgreSQL lease keeps the opt-in test fast. The CLI retains
    # the production default of 30 seconds. SQLite is explicitly a stand-in.
    setup = (
        "real_backend = demo.PostgresBackend\n"
        "demo.PostgresBackend = lambda dsn, **kw: real_backend(dsn, lease_seconds=2, **kw)\n"
        if database == "postgres" else
        "import socket\n"
        "socket.socket.connect = lambda *a: (_ for _ in ()).throw(AssertionError('Network forbidden'))\n"
        "from harnessx import SQLiteBackend\n"
        "demo.PostgresBackend = lambda dsn, **kw: SQLiteBackend(dsn)\n"
    )
    load = (
        "import asyncio, importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('demo', {str(EXAMPLE)!r})\n"
        "demo = importlib.util.module_from_spec(spec)\n"
        "sys.modules['demo'] = demo\n"
        "spec.loader.exec_module(demo)\n"
    )
    script = load + setup + "asyncio.run(demo.main())\n"

    def run(command):
        return subprocess.run(
            [sys.executable, "-c", script, command, "--config", str(config), "--workspace", str(workspace), "--schema", schema],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45,
        )

    async def stored_state():
        store = PostgresBackend(dsn, schema=schema) if database == "postgres" else SQLiteBackend(dsn)
        try:
            await store.initialize()
            session_id = json.loads((workspace / "session.json").read_text())["session_id"]
            return await store.latest_run(session_id)
        finally:
            await store.aclose()

    async def cleanup():
        if database == "postgres":
            store = PostgresBackend(dsn, schema=schema)
            try:
                await store.initialize()
                async with store.transaction() as tx:
                    await tx.execute(f"DROP SCHEMA {store.prefix[:-1]} CASCADE")
            finally:
                await store.aclose()

    try:
        checked = run("check")
        assert checked.returncode == 0, checked.stdout + checked.stderr
        assert "schema" in checked.stdout and "is ready" in checked.stdout
        assert not workspace.exists()  # Checking does not construct an agent or report.
        crashed = run("crash")
        assert crashed.returncode == 42, crashed.stdout + crashed.stderr
        assert "Tool result committed" in crashed.stdout
        state = asyncio.run(stored_state())
        assert state["status"] == "running"
        assert state["tools"][0]["status"] == "raw_completed"
        assert state["tools"][0]["policy"] == "manual"
        assert state["tools"][0]["raw_result"]["content"] == "Report created: 12 orders, $1,200 revenue."
        run_id = state["run_id"]

        status = run("status")
        assert status.returncode == 0, status.stderr
        assert "Saved state: raw_completed" in status.stdout
        assert "Tool invocations: 1" in status.stdout
        assert asyncio.run(stored_state()) == state  # Status does not resume or mutate.

        duplicate = run("crash")
        assert duplicate.returncode == 1 and "already exist" in duplicate.stderr
        deadline = time.monotonic() + 10
        while True:
            resumed = run("resume")
            if "Session lease is still active" not in resumed.stderr or time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert "Status: completed" in resumed.stdout
        assert "The weekly report is ready: 12 orders, $1,200 revenue." in resumed.stdout
        assert "write_report executed" not in resumed.stdout
        assert "Tool invocations: 1" in resumed.stdout
        completed = asyncio.run(stored_state())
        assert completed["status"] == "completed" and completed["run_id"] == run_id
        assert (workspace / "report.txt").read_text() == "Weekly sales report\nOrders: 12\nRevenue: $1,200\n"
        assert (workspace / "tool-invocations.txt").read_text() == "write_report\n"

        again = run("resume")
        assert again.returncode == 0, again.stderr
        assert "No unfinished run" in again.stdout and "Tool invocations: 1" in again.stdout
        assert asyncio.run(stored_state())["run_id"] == run_id
    finally:
        asyncio.run(cleanup())


def test_recovery_example_missing_session_is_actionable(tmp_path, monkeypatch):
    main = load_example("05-durability/durable_crash_recovery.py").main

    monkeypatch.setenv("DATABASE_URL", "unused")
    with pytest.raises(SystemExit, match="Run crash first"):
        asyncio.run(main(["resume", "--workspace", str(tmp_path)]))


def test_postgres_config_encodes_password_and_overrides_environment(tmp_path, monkeypatch):
    load_connection_string = load_example("05-durability/durable_crash_recovery.py").load_connection_string

    password = 'p@ss:/?#% +\\"\'$ä'
    config = tmp_path / "postgres.json"
    config.write_text(json.dumps({
        "connection_string": "postgresql://demo:{password}@example.invalid:5432/harness?sslmode=require&connect_timeout=10",
        "password": password,
    }))
    monkeypatch.setenv("DATABASE_URL", "environment-dsn")
    dsn = urlsplit(load_connection_string(config))
    assert dsn.username == "demo" and unquote(dsn.password) == password
    assert dsn.hostname == "example.invalid" and dsn.port == 5432
    assert dsn.path == "/harness"
    assert parse_qs(dsn.query) == {"sslmode": ["require"], "connect_timeout": ["10"]}
    assert load_connection_string() == "environment-dsn"


@pytest.mark.parametrize("content", [
    '{"password": "secret-value", BROKEN JSON}',
    '["secret-value"]',
    '{"password": "secret-value"}',
    '{"connection_string": "postgresql://demo:{password}@example.invalid/harness", "password": "REPLACE_WITH_PASSWORD"}',
])
def test_bad_postgres_config_fails_before_connecting_without_echoing_secrets(
    tmp_path, monkeypatch, content,
):
    postgres_runtime = load_example("05-durability/durable_crash_recovery.py")

    config = tmp_path / "postgres.json"
    config.write_text(content)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid config attempted a database connection")

    monkeypatch.setattr(postgres_runtime, "PostgresBackend", forbidden)
    with pytest.raises(SystemExit) as error:
        asyncio.run(postgres_runtime.main(["check", "--config", str(config)]))
    assert "secret-value" not in str(error.value)
