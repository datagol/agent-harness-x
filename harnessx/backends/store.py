"""Transactional runtime storage shared by SQLite and PostgreSQL."""

from __future__ import annotations

from typing import Self

from ..errors import HarnessError
import hashlib
import asyncio
from functools import wraps
import json

from ..execution import RunEvent, wire
from .._journal import checkpoint, record


class StorageError(HarnessError, RuntimeError):
    pass


class SessionBusyError(StorageError):
    pass


class LeaseLostError(StorageError):
    pass


class SchemaError(StorageError):
    pass


# Deliberately small explicit migrations, with identical logical tables on both stores.
TABLES = {
    "schema_versions": "version INTEGER PRIMARY KEY, checksum TEXT NOT NULL",
    "sessions": "id TEXT PRIMARY KEY, payload {json} NOT NULL, control TEXT NOT NULL DEFAULT '', owner TEXT, generation BIGINT NOT NULL DEFAULT 0, lease_until DOUBLE PRECISION NOT NULL DEFAULT 0",
    "runs": "id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES {schema}sessions(id), request_id TEXT NOT NULL, status TEXT NOT NULL, payload {json} NOT NULL, UNIQUE(session_id, request_id)",
    "steps": "run_id TEXT NOT NULL REFERENCES {schema}runs(id), id TEXT NOT NULL, payload {json} NOT NULL, PRIMARY KEY(run_id,id)",
    "approvals": "run_id TEXT NOT NULL REFERENCES {schema}runs(id), id TEXT NOT NULL, payload {json} NOT NULL, PRIMARY KEY(run_id,id)",
    "tool_outcomes": "run_id TEXT NOT NULL REFERENCES {schema}runs(id), id TEXT NOT NULL, payload {json} NOT NULL, PRIMARY KEY(run_id,id)",
    "events": "run_id TEXT NOT NULL REFERENCES {schema}runs(id), seq BIGINT NOT NULL, event_id TEXT NOT NULL UNIQUE, payload {json} NOT NULL, PRIMARY KEY(run_id,seq)",
    "snapshots": "session_id TEXT PRIMARY KEY REFERENCES {schema}sessions(id), payload {json} NOT NULL",
    "artifacts": "id TEXT PRIMARY KEY, content {blob} NOT NULL",
}
MIGRATION_CHECKSUM = hashlib.sha256(
    json.dumps(TABLES, sort_keys=True).encode()
).hexdigest()

# Keep migration 1 unchanged so existing installations can be upgraded in place.
JOURNAL_TABLE = "run_id TEXT NOT NULL REFERENCES {schema}runs(id), seq BIGINT NOT NULL, id TEXT NOT NULL UNIQUE, payload {json} NOT NULL, PRIMARY KEY(run_id,seq)"
JOURNAL_CHECKSUM = hashlib.sha256(JOURNAL_TABLE.encode()).hexdigest()


def retry_transaction(method):
    """Retry only PostgreSQL transactions known to have been rolled back."""

    @wraps(method)
    async def wrapped(*args, **kwargs):
        for attempt in range(3):
            try:
                return await method(*args, **kwargs)
            except Exception as exc:
                if (
                    getattr(exc, "sqlstate", None) not in ("40001", "40P01")
                    or attempt == 2
                ):
                    raise
                await asyncio.sleep(0.05 * (attempt + 1))

    return wrapped


class SQLStore:
    """Backends implement transaction(), fencing, initialization and close."""

    async def __aenter__(self) -> Self:
        await self.initialize()
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    postgres = False
    supports_recording = True

    async def _migrate_journal(self, tx, *, validate_only=False):
        rows = await tx.all(
            f"SELECT version,checksum FROM {self.table('schema_versions')} ORDER BY version"
        )
        expected = [(1, MIGRATION_CHECKSUM), (2, JOURNAL_CHECKSUM)]
        if rows == expected:
            return
        if rows != expected[:1]:
            raise SchemaError("Unsupported schema version or migration checksum")
        if validate_only:
            raise SchemaError("Schema migration required; initialize with auto_migrate=True to upgrade")
        definition = JOURNAL_TABLE.format(
            schema=self.prefix, json="JSONB" if self.postgres else "TEXT"
        )
        await tx.execute(f"CREATE TABLE {self.table('journal')} ({definition})")
        await tx.execute(
            f"INSERT INTO {self.table('schema_versions')} VALUES (?,?)", expected[1]
        )

    def table(self, name):
        return f'{self.prefix}"{name}"'

    def encode(self, value):
        raw = wire(value)
        if self.postgres:
            from psycopg.types.json import Jsonb

            return Jsonb(raw)
        return json.dumps(raw, allow_nan=False)

    def decode(self, value):
        return json.loads(value) if isinstance(value, str) else value

    @retry_transaction
    async def create_session(self, session_id, data):
        async with self.transaction() as tx:
            await tx.execute(
                f"INSERT INTO {self.table('sessions')}(id,payload) VALUES (?,?)",
                (session_id, self.encode(data)),
            )

    @retry_transaction
    async def get_session(self, session_id):
        async with self.transaction() as tx:
            row = await tx.one(
                f"SELECT payload FROM {self.table('sessions')} WHERE id=?",
                (session_id,),
            )
            if row is None:
                raise StorageError(
                    "Session not found. Legacy JSON checkpoints require a new session."
                )
            return self.decode(row[0])

    @retry_transaction
    async def create_run(self, session_id, request_id, state, lease):
        async with self.transaction() as tx:
            await self._fence(tx, session_id, lease)
            row = await tx.one(
                f"SELECT payload FROM {self.table('runs')} WHERE session_id=? AND request_id=?",
                (session_id, request_id),
            )
            if row:
                return self.decode(row[0]), False
            row = await tx.one(
                f"SELECT id FROM {self.table('runs')} WHERE session_id=? AND status NOT IN ('completed','cancelled')",
                (session_id,),
            )
            if row:
                raise SessionBusyError(
                    "Session has an unfinished run; resume or cancel it first"
                )
            await tx.execute(
                f"INSERT INTO {self.table('runs')}(id,session_id,request_id,status,payload) VALUES (?,?,?,?,?)",
                (
                    state["run_id"],
                    session_id,
                    request_id,
                    state["status"],
                    self.encode(state),
                ),
            )
            session_row = await tx.one(
                f"SELECT payload FROM {self.table('sessions')} WHERE id=?",
                (session_id,),
            )
            session = self.decode(session_row[0])
            session["last_run_id"] = state["run_id"]
            await tx.execute(
                f"UPDATE {self.table('sessions')} SET control='',payload=? WHERE id=?",
                (self.encode(session), session_id),
            )
            return state, True

    @retry_transaction
    async def find_request(self, session_id, request_id):
        async with self.transaction() as tx:
            row = await tx.one(
                f"SELECT payload FROM {self.table('runs')} WHERE session_id=? AND request_id=?",
                (session_id, request_id),
            )
            return self.decode(row[0]) if row else None

    @retry_transaction
    async def get_run(self, run_id):
        async with self.transaction() as tx:
            row = await tx.one(
                f"SELECT payload FROM {self.table('runs')} WHERE id=?", (run_id,)
            )
            if row is None:
                raise StorageError("Run not found")
            return self.decode(row[0])

    async def latest_run(self, session_id):
        data = await self.get_session(session_id)
        run_id = data.get("last_run_id")
        return await self.get_run(run_id) if run_id else None

    @retry_transaction
    async def save_run(self, state, events, lease, *, records=()):
        async with self.transaction() as tx:
            await self._fence(tx, state["session_id"], lease)
            await tx.execute(
                f"UPDATE {self.table('runs')} SET status=?,payload=? WHERE id=?",
                (state["status"], self.encode(state), state["run_id"]),
            )
            row = await tx.one(
                f"SELECT payload FROM {self.table('sessions')} WHERE id=?",
                (state["session_id"],),
            )
            session = self.decode(row[0])
            session["last_run_id"] = state["run_id"]
            await tx.execute(
                f"UPDATE {self.table('sessions')} SET payload=? WHERE id=?",
                (self.encode(session), state["session_id"]),
            )
            await tx.execute(
                f"INSERT INTO {self.table('snapshots')}(session_id,payload) VALUES (?,?) ON CONFLICT(session_id) DO UPDATE SET payload=excluded.payload",
                (state["session_id"], self.encode(state)),
            )
            step_id = str(state["iterations"])
            step = {
                k: state[k]
                for k in ("phase", "request", "response", "attempt")
                if k in state
            }
            await tx.execute(
                f"INSERT INTO {self.table('steps')}(run_id,id,payload) VALUES (?,?,?) ON CONFLICT(run_id,id) DO UPDATE SET payload=excluded.payload",
                (state["run_id"], step_id, self.encode(step)),
            )
            for tool in state.get("tools", []):
                name = (
                    "tool_outcomes"
                    if "raw_result" in tool or "result" in tool
                    else "approvals"
                )
                await tx.execute(
                    f"INSERT INTO {self.table(name)}(run_id,id,payload) VALUES (?,?,?) ON CONFLICT(run_id,id) DO UPDATE SET payload=excluded.payload",
                    (state["run_id"], tool["execution_key"], self.encode(tool)),
                )
            if state.get("recording"):
                await self._append_records(tx, state["run_id"], [
                    *records, record("checkpoint", checkpoint(state), state),
                ])
            await self._append_events(tx, state["run_id"], events)
            if state.get("recording"):
                await self._append_records(tx, state["run_id"], [
                    record("event", event, state) for event in events
                ])

    async def _append_records(self, tx, run_id, records):
        seq = (await tx.one(
            f"SELECT COALESCE(MAX(seq),0) FROM {self.table('journal')} WHERE run_id=?",
            (run_id,),
        ))[0]
        for item in records:
            existing = await tx.one(
                f"SELECT payload FROM {self.table('journal')} WHERE id=?", (item["id"],)
            )
            if existing:
                previous = self.decode(existing[0])
                previous.pop("seq")
                if previous != item:
                    raise StorageError("Conflicting immutable journal record")
                continue
            if item["run_id"] != run_id:
                raise StorageError("Journal record belongs to a different run")
            seq += 1
            await tx.execute(
                f"INSERT INTO {self.table('journal')} (run_id,seq,id,payload) VALUES (?,?,?,?)",
                (run_id, seq, item["id"], self.encode({**item, "seq": seq})),
            )

    async def _append_events(self, tx, run_id, events):
        seq = (
            await tx.one(
                f"SELECT COALESCE(MAX(seq),0) FROM {self.table('events')} WHERE run_id=?",
                (run_id,),
            )
        )[0]
        for event in events:
            seq += 1
            event.cursor = str(seq)
            await tx.execute(
                f"INSERT INTO {self.table('events')}(run_id,seq,event_id,payload) VALUES (?,?,?,?) ON CONFLICT(event_id) DO NOTHING",
                (run_id, seq, event.event_id, self.encode(event)),
            )
        if self.postgres and events:
            await tx.execute("SELECT pg_notify(?,?)", ("harness_x_events", run_id))

    @retry_transaction
    async def append_events(self, state, events, lease):
        async with self.transaction() as tx:
            await self._fence(tx, state["session_id"], lease)
            await self._append_events(tx, state["run_id"], events)
            if state.get("recording"):
                await self._append_records(tx, state["run_id"], [
                    record("event", event, state) for event in events
                ])

    @retry_transaction
    async def incident_snapshot(self, session_id, run_id, *, max_records=100000):
        """Read a consistent export cut, scoped to the caller's bound session."""
        async with self.transaction() as tx:
            suffix = " FOR UPDATE" if self.postgres else ""
            session = await tx.one(
                f"SELECT payload FROM {self.table('sessions')} WHERE id=?{suffix}", (session_id,)
            )
            if session is None:
                raise StorageError("Session not found")
            row = await tx.one(
                f"SELECT payload FROM {self.table('runs')} WHERE id=? AND session_id=?",
                (run_id, session_id),
            )
            if row is None:
                raise StorageError("Run not found in this session")
            state = self.decode(row[0])
            if not state.get("recording"):
                raise StorageError("Flight recording was not enabled for this run")
            rows = await tx.all(
                f"SELECT payload FROM {self.table('journal')} WHERE run_id=? ORDER BY seq LIMIT ?",
                (run_id, max_records + 1),
            )
            if len(rows) > max_records:
                raise StorageError("Incident exceeds the configured record limit")
            return state, [self.decode(row[0]) for row in rows], self.decode(session[0])

    @retry_transaction
    async def read_events(self, run_id, after="0"):
        async with self.transaction() as tx:
            rows = await tx.all(
                f"SELECT payload FROM {self.table('events')} WHERE run_id=? AND seq>? ORDER BY seq LIMIT 1000",
                (run_id, int(after or 0)),
            )
            return [RunEvent.from_dict(self.decode(r[0])) for r in rows]

    @retry_transaction
    async def set_control(self, session_id, action):
        if action not in ("", "paused", "cancelled"):
            raise ValueError("Invalid runtime control")
        async with self.transaction() as tx:
            await tx.execute(
                f"UPDATE {self.table('sessions')} SET control=? WHERE id=?",
                (action, session_id),
            )

    @retry_transaction
    async def get_control(self, session_id):
        async with self.transaction() as tx:
            row = await tx.one(
                f"SELECT control FROM {self.table('sessions')} WHERE id=?",
                (session_id,),
            )
            return row[0]

    @retry_transaction
    async def put_artifact(self, content: bytes) -> str:
        key = hashlib.sha256(content).hexdigest()
        async with self.transaction() as tx:
            await tx.execute(
                f"INSERT INTO {self.table('artifacts')}(id,content) VALUES (?,?) ON CONFLICT(id) DO NOTHING",
                (key, content),
            )
        return key

    @retry_transaction
    async def get_artifact(self, key: str) -> bytes:
        async with self.transaction() as tx:
            row = await tx.one(
                f"SELECT content FROM {self.table('artifacts')} WHERE id=?", (key,)
            )
            if row is None:
                raise StorageError(f"Artifact missing: {key}")
            return bytes(row[0])
