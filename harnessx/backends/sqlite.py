"""Single-host durability with transactional SQLite and OS session locks."""

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import fcntl
import hashlib
from pathlib import Path
import sqlite3
import uuid

from .store import (
    SQLStore,
    TABLES,
    MIGRATION_CHECKSUM,
    SchemaError,
    SessionBusyError,
    LeaseLostError,
)


class SQLiteTransaction:
    def __init__(self, conn):
        self.conn = conn

    async def execute(self, sql, args=()):
        return self.conn.execute(sql, args)

    async def one(self, sql, args=()):
        return self.conn.execute(sql, args).fetchone()

    async def all(self, sql, args=()):
        return self.conn.execute(sql, args).fetchall()


class SQLiteBackend(SQLStore):
    def __init__(
        self, path: str = ".agent_sessions/runtime.sqlite3", *, auto_migrate=True
    ):
        self.path = Path(path)
        self.auto_migrate = auto_migrate
        self.prefix = ""
        self.conn = None
        self._mutex = asyncio.Lock()
        self._leases = {}

    async def initialize(self):
        if self.conn is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        try:
            async with self.transaction() as tx:
                exists = await tx.one(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_versions'"
                )
                if not exists:
                    if not self.auto_migrate:
                        raise SchemaError("Schema missing; enable auto_migrate")
                    for name, definition in TABLES.items():
                        await tx.execute(
                            f'CREATE TABLE "{name}" ({definition.format(json="TEXT", blob="BLOB", schema="")})'
                        )
                    await tx.execute("CREATE INDEX runs_session ON runs(session_id)")
                    await tx.execute(
                        "INSERT INTO schema_versions VALUES (?,?)",
                        (1, MIGRATION_CHECKSUM),
                    )
                await self._migrate_journal(tx, validate_only=not self.auto_migrate)
        except BaseException:
            self.conn.close()
            self.conn = None
            raise

    @asynccontextmanager
    async def transaction(self):
        async with self._mutex:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield SQLiteTransaction(self.conn)
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise

    async def claim(self, session_id):
        if session_id in self._leases:
            raise SessionBusyError("Session is already executing")
        key = hashlib.sha256(session_id.encode()).hexdigest()
        path = self.path.parent / f".{key}.lock"
        lock = path.open("a+b")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise SessionBusyError("Session is owned by another process") from None
        token = str(uuid.uuid4())
        self._leases[session_id] = (token, lock)
        return token

    async def _fence(self, tx, session_id, lease):
        if session_id not in self._leases or self._leases[session_id][0] != lease:
            raise LeaseLostError("Session ownership lost")

    async def renew(self, session_id, lease):
        await self._fence(None, session_id, lease)

    async def release(self, session_id, lease):
        if session_id in self._leases and self._leases[session_id][0] == lease:
            _, lock = self._leases.pop(session_id)
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    async def aclose(self):
        for session_id, (token, _) in list(self._leases.items()):
            await self.release(session_id, token)
        if self.conn:
            self.conn.close()
            self.conn = None
