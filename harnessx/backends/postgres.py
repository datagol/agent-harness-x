"""PostgreSQL runtime storage; bootstraps an isolated, versioned schema."""

from __future__ import annotations
from contextlib import asynccontextmanager
import hashlib
import asyncio
import uuid

from .store import (
    SQLStore,
    TABLES,
    MIGRATION_CHECKSUM,
    SchemaError,
    StorageError,
    SessionBusyError,
    LeaseLostError,
    retry_transaction,
)


class PostgresTransaction:
    def __init__(self, conn):
        self.conn = conn

    async def execute(self, sql, args=()):
        return await self.conn.execute(postgres_bindings(sql), args)

    async def one(self, sql, args=()):
        return await (await self.execute(sql, args)).fetchone()

    async def all(self, sql, args=()):
        return await (await self.execute(sql, args)).fetchall()


class PostgresBackend(SQLStore):
    postgres = True

    def __init__(
        self,
        connection_string: str,
        *,
        schema="harness_x",  # Preserve existing databases across the package rename.
        auto_migrate=True,
        lease_seconds=30,
        pool_size=10,
        artifact_store=None,
    ):
        if not schema or "\x00" in schema or len(schema.encode()) > 63:
            raise ValueError(
                "Schema must be a nonempty PostgreSQL identifier of at most 63 bytes"
            )
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self._dsn = connection_string
        self.schema = schema
        self.prefix = '"' + schema.replace('"', '""') + '".'
        self.auto_migrate = auto_migrate
        self.lease_seconds = lease_seconds
        self.pool_size = pool_size
        self.pool = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._artifact_store = artifact_store

    def __repr__(self):
        return f"PostgresBackend(schema={self.schema!r}, connection_string=<redacted>)"

    async def initialize(self):
        async with self._init_lock:
            if self._initialized:
                return
            await self._initialize()
            self._initialized = True

    async def _initialize(self):
        try:
            from psycopg_pool import AsyncConnectionPool
        except ImportError:
            raise ImportError("Install harnessx[postgres] to use PostgreSQL") from None
        # Connection failures can include DSN credentials: expose only sanitized diagnostics.
        try:
            self.pool = AsyncConnectionPool(
                self._dsn, min_size=0, max_size=self.pool_size, open=False, timeout=10
            )
            await self.pool.open()
            await self.migrate(validate_only=not self.auto_migrate)
        except SchemaError:
            await self.aclose()
            raise
        except Exception:
            await self.aclose()
            raise StorageError(
                "PostgreSQL initialization failed; check connectivity, TLS, and schema privileges"
            ) from None

    async def migrate(self, *, validate_only=False):
        if self.pool is None:
            # Explicit provisioning entry point for deployments using auto_migrate=False.
            saved = self.auto_migrate
            self.auto_migrate = not validate_only
            try:
                await self.initialize()
            finally:
                self.auto_migrate = saved
            return
        key = int.from_bytes(
            hashlib.sha256(("harness_x:migrations:" + self.schema).encode()).digest()[
                :8
            ],
            "big",
            signed=True,
        )
        try:
            async with self.transaction() as tx:
                await tx.execute("SELECT pg_advisory_xact_lock(?)", (key,))
                row = await tx.one(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_schema=? AND table_name=?)",
                    (self.schema, "schema_versions"),
                )
                if not row[0]:
                    if validate_only:
                        raise SchemaError(
                            "Schema missing; run backend.migrate() or enable auto_migrate"
                        )
                    await tx.execute(f"CREATE SCHEMA IF NOT EXISTS {self.prefix[:-1]}")
                    for name, definition in TABLES.items():
                        await tx.execute(
                            f"CREATE TABLE {self.table(name)} ({definition.format(json='JSONB', blob='BYTEA', schema=self.prefix)})"
                        )
                    await tx.execute(
                        f"CREATE INDEX runs_session ON {self.table('runs')}(session_id)"
                    )
                    await tx.execute(
                        f"INSERT INTO {self.table('schema_versions')} VALUES (?,?)",
                        (1, MIGRATION_CHECKSUM),
                    )
                await self._migrate_journal(tx, validate_only=validate_only)
        except SchemaError:
            raise
        except Exception as exc:
            code = getattr(exc, "sqlstate", "")
            if code == "42501":
                raise SchemaError(
                    "Insufficient PostgreSQL privileges: schema provisioning needs database CREATE; migrations need ownership of Harness objects"
                ) from None
            raise SchemaError(
                "Schema provisioning failed; conflicting objects or unavailable database (SQLSTATE "
                + str(code)
                + ")"
            ) from None

    @asynccontextmanager
    async def transaction(self):
        async with self.pool.connection() as conn:
            async with conn.transaction():
                yield PostgresTransaction(conn)

    @retry_transaction
    async def claim(self, session_id):
        owner = str(uuid.uuid4())
        async with self.transaction() as tx:
            row = await tx.one(
                f"UPDATE {self.table('sessions')} SET owner=?,generation=generation+1,lease_until=EXTRACT(EPOCH FROM clock_timestamp())+? WHERE id=? AND (owner IS NULL OR lease_until<EXTRACT(EPOCH FROM clock_timestamp())) RETURNING generation",
                (owner, self.lease_seconds, session_id),
            )
            if row is None:
                raise SessionBusyError(
                    "Session is owned by another worker or does not exist"
                )
            return (owner, row[0])

    async def _fence(self, tx, session_id, lease):
        owner, generation = lease
        row = await tx.one(
            f"SELECT id FROM {self.table('sessions')} WHERE id=? AND owner=? AND generation=? AND lease_until>EXTRACT(EPOCH FROM clock_timestamp()) FOR UPDATE",
            (session_id, owner, generation),
        )
        if row is None:
            raise LeaseLostError("Session lease expired or was superseded")

    @retry_transaction
    async def renew(self, session_id, lease):
        async with self.transaction() as tx:
            await self._fence(tx, session_id, lease)
            await tx.execute(
                f"UPDATE {self.table('sessions')} SET lease_until=EXTRACT(EPOCH FROM clock_timestamp())+? WHERE id=?",
                (self.lease_seconds, session_id),
            )

    @retry_transaction
    async def release(self, session_id, lease):
        async with self.transaction() as tx:
            await tx.execute(
                f"UPDATE {self.table('sessions')} SET owner=NULL,lease_until=0 WHERE id=? AND owner=? AND generation=?",
                (session_id, *lease),
            )

    async def put_artifact(self, content):
        if self._artifact_store is not None:
            return await self._artifact_store.put_artifact(content)
        return await super().put_artifact(content)

    async def get_artifact(self, key):
        if self._artifact_store is not None:
            return await self._artifact_store.get_artifact(key)
        return await super().get_artifact(key)

    async def aclose(self):
        self._initialized = False
        if self.pool:
            await self.pool.close()
            self.pool = None


def postgres_bindings(sql: str) -> str:
    """Translate our qmark parameters without rewriting quoted identifiers/literals."""
    result = []
    quote = None
    index = 0
    while index < len(sql):
        char = sql[index]
        if quote:
            result.append("%%" if char == "%" else char)
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    result.append(quote)
                    index += 1
                else:
                    quote = None
        elif char in ('"', "'"):
            quote = char
            result.append(char)
        elif char == "?":
            result.append("%s")
        else:
            result.append("%%" if char == "%" else char)
        index += 1
    return "".join(result)
