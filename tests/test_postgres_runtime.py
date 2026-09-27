"""Real PostgreSQL integration tests. Set HARNESS_TEST_POSTGRES_DSN to enable."""

import asyncio
import os
import uuid
import pytest

from harnessx import (
    Agent,
    AgentRuntime,
    PostgresBackend,
    SchemaError,
    LeaseLostError,
    SessionBusyError,
    ProviderResponse,
    ExportPolicy,
    IncidentRecorder,
)
from harnessx.providers import LLMProvider

DSN = os.environ.get("HARNESS_TEST_POSTGRES_DSN")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DSN, reason="HARNESS_TEST_POSTGRES_DSN is not configured"),
]


class P(LLMProvider):
    async def create(self, **kwargs):
        return ProviderResponse(text="postgres works")

    async def count_tokens(self, **kwargs):
        return 0


async def cleanup(store):
    # Every test owns a unique disposable schema.
    if store.pool:
        async with store.transaction() as tx:
            await tx.execute(f"DROP SCHEMA {store.prefix[:-1]} CASCADE")
    await store.aclose()


async def test_concurrent_provisioning_and_roundtrip():
    schema = "harness_test_" + uuid.uuid4().hex
    stores = [PostgresBackend(DSN, schema=schema) for _ in range(3)]
    try:
        await asyncio.gather(*(s.initialize() for s in stores))
        runtime = AgentRuntime(Agent(provider=P()), backend=stores[0])
        await runtime.start()
        result = await runtime.run("hello", request_id="request1")
        assert result.output == "postgres works"
        assert (
            await runtime.run("hello", request_id="request1")
        ).run_id == result.run_id
        loaded = await stores[1].get_run(result.run_id)
        assert loaded["status"] == "completed"
        events = await stores[1].read_events(result.run_id)
        assert events[-1].data.output == "postgres works"
        key = await stores[0].put_artifact(b"payload")
        assert await stores[2].get_artifact(key) == b"payload"
        await runtime.stop()
    finally:
        for s in stores[1:]:
            await s.aclose()
        await cleanup(stores[0])


async def test_schema_validation_and_custom_identifier():
    schema = 'harness test "' + uuid.uuid4().hex
    store = PostgresBackend(DSN, schema=schema)
    try:
        await store.initialize()
        check = PostgresBackend(DSN, schema=schema, auto_migrate=False)
        await check.initialize()
        await check.aclose()
        async with store.transaction() as tx:
            await tx.execute(
                f"UPDATE {store.table('schema_versions')} SET checksum='wrong'"
            )
        with pytest.raises(SchemaError, match="checksum"):
            await check.initialize()
    finally:
        await cleanup(store)


async def test_recorded_run_export(tmp_path):
    store = PostgresBackend(DSN, schema="harness_test_" + uuid.uuid4().hex)
    runtime = AgentRuntime(Agent(provider=P()), backend=store, recording=True)
    try:
        await runtime.start()
        result = await runtime.run("record this")
        bundle = await runtime.export_incident(
            result.run_id, destination=tmp_path / "postgres.hx",
            policy=ExportPolicy(include_payloads=True),
        )
        playback = await IncidentRecorder().playback(bundle)
        assert playback.report.valid and playback.report.complete
        assert len([record for record in playback.records if record["kind"] == "model.response"]) == 1
    finally:
        await runtime.stop()
        await cleanup(store)


async def test_fencing_excludes_stale_writer():
    store = PostgresBackend(DSN, schema="harness_test_" + uuid.uuid4().hex)
    try:
        await store.initialize()
        await store.create_session("s", {"version": 1})
        old = await store.claim("s")
        with pytest.raises(SessionBusyError):
            await store.claim("s")
        async with store.transaction() as tx:
            await tx.execute(
                f"UPDATE {store.table('sessions')} SET lease_until=0 WHERE id='s'"
            )
        new = await store.claim("s")
        with pytest.raises(LeaseLostError):
            await store.renew("s", old)
        await store.release("s", old)
        await store.renew("s", new)
        await store.release("s", new)
    finally:
        await cleanup(store)


async def test_migration_failure_rolls_back_and_existing_objects_are_not_adopted():
    store = PostgresBackend(DSN, schema="harness_test_" + uuid.uuid4().hex)
    try:
        await store.initialize()
        async with store.transaction() as tx:
            await tx.execute(f"DROP TABLE {store.table('schema_versions')}")
        await store.aclose()
        with pytest.raises(SchemaError, match="conflicting"):
            await store.initialize()
        # Reopen only for cleanup; provisioning must not remove existing data.
        from psycopg_pool import AsyncConnectionPool

        store.pool = AsyncConnectionPool(DSN, open=False)
        await store.pool.open()
        async with store.transaction() as tx:
            assert await tx.one(f"SELECT COUNT(*) FROM {store.table('sessions')}") == (
                0,
            )
            assert not (
                await tx.one(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_schema=? AND table_name=?)",
                    (store.schema, "schema_versions"),
                )
            )[0]
    finally:
        await cleanup(store)


async def test_validation_only_does_not_create_schema():
    store = PostgresBackend(
        DSN, schema="harness_test_" + uuid.uuid4().hex, auto_migrate=False
    )
    with pytest.raises(SchemaError, match="missing"):
        await store.initialize()
    await store.migrate()
    await cleanup(store)


async def test_missing_schema_privilege_is_actionable_and_redacted():
    from psycopg.conninfo import make_conninfo

    admin = PostgresBackend(DSN, schema="harness_test_" + uuid.uuid4().hex)
    role = "harness_limited_" + uuid.uuid4().hex
    password = "test_" + uuid.uuid4().hex
    limited = None
    try:
        await admin.initialize()
        async with admin.transaction() as tx:
            # These identifiers are generated exclusively by this test.
            await tx.execute(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{password}'")
        limited = PostgresBackend(
            make_conninfo(DSN, user=role, password=password),
            schema="harness_missing_" + uuid.uuid4().hex,
        )
        with pytest.raises(SchemaError, match="privileges") as error:
            await limited.initialize()
        assert password not in str(error.value)
        assert password not in repr(limited)
    finally:
        if limited:
            await limited.aclose()
        if admin.pool:
            async with admin.transaction() as tx:
                await tx.execute(f'DROP ROLE IF EXISTS "{role}"')
        await cleanup(admin)
