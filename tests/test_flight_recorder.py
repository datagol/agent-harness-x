"""Recorder contracts: fault boundaries, offline disclosure, and untrusted bundles."""

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import textwrap
import zipfile

import pytest

from harnessx import (
    Agent,
    AgentConfig,
    AgentRef,
    AgentRuntime,
    BundleLimits,
    ExportPolicy,
    IncidentError,
    IncidentRecorder,
    Middleware,
    PermissionLevel,
    ProviderResponse,
    ResultSpillExtension,
    SchemaError,
    SQLiteBackend,
    StorageError,
    StreamChunk,
    ToolCall,
    ToolResult,
    export_incident,
)
from harnessx.providers import LLMProvider


class Provider(LLMProvider):
    def __init__(self, responses=None):
        self.responses = list(responses or [ProviderResponse(text="done")])
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def count_tokens(self, **kwargs):
        return 0


def call(name="effect", inputs=None):
    return ProviderResponse(
        tool_calls=[ToolCall("t", name, inputs or {})],
        stop_reason="tool_use",
    )


@asynccontextmanager
async def session(
    tmp_path, *, provider=None, configure=None, recording=True, extensions=None, config=None
):
    agent = Agent(config=config, provider=provider or Provider(), extensions=extensions)
    if configure:
        configure(agent)
    store = SQLiteBackend(tmp_path / "runtime.db")
    runtime = AgentRuntime(agent, backend=store, recording=recording)
    await runtime.start()
    try:
        yield runtime, store
    finally:
        await runtime.stop()
        await store.aclose()


async def export(runtime, result, tmp_path, name="incident.hx", **policy):
    return await runtime.export_incident(
        result.run_id,
        destination=tmp_path / name,
        policy=ExportPolicy(include_payloads=True, **policy),
    )


def of_kind(playback, kind):
    return [item for item in playback.records if item["kind"] == kind]


def rewrite(
    source,
    destination,
    *,
    change_records=None,
    change_manifest=None,
    extra=None,
    rehash=True,
):
    with zipfile.ZipFile(source) as bundle:
        members = {name: bundle.read(name) for name in bundle.namelist()}
    manifest = json.loads(members["manifest.json"])
    if change_records:
        records = [json.loads(line) for line in members["records.jsonl"].splitlines()]
        change_records(records)
        members["records.jsonl"] = b"".join(
            json.dumps(item).encode() + b"\n" for item in records
        )
    if rehash:
        for name, data in members.items():
            if name != "manifest.json":
                manifest["members"][name] = {
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
    if change_manifest:
        change_manifest(manifest)
    members["manifest.json"] = json.dumps(manifest).encode()
    if extra:
        members.update(extra)
    with zipfile.ZipFile(destination, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return destination


@pytest.mark.asyncio
async def test_raw_and_processed_are_private_and_offline(tmp_path, monkeypatch):
    class Transform(Middleware):
        async def after_llm_call(self, response):
            response.text = "approved output"
            return response

    provider = Provider([ProviderResponse(text="private raw output")])
    async with session(
        tmp_path, provider=provider, configure=lambda a: a.middleware.add(Transform())
    ) as (runtime, store):
        result = await runtime.execute("hello")
        assert result.output == "approved output"
        public = await store.read_events(result.run_id)
        assert "private raw output" not in str(public)
        bundle = await export(runtime, result, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline playback must not invoke live dependencies")

    monkeypatch.setattr(Agent, "__init__", forbidden)
    monkeypatch.setattr(Provider, "create", forbidden)
    import socket

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    playback = await IncidentRecorder().playback(bundle)
    assert playback.report.valid and playback.report.complete
    assert (
        of_kind(playback, "model.response")[0]["payload"]["text"]
        == "private raw output"
    )
    assert (
        of_kind(playback, "model.processed")[0]["payload"]["text"] == "approved output"
    )
    assert provider.calls == 1
    assert [item["seq"] for item in playback.records] == list(
        range(1, len(playback.records) + 1)
    )


@pytest.mark.asyncio
async def test_retry_retains_distinct_model_attempts(tmp_path):
    provider = Provider(
        [ConnectionError("lost response"), ProviderResponse(text="recovered")]
    )
    # Provider-level retry is off so the failure reaches the durable attempt journal.
    async with session(
        tmp_path, provider=provider, config=AgentConfig(llm_max_attempts=1)
    ) as (runtime, _):
        result = await runtime.execute("hello")
        bundle = await export(runtime, result, tmp_path)
    playback = await IncidentRecorder().playback(bundle)
    assert playback.report.complete
    assert [item["attempt_id"] for item in of_kind(playback, "model.started")] == [
        "1",
        "2",
    ]
    assert of_kind(playback, "model.failed")[0]["attempt_id"] == "1"
    assert of_kind(playback, "model.response")[0]["attempt_id"] == "2"


@pytest.mark.asyncio
async def test_concurrent_tools_keep_separate_ordered_outcomes(tmp_path):
    effects = []

    def configure(agent):
        @agent.tools.register(permission=PermissionLevel.ALLOW, concurrent=True)
        async def effect(value: int):
            await asyncio.sleep(0.02 if value == 1 else 0)
            effects.append(value)
            return str(value)

    provider = Provider(
        [
            ProviderResponse(
                tool_calls=[
                    ToolCall("a", "effect", {"value": 1}),
                    ToolCall("b", "effect", {"value": 2}),
                ],
                stop_reason="tool_use",
            ),
            ProviderResponse(text="done"),
        ]
    )
    async with session(tmp_path, provider=provider, configure=configure) as (
        runtime,
        _,
    ):
        result = await runtime.execute("parallel")
        playback = await IncidentRecorder().playback(
            await export(runtime, result, tmp_path)
        )
    assert playback.report.complete and effects == [2, 1]
    returned = of_kind(playback, "tool.returned")
    assert [item["payload"]["content"] for item in returned] == ["2", "1"]
    assert len({item["execution_key"] for item in returned}) == 2


@pytest.mark.asyncio
async def test_ambiguous_commit_preserves_one_return_record_and_no_repeated_effect(
    tmp_path,
):
    effects = []

    def configure(agent):
        agent.tools.register(name="effect", permission=PermissionLevel.ALLOW)(
            lambda: effects.append(1) or "written"
        )

    async with session(
        tmp_path,
        provider=Provider([call(), ProviderResponse(text="done")]),
        configure=configure,
    ) as (runtime, store):
        original, interrupted = store.save_run, []

        async def ambiguous(state, events, lease, *, records=()):
            await original(state, events, lease, records=records)
            if not interrupted and any(
                item["kind"] == "tool.returned" for item in records
            ):
                interrupted.append(True)
                raise ConnectionError("commit succeeded but response was lost")

        store.save_run = ambiguous
        failed = await runtime.execute("act")
        assert failed.status == "failed" and effects == [1]
        result = await (await runtime.resume(runtime.session_id)).result()
        playback = await IncidentRecorder().playback(
            await export(runtime, result, tmp_path)
        )
        assert playback.report.complete and effects == [1]
        assert len(of_kind(playback, "tool.returned")) == 1


@pytest.mark.asyncio
async def test_conflicting_journal_record_rolls_back_state_update(tmp_path):
    async with session(tmp_path) as (runtime, store):
        result = await runtime.execute("hello")
        state, records, _ = await store.incident_snapshot(
            runtime.session_id, result.run_id
        )
        conflict = dict(records[0])
        conflict.pop("seq")
        conflict["payload"] = {"message": "overwrite"}
        state["status"] = "failed"
        lease = await store.claim(runtime.session_id)
        try:
            with pytest.raises(StorageError, match="immutable"):
                await store.save_run(state, [], lease, records=[conflict])
        finally:
            await store.release(runtime.session_id, lease)
        assert (await store.get_run(result.run_id))["status"] == "completed"
        assert (await store.incident_snapshot(runtime.session_id, result.run_id))[
            1
        ] == records


@pytest.mark.asyncio
async def test_raw_tool_recovery_and_approval_journal(tmp_path):
    effects, failing = [], [True]

    class Transform(Middleware):
        async def after_tool_execution(self, result):
            if failing[0]:
                raise RuntimeError("middleware unavailable")
            result.content = "processed"
            return result

    def configure(agent):
        agent.middleware.add(Transform())
        agent.tools.register(name="effect", permission=PermissionLevel.ASK)(lambda: effects.append(1) or "raw")

    provider = Provider([call(), ProviderResponse(text="done")])
    async with session(tmp_path, provider=provider, configure=configure) as (
        runtime,
        store,
    ):
        waiting = await runtime.execute("act")
        assert waiting.status == "awaiting_input" and effects == []
        key = waiting.pending[0]["execution_key"]
        await runtime.approve(key)
        failed = await (await runtime.resume(runtime.session_id)).result()
        assert failed.status == "failed" and effects == [1]
        assert (await store.get_run(failed.run_id))["tools"][0][
            "status"
        ] == "raw_completed"
        failing[0] = False
        finished = await (await runtime.resume(runtime.session_id)).result()
        assert finished.status == "completed" and effects == [1]
        bundle = await export(runtime, finished, tmp_path)
    playback = await IncidentRecorder().playback(bundle)
    assert playback.report.complete
    assert len(of_kind(playback, "tool.dispatched")) == 1
    assert of_kind(playback, "tool.returned")[0]["payload"]["content"] == "raw"
    assert of_kind(playback, "tool.processed")[0]["payload"]["content"] == "processed"
    assert of_kind(playback, "tool.resolved")[0]["payload"]["allow"] is True
    assert len(of_kind(playback, "run.resumed")) == 2


@pytest.mark.asyncio
async def test_interrupted_stream_and_live_snapshot_are_incomplete(tmp_path):
    entered = asyncio.Event()

    class Streaming(Provider):
        async def stream(self, **kwargs):
            yield StreamChunk(kind="text_delta", data="partial")
            entered.set()
            await asyncio.Event().wait()

    async with session(tmp_path, provider=Streaming()) as (runtime, _):
        handle = await runtime.submit("stream")
        await asyncio.wait_for(entered.wait(), timeout=3)
        live = await runtime.export_incident(
            handle.run_id,
            destination=tmp_path / "live.hx",
            policy=ExportPolicy(include_payloads=True),
        )
        assert (
            "run_not_terminal_at_export"
            in (await IncidentRecorder().verify(live)).issues
        )
        await runtime.cancel()
        result = await handle.result()
        bundle = await export(runtime, result, tmp_path)
    playback = await IncidentRecorder().playback(bundle)
    assert playback.report.valid and not playback.report.complete
    assert "model_attempt_incomplete" in playback.report.issues
    assert of_kind(playback, "model.interrupted")
    assert any(
        item["payload"].get("data") == "partial" for item in of_kind(playback, "event")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["model.started", "tool.dispatched"])
async def test_required_recording_failure_blocks_dispatch(tmp_path, boundary):
    effects = []
    provider = Provider([call(), ProviderResponse(text="done")])

    def configure(agent):
        agent.tools.register(name="effect", permission=PermissionLevel.ALLOW)(
            lambda: effects.append(1)
        )

    async with session(tmp_path, provider=provider, configure=configure) as (
        runtime,
        store,
    ):
        original, failed = store.save_run, []

        async def unreliable(state, events, lease, *, records=()):
            if not failed and any(item["kind"] == boundary for item in records):
                failed.append(True)
                raise OSError("disk failure")
            return await original(state, events, lease, records=records)

        store.save_run = unreliable
        result = await runtime.execute("act")
        assert result.status in ("failed", "awaiting_input")
        assert effects == [] and provider.calls == (
            0 if boundary == "model.started" else 1
        )
        playback = await IncidentRecorder().playback(
            await export(runtime, result, tmp_path)
        )
        assert len(of_kind(playback, boundary)) == 1
        assert not of_kind(playback, "tool.returned")


@pytest.mark.asyncio
async def test_default_export_omits_payloads_and_custom_redaction_is_copied(tmp_path):
    async with session(
        tmp_path,
        provider=Provider(
            [ProviderResponse(text='{"api_key":"credential","value":7}')]
        ),
    ) as (runtime, store):
        result = await runtime.execute("sensitive business question")
        metadata = await runtime.export_incident(
            result.run_id, destination=tmp_path / "metadata.hx"
        )
        playback = await IncidentRecorder().playback(metadata)
        assert all(
            item["payload"] is None and item["capture"] == "omitted"
            for item in playback.records
        )
        assert not playback.report.complete

        def redact(kind, payload):
            if kind == "run.started":
                payload["message"] = "removed"
            return payload

        bundle = await export(runtime, result, tmp_path, redact=redact)
        with zipfile.ZipFile(bundle) as archive:
            assert b'"credential"' not in archive.read("records.jsonl")
        playback = await IncidentRecorder().playback(bundle)
        assert "[REDACTED]" in of_kind(playback, "model.response")[0]["payload"]["text"]
        assert of_kind(playback, "run.started")[0]["payload"]["message"] == "removed"
        _, original, _ = await store.incident_snapshot(
            runtime.session_id, result.run_id
        )
        assert original[0]["payload"]["message"] == "sensitive business question"
        assert any('"credential"' in str(item["payload"]) for item in original)


@pytest.mark.asyncio
async def test_redacted_final_checkpoint_does_not_validate_against_an_older_state(
    tmp_path,
):
    def redact(kind, payload):
        return (
            None
            if kind == "checkpoint" and payload["status"] == "completed"
            else payload
        )

    async with session(tmp_path) as (runtime, _):
        result = await runtime.execute("hello")
        bundle = await export(runtime, result, tmp_path, redact=redact)
    report = await IncidentRecorder().verify(bundle)
    assert report.valid and not report.complete
    assert "payloads_omitted_or_redacted" in report.issues


@pytest.mark.asyncio
async def test_artifact_export_portability_and_missing_content(tmp_path):
    def configure(agent):
        agent.tools.register(name="effect", permission=PermissionLevel.ALLOW)(
            lambda: "x" * 1000
        )

    extensions = [
        ResultSpillExtension(threshold_bytes=20, spill_root=str(tmp_path / "spills"))
    ]
    async with session(
        tmp_path,
        provider=Provider([call(), ProviderResponse(text="done")]),
        configure=configure,
        extensions=extensions,
    ) as (runtime, store):
        result = await runtime.execute("large result")
        assert result.status == "completed"
        bundle = await export(runtime, result, tmp_path, include_artifacts=True)
        playback = await IncidentRecorder().playback(bundle)
        assert playback.report.complete and list(playback.artifacts.values()) == [
            b"x" * 1000
        ]
        processed = of_kind(playback, "tool.processed")[0]["payload"]["content"]
        result_id = json.loads(processed)["result_id"]
        extensions = of_kind(playback, "checkpoint")[-1]["payload"]["extensions"]
        assert extensions["result_spill"]["results"][result_id]["path"].startswith(
            "artifact://"
        )
        assert str(tmp_path / "spills") not in json.dumps(playback.records)
        excluded = await export(runtime, result, tmp_path, "excluded.hx")
        assert (
            "artifacts_unavailable"
            in (await IncidentRecorder().verify(excluded)).issues
        )
        key = next(iter(playback.artifacts))
        corrupted = tmp_path / "corrupted-artifact.hx"
        with (
            zipfile.ZipFile(bundle) as original,
            zipfile.ZipFile(corrupted, "w") as changed,
        ):
            for name in original.namelist():
                changed.writestr(
                    name,
                    b"tampered" if name == f"artifacts/{key}" else original.read(name),
                )
        assert not (await IncidentRecorder().verify(corrupted)).valid
        # Removing references in export policy must also remove the binary payload.
        redacted = await export(
            runtime,
            result,
            tmp_path,
            "redacted-artifact.hx",
            include_artifacts=True,
            redact=lambda kind, payload: None,
        )
        assert not (await IncidentRecorder().playback(redacted)).artifacts
        async with store.transaction() as tx:
            await tx.execute("DELETE FROM artifacts WHERE id=?", (key,))
        missing = await export(
            runtime, result, tmp_path, "missing.hx", include_artifacts=True
        )
        assert (await IncidentRecorder().playback(missing)).manifest["artifacts"][key][
            "status"
        ] == "missing"


@pytest.mark.asyncio
async def test_portable_raw_file_result_still_gives_middleware_a_local_path(tmp_path):
    class ReadFile(Middleware):
        async def after_tool_execution(self, result):
            result.content = Path(result.content).read_text()
            return result

    def configure(agent):
        agent.middleware.add(ReadFile())

        @agent.tools.register(name="effect", permission=PermissionLevel.ALLOW)
        def effect():
            path = tmp_path / "captured.txt"
            path.write_text("file content")
            agent.memory._artifact_paths.add(str(path))
            return str(path)

    async with session(
        tmp_path,
        provider=Provider([call(), ProviderResponse(text="done")]),
        configure=configure,
    ) as (runtime, _):
        result = await runtime.execute("capture file")
        assert result.status == "completed", result.error
        playback = await IncidentRecorder().playback(
            await export(runtime, result, tmp_path, include_artifacts=True)
        )
        assert of_kind(playback, "tool.returned")[0]["payload"]["content"].startswith(
            "artifact://"
        )
        assert (
            of_kind(playback, "tool.processed")[0]["payload"]["content"]
            == "file content"
        )
        assert playback.report.complete and list(playback.artifacts.values()) == [
            b"file content"
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption", ["checksum", "sequence", "identity", "status", "path", "version"]
)
async def test_bundle_validation_rejects_corruption(tmp_path, corruption):
    async with session(tmp_path) as (runtime, _):
        bundle = await export(runtime, await runtime.execute("hello"), tmp_path)
    options = {
        "checksum": dict(
            change_records=lambda r: r[0].update(step_id="changed"), rehash=False
        ),
        "sequence": dict(change_records=lambda r: r[1].update(seq=1)),
        "identity": dict(
            change_records=lambda r: r[0].update(session_id="someone-else")
        ),
        "status": dict(change_manifest=lambda m: m.update(status="failed")),
        "path": dict(extra={"../escape.py": b"raise RuntimeError('must not execute')"}),
        "version": dict(change_manifest=lambda m: m.update(version=2)),
    }[corruption]
    bad = rewrite(bundle, tmp_path / "bad.hx", **options)
    report = await IncidentRecorder().verify(bad)
    assert not report.valid and not report.complete and report.issues
    with pytest.raises(IncidentError):
        await IncidentRecorder().playback(bad)
    assert not (tmp_path.parent / "escape.py").exists()


@pytest.mark.asyncio
async def test_missing_raw_boundary_cannot_claim_complete_capture(tmp_path):
    async with session(tmp_path) as (runtime, _):
        bundle = await export(runtime, await runtime.execute("hello"), tmp_path)

    def remove_response(records):
        records[:] = [item for item in records if item["kind"] != "model.response"]
        for seq, item in enumerate(records, 1):
            item["seq"] = seq

    def fix_count(manifest):
        manifest["record_count"] -= 1
        manifest["last_sequence"] -= 1

    # Even a coherent archive with recomputed checksums must report this gap.
    missing = rewrite(
        bundle,
        tmp_path / "missing-boundary.hx",
        change_records=remove_response,
        change_manifest=fix_count,
    )
    report = await IncidentRecorder().verify(missing)
    assert (
        report.valid
        and not report.complete
        and "model_boundaries_missing" in report.issues
    )


@pytest.mark.asyncio
async def test_limits_no_overwrite_and_session_scope(tmp_path):
    async with session(tmp_path) as (runtime, store):
        result = await runtime.execute("hello")
        bundle = await export(runtime, result, tmp_path)
        before = bundle.read_bytes()
        with pytest.raises(FileExistsError):
            await export(runtime, result, tmp_path)
        assert bundle.read_bytes() == before
        assert not (
            await IncidentRecorder(limits=BundleLimits(max_bytes=10)).verify(bundle)
        ).valid
        with pytest.raises(IncidentError, match="limit"):
            await runtime.export_incident(
                result.run_id,
                destination=tmp_path / "small.hx",
                limits=BundleLimits(max_member_bytes=10),
            )
        assert not (tmp_path / "small.hx").exists()
        with pytest.raises(StorageError):
            await store.incident_snapshot("another-session", result.run_id)
        with pytest.raises(StorageError, match="record limit"):
            await store.incident_snapshot(
                runtime.session_id, result.run_id, max_records=1
            )


@pytest.mark.asyncio
async def test_recording_off_and_persisted_choice(tmp_path):
    async with session(tmp_path, recording=False) as (runtime, store):
        result = await runtime.execute("hello")
        sid = runtime.session_id
        async with store.transaction() as tx:
            assert (await tx.one("SELECT COUNT(*) FROM journal"))[0] == 0
        with pytest.raises(StorageError, match="not enabled"):
            await export(runtime, result, tmp_path)
    store = SQLiteBackend(tmp_path / "runtime.db")
    runtime = AgentRuntime(Agent(provider=Provider()), backend=store, recording=True)
    try:
        with pytest.raises(ValueError, match="Recording configuration"):
            await runtime.resume(sid)
    finally:
        await runtime._provided_agent.aclose()
        await store.aclose()


@pytest.mark.asyncio
async def test_migrate_v1_preserves_existing_sessions(tmp_path):
    from harnessx.backends.store import TABLES, MIGRATION_CHECKSUM, JOURNAL_CHECKSUM

    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        for name, definition in TABLES.items():
            conn.execute(
                f'CREATE TABLE "{name}" ({definition.format(json="TEXT", blob="BLOB", schema="")})'
            )
        conn.execute(
            "INSERT INTO schema_versions VALUES (?,?)", (1, MIGRATION_CHECKSUM)
        )
        conn.execute(
            "INSERT INTO sessions(id,payload) VALUES (?,?)",
            ("old-session", '{"version":1}'),
        )
    validate = SQLiteBackend(path, auto_migrate=False)
    with pytest.raises(SchemaError, match="migration required"):
        await validate.initialize()
    migrated = SQLiteBackend(path)
    await migrated.initialize()
    assert await migrated.get_session("old-session") == {"version": 1}
    async with migrated.transaction() as tx:
        assert await tx.all("SELECT * FROM schema_versions ORDER BY version") == [
            (1, MIGRATION_CHECKSUM),
            (2, JOURNAL_CHECKSUM),
        ]
    await migrated.aclose()
    await validate.initialize()
    await validate.aclose()


@pytest.mark.asyncio
async def test_real_crash_then_fresh_process_offline_playback(tmp_path):
    script = tmp_path / "crash.py"
    script.write_text(
        textwrap.dedent("""
        import asyncio, os, sys
        from pathlib import Path
        from harnessx import Agent, AgentRuntime, SQLiteBackend, ProviderResponse, ToolCall, PermissionLevel
        from harnessx.providers import LLMProvider
        root = Path(sys.argv[1])
        class P(LLMProvider):
            async def count_tokens(self, **kwargs): return 0
            async def create(self, **kwargs):
                return ProviderResponse(tool_calls=[ToolCall("t", "effect", {})], stop_reason="tool_use")
        async def main():
            agent = Agent(provider=P())
            @agent.tools.register(permission=PermissionLevel.ALLOW)
            def effect():
                (root / "effect").write_text("committed")
                os._exit(42)
            runtime = AgentRuntime(agent, backend=SQLiteBackend(root / "runtime.db"), recording=True)
            (root / "session").write_text(await runtime.start())
            await runtime.execute("act")
        asyncio.run(main())
    """)
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()),
        "LANGSMITH_API_KEY": "",
        "LANGSMITH_TRACING": "false",
        "LANGSMITH_TRACING_V2": "false",
    }
    child = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(script), str(tmp_path)],
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert child.returncode == 42, child.stderr.decode()
    assert (tmp_path / "effect").read_text() == "committed"
    effects = []
    agent = Agent(provider=Provider())
    agent.tools.register(name="effect", permission=PermissionLevel.ALLOW)(
        lambda: effects.append(1)
    )
    store = SQLiteBackend(tmp_path / "runtime.db")
    runtime = AgentRuntime(
        agent, backend=store
    )  # Inherit the persisted recording choice.
    try:
        sid = (tmp_path / "session").read_text()
        await store.initialize()
        crashed = await store.latest_run(sid)
        before_resume = await export_incident(
            store,
            sid,
            crashed["run_id"],
            tmp_path / "before-resume.hx",
            policy=ExportPolicy(include_payloads=True),
        )
        assert (
            "unresolved_tool_outcome"
            in (await IncidentRecorder().verify(before_resume)).issues
        )
        assert effects == [] and agent.provider.calls == 0
        result = await (await runtime.resume(sid)).result()
        assert result.status == "awaiting_input" and effects == [] and runtime.recording
        bundle = await export(runtime, result, tmp_path)
        playback = await IncidentRecorder().playback(bundle)
        assert (
            playback.report.valid
            and "unresolved_tool_outcome" in playback.report.issues
        )
        assert len(of_kind(playback, "tool.dispatched")) == 1 and not of_kind(
            playback, "tool.returned"
        )
        key = result.pending[0]["execution_key"]
        await runtime.resolve_tool(key, result=ToolResult("t", "confirmed externally"))
        finished = await (await runtime.resume(runtime.session_id)).result()
        assert finished.status == "completed" and effects == []
        resolved = await export(runtime, finished, tmp_path, "resolved.hx")
        assert (await IncidentRecorder().verify(resolved)).complete
    finally:
        await runtime.stop()
        await store.aclose()
    offline = textwrap.dedent("""
        import asyncio, json, socket, sys
        from harnessx import Agent, IncidentRecorder
        def forbidden(*a, **kw): raise AssertionError("live execution forbidden")
        Agent.__init__ = forbidden
        socket.socket.connect = forbidden
        async def main():
            playback = await IncidentRecorder().playback(sys.argv[1])
            print(json.dumps([record["id"] for record in playback.records]))
        asyncio.run(main())
    """)
    child = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", offline, str(bundle)],
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert child.returncode == 0, child.stderr.decode()
    assert json.loads(child.stdout) == [item["id"] for item in playback.records]


def test_unsupported_recording_backend_and_policy_validation():
    class Remote:
        remote = True

    with pytest.raises(ValueError, match="SQLite or PostgreSQL"):
        AgentRuntime(AgentRef("registered", "v1"), backend=Remote(), recording=True)
    with pytest.raises(ValueError):
        ExportPolicy(include_artifacts=True)
    with pytest.raises(ValueError):
        BundleLimits(max_bytes=-1)
