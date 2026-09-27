"""Portable flight recordings and offline inspection; never executes bundle code."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import stat
import tempfile
from typing import Any, Callable
import zipfile
import zlib

from .execution import RunEvent


# Persisted v1 protocol identifier; independent of the distribution/import name.
FORMAT = "harness-x.incident"
VERSION = 1
_ARTIFACT = re.compile(r"artifact://([a-f0-9]{64})(?![a-f0-9])")
_KINDS = {
    "run.started",
    "run.resumed",
    "checkpoint",
    "event",
    "command.failed",
    "model.started",
    "model.response",
    "model.processed",
    "model.completed",
    "model.failed",
    "model.interrupted",
    "tool.started",
    "tool.dispatched",
    "tool.returned",
    "tool.processed",
    "tool.failed",
    "tool.interrupted",
    "tool.resolved",
}
_SECRET_KEYS = {
    "authorization",
    "proxyauthorization",
    "apikey",
    "password",
    "secret",
    "accesstoken",
    "refreshtoken",
    "clientsecret",
    "cookie",
    "setcookie",
}


class IncidentError(ValueError):
    """Malformed, oversized, unsupported or corrupted incident bundle."""


@dataclass(frozen=True)
class BundleLimits:
    max_bytes: int = 64 * 1024 * 1024
    max_member_bytes: int = 32 * 1024 * 1024
    max_records: int = 100000
    max_members: int = 10000

    def __post_init__(self):
        for value in (
            self.max_bytes,
            self.max_member_bytes,
            self.max_records,
            self.max_members,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("Bundle limits must be positive integers")


@dataclass(frozen=True)
class ExportPolicy:
    """Payload and artifact disclosure are opt-in, separately from recording.

    `redact(kind, payload)` operates on a copy and may return None to omit it.
    Common credential keys are redacted automatically. Secrets in prose/binary
    data require application policy; this is not a general DLP system.
    """

    include_payloads: bool = False
    include_artifacts: bool = False
    redact: Callable[[str, Any], Any] | None = None

    def __post_init__(self):
        if (
            type(self.include_payloads) is not bool
            or type(self.include_artifacts) is not bool
        ):
            raise TypeError("Export policy flags must be bool")
        if self.include_artifacts and not self.include_payloads:
            raise ValueError("Artifact export also requires include_payloads=True")
        if self.redact is not None and not callable(self.redact):
            raise TypeError("redact must be callable")


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    complete: bool
    record_count: int = 0
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class Playback:
    manifest: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    report: VerificationReport
    artifacts: dict[str, bytes] = field(default_factory=dict, repr=False)


def _json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IncidentError("Duplicate JSON key")
        result[key] = value
    return result


def _load(data):
    def invalid_constant(value):
        raise IncidentError("Non-finite JSON number")

    return json.loads(data, object_pairs_hook=_pairs, parse_constant=invalid_constant)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _refs(value):
    return set(_ARTIFACT.findall(_json(value).decode("utf-8")))


def _redact(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if re.sub(r"[^a-z]", "", key.lower()) in _SECRET_KEYS
            else _redact(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact(child) for child in value]
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            parsed = _load(value)
            redacted = _redact(parsed)
            if parsed != redacted:
                return _json(redacted).decode("utf-8")
        except (ValueError, RecursionError):
            pass
    return value


async def export_incident(
    backend,
    session_id: str,
    run_id: str,
    destination: str | os.PathLike,
    *,
    policy: ExportPolicy | None = None,
    limits: BundleLimits | None = None,
) -> Path:
    """Export from an initialized store without starting/resuming an agent.

    Reads are scoped to session_id. The application authenticates and authorizes
    access to that session; possession of IDs is not an authorization mechanism.
    """
    policy = policy if policy is not None else ExportPolicy()
    limits = limits if limits is not None else BundleLimits()
    if not isinstance(policy, ExportPolicy) or not isinstance(limits, BundleLimits):
        raise TypeError("Expected ExportPolicy and BundleLimits")
    state, source, session = await backend.incident_snapshot(
        session_id,
        run_id,
        max_records=limits.max_records,
    )
    records = []
    for item in source:
        item = copy.deepcopy(item)
        original = item["payload"]
        payload = None
        capture = "omitted"
        if policy.include_payloads:
            payload = _redact(copy.deepcopy(original))
            if policy.redact:
                payload = policy.redact(item["kind"], payload)
            # A redactor must not introduce credential-shaped values either.
            payload = _redact(payload)
            capture = (
                "omitted"
                if payload is None
                else "present"
                if payload == original
                else "redacted"
            )
        item.update(payload=payload, capture=capture)
        records.append(item)

    members = {"records.jsonl": b"".join(_json(item) + b"\n" for item in records)}
    artifacts = {}
    for key in sorted(_refs(records)):
        entry = {"status": "omitted", "reason": "export_policy"}
        if policy.include_artifacts:
            try:
                data = await backend.get_artifact(key)
            except Exception:
                # Do not serialize connection exceptions/credentials into a bundle.
                entry = {"status": "missing", "reason": "artifact_unavailable"}
            else:
                if _digest(data) != key:
                    entry = {"status": "corrupt", "reason": "artifact_checksum"}
                else:
                    path = f"artifacts/{key}"
                    members[path] = data
                    entry = {"status": "included", "path": path}
        artifacts[key] = entry

    try:
        from importlib.metadata import version

        sdk_version = version("harnessx")
    except Exception:
        sdk_version = "unknown"
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "session_id": session_id,
        "status": state["status"],
        "phase": state["phase"],
        "agent": session.get("agent"),
        "sdk_version": sdk_version,
        "python_version": platform.python_version(),
        "record_count": len(records),
        "last_sequence": records[-1]["seq"] if records else 0,
        "unresolved_operations": [
            {
                "execution_key": tool["execution_key"],
                "status": tool["status"],
                "attempt": tool["attempt"],
            }
            for tool in state.get("tools", [])
            if tool["status"] in ("started", "uncertain")
        ],
        "artifacts": artifacts,
        "members": {
            name: {"sha256": _digest(data), "size": len(data)}
            for name, data in members.items()
        },
    }
    members["manifest.json"] = _json(manifest)
    return await asyncio.to_thread(_write_bundle, Path(destination), members, limits)


def _write_bundle(path, members, limits):
    if len(members) > limits.max_members or any(
        len(data) > limits.max_member_bytes for data in members.values()
    ):
        raise IncidentError("Incident exceeds member limits")
    if sum(map(len, members.values())) > limits.max_bytes:
        raise IncidentError("Incident exceeds byte limit")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    content = output.getvalue()
    if len(content) > limits.max_bytes:
        raise IncidentError("Incident exceeds byte limit")
    # Also validate our own output before making it visible to the caller.
    _read_bundle(io.BytesIO(content), limits)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic, exclusive publication: never overwrite an existing incident.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _require(condition, message):
    if not condition:
        raise IncidentError(message)


def _read_bundle(path, limits):
    try:
        if not isinstance(path, io.BytesIO):
            _require(
                Path(path).stat().st_size <= limits.max_bytes,
                "Incident exceeds byte limit",
            )
        with zipfile.ZipFile(path) as bundle:
            infos = bundle.infolist()
            _require(len(infos) <= limits.max_members, "Too many bundle members")
            _require(
                len({item.filename for item in infos}) == len(infos),
                "Duplicate bundle member",
            )
            _require(
                sum(item.file_size for item in infos) <= limits.max_bytes,
                "Expanded incident exceeds byte limit",
            )
            for item in infos:
                _require(
                    item.file_size <= limits.max_member_bytes,
                    "Bundle member exceeds byte limit",
                )
                _require(
                    item.filename in ("manifest.json", "records.jsonl")
                    or re.fullmatch(r"artifacts/[a-f0-9]{64}", item.filename),
                    "Unexpected bundle path",
                )
                _require(not item.flag_bits & 1, "Encrypted members are not supported")
                _require(
                    not stat.S_ISLNK(item.external_attr >> 16),
                    "Symlink members are not supported",
                )
                _require(
                    item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                    "Unsupported compression",
                )
            names = {item.filename for item in infos}
            _require(
                {"manifest.json", "records.jsonl"} <= names,
                "Missing bundle manifest or records",
            )
            manifest = _load(bundle.read("manifest.json"))
            _require(isinstance(manifest, dict), "Invalid manifest")
            _require(
                manifest.get("format") == FORMAT
                and type(manifest.get("version")) is int
                and manifest["version"] == VERSION,
                "Unsupported incident format/version",
            )
            for key in ("run_id", "session_id", "status", "phase"):
                _require(
                    isinstance(manifest.get(key), str) and bool(manifest[key]),
                    "Invalid manifest identity/state",
                )
            _require(
                manifest["status"]
                in (
                    "running",
                    "paused",
                    "awaiting_input",
                    "completed",
                    "failed",
                    "cancelled",
                ),
                "Unknown run status",
            )
            index = manifest.get("members")
            _require(
                isinstance(index, dict) and set(index) == names - {"manifest.json"},
                "Manifest member set mismatch",
            )
            content = {}
            for name, descriptor in index.items():
                _require(isinstance(descriptor, dict), "Invalid member descriptor")
                data = bundle.read(name)
                _require(
                    type(descriptor.get("size")) is int
                    and descriptor["size"] == len(data),
                    "Member size mismatch",
                )
                _require(
                    descriptor.get("sha256") == _digest(data),
                    "Member checksum mismatch",
                )
                content[name] = data

        lines = content["records.jsonl"].splitlines()
        _require(len(lines) <= limits.max_records, "Too many incident records")
        records = tuple(_load(line) for line in lines)
        _require(
            type(manifest.get("record_count")) is int
            and manifest["record_count"] == len(records),
            "Record count mismatch",
        )
        _require(
            type(manifest.get("last_sequence")) is int
            and manifest["last_sequence"] == len(records),
            "Last sequence mismatch",
        )
        ids = set()
        for sequence, item in enumerate(records, 1):
            _require(isinstance(item, dict), "Invalid journal record")
            _require(
                type(item.get("version")) is int and item["version"] == 1,
                "Unsupported journal version",
            )
            _require(
                type(item.get("seq")) is int and item["seq"] == sequence,
                "Journal sequence gap or reordering",
            )
            _require(
                isinstance(item.get("id"), str)
                and item["id"]
                and item["id"] not in ids,
                "Missing or duplicate record identity",
            )
            ids.add(item["id"])
            _require(
                item.get("run_id") == manifest["run_id"]
                and item.get("session_id") == manifest["session_id"],
                "Mixed run/session identities",
            )
            _require(
                isinstance(item.get("kind"), str) and item["kind"] in _KINDS,
                "Unknown journal kind",
            )
            _require(
                item.get("capture") in ("present", "redacted", "omitted"),
                "Invalid capture status",
            )
            _require(
                "payload" in item
                and (item["capture"] != "omitted" or item["payload"] is None),
                "Invalid omitted payload",
            )
            if item["capture"] == "present":
                _require(isinstance(item["payload"], dict), "Invalid journal payload")
            for key in ("step_id", "attempt_id", "recorded_at"):
                _require(isinstance(item.get(key), str), "Missing record metadata")
            _require(
                item.get("execution_key") is None
                or isinstance(item["execution_key"], str),
                "Invalid execution key",
            )
            if item["kind"].startswith("tool."):
                _require(bool(item.get("execution_key")), "Missing tool execution key")
            if item["capture"] == "present" and item["kind"] == "event":
                event = RunEvent.from_dict(item["payload"])
                _require(
                    event.run_id == manifest["run_id"]
                    and event.session_id == manifest["session_id"],
                    "Mixed event identity",
                )

        catalog = manifest.get("artifacts")
        _require(
            isinstance(catalog, dict) and set(catalog) == _refs(records),
            "Artifact reference/catalog mismatch",
        )
        artifacts = {}
        for key, descriptor in catalog.items():
            _require(isinstance(descriptor, dict), "Invalid artifact descriptor")
            if descriptor.get("status") == "included":
                member = f"artifacts/{key}"
                _require(
                    descriptor.get("path") == member and member in content,
                    "Missing declared artifact",
                )
                _require(_digest(content[member]) == key, "Artifact checksum mismatch")
                artifacts[key] = content[member]
            else:
                _require(
                    descriptor.get("status") in ("missing", "corrupt", "omitted"),
                    "Unknown artifact status",
                )
                _require(
                    f"artifacts/{key}" not in content, "Excluded artifact is present"
                )
        _require(
            set(content)
            == {"records.jsonl", *(f"artifacts/{key}" for key in artifacts)},
            "Unreferenced bundle artifact",
        )

        unresolved = manifest.get("unresolved_operations")
        _require(isinstance(unresolved, list), "Invalid unresolved operations")
        for operation in unresolved:
            _require(
                isinstance(operation, dict)
                and isinstance(operation.get("execution_key"), str)
                and operation.get("status") in ("started", "uncertain")
                and type(operation.get("attempt")) is int,
                "Invalid unresolved operation",
            )
        issues = _completeness(manifest, records)
        return Playback(
            manifest,
            records,
            VerificationReport(True, not issues, len(records), tuple(issues)),
            artifacts,
        )
    except IncidentError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RuntimeError,
        zipfile.BadZipFile,
        zlib.error,
    ) as exc:
        raise IncidentError(f"Invalid incident bundle ({type(exc).__name__})") from None


def _completeness(manifest, records):
    issues = []
    if not records or records[0]["kind"] != "run.started":
        issues.append("initial_run_record_missing")
    if any(item["capture"] != "present" for item in records):
        issues.append("payloads_omitted_or_redacted")
    if any(item["status"] != "included" for item in manifest["artifacts"].values()):
        issues.append("artifacts_unavailable")
    if manifest["status"] not in ("completed", "failed", "cancelled"):
        issues.append("run_not_terminal_at_export")
    if manifest["unresolved_operations"]:
        issues.append("unresolved_tool_outcome")
    models, tools = set(), set()
    model_boundaries, tool_boundaries = {}, {}
    gaps = set()
    checkpoint_seen = False
    last_checkpoint = None
    for item in records:
        kind, payload = item["kind"], item["payload"]
        model = (item["step_id"], item["attempt_id"])
        tool = (item["execution_key"], item["attempt_id"])
        if kind == "model.started":
            models.add(model)
            model_boundaries[model] = {"started"}
        elif kind in ("model.response", "model.processed"):
            boundaries = model_boundaries.setdefault(model, set())
            required = "started" if kind == "model.response" else "response"
            if required not in boundaries:
                gaps.add("model_boundaries_missing")
            boundaries.add(kind.split(".")[1])
        elif kind in ("model.completed", "model.failed"):
            if kind == "model.completed" and model_boundaries.get(model) != {
                "started",
                "response",
                "processed",
            }:
                gaps.add("model_boundaries_missing")
            models.discard(model)
        elif kind == "tool.started":
            tool_boundaries[tool] = {"started"}
        elif kind == "tool.dispatched":
            if "started" not in tool_boundaries.get(tool, set()):
                gaps.add("tool_boundaries_missing")
            tools.add(tool)
        elif kind == "tool.returned":
            boundaries = tool_boundaries.setdefault(tool, set())
            if "started" not in boundaries:
                gaps.add("tool_boundaries_missing")
            boundaries.add("returned")
            tools.discard(tool)
        elif kind == "tool.processed":
            if not {"returned", "resolved"} & tool_boundaries.get(tool, set()):
                gaps.add("tool_boundaries_missing")
        elif (
            kind == "tool.resolved"
            and item["capture"] == "present"
            and isinstance(payload, dict)
            and payload.get("result") is not None
        ):
            tools.discard(tool)
            tool_boundaries.setdefault(tool, set()).add("resolved")
        if kind == "checkpoint":
            checkpoint_seen = True
            # Never compare an earlier unredacted checkpoint with the export's
            # final state when the latest checkpoint was deliberately withheld.
            last_checkpoint = payload if item["capture"] == "present" else None
    if not checkpoint_seen:
        issues.append("checkpoint_missing")
    issues.extend(sorted(gaps))
    if models:
        issues.append("model_attempt_incomplete")
    if tools:
        issues.append("tool_attempt_outcome_unresolved")
    if last_checkpoint is not None:
        _require(isinstance(last_checkpoint, dict), "Invalid checkpoint payload")
        _require(
            last_checkpoint.get("status") == manifest["status"]
            and last_checkpoint.get("phase") == manifest["phase"],
            "Manifest/checkpoint state mismatch",
        )
        expected = [
            {key: tool[key] for key in ("execution_key", "status", "attempt")}
            for tool in last_checkpoint.get("tools", [])
            if tool["status"] in ("started", "uncertain")
        ]
        _require(
            expected == manifest["unresolved_operations"],
            "Unresolved outcome metadata mismatch",
        )
    return issues


class IncidentRecorder:
    """Inspect bundles offline. No providers, tools, credentials or runtime needed."""

    def __init__(self, *, limits: BundleLimits | None = None):
        if limits is not None and not isinstance(limits, BundleLimits):
            raise TypeError("Expected BundleLimits")
        self.limits = limits or BundleLimits()

    async def verify(self, bundle: str | os.PathLike) -> VerificationReport:
        try:
            playback = await asyncio.to_thread(_read_bundle, bundle, self.limits)
            return playback.report
        except IncidentError as exc:
            return VerificationReport(False, False, issues=(str(exc),))

    async def playback(self, bundle: str | os.PathLike) -> Playback:
        return await asyncio.to_thread(_read_bundle, bundle, self.limits)
