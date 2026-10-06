"""NVIDIA OpenShell as the execution backend: the agent stays here, its tools run there.

    agent = Agent(config, sandbox=OpenShellSandbox(project="./repo"))
    await agent.run("Run the tests and fix the failure")   # ./repo now has the edits

``run_bash`` and the file tools run inside an OpenShell sandbox. The loop, the
model calls, memory and hooks stay in this process. Nothing else changes for
the caller: the agent creates the sandbox on the first tool call that needs it,
copies the project in, copies changed files back after every run, and deletes
the sandbox when it closes.

Network access is off unless you allow it. ``allow=["github.com"]`` lets any
command the agent runs reach those hosts; ``secrets=["GITHUB_TOKEN"]`` lets
commands use a secret from your environment without ever seeing it (inside
the sandbox the variable holds a placeholder that OpenShell swaps for the real
value only on requests to the secret's hosts). ``policy=`` and ``providers=``
take OpenShell's own policy YAML and provider names instead.

Requires ``pip install harnessx[openshell]`` and a running OpenShell gateway
(``openshell status``). The OpenShell SDK is synchronous; its calls run in
worker threads.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import io
import json
import logging
import os
import posixpath
import re
import shlex
import stat
import tarfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Sequence

from ._openshell_policy import build_policy, denials_from_log, expand_hosts, load_policy, secret_groups
from .types import SandboxResult

logger = logging.getLogger(__name__)

PROJECT_DIR = "/sandbox/project"
INPUTS_DIR = "/sandbox/inputs"
STATE_DIR = "/sandbox/.harnessx"
MANIFEST = f"{STATE_DIR}/manifest.json"
DEFAULT_EXCLUDE = (".venv", "node_modules", "__pycache__", ".DS_Store")

# One gRPC request carries at most 4 MB; stdin larger than this is uploaded in pieces.
_CHUNK_BYTES = 3 * 1024 * 1024
# The gateway's rule for sandbox names: a DNS label of at most 19 characters.
_SANDBOX_NAME = re.compile(r"(?=.{1,19}$)[a-z0-9](?:[a-z0-9]|-(?=[a-z0-9]))*")
# The gateway reports a command that ran out of time as exit code 124.
_TIMED_OUT = 124
_DENIAL = re.compile(r'"error"\s*:\s*"policy_denied"|not permitted by policy', re.IGNORECASE)
_DENIED_TARGET = re.compile(r'CONNECT\s+(\S+)\s+not permitted|"detail"\s*:\s*"([^"]*)"', re.IGNORECASE)


def _require_sdk() -> Any:
    try:
        import openshell
    except ImportError as exc:
        raise RuntimeError("OpenShellSandbox requires harnessx[openshell] (pip install 'harnessx[openshell]')") from exc
    return openshell


def policy_denials(*outputs: str) -> list[str]:
    """What OpenShell's network policy refused, as reported in command output."""
    denials: list[str] = []
    for output in outputs:
        for line in (output or "").splitlines():
            if not _DENIAL.search(line):
                continue
            match = _DENIED_TARGET.search(line)
            detail = (match.group(1) or match.group(2)) if match else line.strip()
            if detail and detail not in denials:
                denials.append(detail[:200])
    return denials


@dataclass(frozen=True)
class PendingRule:
    """A network rule OpenShell drafted from blocked connections, waiting for a decision."""

    id: str
    rule_name: str
    hosts: list[str]
    binary: str
    rationale: str
    review_token: str = field(repr=False, default="")


@dataclass
class SyncReport:
    """What one sync copied back to the local project."""

    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    # Changed in the sandbox and locally since the last sync. The local file is
    # kept; the sandbox's version is beside it as ``<file>.sandbox``.
    conflicts: list[str] = field(default_factory=list)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _excluded(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _pack(root: str, exclude: Sequence[str]) -> tuple[bytes, dict[str, str]]:
    """A gzipped tar of ``root`` (or the single file it names) and the hash of every regular file."""
    buffer, hashes = io.BytesIO(), {}
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        if not os.path.isdir(root):
            name = os.path.basename(root)
            tar.add(root, arcname=name, recursive=False)
            hashes[name] = _sha256(root)
            return buffer.getvalue(), hashes
        for directory, subdirectories, files in os.walk(root):
            relative_directory = os.path.relpath(directory, root)
            kept = []
            for name in sorted(subdirectories):
                path = os.path.join(directory, name)
                if _excluded(name, exclude):
                    continue
                if os.path.islink(path):
                    files.append(name)  # a link to a directory travels as a link
                else:
                    kept.append(name)
            subdirectories[:] = kept
            if relative_directory != ".":
                tar.add(directory, arcname=relative_directory.replace(os.sep, "/"), recursive=False)
            for name in sorted(files):
                if _excluded(name, exclude):
                    continue
                path = os.path.join(directory, name)
                relative = os.path.normpath(os.path.join(relative_directory, name)).replace(os.sep, "/")
                mode = os.lstat(path).st_mode
                if stat.S_ISREG(mode):
                    hashes[relative] = _sha256(path)
                    tar.add(path, arcname=relative, recursive=False)
                elif stat.S_ISLNK(mode):
                    tar.add(path, arcname=relative, recursive=False)
    return buffer.getvalue(), hashes


class OpenShellSandbox:
    """Run an agent's tools in an NVIDIA OpenShell sandbox.

    Args:
        project: Local folder the agent works on (default: the current
            directory). Copied into the sandbox on first use; changed files
            are copied back after every run. A local file edited during a run
            is never overwritten: the sandbox's version lands beside it as
            ``<file>.sandbox``.
        inputs: Extra local files or folders, copied in read-only and never
            copied back. The model reaches them by their local paths.
        allow: Hosts commands may reach, such as ``"github.com"`` or
            ``"example.com:8443"`` (port 443 by default). Well-known
            companions come along (``pypi.org`` brings
            ``files.pythonhosted.org``). Nothing else is reachable.
        secrets: Environment variables commands may use but never read. A
            list names well-known ones (``GITHUB_TOKEN``, ``HF_TOKEN``,
            ``NPM_TOKEN``, ...); a mapping names the hosts for any other:
            ``{"MY_TOKEN": "api.example.com"}``. Their hosts are allowed too.
        policy: An OpenShell policy (YAML path, YAML text, or mapping) to
            start from instead of HarnessX's restrictive default. ``allow`` and
            ``secrets`` add to it.
        providers: Existing OpenShell providers to attach, by name.
        image: OCI image for the sandbox. None is the gateway's default.
        exclude: Names (shell patterns) left out of both copies. None is
            ``DEFAULT_EXCLUDE``.
        keep: Leave the sandbox running when the agent closes.
        name: Sandbox name. None derives one from the agent's session, so a
            resumed session finds its sandbox again.
        workspace: OpenShell workspace.
        gateway: Gateway name. None is the active one (``OPENSHELL_GATEWAY``
            or the CLI's).
        timeout_seconds: Longest a single command may run.
        client: An ``openshell.SandboxClient`` to use instead of connecting to
            ``gateway``.
    """

    owns_filesystem = True
    # The agent it is handed to starts, syncs and closes it.
    agent_owned = True

    def __init__(
        self,
        project: str | os.PathLike[str] = ".",
        *,
        inputs: Sequence[str | os.PathLike[str]] = (),
        allow: Sequence[str] = (),
        secrets: Sequence[str] | Mapping[str, str | Sequence[str]] = (),
        policy: Any | None = None,
        providers: Sequence[str] = (),
        image: str | None = None,
        exclude: Sequence[str] | None = None,
        keep: bool = False,
        name: str | None = None,
        workspace: str = "default",
        gateway: str | None = None,
        timeout_seconds: int = 30,
        client: Any | None = None,
    ) -> None:
        self.project = os.path.realpath(os.path.expanduser(os.fspath(project)))
        if not os.path.isdir(self.project):
            raise ValueError(f"project must be an existing directory: {project!r}")
        self.inputs: dict[str, str] = {}  # local real path -> sandbox path
        for entry in inputs:
            local = os.path.realpath(os.path.expanduser(os.fspath(entry)))
            if not os.path.exists(local):
                raise ValueError(f"input does not exist: {entry!r}")
            base, target, n = os.path.basename(local) or "input", None, 1
            while target is None or target in self.inputs.values():
                target = f"{INPUTS_DIR}/{base}" if n == 1 else f"{INPUTS_DIR}/{base}-{n}"
                n += 1
            self.inputs[local] = target
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        if isinstance(allow, str) or isinstance(providers, str):
            raise TypeError("allow and providers take a list, such as allow=['github.com']")
        self.secret_groups = secret_groups(secrets) if secrets else []
        missing = [name for names, _ in self.secret_groups for name in names if not os.environ.get(name)]
        if missing:
            raise ValueError(f"secrets not set in the environment: {', '.join(missing)}")
        self.allow = expand_hosts(allow)
        for _, hosts in self.secret_groups:
            for host in hosts:
                if (host, 443) not in self.allow:
                    self.allow.append((host, 443))
        self.providers = list(providers)
        self._base_policy = load_policy(policy) if policy is not None else None
        self.image = image
        self.exclude = tuple(DEFAULT_EXCLUDE if exclude is None else exclude)
        self.keep = keep
        self.workspace = workspace
        self.gateway = gateway
        self.timeout_seconds = timeout_seconds
        self.hooks: Any | None = None
        self.last_sync: SyncReport | None = None
        if name is not None and not _SANDBOX_NAME.fullmatch(name):
            raise ValueError(
                f"Sandbox name {name!r}: OpenShell takes 1-19 lowercase letters, digits and single hyphens, "
                "not starting or ending with a hyphen"
            )
        self._name = name
        self._client: Any = client
        self._owns_client = client is None
        self._agent: Any | None = None
        self._started = False
        self._start_lock: asyncio.Lock | None = None
        self._baseline: dict[str, str] = {}  # project file -> hash both sides agreed on at the last sync
        self.sandbox_id: str | None = None

    # ── identity and ownership ───────────────────────────────────────────

    @property
    def name(self) -> str | None:
        """The sandbox's name once known: given, recorded in the session, or derived from it."""
        return self._name

    @property
    def started(self) -> bool:
        return self._started

    def bind_agent(self, agent: Any) -> None:
        """Called by ``Agent`` when it takes this sandbox: sync after every run."""
        from .hooks import HookEvent

        if self._agent is not None and self._agent is not agent:
            raise ValueError("This OpenShellSandbox already belongs to another agent")
        if self._agent is None:
            self._agent = agent
            agent.hooks.on(HookEvent.AGENT_END, self._after_run)

    async def _after_run(self, context: Any) -> None:
        if self._started:
            await self.sync()

    def _session_record(self) -> dict[str, Any] | None:
        metadata = getattr(self._agent, "session_metadata", None)
        return metadata.get("openshell") if isinstance(metadata, dict) else None

    def _choose_name(self) -> str:
        if self._name:
            return self._name
        record = self._session_record()
        if record and record.get("name"):
            return str(record["name"])
        session = getattr(self._agent, "session_id", None) or uuid.uuid4().hex
        return "hx-" + re.sub(r"[^a-z0-9]", "", str(session).lower())[:16]  # 19 characters, the gateway's limit

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Create the sandbox (or find it again) and copy the project in. Idempotent."""
        if self._started:
            return
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            if self._started:
                return
            await self._start()
            self._started = True

    async def _start(self) -> None:
        openshell = _require_sdk()
        import grpc

        if self._client is None:
            self._client = await asyncio.to_thread(openshell.SandboxClient.from_active_cluster, cluster=self.gateway)
        name = self._choose_name()
        existing = None
        try:
            existing = await asyncio.to_thread(self._client.get, name, workspace=self.workspace)
        except grpc.RpcError as exc:
            if exc.code() != grpc.StatusCode.NOT_FOUND:
                raise
        self._name = name
        if existing is None:
            from openshell._proto import openshell_pb2

            spec = openshell_pb2.SandboxSpec()
            if self.image:
                spec.template.image = self.image
            credentials = await self._create_secret_providers()
            try:
                if self.allow or self._base_policy is not None:
                    spec.policy.CopyFrom(build_policy(base=self._base_policy, allow=self.allow, credentials=credentials))
                spec.providers.extend([*self.providers, *sorted(set(credentials.values()))])
                created = await asyncio.to_thread(
                    self._client.create, workspace=self.workspace, spec=spec, name=name, labels={"harnessx": "true"},
                )
            except BaseException:
                await self._delete_secret_providers()
                raise
            self.sandbox_id = created.id
        ready = await asyncio.to_thread(self._client.wait_ready, name, workspace=self.workspace)
        self.sandbox_id = ready.id
        manifest = await self._read_manifest() if existing is not None else None
        if manifest is None:
            await self._seed()
        else:
            self._baseline = manifest
        self._record()

    def _record(self) -> None:
        metadata = getattr(self._agent, "session_metadata", None)
        if isinstance(metadata, dict):
            metadata["openshell"] = {"name": self._name, "workspace": self.workspace, "sandbox_id": self.sandbox_id}

    async def aclose(self) -> None:
        """Copy back what changed, then delete the sandbox unless ``keep``.

        A kept sandbox is found again by name on the next start.
        """
        try:
            if self._started:
                await self.sync()
        finally:
            try:
                if self._started and not self.keep:
                    await asyncio.to_thread(
                        self._client.delete, self._name, workspace=self.workspace, allow_missing=True,
                    )
                    if self.secret_groups:
                        # A provider cannot go while a sandbox still holds it.
                        await asyncio.to_thread(
                            self._client.wait_deleted, self._name, workspace=self.workspace,
                            expected_sandbox_id=self.sandbox_id,
                        )
                        await self._delete_secret_providers()
            finally:
                self._started = False
                if self._owns_client and self._client is not None:
                    await asyncio.to_thread(self._client.close)
                    self._client = None

    async def __aenter__(self) -> OpenShellSandbox:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ── running commands ─────────────────────────────────────────────────

    def _exec(self, script: str, stdin: bytes | None, timeout: int, workdir: str | None) -> tuple[int, bytes, bytes]:
        openshell = _require_sdk()
        stdout: list[bytes] = []
        stderr: list[bytes] = []
        exit_code: int | None = None
        for item in self._client.exec_stream(
            self._name, ["sh", "-c", script], workspace=self.workspace, workdir=workdir,
            stdin=stdin, timeout_seconds=timeout,
        ):
            if isinstance(item, openshell.ExecChunk):
                (stdout if item.stream == "stdout" else stderr).append(item.data)
            elif isinstance(item, openshell.ExecResult):
                exit_code = item.exit_code
        if exit_code is None:
            raise RuntimeError("OpenShell returned no exit status for the command")
        return exit_code, b"".join(stdout), b"".join(stderr)

    async def _run(self, script: str, *, stdin: bytes | None = None, timeout: int | None = None,
                   workdir: str | None = PROJECT_DIR) -> tuple[int, bytes, bytes]:
        limit = timeout or self.timeout_seconds
        if stdin is not None and len(stdin) > _CHUNK_BYTES:
            staged = f"{STATE_DIR}/stdin-{uuid.uuid4().hex}"
            await self._upload(staged, stdin)
            script, stdin = f"(\n{script}\n) < {staged}; s=$?; rm -f {staged}; exit $s", None
        return await asyncio.to_thread(self._exec, script, stdin, limit, workdir)

    async def _upload(self, path: str, data: bytes) -> None:
        quoted = shlex.quote(path)
        await self._checked(f"mkdir -p {shlex.quote(posixpath.dirname(path))} && : > {quoted}")
        for offset in range(0, len(data), _CHUNK_BYTES):
            await self._checked(f"cat >> {quoted}", stdin=data[offset:offset + _CHUNK_BYTES])

    async def _checked(self, script: str, *, stdin: bytes | None = None, timeout: int | None = None) -> bytes:
        exit_code, stdout, stderr = await asyncio.to_thread(
            self._exec, script, stdin, timeout or max(self.timeout_seconds, 300), None,
        )
        if exit_code != 0:
            detail = stderr.decode("utf-8", "replace").strip() or f"exit code {exit_code}"
            raise RuntimeError(f"OpenShell sandbox {self._name}: {detail}")
        return stdout

    def _limit(self, timeout: int | None) -> int:
        if timeout is None:
            return self.timeout_seconds
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("timeout must be a positive integer")
        return min(timeout, self.timeout_seconds)

    async def execute_command(
        self, command: str, *, timeout: int | None = None, stdin: bytes | None = None,
    ) -> SandboxResult:
        """Run a shell command in the project folder inside the sandbox."""
        limit = self._limit(timeout)
        await self.start()
        began, began_at = time.monotonic(), time.time()
        exit_code, out, err = await self._run(command, stdin=stdin, timeout=limit)
        stdout, stderr = out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
        timed_out = exit_code == _TIMED_OUT
        denials = policy_denials(stdout, stderr)
        if exit_code != 0 and not timed_out:
            for host in await self._logged_denials(began_at, stdout + stderr):
                if host not in denials:
                    denials.append(host)
        result = SandboxResult(
            stdout=stdout, stderr=stderr + (f"\nCommand timed out after {limit}s" if timed_out else ""),
            exit_code=exit_code, timed_out=timed_out, execution_time_ms=(time.monotonic() - began) * 1000,
            denials=denials,
        )
        await self._observe("command", result)
        return result

    async def execute(self, code: str, language: str = "python", *, timeout: int | None = None) -> SandboxResult:
        """Run Python code in the sandbox (the image must have ``python3``)."""
        if language != "python":
            return SandboxResult(stderr=f"Unsupported language: {language}", exit_code=1)
        return await self.execute_command("python3 -", timeout=timeout, stdin=code.encode("utf-8"))

    async def _observe(self, kind: str, result: SandboxResult) -> None:
        if self.hooks is None:
            return
        from .hooks import HookContext, HookEvent

        await self.hooks.emit(HookEvent.SANDBOX_EXEC, HookContext(
            event=HookEvent.SANDBOX_EXEC,
            data={
                "kind": kind, "tier": "openshell", "exit_code": result.exit_code,
                "timed_out": result.timed_out, "execution_time_ms": result.execution_time_ms,
                "denials": list(result.denials),
            },
        ))

    # ── the gateway's other services ─────────────────────────────────────

    def _stub(self) -> Any:
        # The SDK wraps sandboxes, not providers, policy or logs; those go
        # through the generated gRPC stub it holds.
        return self._client._stub

    def _scope(self) -> Any:
        from openshell.sandbox import _workspace_scope

        return _workspace_scope(self.workspace)

    def _secret_names(self) -> list[tuple[str, str]]:
        """(provider name, profile id) for each secret group, derived from the sandbox name."""
        return [(f"{self._name}-secrets-{n}", f"harnessx-{self._name}-secrets-{n}") for n in range(len(self.secret_groups))]

    async def _create_secret_providers(self) -> dict[str, str]:
        """One OpenShell provider per secret group; returns host -> provider name.

        Each gets its own endpointless profile, so the secret's hosts in the
        sandbox policy are the only places it can be sent.
        """
        if not self.secret_groups:
            return {}
        from openshell._proto import datamodel_pb2, openshell_pb2

        stub, credentials = self._stub(), {}
        for (names, hosts), (provider, profile) in zip(self.secret_groups, self._secret_names()):
            item = openshell_pb2.ProviderProfileImportItem(source="harnessx", profile=openshell_pb2.ProviderProfile(
                id=profile, display_name=f"HarnessX: {', '.join(names)}",
                category=openshell_pb2.PROVIDER_PROFILE_CATEGORY_OTHER,
                credentials=[openshell_pb2.ProviderProfileCredential(
                    name=name.lower(), env_vars=[name], required=True, auth_style="bearer", header_name="Authorization",
                ) for name in names],
            ))
            response = await asyncio.to_thread(
                stub.ImportProviderProfiles, openshell_pb2.ImportProviderProfilesRequest(profiles=[item]), timeout=60,
            )
            if not response.imported:
                problems = "; ".join(diagnostic.message for diagnostic in response.diagnostics)
                raise RuntimeError(f"OpenShell refused the profile for {', '.join(names)}: {problems}")
            record = datamodel_pb2.Provider(type=profile, credentials={name: os.environ[name] for name in names})
            record.metadata.name = provider
            await asyncio.to_thread(
                stub.CreateProvider, openshell_pb2.CreateProviderRequest(workspace_scope=self._scope(), provider=record),
                timeout=60,
            )
            for host in hosts:
                credentials[host] = provider
        return credentials

    async def _delete_secret_providers(self) -> None:
        if not self.secret_groups:
            return
        import grpc
        from openshell._proto import openshell_pb2

        stub = self._stub()
        for provider, profile in self._secret_names():
            for call, request in (
                (stub.DeleteProvider, openshell_pb2.DeleteProviderRequest(
                    workspace_scope=self._scope(), name=provider, allow_missing=True)),
                (stub.DeleteProviderProfile, openshell_pb2.DeleteProviderProfileRequest(id=profile, allow_missing=True)),
            ):
                try:
                    await asyncio.to_thread(call, request, timeout=60)
                except grpc.RpcError as exc:
                    logger.warning("OpenShell cleanup of %s failed: %s", request, exc)

    async def _logged_denials(self, since: float, output: str) -> list[str]:
        """Destinations the sandbox's audit log refused since ``since``.

        A refused connection reaches the command only as "Permission denied",
        so the log is the reliable record. It can trail the command briefly.
        """
        if not hasattr(self._client, "_stub"):
            return []
        from google.protobuf.timestamp_pb2 import Timestamp
        from openshell._proto import openshell_pb2

        start = Timestamp()
        start.FromMilliseconds(int((since - 1) * 1000))
        request = openshell_pb2.GetSandboxLogsRequest(
            workspace_scope=self._scope(), sandbox=self._name, since_time=start, sources=["sandbox"],
        )
        attempts = 3 if "denied" in output.lower() or "not permitted" in output.lower() else 1
        for attempt in range(attempts):
            try:
                response = await asyncio.to_thread(self._stub().GetSandboxLogs, request, timeout=15)
            except Exception as exc:  # the log is a diagnosis, never a reason to fail the command
                logger.debug("OpenShell log lookup failed: %s", exc)
                return []
            found = denials_from_log([line.message for line in response.logs])
            if found or attempt == attempts - 1:
                # "host:port" from the connection beats the bare host from DNS.
                return [d for d in found if not any(o != d and o.startswith(d + ":") for o in found)]
            await asyncio.sleep(0.5)
        return []

    async def policy(self) -> dict[str, Any]:
        """The sandbox's current OpenShell policy, as a mapping."""
        await self.start()
        from google.protobuf import json_format
        from openshell._proto import sandbox_pb2

        response = await asyncio.to_thread(
            self._stub().GetSandboxConfig,
            sandbox_pb2.GetSandboxConfigRequest(workspace_scope=self._scope(), name=self._name), timeout=60,
        )
        return json_format.MessageToDict(response.policy, preserving_proto_field_name=True)

    async def pending_rules(self) -> list[PendingRule]:
        """Network rules OpenShell drafted from blocked connections, awaiting a decision."""
        await self.start()
        from openshell._proto import openshell_pb2

        response = await asyncio.to_thread(
            self._stub().GetDraftPolicy, openshell_pb2.GetDraftPolicyRequest(
                workspace_scope=self._scope(), sandbox=self._name, status_filter="pending"), timeout=60,
        )
        return [
            PendingRule(
                id=chunk.id, rule_name=chunk.rule_name, binary=chunk.binary, rationale=chunk.rationale,
                hosts=[f"{e.host}:{','.join(map(str, e.ports or [e.port]))}" for e in chunk.proposed_rule.endpoints],
                review_token=chunk.review_token,
            )
            for chunk in response.chunks
        ]

    async def approve_rule(self, rule: PendingRule) -> None:
        """Add a drafted rule to the sandbox's policy; it applies to the next connection."""
        from openshell._proto import openshell_pb2

        await asyncio.to_thread(self._stub().ApproveDraftChunk, openshell_pb2.ApproveDraftChunkRequest(
            workspace_scope=self._scope(), sandbox=self._name, chunk_id=rule.id, review_token=rule.review_token,
        ), timeout=60)

    async def reject_rule(self, rule: PendingRule, reason: str = "") -> None:
        """Turn a drafted rule down; ``reason`` is shown to whoever proposed it."""
        from openshell._proto import openshell_pb2

        await asyncio.to_thread(self._stub().RejectDraftChunk, openshell_pb2.RejectDraftChunkRequest(
            workspace_scope=self._scope(), sandbox=self._name, chunk_id=rule.id, reason=reason,
        ), timeout=60)

    # ── paths ────────────────────────────────────────────────────────────

    def resolve_path(self, path: str, *, write: bool = False) -> str:
        """Map a tool's path into the sandbox.

        Relative paths are in the project. Local paths inside the project or
        an input are translated; sandbox paths inside them are accepted.
        Inputs are read-only. Anything else is not in the sandbox.
        """
        if not isinstance(path, str) or not path or "\0" in path:
            raise ValueError("path must be a nonempty string without NUL bytes")
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded):
            local = os.path.normpath(expanded)
            for root, target in [(self.project, PROJECT_DIR), *self.inputs.items()]:
                for spelling in {root, os.path.normpath(os.path.abspath(root))}:
                    if os.path.commonpath([spelling, local]) == spelling:
                        return self._checked_path(
                            posixpath.join(target, os.path.relpath(local, spelling).replace(os.sep, "/")), path, write,
                        )
            return self._checked_path(posixpath.normpath(expanded), path, write)
        return self._checked_path(posixpath.normpath(posixpath.join(PROJECT_DIR, path)), path, write)

    def _checked_path(self, target: str, path: str, write: bool) -> str:
        target = posixpath.normpath(target)
        if posixpath.commonpath([PROJECT_DIR, target]) == PROJECT_DIR:
            return target
        for mounted in self.inputs.values():
            if posixpath.commonpath([mounted, target]) == mounted:
                if write:
                    raise PermissionError(f"Access denied: {path!r} is a read-only input")
                return target
        raise PermissionError(f"Access denied: {path!r} is not available in the sandbox; only the project folder is")

    # ── copying in and back ──────────────────────────────────────────────

    async def _read_manifest(self) -> dict[str, str] | None:
        exit_code, stdout, _ = await self._run(f"cat {MANIFEST} 2>/dev/null", workdir=None)
        if exit_code != 0:
            return None
        try:
            files = json.loads(stdout).get("files")
        except ValueError:
            return None
        return dict(files) if isinstance(files, dict) else None

    async def _write_manifest(self) -> None:
        await self._checked(
            f"mkdir -p {STATE_DIR} && cat > {MANIFEST}",
            stdin=json.dumps({"files": self._baseline}).encode("utf-8"),
        )

    async def _unpack(self, archive: bytes, destination: str) -> None:
        staged = f"{STATE_DIR}/seed-{uuid.uuid4().hex}.tar.gz"
        await self._upload(staged, archive)
        await self._checked(
            f"mkdir -p {shlex.quote(destination)} && tar -xzf {staged} -C {shlex.quote(destination)}; "
            f"s=$?; rm -f {staged}; exit $s"
        )

    async def _seed(self) -> None:
        archive, hashes = await asyncio.to_thread(_pack, self.project, self.exclude)
        await self._unpack(archive, PROJECT_DIR)
        for local, target in self.inputs.items():
            archive, _ = await asyncio.to_thread(_pack, local, self.exclude)
            await self._unpack(archive, target if os.path.isdir(local) else posixpath.dirname(target))
        self._baseline = hashes
        await self._write_manifest()

    async def _remote_hashes(self) -> dict[str, str]:
        prune = " -o ".join(f"-name {shlex.quote(pattern)}" for pattern in self.exclude)
        walk = f"find . \\( {prune} \\) -prune -o -type f -exec sha256sum {{}} +" if prune else \
            "find . -type f -exec sha256sum {} +"
        listing = await self._checked(f"cd {PROJECT_DIR} && {walk}")
        hashes = {}
        for line in listing.decode("utf-8", "replace").splitlines():
            digest, _, name = line.partition("  ")
            if name.startswith("./") and not digest.startswith("\\"):
                hashes[name[2:]] = digest
        return hashes

    def _local_target(self, relative: str) -> str | None:
        """The local path for a project file, or None when a local symlink would redirect it."""
        target = os.path.join(self.project, *relative.split("/"))
        parent = os.path.dirname(target)
        probe = parent
        while not os.path.exists(probe):
            probe = os.path.dirname(probe)
        if os.path.commonpath([self.project, os.path.realpath(probe)]) != self.project or os.path.islink(target):
            return None
        return target

    async def sync(self) -> SyncReport:
        """Copy files the sandbox changed since the last sync back to the project."""
        report = SyncReport()
        if not self._started:
            return report
        remote = await self._remote_hashes()
        changed = sorted(name for name, digest in remote.items() if self._baseline.get(name) != digest)
        removed = sorted(name for name in self._baseline if name not in remote)
        if changed:
            archive = await self._checked(
                f"cd {PROJECT_DIR} && tar -cf - --null -T -", stdin=b"\0".join(n.encode() for n in changed) + b"\0",
            )
            await asyncio.to_thread(self._apply, archive, remote, report)
        for name in removed:
            target = self._local_target(name)
            if target is not None and os.path.isfile(target):
                if await asyncio.to_thread(_sha256, target) == self._baseline[name]:
                    os.unlink(target)
                    report.deleted.append(name)
                else:
                    report.conflicts.append(name)
        self._baseline = dict(remote)
        if changed or removed:
            await self._write_manifest()
        self.last_sync = report
        if report.updated or report.deleted or report.conflicts:
            logger.info("OpenShell sync: %d updated, %d deleted, %d conflicts",
                        len(report.updated), len(report.deleted), len(report.conflicts))
        return report

    def _apply(self, archive: bytes, remote: dict[str, str], report: SyncReport) -> None:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar:
                if not member.isfile() or member.name not in remote:
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                data = source.read()
                target = self._local_target(member.name)
                if target is None:
                    report.conflicts.append(member.name)
                    continue
                local = _sha256(target) if os.path.isfile(target) else None
                if local == hashlib.sha256(data).hexdigest():
                    continue
                if local is not None and local != self._baseline.get(member.name):
                    target += ".sandbox"  # edited here too: keep the local version
                    report.conflicts.append(member.name)
                else:
                    report.updated.append(member.name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                staged = f"{target}.harnessx-{uuid.uuid4().hex}"
                with open(staged, "wb") as stream:
                    stream.write(data)
                os.chmod(staged, member.mode & 0o777 or 0o644)
                os.replace(staged, target)
