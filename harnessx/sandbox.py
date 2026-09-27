"""Execution backends with explicit, tier-dependent isolation.

Three POSIX execution tiers:
  Tier 1 (process): subprocess + resource limits + working directory; host access
  Tier 2 (docker): Docker container with mem/cpu/network limits
  Tier 3 (seatbelt): macOS Seatbelt OS-enforced sandbox

The sandbox is independent from the tool system. Any tool can opt into
sandboxed execution, but the sandbox doesn't know about tools.
"""

from __future__ import annotations

import asyncio
import os
import platform
import json
import signal
import shutil
import sys
import tempfile
import time
from typing import Any

from .types import SandboxConfig, SandboxResult


class Sandbox:
    """Manages a sandboxed execution context.

    Usage:
        async with Sandbox(config) as sandbox:
            result = await sandbox.execute("print('hello')")
            result = await sandbox.execute_command("ls -la")
    """

    def __init__(self, config: SandboxConfig | None = None, *, hooks: Any | None = None) -> None:
        self.config = config or SandboxConfig()
        self._workdir: str | None = None
        self._temp_dir_obj: tempfile.TemporaryDirectory | None = None
        # A HookManager that receives SANDBOX_EXEC; Agent(sandbox=...) supplies its own.
        self.hooks = hooks

    async def _observe(self, kind: str, result: SandboxResult) -> None:
        if self.hooks is None:
            return
        from .hooks import HookContext, HookEvent

        await self.hooks.emit(HookEvent.SANDBOX_EXEC, HookContext(
            event=HookEvent.SANDBOX_EXEC,
            data={
                "kind": kind, "tier": self.config.tier, "exit_code": result.exit_code,
                "timed_out": result.timed_out, "execution_time_ms": result.execution_time_ms,
            },
        ))

    def _validate_backend(self) -> None:
        self.config.__post_init__()
        if os.name != "posix":
            raise RuntimeError("Sandbox execution requires a POSIX host")
        if self.config.tier == "seatbelt":
            if platform.system() != "Darwin" or shutil.which("sandbox-exec") is None:
                raise RuntimeError("Seatbelt requires macOS and sandbox-exec; no process fallback is permitted")

    async def __aenter__(self) -> Sandbox:
        self._validate_backend()
        self._ensure_workdir()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._temp_dir_obj:
            try:
                self._temp_dir_obj.cleanup()
            except OSError:
                pass
        self._workdir = None
        self._temp_dir_obj = None

    @property
    def workdir(self) -> str:
        if self._workdir is None:
            raise RuntimeError("Sandbox not started. Use 'async with Sandbox() as sb:'")
        return self._workdir

    def _ensure_workdir(self) -> str:
        """Create workdir if not using context manager."""
        if self._workdir is None:
            self._temp_dir_obj = tempfile.TemporaryDirectory(prefix="agent_sandbox_")
            self._workdir = self._temp_dir_obj.name
        return self._workdir

    async def execute(self, code: str, language: str = "python") -> SandboxResult:
        """Execute code in the sandbox."""
        self._validate_backend()
        self._ensure_workdir()

        tier = self.config.tier
        if tier == "docker":
            result = await self._execute_docker(code, language)
        elif tier == "seatbelt":
            result = await self._execute_seatbelt(code, language)
        else:
            result = await self._execute_process(code, language)
        await self._observe("code", result)
        return result

    async def execute_command(self, command: str) -> SandboxResult:
        """Execute a shell command in the sandbox."""
        self._validate_backend()
        self._ensure_workdir()

        tier = self.config.tier
        if tier == "docker":
            result = await self._execute_docker_command(command)
        elif tier == "seatbelt":
            result = await self._execute_seatbelt_command(command)
        else:
            result = await self._execute_process_command(command)
        await self._observe("command", result)
        return result

    # ── Tier 1: Process sandbox ──────────────────────────────────────────

    async def _execute_process(self, code: str, language: str) -> SandboxResult:
        """Run a host subprocess in a temporary working directory."""
        workdir = self._ensure_workdir()
        start = time.monotonic()

        if language == "python":
            # Write code to a temp file to avoid shell escaping issues
            code_file = os.path.join(workdir, "_sandbox_code.py")
            with open(code_file, "w") as f:
                f.write(code)
            cmd = [sys.executable, code_file]
        else:
            return SandboxResult(
                stderr=f"Unsupported language: {language}",
                exit_code=1,
            )

        return await self._run_subprocess(cmd, workdir, start)

    async def _execute_process_command(self, command: str) -> SandboxResult:
        workdir = self._ensure_workdir()
        start = time.monotonic()
        cmd = ["/bin/sh", "-c", command]
        return await self._run_subprocess(cmd, workdir, start)

    async def _run_subprocess(
        self, cmd: list[str], workdir: str, start: float
    ) -> SandboxResult:
        timeout = self.config.timeout_seconds
        env = self._build_env(workdir)

        # Build preexec_fn for resource limits (Unix only)
        preexec = self._make_preexec_fn() if sys.platform != "win32" else None

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=env,
                preexec_fn=preexec,
                start_new_session=True,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
                elapsed = (time.monotonic() - start) * 1000
                return SandboxResult(
                    stderr=f"Process timed out after {timeout}s",
                    exit_code=-1,
                    timed_out=True,
                    execution_time_ms=elapsed,
                    files_created=self._list_new_files(workdir),
                )
            except asyncio.CancelledError:
                if proc.returncode is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                await proc.wait()
                raise

            elapsed = (time.monotonic() - start) * 1000
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            # Check if killed by resource limits
            memory_exceeded = proc.returncode == -9  # SIGKILL often indicates OOM

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode or 0,
                timed_out=False,
                memory_exceeded=memory_exceeded,
                execution_time_ms=elapsed,
                files_created=self._list_new_files(workdir),
            )

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return SandboxResult(
                stderr=f"Sandbox error: {e}",
                exit_code=-1,
                execution_time_ms=elapsed,
            )

    def _make_preexec_fn(self):
        """Create a preexec_fn that sets resource limits."""
        max_mem = self.config.max_memory_mb * 1024 * 1024
        max_cpu = self.config.max_cpu_seconds
        max_fsize = self.config.max_file_size_mb * 1024 * 1024

        def set_limits():
            import resource

            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (max_cpu, max_cpu + 10))
            # Virtual memory limit
            try:
                resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))
            except ValueError:
                pass  # Some systems don't support RLIMIT_AS
            # File size limit
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_fsize, max_fsize))

        return set_limits

    def _build_env(self, workdir: str) -> dict[str, str]:
        """Build restricted environment variables."""
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": workdir,
            "TMPDIR": workdir,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        # Propagate VIRTUAL_ENV if present for access to installed packages
        if "VIRTUAL_ENV" in os.environ:
            env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
            env["PATH"] = os.path.join(os.environ["VIRTUAL_ENV"], "bin") + ":" + env["PATH"]
        return env

    def _list_new_files(self, workdir: str) -> list[str]:
        """List files created in the workdir (excluding our sandbox code file)."""
        files = []
        for root, _, filenames in os.walk(workdir):
            for fn in filenames:
                if fn == "_sandbox_code.py":
                    continue
                rel = os.path.relpath(os.path.join(root, fn), workdir)
                files.append(rel)
        return files

    # ── Tier 2: Docker sandbox ───────────────────────────────────────────

    async def _execute_docker(self, code: str, language: str) -> SandboxResult:
        """Docker container with resource limits and network isolation."""
        try:
            import docker  # noqa: F401  (availability probe)
        except ImportError as exc:
            raise RuntimeError("Docker sandbox requires harnessx[docker]; no process fallback is permitted") from exc

        workdir = self._ensure_workdir()
        start = time.monotonic()

        # Write code to workdir
        code_file = os.path.join(workdir, "_sandbox_code.py")
        with open(code_file, "w") as f:
            f.write(code)

        if language != "python":
            return SandboxResult(stderr=f"Unsupported language: {language}", exit_code=1)

        try:
            return await self._run_docker(["python3", "/sandbox/_sandbox_code.py"], workdir, start)
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            error_msg = str(e)
            timed_out = "timeout" in error_msg.lower() or "deadline" in error_msg.lower()
            return SandboxResult(
                stderr=error_msg,
                exit_code=-1,
                timed_out=timed_out,
                execution_time_ms=elapsed,
            )

    async def _execute_docker_command(self, command: str) -> SandboxResult:
        try:
            import docker  # noqa: F401  (availability probe)
        except ImportError as exc:
            raise RuntimeError("Docker sandbox requires harnessx[docker]; no process fallback is permitted") from exc

        workdir = self._ensure_workdir()
        start = time.monotonic()

        try:
            return await self._run_docker(["/bin/sh", "-c", command], workdir, start)
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return SandboxResult(stderr=str(e), exit_code=-1, execution_time_ms=elapsed)

    async def _run_docker(self, command, workdir, start):
        import docker  # noqa: F401  (availability probe)
        client = await asyncio.to_thread(docker.from_env)
        container = None
        try:
            volumes = {workdir: {"bind": "/sandbox", "mode": "rw"}}
            for path in self.config.allowed_paths:
                resolved = os.path.realpath(path)
                if not os.path.exists(resolved):
                    raise ValueError(f"Allowed path does not exist: {path}")
                volumes[resolved] = {"bind": resolved, "mode": "ro"}
            create = asyncio.create_task(asyncio.to_thread(
                client.containers.run, "python:3.11-slim", command=command,
                volumes=volumes, working_dir="/sandbox", detach=True,
                mem_limit=f"{self.config.max_memory_mb}m", memswap_limit=f"{self.config.max_memory_mb}m",
                network_disabled=not self.config.network_enabled,
                ulimits=[docker.types.Ulimit(name="cpu", soft=self.config.max_cpu_seconds, hard=self.config.max_cpu_seconds),
                         docker.types.Ulimit(name="fsize", soft=self.config.max_file_size_mb * 1024**2, hard=self.config.max_file_size_mb * 1024**2)],
            ))
            try:
                container = await asyncio.shield(create)
            except asyncio.CancelledError:
                container = await create
                raise
            try:
                outcome = await asyncio.to_thread(container.wait, timeout=self.config.timeout_seconds)
            except Exception as exc:
                # The SDK wait uses an HTTP timeout; always remove/kill on failure.
                return SandboxResult(stderr=str(exc), exit_code=-1, timed_out="timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower())
            stdout = await asyncio.to_thread(container.logs, stdout=True, stderr=False)
            stderr = await asyncio.to_thread(container.logs, stdout=False, stderr=True)
            return SandboxResult(stdout=stdout.decode("utf-8", errors="replace"), stderr=stderr.decode("utf-8", errors="replace"),
                                 exit_code=outcome["StatusCode"], execution_time_ms=(time.monotonic() - start) * 1000,
                                 files_created=self._list_new_files(workdir))
        finally:
            try:
                if container is not None:
                    await asyncio.to_thread(container.remove, force=True)
            finally:
                await asyncio.to_thread(client.close)

    # ── Tier 3: macOS Seatbelt sandbox ───────────────────────────────────

    async def _execute_seatbelt(self, code: str, language: str) -> SandboxResult:
        """Run under the configured macOS Seatbelt access profile."""
        if language != "python":
            return SandboxResult(stderr=f"Unsupported language: {language}", exit_code=1)
        workdir = self._ensure_workdir()
        start = time.monotonic()

        # Write code
        code_file = os.path.join(workdir, "_sandbox_code.py")
        with open(code_file, "w") as f:
            f.write(code)

        # Generate Seatbelt profile
        profile = self._generate_seatbelt_profile(workdir)
        profile_file = os.path.join(workdir, "_sandbox.sb")
        with open(profile_file, "w") as f:
            f.write(profile)

        cmd = [
            "sandbox-exec", "-f", profile_file,
            sys.executable, code_file,
        ]

        return await self._run_subprocess(cmd, workdir, start)

    async def _execute_seatbelt_command(self, command: str) -> SandboxResult:
        workdir = self._ensure_workdir()
        start = time.monotonic()

        profile = self._generate_seatbelt_profile(workdir)
        profile_file = os.path.join(workdir, "_sandbox.sb")
        with open(profile_file, "w") as f:
            f.write(profile)

        cmd = ["sandbox-exec", "-f", profile_file, "/bin/sh", "-c", command]
        return await self._run_subprocess(cmd, workdir, start)

    def _generate_seatbelt_profile(self, workdir: str) -> str:
        """Generate a Seatbelt .sb profile for macOS sandboxing."""
        allowed_read_paths = [
            "/usr",
            "/bin",
            "/sbin",
            "/Library",
            "/System",
            "/dev",
            "/etc",
            workdir,
            *[os.path.realpath(path) for path in self.config.allowed_paths],
        ]
        # Allow reading Python installation
        python_prefix = sys.prefix
        if python_prefix not in allowed_read_paths:
            allowed_read_paths.append(python_prefix)

        # Allow read from VIRTUAL_ENV if present
        venv = os.environ.get("VIRTUAL_ENV")
        if venv and venv not in allowed_read_paths:
            allowed_read_paths.append(venv)

        read_rules = "\n".join(
            f'    (subpath {json.dumps(p)})' for p in allowed_read_paths
        )

        network_rule = "(allow network*)" if self.config.network_enabled else "(deny network*)"

        profile = f"""\
(version 1)
(deny default)

;; Allow reading system paths and Python
(allow file-read*
{read_rules}
)

;; Allow writing only to sandbox workdir
(allow file-write*
    (subpath {json.dumps(workdir)})
)

;; Allow process execution
(allow process-exec*)
(allow process-fork)

;; Allow basic system operations
(allow sysctl-read)
(allow mach-lookup)
(allow signal)
(allow ipc-posix-shm-read*)
(allow ipc-posix-shm-write*)

;; Network
{network_rule}
"""
        return profile

    # ── Utility ──────────────────────────────────────────────────────────

    async def cleanup(self) -> None:
        """Manually clean up the sandbox."""
        if self._temp_dir_obj:
            try:
                self._temp_dir_obj.cleanup()
            except OSError:
                pass
            self._workdir = None

    async def write_file(self, relative_path: str, content: str) -> str:
        """Write a file into the sandbox workdir. Returns the absolute path."""
        from .builtin.filesystem import _Filesystem
        workdir = self._ensure_workdir()
        fs = _Filesystem(workdir)
        await asyncio.to_thread(fs.write, relative_path, content, True)
        return os.path.abspath(os.path.join(workdir, relative_path))

    async def read_file(self, relative_path: str) -> str:
        """Read a file from the sandbox workdir."""
        from .builtin.filesystem import _Filesystem
        workdir = self._ensure_workdir()
        return await asyncio.to_thread(_Filesystem(workdir).read, relative_path, 0, 100_000, numbered=False)

    async def list_files(self) -> list[str]:
        """List all files in the sandbox workdir."""
        workdir = self._ensure_workdir()
        return self._list_new_files(workdir)
