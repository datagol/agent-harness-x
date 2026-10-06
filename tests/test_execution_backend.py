"""The execution backend contract: bash timeouts and file tools that run in a backend."""

from __future__ import annotations

import asyncio
import os
import posixpath

import pytest

from harnessx import ExecutionBackend, Sandbox, SandboxConfig, SandboxResult, ToolCall, ToolRegistry
from harnessx.builtin import register_all_tools, register_bash_tools, register_filesystem_tools
from harnessx.types import PermissionLevel

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX shell backend")


class ShellBackend:
    """A backend that owns a directory: what a remote sandbox looks like to the tools.

    Commands run in ``/bin/sh`` with the directory as their working directory;
    paths resolve inside it, and ``read_only`` subdirectories refuse writes.
    """

    owns_filesystem = True
    hooks = None

    def __init__(self, root: str, *, read_only: tuple[str, ...] = ()) -> None:
        self.root = os.path.realpath(root)
        self.read_only = [posixpath.join(self.root, entry) for entry in read_only]
        self.commands: list[tuple[str, int | None, bytes | None]] = []

    async def execute_command(self, command, *, timeout=None, stdin=None):
        self.commands.append((command, timeout, stdin))
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh", "-c", command, cwd=self.root,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(stdin)
        return SandboxResult(stdout=out.decode(), stderr=err.decode(), exit_code=proc.returncode or 0)

    async def execute(self, code, language="python", *, timeout=None):
        raise NotImplementedError

    def resolve_path(self, path, *, write=False):
        target = posixpath.normpath(posixpath.join(self.root, path))
        if posixpath.commonpath([self.root, target]) != self.root:
            raise PermissionError(f"Access denied: path {path!r} is outside the sandbox")
        if write and any(posixpath.commonpath([entry, target]) == entry for entry in self.read_only):
            raise PermissionError(f"Access denied: path {path!r} is read-only")
        return target


def _populate(root) -> None:
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("import os\n\ndef main():\n    return 'hi'\n")
    (root / "src" / "util.py").write_text("def helper():\n    return 42\n")
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text("# Guide\nmain entry point\n")
    (root / "empty").mkdir()
    (root / "unicode.txt").write_text("café\nnaïve\n")
    (root / "link.txt").symlink_to(root / "unicode.txt")


def _registries(tmp_path, **limits):
    host_root, sandbox_root = tmp_path / "host", tmp_path / "sandbox"
    for root in (host_root, sandbox_root):
        root.mkdir()
        _populate(root)
    host = ToolRegistry()
    register_filesystem_tools(host, base_path=str(host_root), permission=PermissionLevel.ALLOW, **limits)
    remote = ToolRegistry()
    backend = ShellBackend(str(sandbox_root))
    register_filesystem_tools(remote, sandbox=backend, permission=PermissionLevel.ALLOW, **limits)
    return host, remote, host_root, sandbox_root


def _run(registry, name, **arguments):
    return asyncio.run(registry.execute(ToolCall("1", name, arguments)))


# Every call is made against both registries; the results must be identical.
PARITY_CALLS = [
    ("read_file", {"path": "src/app.py"}),
    ("read_file", {"path": "src/app.py", "offset": 1, "limit": 2}),
    ("read_file", {"path": "unicode.txt"}),
    ("read_file", {"path": "src/app.py", "limit": 0}),
    ("read_file", {"path": "src"}),
    ("read_file", {"path": "link.txt"}),
    ("list_directory", {"path": "."}),
    ("list_directory", {"path": "empty"}),
    ("glob", {"pattern": "*.py"}),
    ("glob", {"pattern": "src/*"}),
    ("glob", {"pattern": "*.rs"}),
    ("grep", {"pattern": r"def \w+"}),
    ("grep", {"pattern": "main", "include": "*.md"}),
    ("grep", {"pattern": "nothing-here"}),
    ("grep", {"pattern": "("}),
    ("write_file", {"path": "new/deep/file.txt", "content": "fresh\n"}),
    ("read_file", {"path": "new/deep/file.txt"}),
    ("write_file", {"path": "src/app.py", "content": "x"}),
    ("write_file", {"path": "src/app.py", "content": "replaced\n", "overwrite": True}),
    ("read_file", {"path": "src/app.py"}),
    ("write_file", {"path": "link.txt", "content": "x", "overwrite": True}),
    ("edit_file", {"path": "src/util.py", "old_string": "42", "new_string": "43"}),
    ("edit_file", {"path": "src/util.py", "old_string": "absent", "new_string": "y"}),
    ("edit_file", {"path": "src/util.py", "old_string": "e", "new_string": "E"}),
    ("edit_file", {"path": "src/util.py", "old_string": "e", "new_string": "E", "replace_all": True}),
    ("read_file", {"path": "src/util.py"}),
    ("delete", {"path": "link.txt"}),
    ("read_file", {"path": "unicode.txt"}),
    ("delete", {"path": "empty"}),
    ("delete", {"path": "docs/guide.md"}),
    ("delete", {"path": "src", "recursive": True}),
    ("delete", {"path": "."}),
    ("list_directory", {"path": "."}),
]


def _error_type(content: str) -> str:
    return content.split(":")[1].strip()


def test_file_tools_in_a_backend_answer_exactly_as_the_host_tools_do(tmp_path):
    host, remote, _, _ = _registries(tmp_path)
    for name, arguments in PARITY_CALLS:
        expected, actual = _run(host, name, **arguments), _run(remote, name, **arguments)
        if arguments == {"path": "src"}:
            # The host's message names a file descriptor ("Is a directory: 15").
            assert actual.is_error and _error_type(actual.content) == _error_type(expected.content) == "IsADirectoryError"
            continue
        assert (actual.content, actual.is_error) == (expected.content, expected.is_error), (name, arguments)


def test_truncation_limits_match_the_host_tools(tmp_path):
    host, remote, host_root, sandbox_root = _registries(
        tmp_path, max_read_bytes=40, max_directory_entries=2,
    )
    for root in (host_root, sandbox_root):
        (root / "long.txt").write_text("".join(f"line {n}\n" for n in range(50)))
    for name, arguments in [
        ("read_file", {"path": "long.txt"}),
        ("read_file", {"path": "long.txt", "offset": 3}),
        ("glob", {"pattern": "*"}),
        ("grep", {"pattern": "."}),
        ("edit_file", {"path": "long.txt", "old_string": "line 1\n", "new_string": "x"}),
    ]:
        expected, actual = _run(host, name, **arguments), _run(remote, name, **arguments)
        assert (actual.content, actual.is_error) == (expected.content, expected.is_error), (name, arguments)
    # Which entries a truncated listing keeps follows directory order on the
    # host, so only the shape is comparable.
    listing = _run(remote, "list_directory", path=".").content.splitlines()
    assert len(listing) == 4 and listing[-1] == "[truncated: directory entry limit reached]"


def test_backend_file_tools_touch_only_the_backend(tmp_path):
    host_root, sandbox_root = tmp_path / "host", tmp_path / "sandbox"
    host_root.mkdir()
    sandbox_root.mkdir()
    registry = ToolRegistry(sandbox=ShellBackend(str(sandbox_root)))
    register_filesystem_tools(registry, permission=PermissionLevel.ALLOW)  # the registry's backend

    _run(registry, "write_file", path="report.md", content="done\n")
    assert (sandbox_root / "report.md").read_text() == "done\n"
    assert not (host_root / "report.md").exists() and not os.path.exists("report.md")
    # Bytes travel through stdin, never the command line.
    assert all("done" not in command for command, _, _ in registry.sandbox.commands)


def test_backend_paths_are_the_backends_to_refuse(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "data.csv").write_text("a,b\n")
    registry = ToolRegistry()
    register_filesystem_tools(
        registry, sandbox=ShellBackend(str(tmp_path), read_only=("inputs",)), permission=PermissionLevel.ALLOW,
    )
    outside = _run(registry, "read_file", path="../escape.txt")
    assert outside.is_error and "outside the sandbox" in outside.content
    assert not _run(registry, "read_file", path="inputs/data.csv").is_error
    for name, arguments in [
        ("write_file", {"path": "inputs/new.csv", "content": "x"}),
        ("edit_file", {"path": "inputs/data.csv", "old_string": "a", "new_string": "z"}),
        ("delete", {"path": "inputs/data.csv"}),
    ]:
        refused = _run(registry, name, **arguments)
        assert refused.is_error and "read-only" in refused.content, name
    assert (tmp_path / "inputs" / "data.csv").read_text() == "a,b\n"


def test_base_path_cannot_be_combined_with_a_backend_that_owns_its_files(tmp_path):
    with pytest.raises(ValueError, match="base_path does not apply"):
        register_filesystem_tools(ToolRegistry(), sandbox=ShellBackend(str(tmp_path)), base_path=str(tmp_path))


def test_the_local_sandbox_leaves_file_tools_on_the_host(tmp_path):
    sandbox = Sandbox(SandboxConfig())
    assert isinstance(sandbox, ExecutionBackend) and sandbox.owns_filesystem is False
    registry = ToolRegistry(sandbox=sandbox)
    register_filesystem_tools(registry, base_path=str(tmp_path), permission=PermissionLevel.ALLOW)
    _run(registry, "write_file", path="here.txt", content="host")
    assert (tmp_path / "here.txt").read_text() == "host"


def test_register_all_tools_hands_the_backend_to_the_file_tools(tmp_path):
    backend = ShellBackend(str(tmp_path))
    registry = ToolRegistry()
    register_all_tools(registry, sandbox=backend, include=["write_file", "run_bash"], permission=PermissionLevel.ALLOW)
    _run(registry, "write_file", path="a.txt", content="in the backend")
    assert (tmp_path / "a.txt").read_text() == "in the backend"
    assert "in the backend" in _run(registry, "run_bash", command="cat a.txt").content


def test_sandboxed_bash_passes_the_models_timeout_to_the_backend(tmp_path):
    backend = ShellBackend(str(tmp_path))
    registry = ToolRegistry()
    register_bash_tools(registry, sandbox=backend, permission=PermissionLevel.ALLOW)
    _run(registry, "run_bash", command="echo hi", timeout=7)
    assert backend.commands[-1][:2] == ("echo hi", 7)
    # An interrupted remote command is resolved, never silently re-run.
    assert registry.get_tool("run_bash").replay_policy == "manual"


def test_a_call_timeout_shortens_the_local_sandbox_limit_but_cannot_extend_it():
    async def scenario():
        async with Sandbox(SandboxConfig(timeout_seconds=5)) as sandbox:
            quick = await sandbox.execute_command("sleep 3", timeout=1)
            assert quick.timed_out and "1s" in quick.stderr
            capped = await sandbox.execute_command("sleep 30", timeout=600)
            assert capped.timed_out and "5s" in capped.stderr
            with pytest.raises(ValueError):
                await sandbox.execute_command("true", timeout=0)

    asyncio.run(scenario())


def test_the_local_sandbox_feeds_stdin_and_closes_it_otherwise():
    async def scenario():
        async with Sandbox(SandboxConfig()) as sandbox:
            echoed = await sandbox.execute_command("cat", stdin=b"from stdin")
            assert echoed.stdout == "from stdin"
            # Without stdin a reader sees end of input instead of waiting.
            closed = await sandbox.execute_command("cat; echo done", timeout=5)
            assert closed.stdout == "done\n" and not closed.timed_out
            assert sandbox.resolve_path("a/../b.txt") == os.path.join(sandbox.workdir, "b.txt")
            with pytest.raises(PermissionError):
                sandbox.resolve_path("../outside")

    asyncio.run(scenario())
