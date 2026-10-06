"""OpenShell as the execution backend, against a fake gateway that runs commands locally.

The fake stands a temporary directory in for each sandbox's root: ``/sandbox``
in a command becomes ``<root>/<name>/sandbox``. Everything else (the shell
snippets, tar, sha256sum) runs for real.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from typing import Any

import pytest

openshell = pytest.importorskip("openshell")
import grpc  # noqa: E402

from harnessx import Agent, AgentConfig, ProviderResponse, ToolCall, ToolRegistry  # noqa: E402
from harnessx.builtin import register_bash_tools, register_filesystem_tools  # noqa: E402
from harnessx.openshell import PROJECT_DIR, OpenShellSandbox, policy_denials  # noqa: E402
from harnessx.providers import LLMProvider  # noqa: E402
from harnessx.types import PermissionLevel  # noqa: E402

pytestmark = pytest.mark.skipif(os.name != "posix" or shutil.which("sha256sum") is None, reason="POSIX tools")


class NotFound(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.NOT_FOUND


class Ref:
    def __init__(self, name: str, sandbox_id: str) -> None:
        self.name, self.id = name, sandbox_id


class FakeGateway:
    def __init__(self, root) -> None:
        self.root = str(root)
        self.sandboxes: dict[str, str] = {}
        self.created: list[Any] = []
        self.deleted: list[str] = []
        self.commands: list[tuple[str, str, int | None, int]] = []  # name, script, timeout, stdin bytes
        self.closed = False

    def _base(self, name: str) -> str:
        return os.path.join(self.root, name)

    def get(self, name, *, workspace):
        if name not in self.sandboxes:
            raise NotFound()
        return Ref(name, self.sandboxes[name])

    def create(self, *, workspace, spec, name, labels):
        self.sandboxes[name] = f"id-{name}"
        self.created.append(spec)
        os.makedirs(os.path.join(self._base(name), "sandbox"))
        return Ref(name, self.sandboxes[name])

    def wait_ready(self, name, *, workspace):
        return self.get(name, workspace=workspace)

    def exec_stream(self, name, command, *, workspace, workdir=None, stdin=None, timeout_seconds=None):
        assert command[:2] == ["sh", "-c"]
        sandbox = os.path.join(self._base(name), "sandbox")
        script = command[2].replace("/sandbox", sandbox)
        cwd = (workdir or "/sandbox").replace("/sandbox", sandbox, 1)
        self.commands.append((name, command[2], timeout_seconds, len(stdin or b"")))
        try:
            done = subprocess.run(["sh", "-c", script], input=stdin or b"", capture_output=True,
                                  cwd=cwd, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            yield openshell.ExecResult(exit_code=124, stdout="", stderr="")
            return
        yield openshell.ExecChunk(stream="stdout", data=done.stdout)
        yield openshell.ExecChunk(stream="stderr", data=done.stderr)
        yield openshell.ExecResult(exit_code=done.returncode, stdout="", stderr="")

    def delete(self, name, *, workspace, allow_missing=False):
        self.deleted.append(name)
        self.sandboxes.pop(name, None)
        shutil.rmtree(self._base(name), ignore_errors=True)

    def close(self):
        self.closed = True

    def path(self, name: str, inside: str) -> str:
        return os.path.join(self._base(name), inside.lstrip("/"))


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('v1')\n")
    (root / "README.md").write_text("# repo\n")
    (root / ".venv").mkdir()
    (root / ".venv" / "big.bin").write_text("never copied")
    (root / "notes").mkdir()
    (root / "notes" / "todo.txt").write_text("one\n")
    return root


@pytest.fixture
def gateway(tmp_path):
    return FakeGateway(tmp_path / "gateway")


def _sandbox(project, gateway, **options):
    return OpenShellSandbox(project, client=gateway, name=options.pop("name", "box"), **options)


def _tools(sandbox):
    registry = ToolRegistry(sandbox=sandbox)
    register_filesystem_tools(registry, permission=PermissionLevel.ALLOW)
    register_bash_tools(registry, permission=PermissionLevel.ALLOW)
    return registry


def _call(registry, name, **arguments):
    return asyncio.run(registry.execute(ToolCall("1", name, arguments)))


def test_nothing_is_created_until_a_tool_needs_the_sandbox(project, gateway):
    sandbox = _sandbox(project, gateway)
    registry = _tools(sandbox)
    assert gateway.sandboxes == {} and not sandbox.started
    result = _call(registry, "read_file", path="src/app.py")
    assert result.content == "1\tprint('v1')\n"
    assert sandbox.started and list(gateway.sandboxes) == ["box"]


def test_the_project_is_copied_in_without_excluded_folders(project, gateway):
    sandbox = _sandbox(project, gateway)
    asyncio.run(sandbox.start())
    inside = gateway.path("box", PROJECT_DIR)
    assert open(os.path.join(inside, "src", "app.py")).read() == "print('v1')\n"
    assert not os.path.exists(os.path.join(inside, ".venv"))
    # Commands start in the project.
    assert sorted(_call(_tools(sandbox), "run_bash", command="ls").content.split()) == ["README.md", "notes", "src"]


def test_tools_change_the_sandbox_and_a_sync_brings_the_changes_home(project, gateway):
    sandbox = _sandbox(project, gateway)
    registry = _tools(sandbox)
    _call(registry, "edit_file", path="src/app.py", old_string="v1", new_string="v2")
    _call(registry, "write_file", path="src/new.py", content="x = 1\n")
    _call(registry, "delete", path="notes/todo.txt")
    _call(registry, "run_bash", command="echo built > out.txt && chmod 755 out.txt")
    # Nothing local changes until the sync.
    assert (project / "src" / "app.py").read_text() == "print('v1')\n"

    report = asyncio.run(sandbox.sync())
    assert sorted(report.updated) == ["out.txt", "src/app.py", "src/new.py"]
    assert report.deleted == ["notes/todo.txt"] and report.conflicts == []
    assert (project / "src" / "app.py").read_text() == "print('v2')\n"
    assert (project / "src" / "new.py").read_text() == "x = 1\n"
    assert not (project / "notes" / "todo.txt").exists()
    assert os.stat(project / "out.txt").st_mode & 0o777 == 0o755
    # A second sync with nothing new changes nothing.
    assert asyncio.run(sandbox.sync()) == type(report)()


def test_a_file_edited_locally_during_the_run_is_never_overwritten(project, gateway):
    sandbox = _sandbox(project, gateway)
    registry = _tools(sandbox)
    _call(registry, "write_file", path="README.md", content="# from the agent\n", overwrite=True)
    _call(registry, "delete", path="notes/todo.txt")
    (project / "README.md").write_text("# edited by hand\n")
    (project / "notes" / "todo.txt").write_text("one\ntwo\n")

    report = asyncio.run(sandbox.sync())
    assert sorted(report.conflicts) == ["README.md", "notes/todo.txt"] and report.updated == []
    assert (project / "README.md").read_text() == "# edited by hand\n"
    assert (project / "README.md.sandbox").read_text() == "# from the agent\n"
    assert (project / "notes" / "todo.txt").read_text() == "one\ntwo\n"


def test_paths_the_model_may_use(project, gateway, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "rows.csv").write_text("a,b\n")
    sandbox = _sandbox(project, gateway, inputs=[data])
    registry = _tools(sandbox)
    # A local path into the project is translated; so is one into an input.
    assert _call(registry, "read_file", path=str(project / "README.md")).content == "1\t# repo\n"
    assert _call(registry, "read_file", path=str(data / "rows.csv")).content == "1\ta,b\n"
    # Inputs are read-only, and anything else is not in the sandbox.
    refused = _call(registry, "write_file", path=str(data / "rows.csv"), content="x", overwrite=True)
    assert refused.is_error and "read-only input" in refused.content
    for outside in ("/etc/passwd", "../escape.txt", str(tmp_path / "elsewhere.txt")):
        result = _call(registry, "read_file", path=outside)
        assert result.is_error and "not available in the sandbox" in result.content, outside
    # Inputs never come back.
    _call(registry, "run_bash", command="echo changed > /sandbox/inputs/data/rows.csv")
    asyncio.run(sandbox.sync())
    assert (data / "rows.csv").read_text() == "a,b\n"


def test_large_content_is_sent_in_pieces(project, gateway):
    sandbox = _sandbox(project, gateway)
    registry = _tools(sandbox)
    content = "0123456789abcdef\n" * 400_000  # 6.8 MB, above one gRPC message
    assert not _call(registry, "write_file", path="big.txt", content=content).is_error
    assert max(size for *_, size in gateway.commands) <= 3 * 1024 * 1024
    asyncio.run(sandbox.sync())
    assert (project / "big.txt").read_text() == content


def test_timeouts_and_policy_denials_reach_the_model(project, gateway):
    sandbox = _sandbox(project, gateway, timeout_seconds=1)
    registry = _tools(sandbox)
    slow = _call(registry, "run_bash", command="sleep 5", timeout=30)
    assert "timed out" in slow.content and gateway.commands[-1][2] == 1  # capped at the sandbox's limit

    denied = 'HTTP/1.1 403 Forbidden\n{"error":"policy_denied","detail":"POST api.example.com:443 not allowed"}'
    result = _call(registry, "run_bash", command=f"printf '%s' '{denied}'")
    assert "blocked by the sandbox's network policy: POST api.example.com:443 not allowed" in result.content
    assert policy_denials("curl: (56) CONNECT tunnel failed\nCONNECT pypi.org:443 not permitted by policy") == ["pypi.org:443"]


def test_closing_syncs_then_deletes_unless_kept(project, gateway):
    sandbox = _sandbox(project, gateway)
    _call(_tools(sandbox), "write_file", path="late.txt", content="saved\n")
    asyncio.run(sandbox.aclose())
    assert (project / "late.txt").read_text() == "saved\n"
    assert gateway.deleted == ["box"] and not sandbox.started

    kept = _sandbox(project, gateway, name="kept", keep=True)
    asyncio.run(kept.start())
    asyncio.run(kept.aclose())
    assert "kept" in gateway.sandboxes and gateway.deleted == ["box"]


def test_a_kept_sandbox_is_found_again_without_copying_the_project_twice(project, gateway):
    first = _sandbox(project, gateway, keep=True)
    _call(_tools(first), "write_file", path="state.txt", content="1\n")
    asyncio.run(first.aclose())
    assert (project / "state.txt").exists()

    second = _sandbox(project, gateway, keep=True)
    registry = _tools(second)
    seeded = len(gateway.created)
    _call(registry, "write_file", path="state.txt", content="2\n", overwrite=True)
    assert len(gateway.created) == seeded  # reattached, not recreated
    assert not any("tar -xzf" in script for name, script, *_ in gateway.commands[-6:])
    report = asyncio.run(second.sync())
    assert report.updated == ["state.txt"] and (project / "state.txt").read_text() == "2\n"


def test_an_image_is_requested_when_given(project, gateway):
    asyncio.run(_sandbox(project, gateway, image="ghcr.io/example/tools:1").start())
    assert gateway.created[-1].template.image == "ghcr.io/example/tools:1"


# ── with an agent ────────────────────────────────────────────────────────────


class Scripted(LLMProvider):
    name = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)

    async def create(self, **kwargs):
        return self.responses.pop(0)

    async def count_tokens(self, **kwargs):
        return 0


def test_the_agent_owns_the_sandbox_and_the_user_never_touches_it(project, gateway):
    provider = Scripted([
        ProviderResponse(tool_calls=[ToolCall("c1", "edit_file", {
            "path": "src/app.py", "old_string": "v1", "new_string": "v2"})]),
        ProviderResponse(text="Fixed."),
    ])
    sandbox = OpenShellSandbox(project, client=gateway)

    async def scenario():
        async with Agent(AgentConfig(model="m", planning=False), provider=provider,
                         tools=["filesystem", "bash"], sandbox=sandbox) as agent:
            result = await agent.run("fix it")
            assert result.output == "Fixed."
            # Synced when the run ended, before the agent closed.
            assert (project / "src" / "app.py").read_text() == "print('v2')\n"
            name = agent.session_metadata["openshell"]["name"]
            assert name == "hx-" + agent.session_id.replace("-", "")[:20]
            return name

    name = asyncio.run(scenario())
    assert gateway.deleted == [name] and gateway.closed is False  # an injected client stays the caller's


def test_tools_registered_before_the_agent_move_into_the_sandbox(project, gateway, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry()
    register_filesystem_tools(registry, permission=PermissionLevel.ALLOW, max_read_bytes=5_000)
    register_bash_tools(registry, permission=PermissionLevel.ASK)
    sandbox = OpenShellSandbox(project, client=gateway, name="moved")
    Agent(AgentConfig(model="m", planning=False), provider=Scripted([]), tools=registry, sandbox=sandbox)

    assert _call(registry, "write_file", path="here.txt", content="in the sandbox").content.startswith("Written")
    _call(registry, "write_file", path="big.txt", content="x\n" * 4_000)  # above max_read_bytes=5_000
    assert not (tmp_path / "here.txt").exists()
    assert os.path.exists(gateway.path("moved", PROJECT_DIR + "/here.txt"))
    # The registration options came along, and bash runs there too.
    assert registry.get_tool("run_bash").permission_level == PermissionLevel.ASK
    assert registry.get_tool("run_bash").handler._harnessx_backend is sandbox
    assert "[truncated" in _call(registry, "read_file", path="big.txt").content


def test_one_sandbox_belongs_to_one_agent(project, gateway):
    sandbox = OpenShellSandbox(project, client=gateway)
    Agent(AgentConfig(model="m", planning=False), provider=Scripted([]), sandbox=sandbox)
    with pytest.raises(ValueError, match="another agent"):
        Agent(AgentConfig(model="m", planning=False), provider=Scripted([]), sandbox=sandbox)
