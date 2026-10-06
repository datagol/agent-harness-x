"""Regression coverage for tool contracts and filesystem access boundaries."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal, Optional
from unittest.mock import AsyncMock

import pytest
from jsonschema import ValidationError, validate

from harnessx import PermissionLevel, ToolRegistry
from harnessx.builtin import register_all_tools, register_filesystem_tools
from harnessx.builtin import filesystem
from harnessx.extensions.base import ExtensionContext
from harnessx.permissions import PermissionManager
from harnessx.types import ToolCall, ToolDefinition


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def scoped(root, **options):
    registry = ToolRegistry()
    register_filesystem_tools(registry, base_path=str(root), permission=PermissionLevel.ALLOW, **options)
    return registry


async def call(registry, name, **inputs):
    return await registry.execute(ToolCall("test", name, inputs))


@pytest.mark.asyncio
@pytest.mark.parametrize("loader", ["bundle", "single", "all"])
async def test_all_registration_paths_confine_reads(workspace, loader):
    outside = workspace.parent / "outside.txt"
    outside.write_text("outside fixture")
    (workspace / "inside.txt").write_text("inside fixture")
    registry = ToolRegistry()
    if loader == "bundle":
        registry.load_builtin("filesystem", base_path=str(workspace), include=["read_file"])
    elif loader == "single":
        registry.load_builtin("read_file", base_path=str(workspace))
    else:
        register_all_tools(registry, base_path=str(workspace), include=["read_file"])
    for path in (str(outside), "../outside.txt"):
        result = await call(registry, "read_file", path=path)
        assert result.is_error
        assert "outside fixture" not in result.content
    result = await call(registry, "read_file", path=str(workspace / "inside.txt"))
    assert not result.is_error
    assert "inside fixture" in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["inside", "outside", "dangling"])
async def test_scoped_tools_reject_final_symlinks(workspace, destination):
    target = (workspace if destination == "inside" else workspace.parent) / "target.txt"
    if destination != "dangling":
        target.write_text("original")
    (workspace / "link.txt").symlink_to(target)
    registry = scoped(workspace)
    for name, inputs in (
        ("read_file", {}),
        ("write_file", {"content": "replacement", "overwrite": True}),
        ("generate_file", {"content": "replacement", "overwrite": True}),
    ):
        result = await call(registry, name, path="link.txt", **inputs)
        assert result.is_error
    if destination == "dangling":
        assert not target.exists()
    else:
        assert target.read_text() == "original"


@pytest.mark.asyncio
async def test_parent_symlinks_block_reads_writes_lists_and_downloads(workspace):
    outside = workspace.parent / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("original")
    (workspace / "linked").symlink_to(outside, target_is_directory=True)
    registry = scoped(workspace)
    for name, inputs in (
        ("read_file", {"path": "linked/secret.txt"}),
        ("write_file", {"path": "linked/new/deep.txt", "content": "new"}),
        ("generate_file", {"path": "linked/report.txt", "content": "new"}),
        ("list_directory", {"path": "linked"}),
    ):
        result = await call(registry, name, **inputs)
        assert result.is_error
    assert list(outside.iterdir()) == [outside / "secret.txt"]
    result = await call(registry, "list_directory")
    assert not result.is_error
    assert "symlink; access disabled" in result.content
    assert "secret.txt" not in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("swap_parent", [False, True])
async def test_symlink_swap_immediately_before_open_is_rejected(workspace, monkeypatch, swap_parent):
    outside = workspace.parent / "outside"
    outside.mkdir()
    (outside / "data.txt").write_text("outside fixture")
    folder = workspace / "folder"
    folder.mkdir()
    target = folder / "data.txt"
    target.write_text("inside")
    registry = scoped(workspace)
    original_open = filesystem.os.open
    changed = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal changed
        if not changed and path == ("folder" if swap_parent else "data.txt") and kwargs.get("dir_fd") is not None:
            changed = True
            if swap_parent:
                folder.rename(workspace / "held")
                folder.symlink_to(outside, target_is_directory=True)
            else:
                target.unlink()
                target.symlink_to(outside / "data.txt")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(filesystem.os, "open", racing_open)
    result = await call(registry, "read_file", path="folder/data.txt")
    assert changed and result.is_error
    assert "outside fixture" not in result.content


@pytest.mark.asyncio
async def test_write_keeps_open_parent_when_directory_is_replaced_by_link(workspace, monkeypatch):
    outside = workspace.parent / "outside"
    outside.mkdir()
    folder = workspace / "folder"
    folder.mkdir()
    registry = scoped(workspace)
    original_open = filesystem.os.open
    changed = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal changed
        if not changed and str(path).startswith(".harness-"):
            changed = True
            folder.rename(workspace / "held")
            folder.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(filesystem.os, "open", racing_open)
    result = await call(registry, "write_file", path="folder/new.txt", content="inside")
    assert changed and not result.is_error
    assert not (outside / "new.txt").exists()
    assert (workspace / "held/new.txt").read_text() == "inside"


@pytest.mark.asyncio
async def test_relative_base_is_fixed_at_registration(workspace, monkeypatch):
    (workspace / "data.txt").write_text("correct")
    monkeypatch.chdir(workspace.parent)
    registry = scoped("workspace")
    monkeypatch.chdir(workspace)
    assert "correct" in (await call(registry, "read_file", path="data.txt")).content


@pytest.mark.asyncio
async def test_writes_require_explicit_overwrite_and_count_utf8_bytes(workspace):
    registry = scoped(workspace)
    target = workspace / "nested/file.txt"
    result = await call(registry, "write_file", path="nested/file.txt", content="é")
    assert not result.is_error and "2 bytes" in result.content
    result = await call(registry, "write_file", path="nested/file.txt", content="new")
    assert result.is_error and target.read_text() == "é"
    result = await call(registry, "write_file", path="nested/file.txt", content="new", overwrite=True)
    assert not result.is_error and target.read_text() == "new"
    assert not list(target.parent.glob(".harness-*"))


@pytest.mark.asyncio
async def test_failed_atomic_replace_preserves_previous_file(workspace, monkeypatch):
    target = workspace / "data.txt"
    target.write_text("original")
    registry = scoped(workspace)

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(filesystem.os, "replace", fail)
    result = await call(registry, "write_file", path="data.txt", content="new", overwrite=True)
    assert result.is_error and target.read_text() == "original"
    assert not list(workspace.glob(".harness-*"))


@pytest.mark.asyncio
async def test_bounded_reads_offsets_listing_and_writes(workspace):
    (workspace / "large.txt").write_text("abcdefghij" * 100)
    (workspace / "lines.txt").write_text("one\ntwo\nthree\n")
    registry = scoped(workspace, max_read_bytes=20, max_write_bytes=3, max_directory_entries=1)
    result = await call(registry, "read_file", path="large.txt", limit=1)
    assert not result.is_error and "truncated" in result.content and len(result.content) < 100
    result = await call(registry, "read_file", path="lines.txt", offset=1, limit=1)
    assert result.content.startswith("2\ttwo\n") and "truncated" in result.content
    for inputs in ({"offset": -1}, {"limit": -1}):
        assert (await call(registry, "read_file", path="lines.txt", **inputs)).is_error
    assert (await call(registry, "read_file", path="lines.txt", limit=0)).content == ""
    assert "truncated" in (await call(registry, "list_directory")).content
    assert (await call(registry, "write_file", path="new.txt", content="éé")).is_error
    assert not (workspace / "new.txt").exists()


@pytest.mark.asyncio
async def test_fifo_is_rejected_without_blocking(workspace):
    filesystem.os.mkfifo(workspace / "pipe")
    result = await asyncio.wait_for(call(scoped(workspace), "read_file", path="pipe"), timeout=2)
    assert result.is_error and "regular files" in result.content


@pytest.mark.asyncio
async def test_downloads_use_configured_output_and_require_permission(workspace):
    output = workspace.parent / "downloads"
    registry = ToolRegistry()
    register_filesystem_tools(registry, base_path=str(workspace), output_dir=str(output), permission=PermissionLevel.ASK)
    assert registry.get_tool("generate_file").permission_level == PermissionLevel.ASK
    assert not registry.get_tool("write_file").concurrent
    denied = await call(registry, "generate_file", path="report.txt", content="hello")
    assert denied.is_error and not (workspace / "report.txt").exists()
    permissions = PermissionManager()
    permissions.grant_session("generate_file")
    result = await registry.execute(ToolCall("download", "generate_file", {"path": "report.txt", "content": "hello"}), permissions=permissions)
    assert not result.is_error and result.content.startswith("__FILE__:/files/")
    downloads = list(output.glob("*/report.txt"))
    assert len(downloads) == 1 and downloads[0].read_text() == "hello"


def sample(value: str) -> str:
    return value


def test_definition_overrides_are_isolated_and_schemas_are_copied():
    original = ToolDefinition("sample", "original", {"type": "object", "properties": {"value": {"type": "string"}}}, sample, permission_level=PermissionLevel.ASK)
    first, second = ToolRegistry(), ToolRegistry()
    first.register_tool(original)
    changed = second.register_tool(original, name="renamed", description="changed", permission=PermissionLevel.ALLOW,
                                   concurrent=False, replay_policy="safe", timeout_seconds=1)
    assert changed.name == "renamed" and changed.description == "changed"
    assert changed.timeout_seconds == 1 and changed.replay_policy == "safe" and not changed.concurrent
    assert original.permission_level == first.get_tool("sample").permission_level == PermissionLevel.ASK
    original.input_schema["properties"]["value"]["type"] = "integer"
    assert changed.input_schema["properties"]["value"]["type"] == "string"
    exported = second.get_tool_params()
    exported[0]["input_schema"]["properties"].clear()
    assert "value" in changed.input_schema["properties"]


@pytest.mark.parametrize("registration", ["function", "definition", "schema", "decorator"])
def test_duplicates_require_explicit_replacement(registration):
    registry = ToolRegistry()
    registry.register_tool(sample)

    def register(replace=False):
        if registration == "function":
            return registry.register_tool(sample, replace=replace)
        if registration == "definition":
            return registry.register_tool(registry.get_tool("sample"), replace=replace)
        if registration == "schema":
            return registry.register_with_schema("sample", "sample", {"type": "object"}, sample, replace=replace)
        return registry.register(replace=replace)(sample)

    with pytest.raises(ValueError, match="already registered"):
        register()
    register(replace=True)
    assert registry.list_tools() == ["sample"]


@pytest.mark.parametrize("bundle", ["filesystem", "read_file", "bash", "web", "memory", "all"])
def test_unknown_filters_and_options_are_rejected(bundle):
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="Unknown tool"):
        registry.load_builtin(bundle, include=["read_flie"])
    with pytest.raises(ValueError, match="Unknown tool"):
        registry.load_builtin(bundle, exclude=["read_flie"])
    with pytest.raises(TypeError):
        registry.load_builtin(bundle, imaginary_option=True)
    assert not registry.list_tools()


def test_bundle_collision_preflight_does_not_partially_register():
    registry = ToolRegistry()
    registry.load_builtin("write_file")
    with pytest.raises(ValueError):
        registry.load_builtin("filesystem")
    assert registry.list_tools() == ["write_file"]
    registry.load_builtin("filesystem", replace=True)
    assert len(registry.list_tools()) == 8


@pytest.mark.asyncio
async def test_individual_bash_preserves_sandbox_option():
    sandbox = SimpleNamespace(execute_command=AsyncMock(return_value=SimpleNamespace(stdout="sandbox", stderr="", timed_out=False, exit_code=0)))
    registry = ToolRegistry()
    registry.load_builtin("run_bash", sandbox=sandbox, permission=PermissionLevel.ALLOW)
    result = await call(registry, "run_bash", command="fixture command")
    assert result.content == "sandbox"
    sandbox.execute_command.assert_awaited_once_with("fixture command")


def test_nullable_required_literal_and_nested_schemas():
    def typed(required: Optional[int], modern: int | None, choice: Literal["a", "b"], values: list[dict[str, int]], defaulted: str | None = None):
        return required

    schema = ToolRegistry().register_tool(typed).input_schema
    inputs = {"required": None, "modern": 3, "choice": "a", "values": [{"count": 2}]}
    validate(inputs, schema)
    for invalid in (
        {k: v for k, v in inputs.items() if k != "required"},
        {**inputs, "modern": "wrong"}, {**inputs, "choice": "c"},
        {**inputs, "values": [{"count": "wrong"}]}, {**inputs, "extra": 1},
    ):
        with pytest.raises(ValidationError):
            validate(invalid, schema)


def test_annotated_and_mixed_literal_keep_json_value_types():
    def typed(value: Annotated[int, {"description": "count"}], choice: Literal["yes", 1, None]):
        pass

    schema = ToolRegistry().register_tool(typed).input_schema
    for choice in ("yes", 1, None):
        validate({"value": 2, "choice": choice}, schema)
    for choice in (True, "1", "no"):
        with pytest.raises(ValidationError):
            validate({"value": 2, "choice": choice}, schema)


def test_unsupported_signatures_and_annotations_fail_registration():
    def positional(value: str, /): pass
    def variadic(*values: str): pass
    def custom(value: Path): pass
    def unresolved(value: "MissingToolType"): pass  # noqa: F821  (deliberately unresolvable)
    for handler in (positional, variadic, custom, unresolved):
        with pytest.raises(TypeError):
            ToolRegistry().register_tool(handler)
    for timeout in (float("nan"), float("inf"), -1, 0):
        with pytest.raises(ValueError):
            ToolRegistry().register_tool(sample, timeout_seconds=timeout)


@pytest.mark.asyncio
async def test_direct_execution_validates_and_authorizes_before_side_effects():
    calls = []

    def side_effect(value: int):
        calls.append(value)
        return {"value": value}

    registry = ToolRegistry()
    registry.register_tool(side_effect)
    result = await call(registry, "side_effect", value=1)
    assert not result.is_error and result.content == '{"value": 1}'
    permissions = PermissionManager(PermissionLevel.ASK)
    assert (await registry.execute(ToolCall("ask", "side_effect", {"value": 2}), permissions=permissions)).is_error
    permissions.grant_session("side_effect")
    result = await registry.execute(ToolCall("allowed", "side_effect", {"value": 2}), permissions=permissions)
    assert not result.is_error and result.content == '{"value": 2}'
    invalid = await registry.execute(ToolCall("invalid", "side_effect", {"value": "bad"}), permissions=permissions)
    assert invalid.is_error
    permissions.set_permission("side_effect", PermissionLevel.DENY)
    assert (await registry.execute(ToolCall("denied", "side_effect", {"value": 2}), permissions=permissions)).is_error
    registry.register_tool(side_effect, permission=PermissionLevel.ASK, replace=True)
    assert (await call(registry, "side_effect", value=3)).is_error
    registry.register_tool(side_effect, permission=PermissionLevel.DENY, replace=True)
    assert (await call(registry, "side_effect", value=3)).is_error
    assert calls == [1, 2]


@pytest.mark.asyncio
async def test_direct_execution_applies_async_timeout():
    stopped = asyncio.Event()

    async def slow():
        try:
            await asyncio.sleep(10)
        finally:
            stopped.set()

    registry = ToolRegistry()
    registry.register_tool(slow, permission=PermissionLevel.ALLOW, timeout_seconds=0.01)
    result = await call(registry, "slow")
    assert result.is_error and "TimeoutError" in result.content and stopped.is_set()


def test_extension_registration_preserves_execution_metadata():
    registry = ToolRegistry()
    ctx = ExtensionContext(SimpleNamespace(tools=registry))
    ctx.register_tool(sample, replay_policy="safe", timeout_seconds=2, concurrent=False)
    definition = registry.get_tool("sample")
    assert definition.replay_policy == "safe" and definition.timeout_seconds == 2 and not definition.concurrent


@pytest.mark.asyncio
async def test_example_calculator_rejects_code_and_reports_tool_errors():
    from examples._calculator import calculate

    assert calculate("(25 * 4) + 50") == "150"
    assert calculate("144 / 12") == "12.0"
    registry = ToolRegistry()
    registry.register_tool(calculate, permission=PermissionLevel.ALLOW)
    for expression in ("().__class__", "__import__('os')", "'a' * 3", "2 ** 100000", "1 / 0", "1e999", "True + 1"):
        assert (await call(registry, "calculate", expression=expression)).is_error


# ── edit_file, delete, glob, grep ────────────────────────────────────────────


async def _call(registry, name, **arguments):
    return (await registry.execute(ToolCall("1", name, arguments))).content.strip()


@pytest.mark.asyncio
async def test_edit_file_requires_a_unique_match_unless_told_otherwise(workspace):
    (workspace / "dup.txt").write_text("x = 1\ny = 2\nx = 1\n")
    registry = scoped(workspace)

    ambiguous = await _call(registry, "edit_file", path="dup.txt", old_string="x = 1", new_string="x = 9")
    assert "matches 2 times" in ambiguous, "a silent partial edit is worse than a refusal"
    assert (workspace / "dup.txt").read_text().count("x = 1") == 2, "nothing was written"

    assert "2 occurrence" in await _call(
        registry, "edit_file", path="dup.txt", old_string="x = 1", new_string="x = 9", replace_all=True
    )
    assert (workspace / "dup.txt").read_text() == "x = 9\ny = 2\nx = 9\n"


@pytest.mark.asyncio
async def test_edit_file_rejects_a_missing_or_pointless_edit(workspace):
    (workspace / "a.txt").write_text("hello\n")
    registry = scoped(workspace)
    assert "was not found" in await _call(registry, "edit_file", path="a.txt", old_string="nope", new_string="x")
    assert "identical" in await _call(registry, "edit_file", path="a.txt", old_string="hello", new_string="hello")
    assert "nonempty" in await _call(registry, "edit_file", path="a.txt", old_string="", new_string="x")
    assert (workspace / "a.txt").read_text() == "hello\n"


@pytest.mark.asyncio
async def test_edit_file_is_confined_and_refuses_symlinks(workspace, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    (workspace / "link.txt").symlink_to(outside)
    registry = scoped(workspace)

    assert "escapes base directory" in await _call(
        registry, "edit_file", path="../outside.txt", old_string="secret", new_string="x"
    )
    assert "symlink" in await _call(registry, "edit_file", path="link.txt", old_string="secret", new_string="x")
    assert outside.read_text() == "secret\n"


@pytest.mark.asyncio
async def test_delete_removes_files_links_and_trees_but_never_the_root(workspace):
    (workspace / "file.txt").write_text("x")
    (workspace / "tree").mkdir()
    (workspace / "tree/nested").mkdir()
    (workspace / "tree/nested/deep.txt").write_text("y")
    (workspace / "empty").mkdir()
    registry = scoped(workspace)

    assert "Deleted file.txt" in await _call(registry, "delete", path="file.txt")
    assert not (workspace / "file.txt").exists()

    assert "empty directory" in await _call(registry, "delete", path="empty")
    assert not (workspace / "empty").exists()

    refused = await _call(registry, "delete", path="tree")
    assert "not empty" in refused.lower(), refused
    assert (workspace / "tree/nested/deep.txt").exists(), "a non-empty tree needs recursive=True"

    assert "2 entries" in await _call(registry, "delete", path="tree", recursive=True)
    assert not (workspace / "tree").exists()

    assert "base directory" in await _call(registry, "delete", path=".")
    assert workspace.exists()


@pytest.mark.asyncio
async def test_delete_removes_a_symlink_without_touching_its_target(workspace, tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("keep me\n")
    (workspace / "link.txt").symlink_to(target)
    registry = scoped(workspace)

    assert "symlink" in await _call(registry, "delete", path="link.txt")
    assert not (workspace / "link.txt").is_symlink()
    assert target.read_text() == "keep me\n", "the link was removed, not what it pointed at"


@pytest.mark.asyncio
async def test_glob_matches_by_name_and_relative_path_and_skips_links(workspace, tmp_path):
    (workspace / "pkg").mkdir()
    (workspace / "pkg/a.py").write_text("")
    (workspace / "pkg/b.txt").write_text("")
    (workspace / "top.py").write_text("")
    (workspace / "escape").symlink_to(tmp_path)
    registry = scoped(workspace)

    matched = await _call(registry, "glob", pattern="*.py")
    assert "pkg/a.py" in matched and "top.py" in matched and "b.txt" not in matched
    assert "pkg/" in await _call(registry, "glob", pattern="pkg*")
    assert "No matches" in await _call(registry, "glob", pattern="*.rs")
    assert "escape" not in matched, "a symlinked directory is never traversed"


@pytest.mark.asyncio
async def test_grep_reports_path_and_line_and_honors_include(workspace):
    (workspace / "a.py").write_text("import os\nvalue = 1\n")
    (workspace / "b.md").write_text("value = 1\n")
    registry = scoped(workspace)

    hits = await _call(registry, "grep", pattern=r"value = \d")
    assert "a.py:2:" in hits and "b.md:1:" in hits

    only_markdown = await _call(registry, "grep", pattern="value", include="*.md")
    assert "b.md" in only_markdown and "a.py" not in only_markdown

    assert "No matches" in await _call(registry, "grep", pattern="nowhere")
    assert "Invalid regular expression" in await _call(registry, "grep", pattern="(unclosed")


@pytest.mark.asyncio
async def test_grep_does_not_read_through_a_symlink(workspace, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET\n")
    (workspace / "link.txt").symlink_to(secret)
    registry = scoped(workspace)
    assert "No matches" in await _call(registry, "grep", pattern="TOPSECRET")


def test_the_builtin_bundle_offers_the_full_file_tool_set():
    registry = ToolRegistry()
    names = register_filesystem_tools(registry, permission=PermissionLevel.ALLOW)
    assert set(names) >= {"read_file", "write_file", "edit_file", "delete", "glob", "grep", "list_directory"}


def test_bash_can_start_in_a_given_directory(tmp_path):
    """A server holding many conversations gives each its own directory; the
    commands of one must not run in the server's own working directory."""
    import asyncio

    from harnessx.builtin.bash import register_bash_tools
    from harnessx.tools import ToolRegistry
    from harnessx.types import PermissionLevel, ToolCall

    (tmp_path / "marker.txt").write_text("here")
    registry = ToolRegistry()
    register_bash_tools(registry, cwd=str(tmp_path), permission=PermissionLevel.ALLOW)
    result = asyncio.run(registry.execute(ToolCall("1", "run_bash", {"command": "pwd && ls"})))
    assert str(tmp_path.resolve()) in result.content and "marker.txt" in result.content
    assert set(registry.get_tool("run_bash").input_schema["properties"]) == {"command", "timeout"}
