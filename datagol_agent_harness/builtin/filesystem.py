"""Built-in filesystem tools: read, write, list directory."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

from datagol_agent_harness.types import PermissionLevel

if TYPE_CHECKING:
    from datagol_agent_harness.tools import ToolRegistry


def _resolve_path(path: str, base_path: str | None = None) -> str:
    """Resolve a path safely, optionally enforcing a base directory boundary."""
    if base_path is None:
        return os.path.abspath(path)

    abs_base = os.path.abspath(base_path)
    if os.path.isabs(path):
        abs_target = os.path.abspath(path)
    else:
        abs_target = os.path.abspath(os.path.join(abs_base, path))

    # Path traversal check
    if os.path.commonpath([abs_base, abs_target]) != abs_base:
        raise PermissionError(f"Access denied: path '{path}' escapes base directory '{base_path}'")

    return abs_target


async def generate_file(path: str, content: str, display_name: str = "") -> str:
    """Generate a file and make it available for download in the UI.

    Use this when the user asks for a downloadable file (CSV, JSON, text, etc.).

    Args:
        path: Filename for the generated file (e.g., 'report.csv').
        content: The file content to write.
        display_name: Human-readable name for the download link.
    """
    from datagol_agent_harness.builtin.file_output import make_downloadable

    full_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else ".", exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    return make_downloadable(full_path, display_name or path)


async def read_file(path: str, offset: int = 0, limit: int = 100_000) -> str:
    """Read contents of a file with line numbers.

    Args:
        path: Absolute or relative path to the file.
        offset: Line number to start from (0-based).
        limit: Maximum number of lines to read (default: 100000).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    selected = lines[offset : offset + limit]
    numbered = [f"{offset + i + 1}\t{line}" for i, line in enumerate(selected)]
    return "".join(numbered)


async def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates the file and parent directories if they don't exist.

    Args:
        path: Path to the file.
        content: Content to write.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written {len(content)} bytes to {path}"


async def list_directory(path: str = ".") -> str:
    """List files and directories at the given path.

    Args:
        path: Directory path to list. Defaults to current directory.
    """
    entries = os.listdir(path)
    result_lines: list[str] = []
    for entry in sorted(entries):
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            result_lines.append(f"  {entry}/")
        else:
            size = os.path.getsize(full)
            result_lines.append(f"  {entry} ({size} bytes)")
    return f"Directory: {path}\n" + "\n".join(result_lines) if result_lines else f"Directory: {path} (empty)"


def register_filesystem_tools(
    registry: ToolRegistry,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    permission: PermissionLevel | None = None,
    base_path: str | None = None,
) -> list[str]:
    """Register file system tools onto a registry.

    Args:
        registry: Target tool registry.
        include: Specific tool names to register ('read_file', 'write_file', 'list_directory', 'generate_file').
        exclude: Tool names to omit.
        permission: Override default permission level for registered tools.
        base_path: Optional directory to restrict filesystem access to.

    Returns:
        List of registered tool names.
    """
    tools_map: dict[str, tuple[Callable, PermissionLevel]] = {
        "read_file": (read_file, PermissionLevel.ALLOW),
        "write_file": (write_file, PermissionLevel.ASK),
        "list_directory": (list_directory, PermissionLevel.ALLOW),
        "generate_file": (generate_file, PermissionLevel.ALLOW),
    }

    if base_path is not None:
        async def scoped_read_file(path: str, offset: int = 0, limit: int = 100_000) -> str:
            """Read contents of a file with line numbers.

            Args:
                path: Relative or absolute path to the file.
                offset: Line number to start from (0-based).
                limit: Maximum number of lines to read.
            """
            resolved = _resolve_path(path, base_path)
            return await read_file(resolved, offset=offset, limit=limit)

        async def scoped_write_file(path: str, content: str) -> str:
            """Write content to a file. Creates the file if it doesn't exist.

            Args:
                path: Path to the file.
                content: Content to write.
            """
            resolved = _resolve_path(path, base_path)
            return await write_file(resolved, content)

        async def scoped_list_directory(path: str = ".") -> str:
            """List files and directories at the given path.

            Args:
                path: Directory path to list.
            """
            resolved = _resolve_path(path, base_path)
            return await list_directory(resolved)

        async def scoped_generate_file(path: str, content: str, display_name: str = "") -> str:
            """Generate a file and make it available for download in the UI.

            Args:
                path: Filename for the generated file.
                content: The file content to write.
                display_name: Human-readable name for the download link.
            """
            resolved = _resolve_path(path, base_path)
            return await generate_file(resolved, content, display_name=display_name)

        tools_map = {
            "read_file": (scoped_read_file, PermissionLevel.ALLOW),
            "write_file": (scoped_write_file, PermissionLevel.ASK),
            "list_directory": (scoped_list_directory, PermissionLevel.ALLOW),
            "generate_file": (scoped_generate_file, PermissionLevel.ALLOW),
        }

    registered: list[str] = []
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude) if exclude is not None else set()

    for name, (fn, default_perm) in tools_map.items():
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue

        perm = permission if permission is not None else default_perm
        registry.register_tool(fn, name=name, permission=perm)
        registered.append(name)

    return registered
