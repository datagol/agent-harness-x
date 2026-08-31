"""Built-in filesystem tools: read, write, list directory."""

from __future__ import annotations

import os

from agent_harness.tools import ToolRegistry
from agent_harness.types import PermissionLevel


def register_filesystem_tools(registry: ToolRegistry) -> None:
    """Register file system tools onto a registry."""

    @registry.register(permission=PermissionLevel.ALLOW)
    async def generate_file(path: str, content: str, display_name: str = "") -> str:
        """Generate a file and make it available for download in the UI.

        Use this when the user asks for a downloadable file (CSV, JSON, text, etc.).

        Args:
            path: Filename for the generated file (e.g., 'report.csv').
            content: The file content to write.
            display_name: Human-readable name for the download link.
        """
        from agent_harness.builtin.file_output import make_downloadable

        full_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else ".", exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return make_downloadable(full_path, display_name or path)

    @registry.register(permission=PermissionLevel.ALLOW)
    async def read_file(path: str, offset: int = 0, limit: int = 100_000) -> str:
        """Read contents of a file.

        Args:
            path: Absolute or relative path to the file.
            offset: Line number to start from (0-based).
            limit: Maximum number of lines to read (default: 100000).
        """
        with open(path) as f:
            lines = f.readlines()

        selected = lines[offset : offset + limit]
        # Add line numbers like cat -n
        numbered = [f"{offset + i + 1}\t{line}" for i, line in enumerate(selected)]
        return "".join(numbered)

    @registry.register(permission=PermissionLevel.ASK)
    async def write_file(path: str, content: str) -> str:
        """Write content to a file. Creates the file if it doesn't exist.

        Args:
            path: Path to the file.
            content: Content to write.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Written {len(content)} bytes to {path}"

    @registry.register(permission=PermissionLevel.ALLOW)
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
