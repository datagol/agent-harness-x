"""Built-in tools for the agent harness."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .ask import register_ask_user_tool
from .bash import register_bash_tools, run_bash
from .file_output import make_downloadable
from .filesystem import (
    delete,
    edit_file,
    generate_file,
    glob,
    grep,
    list_directory,
    read_file,
    register_filesystem_tools,
    write_file,
)
from .memory import recall_memories, register_memory_tools, save_memory
from .planning import register_planning_tools
from .web import fetch_url, register_web_tools
from ._registration import select_tools

if TYPE_CHECKING:
    from harnessx.tools import ToolRegistry
    from harnessx.types import PermissionLevel


def register_all_tools(
    registry: ToolRegistry,
    sandbox: Any | None = None,
    long_term: Any | None = None,
    agent_memory: Any | None = None,
    vector_store: Any | None = None,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    permission: PermissionLevel | None = None,
    base_path: str | None = None,
    max_read_bytes: int = 1_000_000,
    max_write_bytes: int = 10_000_000,
    max_directory_entries: int = 1_000,
    output_dir: str | None = None,
    replace: bool = False,
) -> list[str]:
    """Convenience: register all built-in tools with optional filtering and permission overrides.

    Args:
        registry: Target tool registry.
        sandbox: Optional sandbox execution engine for bash.
        long_term: Optional LongTermMemory instance.
        agent_memory: Optional AgentMemory instance.
        vector_store: Optional VectorMemoryStore instance.
        include: Specific tool names to register.
        exclude: Tool names to skip.
        permission: Global permission override for all registered tools.
        base_path: Optional directory to restrict filesystem tools to.

    Returns:
        List of registered tool names.
    """
    filesystem_names = ["read_file", "write_file", "edit_file", "delete", "glob", "grep",
                        "list_directory", "generate_file"]
    selected = select_tools(registry, filesystem_names + ["run_bash", "fetch_url", "save_memory", "recall_memories"],
                            include, exclude, replace=replace)
    registered: list[str] = []
    registered.extend(
        register_filesystem_tools(
            registry,
            include=[name for name in selected if name in filesystem_names],
            permission=permission,
            base_path=base_path,
            max_read_bytes=max_read_bytes,
            max_write_bytes=max_write_bytes,
            max_directory_entries=max_directory_entries,
            output_dir=output_dir,
            replace=replace,
        )
    )
    registered.extend(
        register_bash_tools(
            registry,
            sandbox=sandbox,
            include=[name for name in selected if name == "run_bash"],
            permission=permission,
            replace=replace,
        )
    )
    registered.extend(
        register_web_tools(
            registry,
            include=[name for name in selected if name == "fetch_url"],
            permission=permission,
            replace=replace,
        )
    )
    registered.extend(
        register_memory_tools(
            registry,
            long_term=long_term,
            agent_memory=agent_memory,
            vector_store=vector_store,
            include=[name for name in selected if name in ("save_memory", "recall_memories")],
            permission=permission,
            replace=replace,
        )
    )
    return registered


__all__ = [
    # Registration helpers
    "register_all_tools",
    "register_filesystem_tools",
    "register_bash_tools",
    "register_web_tools",
    "register_memory_tools",
    "register_planning_tools",
    # Standalone tools
    "generate_file",
    "read_file",
    "write_file",
    "edit_file",
    "delete",
    "glob",
    "grep",
    "list_directory",
    "run_bash",
    "fetch_url",
    "save_memory",
    "recall_memories",
    "make_downloadable",
]
