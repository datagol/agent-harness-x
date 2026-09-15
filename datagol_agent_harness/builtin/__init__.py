"""Built-in tools for the agent harness."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .bash import register_bash_tools, run_bash
from .file_output import make_downloadable
from .filesystem import (
    generate_file,
    list_directory,
    read_file,
    register_filesystem_tools,
    write_file,
)
from .memory import recall_memories, register_memory_tools, save_memory
from .web import fetch_url, register_web_tools

if TYPE_CHECKING:
    from datagol_agent_harness.tools import ToolRegistry
    from datagol_agent_harness.types import PermissionLevel


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
    registered: list[str] = []
    registered.extend(
        register_filesystem_tools(
            registry,
            include=include,
            exclude=exclude,
            permission=permission,
            base_path=base_path,
        )
    )
    registered.extend(
        register_bash_tools(
            registry,
            sandbox=sandbox,
            include=include,
            exclude=exclude,
            permission=permission,
        )
    )
    registered.extend(
        register_web_tools(
            registry,
            include=include,
            exclude=exclude,
            permission=permission,
        )
    )
    registered.extend(
        register_memory_tools(
            registry,
            long_term=long_term,
            agent_memory=agent_memory,
            vector_store=vector_store,
            include=include,
            exclude=exclude,
            permission=permission,
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
    # Standalone tools
    "generate_file",
    "read_file",
    "write_file",
    "list_directory",
    "run_bash",
    "fetch_url",
    "save_memory",
    "recall_memories",
    "make_downloadable",
]
