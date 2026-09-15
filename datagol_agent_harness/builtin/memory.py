"""Built-in memory tools: save and recall across memory layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from datagol_agent_harness.memory import AgentMemory, LongTermMemory, VectorMemoryStore
from datagol_agent_harness.types import PermissionLevel

if TYPE_CHECKING:
    from datagol_agent_harness.tools import ToolRegistry


def make_memory_tools(
    long_term: LongTermMemory | None = None,
    agent_memory: AgentMemory | None = None,
    vector_store: VectorMemoryStore | None = None,
) -> tuple[Callable, Callable]:
    """Create a pair of bound (save_memory, recall_memories) tool handlers."""
    lt = long_term or LongTermMemory()
    am = agent_memory or AgentMemory()

    async def save_memory(
        content: str,
        layer: str = "long_term",
        category: str = "general",
        title: str = "",
        tags: str = "",
    ) -> str:
        """Save information to persistent memory.

        Args:
            content: The information to remember.
            layer: Where to store it — "long_term" for facts/data (DB-backed),
                   "agent" for learned context/decisions (markdown),
                   "preference" for personal preferences (vector DB).
            category: Category for long_term memories (e.g., "project", "user", "decision").
            title: Title for agent memories (required when layer="agent").
            tags: Comma-separated tags for agent memories.
        """
        if layer == "long_term":
            memory_id = lt.save(content, category=category)
            return f"Saved to long-term memory (id: {memory_id}, category: {category})"

        elif layer == "agent":
            if not title:
                title = content[:60]
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            filename = am.save(title, content, tags=tag_list)
            return f"Saved agent memory: {filename}"

        elif layer == "preference":
            if vector_store is None:
                return "Error: vector store not configured for preference storage."
            memory_id = await vector_store.add(content, metadata={"category": category})
            return f"Saved preference (id: {memory_id})"

        else:
            return f"Error: unknown layer '{layer}'. Use 'long_term', 'agent', or 'preference'."

    async def recall_memories(
        query: str = "",
        layer: str = "all",
        category: str = "",
        limit: int = 10,
    ) -> str:
        """Recall information from persistent memory.

        Args:
            query: Search query (keyword match for long_term/agent, semantic for preference).
            layer: Which layer to search — "long_term", "agent", "preference", or "all".
            category: Filter by category (long_term only).
            limit: Maximum results to return.
        """
        results: list[str] = []

        if layer in ("long_term", "all"):
            lt_results = lt.search(query=query, category=category or None, limit=limit)
            for r in lt_results:
                results.append(
                    f"[long_term/{r['id']}] ({r.get('category', '')}) {r['content']}"
                )

        if layer in ("agent", "all"):
            am_results = am.search(query=query)
            for r in am_results[:limit]:
                results.append(
                    f"[agent/{r.get('filename', '?')}] {r.get('title', '')} — {r.get('content', '')[:200]}"
                )

        if layer in ("preference", "all") and vector_store is not None:
            v_results = await vector_store.search(query=query, limit=limit)
            for r in v_results:
                results.append(
                    f"[preference/{r.get('id', '?')}] {r.get('text', '')}"
                )

        if not results:
            return "No memories found."

        return "\n\n".join(results)

    return save_memory, recall_memories


# Default unbound memory tools (instantiates default stores on call)
_default_save_memory, _default_recall_memories = make_memory_tools()
save_memory = _default_save_memory
recall_memories = _default_recall_memories


def register_memory_tools(
    registry: ToolRegistry,
    long_term: LongTermMemory | None = None,
    agent_memory: AgentMemory | None = None,
    vector_store: VectorMemoryStore | None = None,
    *,
    permission: PermissionLevel | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Register memory tools onto a registry.

    Args:
        registry: Target tool registry.
        long_term: Optional LongTermMemory instance.
        agent_memory: Optional AgentMemory instance.
        vector_store: Optional VectorMemoryStore instance.
        permission: Override default permission level (default PermissionLevel.ALLOW).
        include: Specific tool names to register ('save_memory', 'recall_memories').
        exclude: Tool names to omit.

    Returns:
        List of registered tool names.
    """
    save_fn, recall_fn = make_memory_tools(
        long_term=long_term,
        agent_memory=agent_memory,
        vector_store=vector_store,
    )

    tools_map = {
        "save_memory": save_fn,
        "recall_memories": recall_fn,
    }

    registered: list[str] = []
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude) if exclude is not None else set()
    perm = permission if permission is not None else PermissionLevel.ALLOW

    for name, fn in tools_map.items():
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue

        registry.register_tool(fn, name=name, permission=perm)
        registered.append(name)

    return registered
