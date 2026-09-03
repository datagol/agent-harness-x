"""Built-in memory tools: save and recall across memory layers."""

from __future__ import annotations

import json
from typing import Any

from datagol_agent_harness.memory import AgentMemory, LongTermMemory, VectorMemoryStore
from datagol_agent_harness.tools import ToolRegistry
from datagol_agent_harness.types import PermissionLevel


def register_memory_tools(
    registry: ToolRegistry,
    long_term: LongTermMemory | None = None,
    agent_memory: AgentMemory | None = None,
    vector_store: VectorMemoryStore | None = None,
) -> None:
    """Register memory tools onto a registry."""

    lt = long_term or LongTermMemory()
    am = agent_memory or AgentMemory()

    @registry.register(permission=PermissionLevel.ALLOW)
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

    @registry.register(permission=PermissionLevel.ALLOW)
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
