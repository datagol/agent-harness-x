"""Agent with persistent memory: saves and recalls learned context.

Demonstrates AgentMemory — the agent can save notes, decisions, and
context as markdown files, then search and recall them later in the
conversation. Memories persist across sessions on disk.

Run: python -m examples.memory_agent
"""

import asyncio
import json

from datagol_agent_harness import AgentConfig, AgentMemory, LongTermMemory, PermissionLevel, StreamingAgent
from examples._console import (
    console,
    get_user_input,
    handle_stream_event,
    print_banner,
    print_error,
    print_status,
)
from datagol_agent_harness.builtin import register_all_tools

MEMORY_DIR = ".agent_memory/agent"
LONG_TERM_DIR = ".agent_memory/long_term"


async def main():
    memory = AgentMemory(storage_dir=MEMORY_DIR)
    long_term = LongTermMemory(storage_dir=LONG_TERM_DIR)

    agent = StreamingAgent(
        config=AgentConfig(
            system_prompt=(
                "You are a helpful assistant with two layers of persistent memory.\n\n"
                "AGENT MEMORY (markdown notes — for context, decisions, project notes):\n"
                "- save_memory: save a titled note with tags\n"
                "- search_memory / list_memories: find past notes\n"
                "- read_memory / delete_memory: read or remove a note\n\n"
                "LONG-TERM MEMORY (structured facts — for specific facts, preferences, learned info):\n"
                "- save_fact: store a categorized fact\n"
                "- search_facts: search by keyword or category\n"
                "- list_facts / read_fact / update_fact / delete_fact\n\n"
                "Use agent memory for rich notes and context. "
                "Use long-term memory for discrete facts (user preferences, "
                "project details, decisions). Proactively save useful information "
                "so you can reference it later."
            ),
        ),
    )
    register_all_tools(agent.tools)

    # Register memory tools
    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def save_memory(title: str, content: str, tags: str = "") -> str:
        """Save a note to persistent memory.

        Args:
            title: Short title for the memory (e.g., 'User preferences').
            content: The information to remember.
            tags: Comma-separated tags for categorization (e.g., 'project,decision').
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        filename = memory.save(title, content, tags=tag_list)
        return f"Saved memory '{title}' -> {filename}"

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def search_memory(query: str = "", tags: str = "") -> str:
        """Search saved memories by keyword or tags.

        Args:
            query: Keyword to search for in memory content.
            tags: Comma-separated tags to filter by.
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        results = memory.search(query=query, tags=tag_list)
        if not results:
            return "No memories found."
        out = []
        for r in results:
            out.append(
                f"- [{r.get('filename')}] {r.get('title', '?')} "
                f"(tags: {r.get('tags', [])}, updated: {r.get('updated', '?')})"
            )
        return "\n".join(out)

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def list_memories() -> str:
        """List all saved memories."""
        all_mem = memory.list_all()
        if not all_mem:
            return "No memories saved yet."
        out = []
        for m in all_mem:
            out.append(
                f"- [{m.get('filename')}] {m.get('title', '?')} "
                f"(tags: {m.get('tags', [])}, updated: {m.get('updated', '?')})"
            )
        return "\n".join(out)

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def read_memory(filename: str) -> str:
        """Read a specific memory file in full.

        Args:
            filename: The filename of the memory (e.g., 'user-preferences.md').
        """
        result = memory.read(filename)
        if result is None:
            return f"Memory '{filename}' not found."
        return json.dumps(result, indent=2)

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def delete_memory(filename: str) -> str:
        """Delete a memory file.

        Args:
            filename: The filename of the memory to delete.
        """
        if memory.delete(filename):
            return f"Deleted memory '{filename}'."
        return f"Memory '{filename}' not found."

    # Register long-term memory tools
    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def save_fact(content: str, category: str = "general") -> str:
        """Save a structured fact to long-term memory.

        Args:
            content: The fact to remember (e.g., 'User prefers dark mode').
            category: Category for organization (e.g., 'preference', 'project', 'decision').
        """
        memory_id = long_term.save(content, category=category)
        return f"Saved fact [{memory_id}] in category '{category}'"

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def search_facts(query: str = "", category: str = "") -> str:
        """Search long-term memory by keyword or category.

        Args:
            query: Keyword to search for.
            category: Filter by category (leave empty for all).
        """
        results = long_term.search(query=query, category=category or None)
        if not results:
            return "No facts found."
        out = []
        for r in results:
            out.append(
                f"- [{r['id']}] ({r.get('category', '?')}) {r['content']}"
            )
        return "\n".join(out)

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def list_facts() -> str:
        """List all facts in long-term memory."""
        all_facts = long_term.list_all()
        if not all_facts:
            return "No facts saved yet."
        out = []
        for f in all_facts:
            out.append(f"- [{f['id']}] ({f.get('category', '?')}) {f['content']}")
        return "\n".join(out)

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def read_fact(memory_id: str) -> str:
        """Read a specific fact by ID.

        Args:
            memory_id: The ID of the fact to read.
        """
        result = long_term.get(memory_id)
        if result is None:
            return f"Fact '{memory_id}' not found."
        return json.dumps(result, indent=2)

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def update_fact(memory_id: str, content: str) -> str:
        """Update an existing fact.

        Args:
            memory_id: The ID of the fact to update.
            content: The new content.
        """
        if long_term.update(memory_id, content):
            return f"Updated fact [{memory_id}]."
        return f"Fact '{memory_id}' not found."

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def delete_fact(memory_id: str) -> str:
        """Delete a fact from long-term memory.

        Args:
            memory_id: The ID of the fact to delete.
        """
        if long_term.delete(memory_id):
            return f"Deleted fact [{memory_id}]."
        return f"Fact '{memory_id}' not found."

    # Show existing memories on startup
    existing = memory.list_all()
    if existing:
        console.print(f"  [info]Loaded {len(existing)} agent memories from {MEMORY_DIR}/[/info]")
        for m in existing[:5]:
            console.print(f"    [dim]-[/dim] {m.get('title', m.get('filename', '?'))}")
        if len(existing) > 5:
            console.print(f"    [dim]... and {len(existing) - 5} more[/dim]")
    else:
        console.print(f"  [info]No agent memories in {MEMORY_DIR}/[/info]")

    existing_facts = long_term.list_all()
    if existing_facts:
        console.print(f"  [info]Loaded {len(existing_facts)} long-term facts from {LONG_TERM_DIR}/[/info]")
        for f in existing_facts[:5]:
            console.print(f"    [dim]-[/dim] [{f['id']}] {f['content'][:60]}")
        if len(existing_facts) > 5:
            console.print(f"    [dim]... and {len(existing_facts) - 5} more[/dim]")
    else:
        console.print(f"  [info]No long-term facts in {LONG_TERM_DIR}/[/info]")

    print_banner(
        "DataGOL Agent Harness — Memory Agent",
        subtitle="Agent saves and recalls context across conversations",
        commands={"quit": "Exit", "usage": "Token stats", "memories": "List saved memories"},
    )

    while True:
        user_input = get_user_input()
        if user_input is None or user_input.lower() == "quit":
            break
        if not user_input:
            continue
        if user_input.lower() == "usage":
            print_status({"Usage": agent.guardrails.usage_summary})
            continue
        if user_input.lower() == "memories":
            all_mem = memory.list_all()
            all_facts = long_term.list_all()
            console.print("  [bold]Agent Memories (notes):[/bold]")
            if not all_mem:
                console.print("    [dim]None[/dim]")
            else:
                for m in all_mem:
                    tags = m.get("tags", [])
                    tag_str = f" [dim]({', '.join(tags)})[/dim]" if tags else ""
                    console.print(f"    [bold cyan]{m.get('filename')}[/bold cyan]  {m.get('title', '?')}{tag_str}")
            console.print("  [bold]Long-Term Facts:[/bold]")
            if not all_facts:
                console.print("    [dim]None[/dim]")
            else:
                for f in all_facts:
                    console.print(f"    [bold cyan][{f['id']}][/bold cyan]  ({f.get('category', '?')}) {f['content'][:80]}")
            continue

        try:
            console.print()
            async for event in agent.run_stream(user_input):
                handle_stream_event(event)
        except Exception as e:
            print_error(e)

    print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
