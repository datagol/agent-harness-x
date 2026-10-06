"""Persistent memory: the built-in memory tools, plus your own tools over the same stores.

`register_memory_tools` gives the agent `save_memory` and `recall_memories`, bound to the stores you pass:
`LongTermMemory` for discrete facts and `AgentMemory` for titled markdown notes. Both live on disk, so a
later session recalls what an earlier one saved. Your own tools can share those stores: `save_fact` is a
narrow shortcut the model picks reliably for facts, and `forget_fact` adds the delete the built-ins leave out.

Run:
    python examples/04-context/custom_memory_tools.py

Memories are written under .agent_memory/ in the current directory. Type `memories` to list them.

Needs: ANTHROPIC_API_KEY
"""

import asyncio
import json

from dotenv import load_dotenv

from harnessx import Agent, AgentConfig, AgentMemory, LongTermMemory, PermissionLevel, RunEventType, ToolCall, ToolResult
from harnessx.builtin import register_memory_tools

load_dotenv()  # finds the repository's .env from this file's folder; never overrides set variables

MEMORY_DIR = ".agent_memory/agent"
LONG_TERM_DIR = ".agent_memory/long_term"


async def stream_reply(agent: Agent, text: str) -> None:
    midline = False  # streamed text has no trailing newline until the answer ends
    async with agent.run_stream(text) as stream:
        async for event in stream:
            if event.type is RunEventType.TEXT_DELTA:
                print(event.data, end="", flush=True)
                midline = True
            elif event.type is RunEventType.TOOL_CALL_START and isinstance(event.data, ToolCall):
                args = json.dumps(event.data.input, default=str)[:120]
                print(("\n" if midline else "") + f"  > {event.data.name}({args})")
                midline = False
            elif event.type is RunEventType.TOOL_RESULT and isinstance(event.data, ToolResult) and event.data.is_error:
                print(f"  ! {event.data.content[:200]}")
        result = await stream.result()
    if midline:
        print()
    if result.status != "completed":
        print(f"Error: {result.error['message'] if result.error else result.status.value}")


async def main() -> None:
    notes = AgentMemory(storage_dir=MEMORY_DIR)
    facts = LongTermMemory(storage_dir=LONG_TERM_DIR)

    agent = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are a helpful assistant with persistent memory. Use save_fact for discrete facts "
                "(preferences, project details, decisions) and save_memory with layer='agent' for longer "
                "notes. Check recall_memories before answering questions about the user or past work. "
                "Remove a fact with forget_fact when the user says it no longer holds."
            ),
        ),
    )
    async with agent:
        # The built-ins, bound to these stores rather than the default ones.
        register_memory_tools(agent.tools, long_term=facts, agent_memory=notes)

        @agent.tools.register(permission=PermissionLevel.ALLOW)
        async def save_fact(content: str, category: str = "general") -> str:
            """Save a fact to long-term memory.

            Args:
                content: The fact to remember (e.g., 'User prefers dark mode').
                category: Category for organization (e.g., 'preference', 'project', 'decision').
            """
            memory_id = facts.save(content, category=category)
            return f"Saved fact [{memory_id}] in category '{category}'"

        @agent.tools.register(permission=PermissionLevel.ALLOW)
        async def forget_fact(memory_id: str) -> str:
            """Delete a fact from long-term memory.

            Args:
                memory_id: The fact's id, as shown in brackets by recall_memories (long_term/<id>).
            """
            return f"Deleted fact [{memory_id}]." if facts.delete(memory_id) else f"Fact '{memory_id}' not found."

        print(f"Memory agent: {len(notes.list_all())} notes, {len(facts.list_all())} facts on disk.")
        print("Commands: memories, usage, quit")
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit"):
                break
            if not text:
                continue
            if text.lower() == "usage":
                print(f"Usage: {agent.guardrails.usage_summary}")
                continue
            if text.lower() == "memories":
                for note in notes.list_all():
                    print(f"  note {note.get('filename')}: {note.get('title', '?')}")
                for fact in facts.list_all():
                    print(f"  fact [{fact['id']}] ({fact.get('category', '?')}) {fact['content'][:80]}")
                continue
            try:
                await stream_reply(agent, text)
            except Exception as exc:
                print(f"Error: {exc}")

        print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
