"""Memory system for the agent harness.

Four layers:
  1. ConversationMemory — short-term, in-session message history
  2. PersistentMemory   — session save/resume (JSON files, simulating a DB)
  3. AgentMemory        — agent-specific learned context (markdown files)
  4. VectorMemoryStore  — personal preferences & semantic recall (vector DB)

Large tool results are evicted to disk and replaced with a head/tail
preview plus a file path reference, so the agent can re-read the full
output on demand via read_file.
"""

from __future__ import annotations

import glob as globmod
import json
import os
import re
import tempfile
import shutil
from dataclasses import asdict
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Protocol

import logging

from .types import SessionState, TokenUsage
from .messages import Message

logger = logging.getLogger(__name__)

# Defaults for tool result eviction
DEFAULT_MAX_RESULT_CHARS = 12_000  # ~3k tokens at ~4 chars/token
DEFAULT_PREVIEW_LINES = 20         # head and tail lines in preview


class ConversationMemory:
    """Short-term memory: the messages list for the current session.

    Stores validated SDK Message objects; provider adapters consume detached dict views.
    Large tool results are automatically evicted to disk when they exceed
    max_result_chars, replaced with a preview + file path reference.
    """

    def __init__(
        self,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
        preview_lines: int = DEFAULT_PREVIEW_LINES,
        eviction_dir: str | None = None,
    ) -> None:
        self._messages: list[Message] = []
        self._artifact_paths: set[str] = set()
        self._owned_artifact_paths: set[str] = set()
        self._max_result_chars = max_result_chars
        self._preview_lines = preview_lines
        self._eviction_dir = eviction_dir or os.path.join(
            tempfile.gettempdir(), "harnessx_tool_results"
        )
        os.makedirs(self._eviction_dir, exist_ok=True)

    def add_user_message(self, content: str) -> None:
        self._messages.append(Message("user", content))

    def add_assistant_message(self, content: list[Any]) -> None:
        """Convert SDK ContentBlock objects to MessageParam format."""
        content_params: list[dict[str, Any]] = []
        for block in content:
            if hasattr(block, "type"):
                if block.type == "text":
                    content_params.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    content_params.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                elif block.type == "thinking":
                    content_params.append({
                        "type": "thinking",
                        "thinking": getattr(block, "thinking", ""),
                    })
            elif isinstance(block, dict):
                content_params.append(block)

        self._messages.append(Message("assistant", content_params))

    def add_tool_results(self, results: list[Any]) -> None:
        """Tool results go in a user message with tool_result content blocks.

        If a result exceeds max_result_chars, the full content is saved to
        disk and the in-context content is replaced with a head/tail preview
        plus the file path so the agent can re-read it via read_file.
        """
        if not results:
            return

        from .types import ToolResult

        content: list[dict[str, Any]] = []
        for r in results:
            if isinstance(r, ToolResult):
                result_content = r.content
                call_id = getattr(r, "tool_call_id", "") or getattr(r, "tool_use_id", "")
                if not r.is_error and len(result_content) > self._max_result_chars:
                    result_content = self._evict_large_result(
                        call_id, result_content
                    )
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": result_content,
                }
                if r.is_error:
                    block["is_error"] = True
                content.append(block)
            elif isinstance(r, dict):
                content.append(r)

        if content:
            self._messages.append(Message("user", content))

    def _evict_large_result(self, tool_use_id: str, content: str) -> str:
        """Save full result to disk, return a preview with file path reference."""
        safe_id = tool_use_id.replace("/", "_").replace("..", "_")
        filepath = os.path.join(self._eviction_dir, f"{safe_id}-{uuid.uuid4().hex}.txt")
        self._artifact_paths.add(filepath)
        self._owned_artifact_paths.add(filepath)
        with open(filepath, "w") as f:
            f.write(content)

        lines = content.splitlines()
        total_lines = len(lines)
        n = self._preview_lines

        if total_lines <= n * 2:
            preview = content
        else:
            head = "\n".join(lines[:n])
            tail = "\n".join(lines[-n:])
            omitted = total_lines - (n * 2)
            preview = (
                f"{head}\n"
                f"\n... ({omitted} lines omitted) ...\n\n"
                f"{tail}"
            )

        # Cap preview by character count to handle wide-line files (minified JSON, CSV)
        max_preview_chars = 3_000
        if len(preview) > max_preview_chars:
            preview = preview[:max_preview_chars] + "\n... [preview truncated]"

        logger.info(
            "Evicted tool result %s: %d chars / %d lines -> preview (%d chars). Saved to %s",
            tool_use_id, len(content), total_lines, len(preview), filepath,
        )

        return (
            f"[Tool result too large ({total_lines} lines, {len(content)} chars). "
            f"Full output saved to: {filepath}]\n"
            f"[Use read_file with offset and limit parameters to access specific sections.]\n\n"
            f"{preview}"
        )

    def get_messages(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self._messages]

    def set_messages(self, messages: list[dict[str, Any]]) -> None:
        self._messages = [Message.from_dict(message) for message in messages]

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(Message.from_dict(message) for message in self.get_messages())

    def clear(self) -> None:
        self._messages.clear()

    async def aclose(self) -> None:
        """Remove only files created by this conversation; persisted artifacts are retained."""
        for path in self._owned_artifact_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self._owned_artifact_paths.clear()
        self._artifact_paths.clear()

    @property
    def message_count(self) -> int:
        return len(self._messages)

    async def trim_if_needed(
        self,
        provider_or_client: Any,
        model: str,
        system: str,
        tools: list[dict[str, Any]],
        max_context_tokens: int = 150_000,
    ) -> bool:
        """If token count approaches limit, summarize older messages.

        Accepts either an LLMProvider (preferred) or a raw Anthropic client
        (back-compat). Returns True if trimming was performed.
        """
        if len(self._messages) < 6:
            return False

        try:
            if hasattr(provider_or_client, "count_tokens") and not hasattr(
                provider_or_client, "messages"
            ):
                # LLMProvider path
                token_count = await provider_or_client.count_tokens(
                    model=model,
                    system=system,
                    tools=tools or [],
                    messages=self.get_messages(),
                )
            else:
                # Raw Anthropic client path (back-compat)
                resp = await provider_or_client.messages.count_tokens(
                    model=model,
                    system=system,
                    tools=tools if tools else [],
                    messages=self.get_messages(),
                )
                token_count = resp.input_tokens
        except Exception:
            token_count = len(str(self._messages)) // 3

        if token_count < max_context_tokens * 0.8:
            return False

        # Find a clean boundary that preserves alternating roles and tool call pairs
        split_point = self._find_safe_trim_boundary(min_keep=4)
        if split_point == 0:
            return False

        to_summarize = self._messages[:split_point]
        to_keep = self._messages[split_point:]

        summary_text = self._create_summary(to_summarize)

        self._messages = [
            Message("user", f"[Previous conversation summary]\n{summary_text}"),
            Message("assistant", [{"type": "text", "text": "Understood, I have the context from our previous conversation. How can I help?"}]),
            *to_keep,
        ]
        return True

    def _find_safe_trim_boundary(self, min_keep: int = 4) -> int:
        """Walk backwards to find a safe boundary where no tool call/result pairs are severed."""
        total = len(self._messages)
        if total <= min_keep:
            return 0

        split_idx = total - min_keep
        while split_idx > 0:
            msg = self._messages[split_idx]
            prev_msg = self._messages[split_idx - 1]

            # Check if msg is a user message containing tool results
            is_tool_result = False
            content = msg.get("content")
            if isinstance(content, list):
                is_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                )

            # Safe boundary: target message is a user message that does NOT contain tool results,
            # and the previous message was an assistant message without tool use.
            if msg.get("role") == "user" and not is_tool_result:
                if prev_msg.get("role") == "assistant":
                    prev_content = prev_msg.get("content")
                    prev_has_tool_use = False
                    if isinstance(prev_content, list):
                        prev_has_tool_use = any(
                            isinstance(b, dict) and b.get("type") == "tool_use" for b in prev_content
                        )
                    if not prev_has_tool_use:
                        return split_idx
            split_idx -= 1

        return 0

    def _create_summary(self, messages: list[dict[str, Any]]) -> str:
        """Create a simple text summary of messages."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}: {content[:200]}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(f"{role}: {block['text'][:200]}")
                        elif block.get("type") == "tool_use":
                            parts.append(f"{role}: [called {block.get('name', '?')}]")
                        elif block.get("type") == "tool_result":
                            parts.append(f"{role}: [tool result: {str(block.get('content', ''))[:100]}]")
        return "\n".join(parts[-20:])  # Keep last 20 items max


class PersistentMemory:
    """Long-term memory: persisted to disk as JSON files.

    Each session is saved as a separate file in a sessions/ directory.
    """

    def __init__(self, storage_dir: str = ".agent_sessions") -> None:
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save_session(self, state: SessionState) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        if not state.created_at:
            state.created_at = state.updated_at
        if not state.session_id:
            state.session_id = str(uuid.uuid4())

        filepath = self._session_file(state.session_id)
        serialized = json.dumps(asdict(state), indent=2, allow_nan=False)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.storage_dir, delete=False) as output:
            temporary = output.name
            try:
                output.write(serialized)
                output.flush()
                os.fsync(output.fileno())
                os.replace(temporary, filepath)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def _session_file(self, session_id: str) -> str:
        if not isinstance(session_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id) is None:
            raise ValueError("Invalid session ID")
        return os.path.join(self.storage_dir, f"{session_id}.json")

    def artifact_store(self, session_id: str):
        from .artifacts import DirectoryArtifactStore
        self._session_file(session_id)
        return DirectoryArtifactStore(os.path.join(self.storage_dir, "artifacts", session_id))

    def load_session(self, session_id: str) -> SessionState:
        filepath = self._session_file(session_id)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Session '{session_id}' not found at {filepath}")

        with open(filepath) as f:
            data = json.load(f)
        if data.get("version", 1) not in (1, 2):
            raise ValueError("Unsupported session snapshot version")
        if data["session_id"] != session_id:
            raise ValueError("Session identity does not match snapshot filename")

        usage_data = data.get("total_usage", {})
        return SessionState(
            session_id=data["session_id"],
            messages=data.get("messages", []),
            total_usage=TokenUsage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
            ),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            metadata=data.get("metadata", {}),
            config=data.get("config", {}),
            extensions=data.get("extensions", data.get("metadata", {}).get("extensions", {})),
            lifetime_iterations=data.get("lifetime_iterations", 0),
            version=data.get("version", 1),
        )

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    sessions.append({
                        "session_id": data.get("session_id", filename[:-5]),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "message_count": len(data.get("messages", [])),
                    })
                except (json.JSONDecodeError, OSError):
                    continue
        return sorted(sessions, key=lambda s: s.get("updated_at", ""), reverse=True)

    def delete_session(self, session_id: str) -> None:
        filepath = self._session_file(session_id)
        if os.path.exists(filepath):
            os.remove(filepath)
        shutil.rmtree(os.path.join(self.storage_dir, "artifacts", session_id), ignore_errors=True)


# ── Long-term memory (DB, simulated as JSON) ─────────────────────────────


class LongTermMemory:
    """Long-term factual memory stored in a DB (simulated as JSON files).

    Stores structured facts, summaries, and learned information that
    persists across sessions. In production, swap the file backend for
    a real database (Postgres, SQLite, etc.).
    """

    def __init__(self, storage_dir: str = ".agent_memory/long_term") -> None:
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save(self, content: str, category: str = "general", metadata: dict[str, Any] | None = None) -> str:
        """Save a fact or learned information. Returns the memory ID."""
        memory_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "id": memory_id,
            "content": content,
            "category": category,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
        }
        filepath = os.path.join(self.storage_dir, f"{memory_id}.json")
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2)
        return memory_id

    def search(self, query: str = "", category: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories by keyword and/or category."""
        results: list[dict[str, Any]] = []
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.storage_dir, filename)) as f:
                    record = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if category and record.get("category") != category:
                continue
            if query and query.lower() not in record.get("content", "").lower():
                continue
            results.append(record)

        results.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
        return results[:limit]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        filepath = os.path.join(self.storage_dir, f"{memory_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath) as f:
            return json.load(f)

    def update(self, memory_id: str, content: str) -> bool:
        record = self.get(memory_id)
        if record is None:
            return False
        record["content"] = content
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        filepath = os.path.join(self.storage_dir, f"{memory_id}.json")
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2)
        return True

    def delete(self, memory_id: str) -> bool:
        filepath = os.path.join(self.storage_dir, f"{memory_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def list_all(self) -> list[dict[str, Any]]:
        return self.search(limit=1000)


# ── Agent-specific memory (markdown files) ────────────────────────────────


class AgentMemory:
    """Agent-specific learned context stored as markdown files.

    Captures decisions, project notes, architectural context, and
    anything the agent learns that isn't part of a single conversation.
    Each memory is a .md file with YAML-style frontmatter.

    Storage: persistent directory (e.g., .agent_memory/agent/)
    """

    def __init__(self, storage_dir: str = ".agent_memory/agent") -> None:
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save(self, title: str, content: str, tags: list[str] | None = None) -> str:
        """Save a markdown memory file. Returns the filename."""
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
        filename = f"{slug}.md"
        filepath = os.path.join(self.storage_dir, filename)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tag_str = ", ".join(tags) if tags else ""

        md = f"""---
title: {title}
tags: [{tag_str}]
created: {now}
updated: {now}
---

{content}
"""
        with open(filepath, "w") as f:
            f.write(md)
        return filename

    def read(self, filename: str) -> dict[str, Any] | None:
        """Read a memory file, parsing frontmatter and content."""
        filepath = os.path.join(self.storage_dir, filename)
        if not os.path.exists(filepath):
            return None
        with open(filepath) as f:
            text = f.read()
        return self._parse_md(filename, text)

    def list_all(self) -> list[dict[str, Any]]:
        """List all memory files with their frontmatter."""
        memories: list[dict[str, Any]] = []
        for filepath in globmod.glob(os.path.join(self.storage_dir, "*.md")):
            with open(filepath) as f:
                text = f.read()
            parsed = self._parse_md(os.path.basename(filepath), text)
            if parsed:
                memories.append(parsed)
        memories.sort(key=lambda m: m.get("updated", ""), reverse=True)
        return memories

    def search(self, query: str = "", tags: list[str] | None = None) -> list[dict[str, Any]]:
        """Search memories by keyword in content or by tags."""
        results: list[dict[str, Any]] = []
        for mem in self.list_all():
            if query and query.lower() not in mem.get("content", "").lower():
                continue
            if tags:
                mem_tags = mem.get("tags", [])
                if not any(t in mem_tags for t in tags):
                    continue
            results.append(mem)
        return results

    def delete(self, filename: str) -> bool:
        filepath = os.path.join(self.storage_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def _parse_md(self, filename: str, text: str) -> dict[str, Any] | None:
        """Parse a markdown file with YAML frontmatter."""
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {"filename": filename, "content": text, "title": filename}

        frontmatter = parts[1].strip()
        content = parts[2].strip()

        meta: dict[str, Any] = {"filename": filename, "content": content}
        for line in frontmatter.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key == "tags":
                    val = [t.strip() for t in val.strip("[]").split(",") if t.strip()]
                meta[key] = val
        return meta


# ── Vector memory store (interface + in-memory default) ───────────────────


class VectorMemoryStore(ABC):
    """Abstract interface for vector-backed memory.

    Used for personal preferences, user corrections, and anything
    that benefits from semantic retrieval. Plug in your provider:
    Pinecone, Weaviate, Chroma, pgvector, etc.
    """

    @abstractmethod
    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Store a memory. Returns the memory ID."""
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Semantic search. Returns memories ranked by relevance."""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        ...


class InMemoryVectorStore(VectorMemoryStore):
    """Simple in-memory store for development. No embeddings — keyword fallback.

    In production, replace with a real vector store (Chroma, Pinecone, etc.)
    that provides actual embedding-based semantic search.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        memory_id = str(uuid.uuid4())[:8]
        self._store[memory_id] = {
            "id": memory_id,
            "text": text,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return memory_id

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Keyword-based fallback. Replace with embedding similarity in production."""
        results: list[dict[str, Any]] = []
        query_lower = query.lower()
        for record in self._store.values():
            text = record.get("text", "")
            if query_lower in text.lower():
                results.append(record)
        return results[:limit]

    async def delete(self, memory_id: str) -> bool:
        if memory_id in self._store:
            del self._store[memory_id]
            return True
        return False
