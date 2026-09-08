"""FastAPI wrapper for the agent harness.

Exposes the agent as an HTTP API with streaming SSE support.

Run from the project root:
    uvicorn examples.web_app.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from datagol_agent_harness import (
    AgentConfig,
    MCPManager,
    PermissionLevel,
    SkillManager,
    StreamingAgent,
    StreamEventType,
)
from datagol_agent_harness.builtin import register_all_tools

HERE = os.path.dirname(__file__)
STATIC_DIR = os.path.join(HERE, "static")
OUTPUT_DIR = os.environ.get("AGENT_OUTPUT_DIR", os.path.join(os.getcwd(), "output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _resolve_skill_paths() -> list[str]:
    """Discover skills via AGENT_SKILLS env var, or fall back to a default dir.

    AGENT_SKILLS is a colon-separated list of paths (files or folders). Each
    entry is either a SKILL.md file, a folder containing SKILL.md, or a folder
    of skill folders (auto-expanded one level).
    """
    raw = os.environ.get("AGENT_SKILLS", "")
    if raw:
        entries = [p.strip() for p in raw.split(":") if p.strip()]
    else:
        # Default: a sibling examples/skills/ next to the repo
        default = os.path.join(os.getcwd(), "examples", "skills")
        entries = [default] if os.path.isdir(default) else []

    resolved: list[str] = []
    for entry in entries:
        if not os.path.exists(entry):
            print(f"Warning: skill path does not exist: {entry}")
            continue
        if os.path.isfile(entry):
            resolved.append(entry)
            continue
        # Folder: SKILL.md directly inside, or one-level expand into subfolders
        if os.path.isfile(os.path.join(entry, "SKILL.md")):
            resolved.append(entry)
        else:
            for child in sorted(os.listdir(entry)):
                child_path = os.path.join(entry, child)
                if os.path.isfile(child_path) and child.endswith(".md"):
                    resolved.append(child_path)
                elif os.path.isdir(child_path) and os.path.isfile(
                    os.path.join(child_path, "SKILL.md")
                ):
                    resolved.append(child_path)
    return resolved

# ── Global state ─────────────────────────────────────────────────────────

streaming_agent: StreamingAgent | None = None
mcp_manager: MCPManager | None = None


def create_streaming_agent() -> StreamingAgent:
    """Create the single streaming agent with all tools and skills registered."""
    from anthropic import AsyncAnthropic
    from datagol_agent_harness import PermissionManager, ConversationMemory, ToolRegistry

    config = AgentConfig(
        system_prompt=(
            "You are a helpful AI assistant with access to tools. "
            "Use tools when they help answer the user's question. "
            "When a user's request matches a skill in <available-skills>, "
            "call the Skill tool first to load its instructions. "
            "Be concise and direct."
        ),
        max_iterations=0,
    )

    tools = ToolRegistry()
    register_all_tools(tools)

    permissions = PermissionManager()
    for t in ["read_file", "list_directory", "run_bash", "fetch_url", "write_file", "generate_file"]:
        permissions.set_permission(t, PermissionLevel.ALLOW)

    # Register MCP tools if manager exists
    if mcp_manager and mcp_manager.tool_count > 0:
        registered = mcp_manager.register_tools(tools, permission=PermissionLevel.ALLOW)
        for name in registered:
            permissions.set_permission(name, PermissionLevel.ALLOW)

    skill_paths = _resolve_skill_paths()
    skills = SkillManager.from_paths(skill_paths) if skill_paths else None

    sa = StreamingAgent(
        config=config,
        client=AsyncAnthropic(),
        tools=tools,
        memory=ConversationMemory(max_result_chars=config.max_result_chars),
        permissions=permissions,
        skills=skills,
    )
    # The Skill tool itself was registered by SkillManager.install() with
    # PermissionLevel.ALLOW already; this line is a safety net in case
    # the permission manager is rebuilt elsewhere.
    if skills is not None:
        permissions.set_permission("Skill", PermissionLevel.ALLOW)
    return sa


@asynccontextmanager
async def lifespan(app: FastAPI):
    global streaming_agent, mcp_manager
    mcp_manager = MCPManager()

    # Connect to MCP servers from environment
    mcp_config = os.environ.get("MCP_SERVERS")
    if mcp_config:
        try:
            servers = json.loads(mcp_config)
            await mcp_manager.connect_from_config(servers)
        except Exception as e:
            print(f"Warning: Failed to connect MCP servers: {e}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    streaming_agent = create_streaming_agent()
    yield

    if mcp_manager:
        await mcp_manager.disconnect_all()


app = FastAPI(title="Agent Harness API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    response: str
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}


class MCPConnectRequest(BaseModel):
    name: str
    command: str | None = None
    args: list[str] = []
    url: str | None = None
    headers: dict[str, str] = {}


class SessionResponse(BaseModel):
    session_id: str


# ── API routes ───────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming chat endpoint. Uses the streaming agent, collects all events."""
    if req.system_prompt is not None:
        streaming_agent.config.system_prompt = req.system_prompt

    tool_calls_log: list[dict[str, Any]] = []
    final_text = ""

    async for event in streaming_agent.run_stream(req.message):
        if event.type == StreamEventType.TEXT_COMPLETE:
            final_text = event.data or ""
        elif event.type == StreamEventType.TOOL_CALL_START:
            tc = event.data
            tool_calls_log.append({"name": tc.name, "input": tc.input, "status": "running"})
        elif event.type == StreamEventType.TOOL_RESULT:
            r = event.data
            for entry in tool_calls_log:
                if entry["status"] == "running":
                    entry["status"] = "error" if r.is_error else "done"
                    entry["result"] = r.content
                    break

    return ChatResponse(
        response=final_text,
        tool_calls=tool_calls_log,
        usage=streaming_agent.guardrails.usage_summary,
    )


@app.post("/api/stream")
async def stream(req: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    if req.system_prompt is not None:
        streaming_agent.config.system_prompt = req.system_prompt

    async def event_generator():
        async for event in streaming_agent.run_stream(req.message):
            if event.type == StreamEventType.TEXT_DELTA:
                yield f"data: {json.dumps({'type': 'text_delta', 'content': event.data})}\n\n"

            elif event.type == StreamEventType.TEXT_COMPLETE:
                yield f"data: {json.dumps({'type': 'text_complete', 'content': event.data})}\n\n"

            elif event.type == StreamEventType.TOOL_CALL_START:
                tc = event.data
                yield f"data: {json.dumps({'type': 'tool_call_start', 'name': tc.name, 'input': tc.input, 'id': tc.id})}\n\n"

            elif event.type == StreamEventType.TOOL_RESULT:
                r = event.data
                yield f"data: {json.dumps({'type': 'tool_result', 'tool_use_id': r.tool_use_id, 'content': r.content, 'is_error': r.is_error})}\n\n"

            elif event.type == StreamEventType.TURN_COMPLETE:
                usage = streaming_agent.guardrails.usage_summary
                yield f"data: {json.dumps({'type': 'turn_complete', 'usage': usage})}\n\n"

            elif event.type == StreamEventType.ERROR:
                yield f"data: {json.dumps({'type': 'error', 'content': str(event.data)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/status")
async def status():
    """Get agent status, tools, and MCP servers."""
    sa = streaming_agent
    tools = sa.tools.list_tools() if sa else []
    mcp_servers = mcp_manager.list_servers() if mcp_manager else {}
    mcp_tools = [
        {"server": t.server_name, "name": t.tool_name, "description": t.description[:100]}
        for t in (mcp_manager.list_tools() if mcp_manager else [])
    ]

    skills = (
        [{"name": s.name, "description": s.description} for s in sa.skills.list()]
        if sa and sa.skills
        else []
    )

    return {
        "status": "running" if sa else "stopped",
        "model": sa.config.model if sa else "",
        "system_prompt": sa.config.system_prompt[:200] if sa else "",
        "session_id": "",
        "tools": tools,
        "mcp_servers": mcp_servers,
        "mcp_tools": mcp_tools,
        "skills": skills,
        "usage": sa.guardrails.usage_summary if sa else {},
    }


@app.get("/api/skills")
async def list_skills():
    """List all skills available to the agent."""
    sa = streaming_agent
    if not sa or not sa.skills:
        return {"skills": []}
    return {
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "source_path": str(s.source_path),
                "body_chars": len(s.body),
            }
            for s in sa.skills.list()
        ]
    }


@app.post("/api/mcp/connect")
async def mcp_connect(req: MCPConnectRequest):
    """Connect to an MCP server at runtime."""
    global streaming_agent
    try:
        tools = await mcp_manager.connect(
            req.name,
            command=req.command,
            args=req.args,
            url=req.url,
            headers=req.headers,
            permission=PermissionLevel.ALLOW,
        )
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "detail": f"Failed to connect to MCP server '{req.name}'"},
        )
    # Re-create agent with new tools
    streaming_agent = create_streaming_agent()
    return {
        "server": req.name,
        "tools": [{"name": t.tool_name, "description": t.description[:100]} for t in tools],
    }


@app.post("/api/mcp/disconnect/{name}")
async def mcp_disconnect(name: str):
    """Disconnect an MCP server."""
    global streaming_agent
    await mcp_manager.disconnect(name)
    streaming_agent = create_streaming_agent()
    return {"disconnected": name}


@app.post("/api/session/save", response_model=SessionResponse)
async def save_session():
    # Save streaming agent's memory
    from datagol_agent_harness import PersistentMemory, SessionState
    import uuid
    storage = PersistentMemory()
    session_id = str(uuid.uuid4())
    state = SessionState(
        session_id=session_id,
        messages=streaming_agent.memory.get_messages(),
        total_usage=streaming_agent.guardrails.total_usage,
    )
    storage.save_session(state)
    return SessionResponse(session_id=session_id)


@app.post("/api/session/load/{session_id}")
async def load_session(session_id: str):
    from datagol_agent_harness import PersistentMemory
    storage = PersistentMemory()
    state = storage.load_session(session_id)
    streaming_agent.memory.set_messages(state.messages)
    return {"loaded": session_id}


@app.post("/api/session/clear")
async def clear_session():
    global streaming_agent
    streaming_agent = create_streaming_agent()
    return {"status": "cleared"}


@app.get("/api/sessions")
async def list_sessions():
    from datagol_agent_harness import PersistentMemory
    storage = PersistentMemory()
    return {"sessions": storage.list_sessions()}


# ── Serve the UI ─────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    with open(os.path.join(STATIC_DIR, "index.html")) as f:
        return HTMLResponse(f.read())


# ── Serve generated files ───────────────────────────────────────────────

app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="generated_files")
