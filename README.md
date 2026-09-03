# DataGOL Agent Harness — Developer Guide

A provider-agnostic toolkit for building LLM agents in Python. This guide
covers the core building blocks with working code examples: creating agents,
registering tools, streaming, multi-agent orchestration, skills, memory,
permissions, hooks, middleware, sandboxing, MCP, and shipping an agent as a
web app.

- [Setup](#setup)
- [1. Creating an agent](#1-creating-an-agent)
- [2. Registering tools](#2-registering-tools)
- [3. Streaming agents](#3-streaming-agents)
- [4. Multi-agent orchestration](#4-multi-agent-orchestration)
- [5. Skills](#5-skills)
- [6. Memory](#6-memory)
- [7. Permissions and guardrails](#7-permissions-and-guardrails)
- [8. Hooks and middleware](#8-hooks-and-middleware)
- [9. MCP servers](#9-mcp-servers)
- [10. Session persistence](#10-session-persistence)
- [11. Shipping a web app](#11-shipping-a-web-app)
- [API reference (quick)](#api-reference-quick)

---

## Setup

```bash
# Core (Anthropic)
pip install -e .

# With OpenAI support / everything
pip install -e ".[openai]"
pip install -e ".[all]"
```

Set your key (a root `.env` is auto-loaded by the `examples` package):

```bash
ANTHROPIC_API_KEY=sk-ant-...
# or, for OpenAI-backed agents:
OPENAI_API_KEY=sk-...
```

Run examples as modules from the repo root (not as plain scripts):

```bash
python -m examples.simple_chat
```

---

## 1. Creating an agent

`Agent` is the core agentic loop: LLM reasoning → tool execution → repeat,
until the model stops with `end_turn`.

```python
import asyncio
from datagol_agent_harness import Agent, AgentConfig

agent = Agent(
    config=AgentConfig(
        model="claude-sonnet-4-6",      # or "gpt-5"
        provider="anthropic",            # or "openai"
        system_prompt="You are a helpful assistant.",
        max_tokens=8192,
        max_iterations=50,               # loop guardrail
        temperature=0.0,
    )
)

print(asyncio.run(agent.run("What is 17 + 25?")))
```

`AgentConfig` fields (all optional, these are the defaults):

| Field | Default | Purpose |
|---|---|---|
| `model` | `"claude-sonnet-4-6"` | Model id passed to the provider |
| `provider` | `"anthropic"` | `"anthropic"` or `"openai"` |
| `max_tokens` | `8192` | Per-response token cap |
| `max_iterations` | `50` | Max loop iterations before `MaxIterationsError` |
| `system_prompt` | `"You are a helpful assistant."` | System prompt |
| `temperature` | `0.0` | Sampling temperature |
| `max_result_chars` | `12000` | Tool-result eviction threshold in memory |

You can also inject your own pieces — everything is a constructor parameter:

```python
from datagol_agent_harness import (
    Agent, AgentConfig, ConversationMemory, HookManager,
    MiddlewarePipeline, PermissionManager, ToolRegistry,
)

agent = Agent(
    config=AgentConfig(system_prompt="..."),
    tools=ToolRegistry(),               # your own registry
    memory=ConversationMemory(),        # your own memory
    permissions=PermissionManager(),    # your own permission policy
    hooks=HookManager(),                # lifecycle hooks
    middleware=MiddlewarePipeline(),    # request/response transforms
)
```

### Conversation loop

`agent.run()` keeps the conversation in memory, so calling it repeatedly is a
multi-turn chat:

```python
async def chat(agent):
    while True:
        text = input("You: ").strip()
        if text in ("", "quit"):
            break
        print("Agent:", await agent.run(text))
```

---

## 2. Registering tools

A tool is any Python function. The registry converts its signature, type
hints, and docstring into a JSON schema for the model, and dispatches calls
back to the function.

### Decorator registration

```python
from datagol_agent_harness import PermissionLevel

@agent.tools.register(permission=PermissionLevel.ALLOW)
def add(a: int, b: int) -> int:
    """Add two numbers."""          # becomes the tool description
    return a + b

@agent.tools.register(permission=PermissionLevel.ASK)
async def read_file(path: str) -> str:
    """Read a file from disk.

    Args:
        path: Absolute path to the file.   # becomes the param description
    """
    with open(path) as f:
        return f.read()
```

Rules of thumb:

- The **first line of the docstring** is the tool description the model sees.
- An `Args:` section documents parameters.
- Type hints drive the JSON schema: `str`, `int`, `float`, `bool`, `list[X]`,
  `dict`. Optional params (with defaults) are not required.
- **Pitfall:** avoid `list[str] | None` style unions — the schema generator
  can't express them and will fall back to `"type": "string"`. Give optional
  params a plain default and type instead, or use `register_with_schema`
  (below) for exact control.
- Sync and async handlers both work. Errors never crash the agent — they are
  returned to the model as `is_error` tool results.

### Explicit schema registration

When you need a schema the signature can't express (nested objects, arrays
with item types, `oneOf`-style choices):

```python
async def propose_items(query: str = "", items=None, intro_text: str = "") -> str:
    if (query and items) or (not query and not items):
        return "Error: provide exactly one of `query` or `items`."
    resolved = [query] if query else items
    return f"Presented {len(resolved)} item(s)."

agent.tools.register_with_schema(
    name="propose_items",
    description="Propose items, via query (user's words) or items (your strings).",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "User's words, untouched."},
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Item strings you wrote yourself.",
            },
            "intro_text": {"type": "string"},
        },
    },
    handler=propose_items,
    permission=PermissionLevel.ALLOW,
)
```

### Built-in tools

`datagol_agent_harness.builtin` ships ready-made tool sets:

```python
from datagol_agent_harness.builtin import register_all_tools

register_all_tools(agent.tools)
# read_file, write_file, list_directory, run_bash, fetch_url, memory tools...
```

Or individually: `register_filesystem_tools`, `register_bash_tools`,
`register_web_tools`, `register_memory_tools`.

### Inspecting and executing

```python
agent.tools.list_tools()                 # ['add', 'read_file', ...]
agent.tools.get_tool_params()            # Anthropic-format tool schemas
result = await agent.tools.execute(call) # manual dispatch (ToolCall -> ToolResult)
```

---

## 3. Streaming agents

`StreamingAgent` yields typed events as the model generates, instead of
returning one final string. It mirrors `Agent`'s constructor.

```python
import asyncio
from datagol_agent_harness import AgentConfig, StreamingAgent, StreamEventType

agent = StreamingAgent(config=AgentConfig(system_prompt="You are helpful."))

async def main():
    async for event in agent.run_stream("Tell me a story"):
        if event.type == StreamEventType.TEXT_DELTA:
            print(event.data, end="", flush=True)
        elif event.type == StreamEventType.TOOL_CALL_START:
            print(f"\n> calling {event.data.name}({event.data.input})")
        elif event.type == StreamEventType.TOOL_RESULT:
            if event.data.is_error:
                print(f"  tool error: {event.data.content}")
        elif event.type == StreamEventType.TURN_COMPLETE:
            print()

asyncio.run(main())
```

Event types: `TEXT_DELTA`, `TEXT_COMPLETE`, `TOOL_CALL_START`,
`TOOL_CALL_COMPLETE`, `TOOL_RESULT`, `THINKING_DELTA`, `TURN_COMPLETE`,
`ERROR`.

Usage stats accumulate on the guardrails engine:

```python
agent.guardrails.usage_summary
# {'iterations': 3, 'input_tokens': 5120, 'output_tokens': 640, 'estimated_cost': '$0.0250'}
```

---

## 4. Multi-agent orchestration

There is no special "multi-agent" class — a sub-agent is just an `Agent`
wrapped in a tool. The orchestrator's LLM decides when to delegate; the tool
runs the specialist to completion and returns its text as the tool result.

```python
import asyncio
from datagol_agent_harness import Agent, AgentConfig, PermissionLevel
from datagol_agent_harness.builtin.web import register_web_tools


async def run_research_agent(query: str) -> str:
    """A specialist with its own prompt, tools, and iteration budget."""
    researcher = Agent(
        config=AgentConfig(
            system_prompt="You are a research specialist. Cite sources.",
            max_iterations=10,
        )
    )
    register_web_tools(researcher.tools)
    researcher.permissions.set_permission("fetch_url", PermissionLevel.ALLOW)
    return await researcher.run(query)


orchestrator = Agent(
    config=AgentConfig(
        system_prompt=(
            "You manage a team of specialists. Delegate research to "
            "delegate_research; synthesize results for the user."
        ),
        max_iterations=20,
    )
)

@orchestrator.tools.register(permission=PermissionLevel.ALLOW)
async def delegate_research(query: str) -> str:
    """Delegate a research question to the research specialist.

    Args:
        query: The research question.
    """
    return await run_research_agent(query)


print(asyncio.run(orchestrator.run("Compare SQLite and DuckDB for analytics.")))
```

Patterns that work well:

- **Give each specialist a tight system prompt and a small tool set.** The
  orchestrator's prompt should list its delegation tools and when to use them.
- **Cap `max_iterations` per specialist** so a runaway sub-agent can't burn
  the budget.
- Sub-agents are created per call (stateless). If a specialist needs
  continuity, keep one instance and reuse it across calls.
- Sub-agents can themselves delegate — the pattern composes.

A full working version with three specialists lives in
`examples/multi_agent.py` (`python -m examples.multi_agent`).

---

## 5. Skills

A skill is a markdown file with YAML frontmatter — a way to ship reusable
instruction packs that load **lazily**: the agent only sees each skill's
name + description in its system prompt, and pulls the full body into context
by calling the `Skill` tool when a request matches.

### Authoring a skill

`skills/commit-message/SKILL.md`:

```markdown
---
name: commit-message
description: Write a conventional commit message for a described change.
---

You write commit messages. Rules:
- Conventional Commits format: type(scope): subject
- Subject in imperative mood, <= 72 chars, no trailing period
- Add a body only when the "why" isn't obvious from the subject
```

A path can be a folder containing `SKILL.md`, or a markdown file directly.

### Loading skills into an agent

```python
from datagol_agent_harness import Agent, AgentConfig

agent = Agent(
    config=AgentConfig(
        system_prompt=(
            "You are an engineering assistant. When a request matches a "
            "skill in <available-skills>, call the Skill tool first to load "
            "its instructions, then follow them."
        ),
    ),
    skills=["./skills/commit-message", "./skills/code-review"],
)
```

The `skills=` argument accepts a list of paths or a `SkillManager`:

```python
from datagol_agent_harness import SkillManager

manager = SkillManager.from_paths(["./skills/commit-message"])
agent = Agent(config=AgentConfig(system_prompt="..."), skills=manager)

for s in agent.skills.list():
    print(s.name, "-", s.description)
```

### Reacting to skill invocation

A `SKILL_INVOKED` hook fires when the model loads a skill:

```python
from datagol_agent_harness import HookContext, HookEvent

async def on_skill(ctx: HookContext):
    if ctx.data.get("found"):
        print(f"loaded skill: {ctx.data['skill']} ({ctx.data['body_chars']} chars)")

agent.hooks.on(HookEvent.SKILL_INVOKED, on_skill)
```

Full demo: `examples/skills_agent.py`, with sample skills under
`examples/skills/`.

---

## 6. Memory

Three layers, all optional:

### Conversation memory (default)

Every agent has a `ConversationMemory` that stores the message list and trims
itself when approaching the context limit. You rarely touch it directly, but
you can:

```python
agent.memory.get_messages()        # current message list
agent.memory.set_messages(saved)   # restore
```

### Agent memory — markdown notes on disk

`AgentMemory` persists titled, tagged notes as markdown files. You expose it
to the model through tools you write:

```python
from datagol_agent_harness import AgentMemory, PermissionLevel

memory = AgentMemory(storage_dir=".agent_memory/agent")

@agent.tools.register(permission=PermissionLevel.ALLOW)
async def save_memory(title: str, content: str, tags: str = "") -> str:
    """Save a note to persistent memory.

    Args:
        title: Short title for the memory.
        content: The information to remember.
        tags: Comma-separated tags.
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    filename = memory.save(title, content, tags=tag_list)
    return f"Saved '{title}' -> {filename}"

@agent.tools.register(permission=PermissionLevel.ALLOW)
async def search_memory(query: str = "") -> str:
    """Search saved memories by keyword."""
    results = memory.search(query=query)
    return "\n".join(f"- {r['title']} [{r['filename']}]" for r in results) or "None found."
```

### Long-term memory — structured facts

`LongTermMemory` stores categorized facts with ids:

```python
from datagol_agent_harness import LongTermMemory

long_term = LongTermMemory(storage_dir=".agent_memory/long_term")
fact_id = long_term.save("User prefers dark mode", category="preference")
long_term.search(query="dark mode")
long_term.update(fact_id, "User prefers light mode")
```

Tell the agent about these tools in the system prompt and when to use them
("proactively save useful information"). A complete two-layer example is in
`examples/memory_agent.py`.

---

## 7. Permissions and guardrails

### Permission levels

Every tool has a level: `ALLOW` (run silently), `ASK` (prompt the user
y/n/always), or `DENY` (never run). The default at registration is `ASK`.

```python
from datagol_agent_harness import PermissionLevel

# At registration
@agent.tools.register(permission=PermissionLevel.ALLOW)
def add(a: int, b: int) -> int: ...

# Or override later
agent.permissions.set_permission("run_bash", PermissionLevel.ASK)
agent.permissions.grant_session("read_file")   # pre-approve for this session
```

`ASK` prompts on stdin in terminal apps. In a server context, register tools
as `ALLOW` (or build your own `PermissionManager` subclass that asks over
your transport).

### Guardrails

```python
from datagol_agent_harness import GuardrailsEngine, MaxIterationsError, CostLimitError

agent.guardrails.max_cost_dollars = 0.50   # raises CostLimitError past this
agent.guardrails.usage_summary             # tokens + estimated cost so far
agent.guardrails.reset()                   # reset counters
```

`max_iterations` comes from `AgentConfig` and raises `MaxIterationsError`.

---

## 8. Hooks and middleware

### Hooks — observe the lifecycle

```python
from datagol_agent_harness import HookContext, HookEvent

@agent.hooks.before_tool
async def log_call(ctx: HookContext):
    tc = ctx.data["tool_call"]
    print(f"-> {tc.name}({tc.input})")

@agent.hooks.after_tool
async def log_result(ctx: HookContext):
    result = ctx.data["result"]
    if result.is_error:
        print(f"tool failed: {result.content}")

agent.hooks.on(HookEvent.LLM_RESPONSE, lambda ctx: print(ctx.data["stop_reason"]))
```

Events: `AGENT_START`, `AGENT_END`, `LOOP_ITERATION_START`,
`LOOP_ITERATION_END`, `LLM_REQUEST`, `LLM_RESPONSE`, `TOOL_CALL_START`,
`TOOL_CALL_END`, `SKILL_INVOKED`, `SANDBOX_EXEC`, `CHECKPOINT`, `ERROR`.

### Middleware — transform requests and results

Subclass `Middleware` to mutate messages, tool calls, or results as they flow:

```python
from datagol_agent_harness import Middleware

class RedactSecrets(Middleware):
    async def before_tool_execution(self, tool_call):
        if "api_key" in str(tool_call.input):
            tool_call.input = {k: ("***" if "key" in k else v)
                               for k, v in tool_call.input.items()}
        return tool_call

agent.middleware.add(RedactSecrets())
```

Overridable stages: `before_llm_call`, `after_llm_call`,
`before_tool_execution`, `after_tool_execution`.

---

## 9. MCP servers

`MCPManager` connects to Model Context Protocol servers (stdio subprocess or
remote SSE) and registers their tools into your agent's registry, so MCP
tools look exactly like native tools to the model.

```python
from datagol_agent_harness import Agent, AgentConfig, MCPManager

mcp = MCPManager()
await mcp.connect(
    "filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)
# or a remote server:
# await mcp.connect("remote", url="http://localhost:8000/sse")

agent = Agent(config=AgentConfig(system_prompt="..."), mcp=mcp)
mcp.register_tools(agent.tools)

print(await agent.run("What's in /tmp?"))
await mcp.disconnect_all()
```

See `examples/mcp_agent.py` for the interactive version.

---

## 10. Session persistence

Save and restore an agent's conversation:

```python
session_id = await agent.save_session()            # -> writes .agent_sessions/

# later, or in another process:
from datagol_agent_harness import Agent
restored = await Agent.load_session(session_id, config=AgentConfig(system_prompt="..."))
await restored.run("Where were we?")
```

For managed multi-session services (checkpointing, pause/resume, expiry),
see `AgentRuntime` in `datagol_agent_harness/runtime.py`.

---

## 11. Shipping a web app

The harness is transport-agnostic. The pattern used by
`apps/grocery_buddy` (a complete reference implementation):

1. **One `StreamingAgent` per browser session**, held in a registry keyed by
   a session id (cookie, localStorage value, etc.).
2. **Translate `StreamEvent`s into your wire format** over a WebSocket:

```python
from fastapi import FastAPI, WebSocket
from datagol_agent_harness import AgentConfig, StreamingAgent, StreamEventType

app = FastAPI()
agents: dict[str, StreamingAgent] = {}

@app.websocket("/ws")
async def chat(ws: WebSocket):
    await ws.accept()
    sid = ws.query_params.get("session") or "default"
    agent = agents.setdefault(sid, StreamingAgent(config=AgentConfig(system_prompt="...")))

    while True:
        text = (await ws.receive_json())["text"]
        async for event in agent.run_stream(text):
            if event.type == StreamEventType.TEXT_DELTA:
                await ws.send_json({"type": "delta", "text": event.data})
            elif event.type == StreamEventType.TOOL_CALL_START:
                await ws.send_json({"type": "tool", "name": event.data.name,
                                    "input": event.data.input})
        await ws.send_json({"type": "done"})
```

3. **Register tools as `PermissionLevel.ALLOW`** — there is no stdin to ask
   on in a server.
4. **Use tool calls as structured UI actions.** In GroceryBuddy the model's
   `propose_items` / `present_choice` / `respond_text` calls become proposal
   cards and choice chips in the browser; `client_action` arguments drive
   client-side behavior (open a screen, focus input). Tools that only fetch
   (`get_staples`) stay silent.

Run the full example:

```bash
.venv/bin/uvicorn apps.grocery_buddy.server.main:app --port 4100
# open http://localhost:4100
```

---

## API reference (quick)

| Class / function | Module | Purpose |
|---|---|---|
| `Agent` | `datagol_agent_harness` | Core agentic loop (`await agent.run(msg)`) |
| `StreamingAgent` | `datagol_agent_harness` | Event-streaming variant (`run_stream`) |
| `AgentConfig` | `datagol_agent_harness` | Model, provider, limits, prompt |
| `ToolRegistry` | `datagol_agent_harness` | `register`, `register_with_schema`, `execute` |
| `PermissionLevel` | `datagol_agent_harness` | `ALLOW` / `ASK` / `DENY` |
| `PermissionManager` | `datagol_agent_harness` | Per-tool overrides, session grants |
| `GuardrailsEngine` | `datagol_agent_harness` | Iteration/cost limits, usage stats |
| `ConversationMemory` | `datagol_agent_harness` | Message list with auto-trimming |
| `AgentMemory` / `LongTermMemory` | `datagol_agent_harness` | Disk-backed notes / facts |
| `PersistentMemory` | `datagol_agent_harness` | Session save/load |
| `HookManager` / `HookEvent` | `datagol_agent_harness` | Lifecycle hooks |
| `Middleware` / `MiddlewarePipeline` | `datagol_agent_harness` | Request/result transforms |
| `SkillManager` | `datagol_agent_harness` | Lazy skill loading |
| `MCPManager` | `datagol_agent_harness` | MCP server connections |
| `Sandbox` | `datagol_agent_harness` | Sandboxed code execution |
| `AgentRuntime` | `datagol_agent_harness` | Managed sessions, checkpoints |
| `register_all_tools` | `datagol_agent_harness.builtin` | Filesystem, bash, web, memory tools |

## Runnable examples

| Example | Shows |
|---|---|
| `python -m examples.simple_chat` | Streaming chat + GroceryBuddy prompt/tools |
| `python -m examples.multi_agent` | Orchestrator + specialist agents |
| `python -m examples.skills_agent` | Lazy skill loading |
| `python -m examples.memory_agent` | Two-layer persistent memory |
| `python -m examples.mcp_agent` | MCP tool integration |
| `python -m examples.coding_agent` | Full coding assistant |
| `python -m examples.sandboxed_coder` | Sandboxed execution |
| `uvicorn apps.grocery_buddy.server.main:app --port 4100` | Web app over WebSocket |
