# DataGOL Agent Harness

[![PyPI version](https://img.shields.io/pypi/v/datagol-agent-harness.svg)](https://pypi.org/project/datagol-agent-harness/)
[![Python versions](https://img.shields.io/pypi/pyversions/datagol-agent-harness.svg)](https://pypi.org/project/datagol-agent-harness/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A provider-agnostic, library-first Python toolkit for building production-grade LLM agents.**

Rather than locking developers into rigid state-machine graphs or opaque persona prompts, `datagol-agent-harness` gives you a composable set of building blocks: real-time streaming, typed tool registration with schema inference, lazy-loaded skills, multi-agent delegation, 4-tier memory, sandboxed code execution, Model Context Protocol (MCP) tool bridges, native LangSmith tracing, and evaluation suites.

### Why DataGOL Agent Harness?

- **No Graph Boilerplate:** Write simple async Python functions instead of complex state graphs.
- **Provider-Agnostic:** First-class support for Anthropic (Claude 3.5/3.7/Sonnet/Opus) and OpenAI (GPT-4o/GPT-5).
- **Streaming-First:** First-class typed event stream (`run_stream`) for WebSocket and SSE frontends.
- **Lazy Instruction Packs (Skills):** Load specialized guidelines only when needed via YAML-frontmatter `SKILL.md` packs.
- **Enterprise Guardrails & Sandboxing:** Execute untrusted code safely via Process, Docker, or macOS Seatbelt isolation with granular budget and iteration caps.
- **MCP Native:** Connect to any Model Context Protocol server (stdio subprocess or remote SSE) in 3 lines of code.
- **Full Observability & Evals:** Zero-overhead lifecycle hooks, middleware transforms, native LangSmith tracing, and benchmark evaluations.

---

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
- [12. Evaluations with LangSmith](#12-evaluations-with-langsmith)
- [API reference (quick)](#api-reference-quick)

---

## Setup

```bash
# From PyPI
pip install datagol-agent-harness

# With optional extras (LangSmith, OpenAI, MCP, Docker, or everything)
pip install "datagol-agent-harness[langsmith]"
pip install "datagol-agent-harness[openai]"
pip install "datagol-agent-harness[all]"

# For local development (editable)
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

The harness is transport-agnostic. The reference implementation in
`examples/web_app` demonstrates:

1. **Streaming agent execution** translated to Server-Sent Events (SSE) or WebSockets:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from datagol_agent_harness import AgentConfig, StreamingAgent, StreamEventType
import json

app = FastAPI()
agent = StreamingAgent(config=AgentConfig(system_prompt="You are helpful."))

@app.post("/api/stream")
async def stream(req: dict):
    async def event_generator():
        async for event in agent.run_stream(req["message"]):
            if event.type == StreamEventType.TEXT_DELTA:
                yield f"data: {json.dumps({'type': 'text_delta', 'content': event.data})}\n\n"
            elif event.type == StreamEventType.TOOL_CALL_START:
                yield f"data: {json.dumps({'type': 'tool_call', 'name': event.data.name})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

2. **Register tools as `PermissionLevel.ALLOW`** — in a server context, tools should be pre-approved or gated via an authorization middleware rather than interactive stdin prompts.
3. **Built-in UI and MCP integrations** — the web app serves a complete browser UI that displays tool calls, token usage, MCP servers, and dynamic skills.

Run the full example from the repo root:

```bash
uvicorn examples.web_app.server:app --port 8000 --reload
# open http://localhost:8000
```

---

## 12. Evaluations with LangSmith

The harness includes first-class evaluation capabilities powered by the
**LangSmith evaluation framework** (`evaluate` / `aevaluate`). You can benchmark
tool selection, skill routing, multi-agent delegation, and guardrails either
locally (offline, zero-cost) or in the LangSmith Web UI.

### Running an evaluation

```python
from datagol_agent_harness import Agent, AgentConfig, PermissionLevel
from datagol_agent_harness.evals import (
    build_example,
    default_evaluators,
    evaluate_agent,
    tool_selection_evaluator,
    contains_evaluator,
)

# 1. Define your agent factory or instance
def make_agent(inputs):
    agent = Agent(config=AgentConfig(model="claude-sonnet-4-6"))
    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def calculate(expression: str) -> str:
        return str(eval(expression, {"__builtins__": None}, {}))
    return agent

# 2. Define test cases or use built-in suites ("tool_calling", "skills", "all")
dataset = [
    build_example(
        inputs={"prompt": "What is 144 / 12?"},
        outputs={
            "expected_tools": ["calculate"],
            "contains_all": ["12"],
            "max_allowed_iterations": 3,
        },
    )
]

# 3. Evaluate (offline for fast CI/CD tests, or live to sync to LangSmith)
summary = evaluate_agent(
    agent=make_agent,
    dataset=dataset,
    evaluators=[tool_selection_evaluator, contains_evaluator],
    experiment_prefix="math-agent-benchmark",
    offline=True,   # set False or omit when LANGSMITH_API_KEY is present
)

print(f"Pass rate: {summary.pass_rate * 100:.1f}%")
```

### Running evals from the CLI

```bash
# Run tool calling benchmarks offline (no API keys required)
python -m datagol_agent_harness.evals.cli --suite tool_calling --offline

# Run against Anthropic and upload results + traces to LangSmith
export LANGSMITH_API_KEY="lsv2_pt_..."
python -m datagol_agent_harness.evals.cli --suite skills --model claude-sonnet-4-6 --upload

# Run all benchmark suites with concurrency
python -m datagol_agent_harness.evals.cli --suite all --concurrency 2
```

When uploaded, the CLI prints a clickable LangSmith URL to inspect row-level scores,
side-by-side prompt diffs, and the complete nested execution tree for every turn.

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
| `Extension` / `LangSmithExtension` | `datagol_agent_harness` | Pluggable runtime extensions / LangSmith tracing |
| `evaluate_agent` | `datagol_agent_harness.evals` | LangSmith evaluation runner |
| `AgentTarget` | `datagol_agent_harness.evals` | Target adapter with telemetry & trace linking |
| `default_evaluators` | `datagol_agent_harness.evals` | Standard suite of evaluators |
| `register_all_tools` | `datagol_agent_harness.builtin` | Filesystem, bash, web, memory tools |

## Runnable examples

| Example | Shows |
|---|---|
| `python -m examples.simple_chat` | Streaming interactive chat + basic tools |
| `python -m examples.multi_agent` | Orchestrator + specialist agents |
| `python -m examples.langsmith_tracing` | LangSmith lifecycle tracing + multi-agent nesting |
| `python -m examples.run_evals` | Agent evaluation suite with LangSmith |
| `python -m examples.skills_agent` | Lazy skill loading |
| `python -m examples.memory_agent` | Two-layer persistent memory |
| `python -m examples.mcp_agent` | MCP tool integration |
| `python -m examples.coding_agent` | Full coding assistant |
| `python -m examples.sandboxed_coder` | Sandboxed execution |
| `uvicorn examples.web_app.server:app --port 8000` | Web app with streaming UI, MCP, and skills |
