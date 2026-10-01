# HarnessX

[![PyPI version](https://img.shields.io/pypi/v/harnessx.svg)](https://pypi.org/project/harnessx/)
[![Python versions](https://img.shields.io/pypi/pyversions/harnessx.svg)](https://pypi.org/project/harnessx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A provider-agnostic, library-first Python toolkit for building production-grade LLM agents.**

Try [harness-web](https://github.com/datagol/agent-harness-x/blob/main/harness-web/README.md) to launch all examples from a browser,
review approvals and live output, or use a separate general chat workspace.

Rather than locking developers into rigid state-machine graphs or opaque persona prompts, `harnessx` gives you a composable set of building blocks: real-time streaming, typed tool registration with schema inference, lazy-loaded skills, multi-agent delegation, 4-tier memory, sandboxed code execution, Model Context Protocol (MCP) tool bridges, native LangSmith tracing, and evaluation suites.

### Why HarnessX?

- **No Graph Boilerplate:** Write simple async Python functions instead of complex state graphs.
- **Provider-Agnostic:** First-class support for Anthropic (Claude 3.5/3.7/Sonnet/Opus) and OpenAI (GPT-4o/GPT-5).
- **Streaming-First:** First-class typed event stream (`run_stream`) for WebSocket and SSE frontends.
- **Lazy Instruction Packs (Skills):** Load specialized guidelines only when needed via YAML-frontmatter `SKILL.md` packs.
- **Guardrails & Execution Backends:** Budget and iteration limits, host subprocess execution, and Docker or macOS Seatbelt access isolation.
- **MCP Native:** Connect to any Model Context Protocol server (stdio subprocess or remote SSE) in 3 lines of code.
- **Full Observability & Evals:** Zero-overhead lifecycle hooks, middleware transforms, native LangSmith tracing, and benchmark evaluations.
- **Typed Decisions:** Optional Jev SDK for Choice routing/classification, Score rubrics, and Noul evidence checks. See the [decision SDK guide](https://harnessx-site.vercel.app/docs/decisions/).

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

The distribution and Python import are both named `harnessx`:
`from harnessx import Agent`. Upgrading from `datagol-agent-harness`? Update the import
to `harnessx` (there is no compatibility alias) and see the [changelog](https://github.com/datagol/agent-harness-x/blob/main/CHANGELOG.md).

```bash
# Add it to a uv project (extras: openai, azure, gemini, openrouter, jev, mcp,
# postgres, temporal, docker, langsmith, server, all)
uv add harnessx
uv add "harnessx[openai]"
uv add "harnessx[all]"

# Or with pip, into any environment
pip install harnessx
```

To work on the SDK itself, clone the repository and let uv create the
environment from the committed lockfile:

```bash
git clone https://github.com/datagol/agent-harness-x.git
cd agent-harness-x
uv sync --all-extras          # SDK, every integration, and the dev tools
uv run pytest -q              # the suite runs offline
```

Set your key (a root `.env` is auto-loaded by the `examples` package):

```bash
ANTHROPIC_API_KEY=sk-ant-...
# or, for OpenAI-backed agents:
OPENAI_API_KEY=sk-...
```

The examples live in the repository, not in the package. Run them as modules
from the checkout root, through the environment uv created:

```bash
uv run python -m examples.simple_chat
```

See the [examples guide](https://github.com/datagol/agent-harness-x/blob/main/examples/README.md) for every entry point, optional
dependencies, and service requirements. For a first run without API keys, use
`uv run python -m examples.skills_demo`, `uv run python -m examples.prompt_caching`, or
`uv run python -m examples.flight_recorder --output incident.hx`.

`Agent()` and `examples.simple_chat` use Anthropic by default. To choose Anthropic,
OpenAI, Gemini, OpenRouter, or Azure OpenAI, use [provider chat](https://github.com/datagol/agent-harness-x/blob/main/examples/provider_chat.py):
`python -m examples.provider_chat --provider openai --model YOUR_MODEL_ID`.
See the [provider setup guide](https://github.com/datagol/agent-harness-x/blob/main/examples/README.md#choose-a-provider) for API keys,
optional dependencies, and streaming options.

The standalone [decision SDK](https://harnessx-site.vercel.app/docs/decisions/) provides typed Jev assessments
alongside agents. Try its offline examples with `python -m examples.jev_routing
--min-confidence 0.8`, `python -m examples.jev_classification`, or
`python -m examples.jev_answer_review`. Live calls require the `jev` extra,
`TYPESAFE_API_KEY`, and an explicit `--live` flag.

Upgrading from 0.3? [MIGRATING.md](https://github.com/datagol/agent-harness-x/blob/main/MIGRATING.md) lists every
renamed, deprecated, and removed name in 0.4. The [durable runtime guide](https://harnessx-site.vercel.app/docs/durable-agent-runs/)
covers the 0.3 runtime changes.

---

## 1. Creating an agent

`Agent` is the core agentic loop: LLM reasoning → tool execution → repeat,
until the model stops with `end_turn`.

```python
import asyncio
from harnessx import Agent, AgentConfig, Limits

agent = Agent(
    config=AgentConfig(
        model="claude-sonnet-4-6",      # or "gpt-4o", "anthropic/claude-3.7-sonnet"
        provider="anthropic",            # "anthropic", "openai", "gemini", "openrouter", or "azure"
        system_prompt="You are a helpful assistant.",
        max_tokens=8192,
        limits=Limits(max_iterations=50),  # loop guardrail
        temperature=None,                # omitted by default for safety
    )
)

async def main():
    async with agent:
        result = await agent.run("What is 17 + 25?")
        print(result.raise_for_status().output)   # raises RunFailed if the run did not complete

asyncio.run(main())
```

`AgentConfig` fields (all optional, these are the defaults):

| Field | Default | Purpose |
|---|---|---|
| `model` | `"claude-sonnet-4-6"` | Model id passed to the provider |
| `provider` | `"anthropic"` | `"anthropic"`, `"openai"`, `"gemini"`, `"openrouter"`, `"azure"`, or a name passed to `register_provider()` |
| `max_tokens` | `None` | Reply token budget. Unset, the provider chooses for the model: 20,000 on Claude 4 models, 8,192 elsewhere |
| `system_prompt` | `"You are a helpful assistant."` | System prompt |
| `temperature` | `None` | Sampling temperature (omitted by default for safety) |
| `limits` | `Limits()` | Budgets: `max_iterations=50` (`0` is unlimited), `max_context_tokens=150_000`, `max_result_chars=12_000`, `max_cost_dollars=None`, `input_cost_per_m=None`, `output_cost_per_m=None` |
| `fallbacks` | `()` | Providers to try, in order, when the primary fails transiently: `Fallback(provider, model=None, max_tokens=None)` |
| `retry` | `RetryPolicy()` | Transient model failures (429, 5xx, timeouts, connection loss): `attempts=2` (`1` disables retry), `backoff_seconds=0.5` doubled each time, `max_backoff_seconds=30.0` caps one wait, `call_timeout_seconds=None` sizes the per-attempt timeout to the reply budget, `switch_after=1` and `cooldown_seconds=0.0` govern failover |
| `prompt_cache` | `PromptCachePolicy()` | Prompt caching of the stable prefix; `None` disables it |
| `tools` | `ToolPolicy()` | Registry-wide tool options: `default_timeout_seconds=None` (300 s), `dedupe_calls=False`, `retry=None` (a `ToolRetry` applied to tools registered without one) |

The sub-policies are frozen dataclasses; change one with `dataclasses.replace`.
The 0.3 flat names (`max_iterations=`, `llm_max_attempts=`, ...) still work and
warn until 0.5; see [MIGRATING.md](https://github.com/datagol/agent-harness-x/blob/main/MIGRATING.md).

### Prompt caching

Every iteration of the loop resends the system prompt, the tool list, and the
conversation so far. Caching is on by default: the engine computes a stable
prefix key and marks where the reusable prefix ends, and each provider
translates that into its vendor's mechanism. Caching is an optimization, never
a correctness dependency, so a provider whose vendor rejects the request runs
it uncached.

```python
from harnessx import Agent, AgentConfig, PromptCachePolicy

# Default: on, vendor default lifetime, history breakpoint after the last turn.
agent = Agent(config=AgentConfig(model="claude-sonnet-4-6"))

# Tuned: a one-hour lifetime and per-tenant separation of identical prompts.
agent = Agent(config=AgentConfig(
    provider="gemini",
    model="gemini-3.6-flash",
    prompt_cache=PromptCachePolicy(ttl_seconds=3600, key_salt="tenant-a"),
))

# Off, for every provider.
agent = Agent(config=AgentConfig(prompt_cache=None))
```

| Provider | What the hint becomes |
|---|---|
| `anthropic` | `cache_control` markers after the system prompt, the last tool, and the last message. `ttl_seconds >= 3600` selects the one-hour cache; otherwise the five-minute default applies. Requests are only cached above Anthropic's minimum prefix size |
| `openai` | Automatic prefix caching, plus `prompt_cache_key` set to the prefix key so identical prefixes route to the same cache |
| `azure`, `openrouter` | Automatic prefix caching only; no routing key is sent |
| `gemini` | Implicit caching by default. With `ttl_seconds` set (or the `GEMINI_PROMPT_CACHE_TTL` variable), the system prompt and tools are uploaded once as an explicit cache and referenced per call. A prefix below Gemini's minimum cacheable size runs uncached, and that prefix is not retried for ten minutes |

Cache hits show up in the usual usage fields: `RunResult.usage.cache_read_input_tokens`
and `cache_creation_input_tokens`, both counted inside `input_tokens`, which is
the full prompt size on every provider. Every `LLM_REQUEST` hook carries
`data["prefix_key"]`, so an observer can assert that consecutive iterations
share one key. A prefix that drifts, for example a timestamp in the system
prompt or tools registered in a different order, silently defeats every
vendor's cache. Custom providers receive the hint as an optional `cache`
keyword on `create` and `stream`; ignoring it is fine, honoring it means
failing open.

### Azure OpenAI

Install the extra (`uv add "harnessx[azure]"`; the `openai`
package covers Azure). Azure reuses the OpenAI translation layer through the
`AsyncAzureOpenAI` client, so completion **and streaming** both work. On Azure
the `model` you pass is the **deployment name**, and both an endpoint and an
API version are required for provider-created clients. There are two ways to
supply credentials — the harness never loads a `.env` file itself, so your app
is responsible for populating the environment or passing values in code.

**1. Config-driven, credentials from the environment** (`AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`):

```python
from harnessx import Agent, AgentConfig

# env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION
agent = Agent(config=AgentConfig(provider="azure", model="<your-deployment>"))
```

**2. Explicit construction + injection** (pass keys/endpoint in code, or hand in
a fully preconfigured client):

```python
from harnessx import Agent, AgentConfig
from harnessx.providers.azure_openai import AzureOpenAIProvider

provider = AzureOpenAIProvider(
    azure_endpoint="https://<resource>.openai.azure.com",
    api_key="<key>",
    api_version="2024-10-21",        # use an API version your resource supports
    azure_deployment="<deployment>", # optional; overrides AgentConfig.model per request
)
agent = Agent(
    config=AgentConfig(provider="azure", model="<deployment>"),
    provider=provider,
)
```

Token-based auth is supported: omit `api_key` and pass `azure_ad_token` or
`azure_ad_token_provider` — they are forwarded straight to `AsyncAzureOpenAI`.
Token counting is approximate for Azure, because a deployment name does not
identify the underlying tokenizer.

You can also inject your own pieces — everything is a constructor parameter:

```python
from harnessx import (
    Agent, AgentConfig, ConversationMemory, HookManager,
    MiddlewarePipeline, PermissionManager, ToolRegistry,
)

agent = Agent(
    config=AgentConfig(system_prompt="..."),
    tools=ToolRegistry(),               # your own registry
    subagents=[],                      # declarative SubAgent specialists
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
        print("Agent:", (await agent.run(text)).output)
```

---

## 2. Registering tools

A tool has two parts: a Python handler that performs an operation, and a registered
name, description, input schema, and execution policy that the agent uses to call
it. Registration makes a handler available; it does not execute it.

### Choose a registration style

| Style | Example | Best use |
| --- | --- | --- |
| Constructor function | `Agent(tools=[fetch_url])` | A small, explicit list of handlers |
| Constructor built-in name | `Agent(tools=["fetch_url"])` | Load a shipped tool without importing its handler |
| Constructor bundle | `Agent(tools=["web"])` | Load a built-in group |
| Later registration | `agent.tools.register_tool(fetch_url)` | Assemble tools after creating the agent |
| Decorator | `@agent.tools.register(...)` | Define a custom handler beside its registration |
| Built-in helper | `register_web_tools(agent.tools, ...)` | Configure and select tools in a bundle |
| Registry loading | `registry.load_builtin("web", ...)` | The same built-in helper options through one entry point |
| Tool definition | `Agent(tools=[definition])` | Reuse an explicit schema and execution policy |

All of these inherit the agent's permission policy when permission is omitted.
A normal `Agent()` defaults to **ALLOW**. A custom manager can instead default to
ASK or DENY. A preconfigured `ToolDefinition` retains its explicit permission.
Registration style does not choose the permission default.

Prefer functions for simple registration and helpers for configured capabilities.
Bundle names currently include `filesystem`, `bash`, `web`, `memory`, and `all`;
a bundle can contain more tools than one standalone function.

### Constructor registration and registration afterward

```python
from harnessx import Agent
from harnessx.builtin.web import fetch_url

agent = Agent(tools=[fetch_url])
```

The constructor infers the tool name, schema, and description from the function.
You can also register afterward, before starting the run:

```python
from harnessx import Agent
from harnessx.builtin.web import fetch_url

agent = Agent()
agent.tools.register_tool(fetch_url)
```

Both expose `fetch_url` to subsequent model calls. Finish changing bindings before
starting a run; mutating a registry during execution can invalidate an in-flight
call. Use `async with agent:` or `await agent.aclose()` to close an agent when done.

A constructor list may mix functions, built-in names, and `ToolDefinition` objects.
Passing an existing `ToolRegistry` attaches that same registry to an ordinary
agent; it does not clone the registry. Declare subagent tools separately: each
specialist task receives a fresh registry with copied schemas and metadata.

### Direct function versus a registration helper

These imports serve different purposes:

```python
from harnessx.builtin.web import fetch_url           # operation
from harnessx.builtin.web import register_web_tools  # registration/configuration
```

`fetch_url` is an async Python function. `register_web_tools` installs the web
bundle into a supplied registry and returns the list of registered names.
The web bundle currently contains `fetch_url`.

```python
from harnessx import Agent
from harnessx.builtin.web import register_web_tools

agent = Agent()
registered = register_web_tools(
    agent.tools,
    include=["fetch_url"],
    max_length=20_000,
)
assert registered == ["fetch_url"]
```

This uses the same HTTP-fetch implementation, with configured response truncation.
With a non-default `max_length`, the helper creates a wrapper whose model input is
only `url`; the model cannot override that configured limit. Direct `fetch_url`
registration exposes its `url` and optional `max_length` function parameters.
With the helper's default `max_length=500_000`, it currently registers the direct
function, so that parameter remains visible. Truncation limits the returned text,
not bytes downloaded from the network.

Both approaches inherit permissions. Configuration can still change the handler
and schema; equivalent permissions do not imply identical capabilities.

Do not pass `register_web_tools` as an agent tool: its argument is a registry, not
an input for the model. Call the helper during application setup.

### Permission defaults and precedence

| Registration | Ordinary agent | Manager default ASK | Manager default DENY |
| --- | --- | --- | --- |
| Function, name, bundle, or helper; permission omitted | ALLOW | ASK | DENY |
| Explicit `permission=ALLOW` | ALLOW | ALLOW | ALLOW |
| Explicit `permission=ASK` | ASK | ASK | ASK |
| Explicit `permission=DENY` | DENY | DENY | DENY |

A **manager override for the tool name takes precedence over every row**. Otherwise,
an explicit tool permission takes precedence over the manager default. After
resolving the effective level, DENY blocks execution even if there is a previous
session grant. A session grant can satisfy ASK without another callback.

For example, register an explicit policy or override it later:

```python
from harnessx import Agent, PermissionLevel
from harnessx.builtin.web import fetch_url

agent = Agent()
agent.tools.register_tool(fetch_url, permission=PermissionLevel.ALLOW)
agent.permissions.set_permission("fetch_url", PermissionLevel.DENY)
```

#### ASK needs an approval mechanism

A plain agent does not display a prompt automatically. Direct execution and
ordinary runs deny ASK calls when no approval callback is provided.

For a terminal application:

```python
from harnessx import Agent, CliPermissionManager
from harnessx.builtin.web import fetch_url

agent = Agent(tools=[fetch_url], permissions=CliPermissionManager())
```

`CliPermissionManager` defaults to ASK. That includes built-in tools registered
without an explicit policy. Alternatively, supply a synchronous or asynchronous
callback. This complete callback example approves only one known URL:

```python
from harnessx import Agent, PermissionLevel, PermissionManager, ToolCall, ToolDefinition
from harnessx.builtin.web import fetch_url

async def approve(call: ToolCall, definition: ToolDefinition) -> bool:
    return call.name == "fetch_url" and call.input.get("url") == "https://example.com"

agent = Agent(
    tools=[fetch_url],
    permissions=PermissionManager(
        default_level=PermissionLevel.ASK,
        approval_callback=approve,
    ),
)
```

A UI can implement the same callback by awaiting the user's decision. Providing
only `approval_callback` leaves the default at ALLOW; also select ASK for the
tools that should request approval. Callbacks must return an actual boolean.

Durable `AgentRuntime` runs persist ASK requests and return `awaiting_input`.
Use the returned execution key with `runtime.approve(...)`, then resume the run;
see the recovery documentation. This persisted approval behavior applies to the
outer delegation call, not to the non-durable specialist's internal tools.
MCP tools retain their explicitly configured per-server permission policy.

### Decorators and schema inference

```python
from harnessx import Agent, PermissionLevel

agent = Agent()

@agent.tools.register(permission=PermissionLevel.ALLOW, replay_policy="safe")
def calculate(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a * b
```

The description before an `Args:` or other recognized section becomes the tool
description, including multiple lines. An `Args:` section documents parameters.
Type hints drive JSON Schema: primitive types, nullable unions, `Literal`, nested
lists, and dictionaries with string keys are supported. Nullable parameters remain
required unless they have a Python default. Unsupported annotations fail during
registration; use `register_with_schema` for a custom contract.

Sync and async handlers both work. Synchronous handlers run outside the event
loop. A timeout stops waiting for a synchronous handler; Python cannot forcibly
stop its worker thread or reverse an external effect.

### Explicit schemas and reusable definitions

Use a schema when type inference cannot express the input contract:

```python
from harnessx import Agent, ToolRegistry

async def label(value: str) -> str:
    return value.upper()

registry = ToolRegistry()
definition = registry.register_with_schema(
    name="label",
    description="Uppercase a short label.",
    input_schema={
        "type": "object",
        "properties": {"value": {"type": "string", "maxLength": 40}},
        "required": ["value"],
        "additionalProperties": False,
    },
    handler=label,
)
agent = Agent(tools=[definition])
```

Registering a `ToolDefinition` copies its schema and metadata while retaining the
handler. Registration options override only specified metadata; registering an
existing definition does not silently reset its permissions or replay policy.

### Configured built-in tools

Helpers support `include`, `exclude`, `permission`, and `replace`, plus options
specific to the bundle. Unknown names, filters, and unsupported options fail.

For filesystem inspection, configure a root and only the operations you need:

```python
from pathlib import Path
from harnessx import Agent
from harnessx.builtin import register_filesystem_tools

agent = Agent()
register_filesystem_tools(
    agent.tools,
    include=["read_file", "list_directory"],
    base_path=str(Path.cwd()),
    max_read_bytes=1_000_000,
    max_directory_entries=1_000,
)
```

The root must exist. Scoped operations reject symlinks in files and parent
directories using descriptor-relative, no-follow access on POSIX. This is an
application boundary, not an OS sandbox for arbitrary code: hard links, mounts,
and other processes relocating directories require stronger isolation. Without
`base_path`, filesystem tools use host filesystem access under process permissions.

Configured filesystem limits are hidden from model input. Existing files require
`overwrite=True`; writes publish atomically. Writes and downloadable-file
creation remain non-concurrent when registered through filesystem helpers,
regardless of permission. Direct standalone functions use ordinary registration
metadata, so choose `concurrent=False` explicitly if needed.

Download copies use configured `output_dir`, otherwise `AGENT_OUTPUT_DIR` or
`./output`. Keep the output directory application-owned. Filesystem reads and
writes run outside the event loop.

Other options include a sandbox for the bash helper and memory stores for the
memory helper. Single-tool loading accepts the corresponding bundle options:

```python
from pathlib import Path
from harnessx import ToolRegistry

registry = ToolRegistry()
registry.load_builtin("read_file", base_path=str(Path.cwd()))
```

### Replacement, timeouts, and replay

Duplicate tool names fail by default. Use `replace=True` to deliberately replace a
binding during setup. Inspect `get_tool(name)`, `list_tools()`, or
`get_tool_params()`; the last returns canonical provider-neutral tool schemas.

Ordinary function registrations default to `concurrent=True`, a 300-second timeout,
and `replay_policy="manual"`. Configure these with `register_tool` or the decorator.
Built-in helpers do not automatically mark fetching or other operations replay-safe.

Durable execution uses `manual` to stop for recovery when an interrupted operation's
outcome is unknown. Select `safe` only when repeating an operation is safe, or
`idempotent` when the integration actually enforces idempotency. Neither setting
provides exactly-once external effects on its own.

### Retrying a flaky tool

A tool that calls a rate-limited API fails for a reason that clears on its own.
Raise `TransientToolError` from the handler and the engine runs the call again
under the tool's `ToolRetry`, exactly the way it retries a model call. Every
other exception still becomes a one-line error result for the model, as before.

```python
from harnessx import ToolRetry, TransientToolError

@agent.tools.register(replay_policy="idempotent", retry=ToolRetry(attempts=3, backoff_seconds=2))
async def fetch_page(url: str) -> str:
    """Fetch a page."""
    response = await http.get(url)
    if response.status in (429, 503):
        raise TransientToolError(f"{response.status} from {url}")
    return response.text
```

`ToolRetry(attempts=3, backoff_seconds=0.5, max_backoff_seconds=30.0,
retry_error_results=False)` is the whole policy; `attempts=1` disables retry.
Set a default for a whole registry with `ToolPolicy(retry=...)`; a tool
registered with its own `retry=` keeps it.

Retry only happens when `replay_policy` is `safe` or `idempotent`: repeating a
`manual` tool could duplicate a side effect, so a declared transient failure
there goes straight to the model instead. Two more ways to spot a failure that
arrived as a successful result:

- `retry_error_results=True` retries an error result whose text reads as
  throttling, overload, or a timeout.
- `retry_if_result=lambda result: ...` is your own check, for an API that
  reports `{"error": ..., "status": 429}` inside a 200.

Each attempt is journaled and emits the `RETRY` hook, so the retries are
visible in the flight recorder and in tracing rather than hidden in a handler.

### Calling a function versus executing an agent tool

`await fetch_url(url)` is an ordinary Python call. It bypasses the registry's
schema validation, permission manager, timeout, and durable execution records.

`await registry.execute(call, permissions=agent.permissions)` validates the input,
checks permission, and applies the registered timeout. Without a manager it uses
the ordinary ALLOW default for unspecified policies; explicit ASK without a
callback and explicit DENY fail. It does not provide durable replay or the agent's
complete middleware/event pipeline.

`agent.run(...)`, `agent.run_stream(...)`, and `AgentRuntime` use the shared engine
for model-requested tools. They apply execution checks before dispatch, with
persisted approvals and recovery when using a durable runtime. Handler exceptions
usually become error tool results; runtime/storage errors may fail the run.


---

## 3. Streaming agents

`Agent.stream_text()` yields the answer as text and raises `RunFailed` if the
run does not complete:

```python
async with agent:
    async for text in agent.stream_text("Tell me a story"):
        print(text, end="", flush=True)
```

Pass `on_reset=callback` to be told when a retried model call restarts the
answer, so a display can discard the provisional text it has shown.

`Agent.run_stream()` yields typed events (tool calls, results, thinking) and
exposes the final `RunResult`. Use its async context manager to close or cancel
an abandoned stream; `stream.text()` is the same text view over its events.

```python
import asyncio
from harnessx import AgentConfig, Agent, RunEventType

agent = Agent(config=AgentConfig(system_prompt="You are helpful."))

async def main():
    async with agent, agent.run_stream("Tell me a story") as stream:
        async for event in stream:
            if event.type == RunEventType.TEXT_DELTA:
                print(event.data, end="", flush=True)
            elif event.type == RunEventType.TOOL_CALL_START:
                print(f"\n> calling {event.data.name}({event.data.input})")
            elif event.type == RunEventType.TOOL_RESULT:
                if event.data.is_error:
                    print(f"  tool error: {event.data.content}")
            elif event.type == RunEventType.TURN_COMPLETE:
                print()

asyncio.run(main())
```

Event types include `TEXT_DELTA`, `TEXT_COMPLETE`, `TOOL_CALL_START`,
`TOOL_CALL_COMPLETE`, `TOOL_RESULT`, `THINKING_DELTA`, `TURN_COMPLETE`,
`ERROR`, `ATTEMPT_RESET`, and `RUN_RESULT`. Durable runs also emit approval and
recovery events. Text deltas are provisional; discard an interrupted attempt's
text when `ATTEMPT_RESET` arrives.

Usage stats accumulate on the guardrails engine:

```python
agent.guardrails.usage_summary
# {'iterations': 3, 'input_tokens': 5120, 'output_tokens': 640, 'estimated_cost': '$0.0250'}
```

---

## 4. Multi-agent orchestration

Declare specialists when creating an orchestrator. HarnessX handles creating,
running, and closing the specialist for each task; you do not write or register
delegation wrappers.

```python
import asyncio
from harnessx import Agent, AgentConfig, SubAgent
from harnessx.builtin.web import fetch_url

async def main():
    async with Agent(
        config=AgentConfig(
            system_prompt="Delegate research when useful, then synthesize the findings.",
        ),
        subagents=[
            SubAgent(
                name="researcher",
                description="Read public URLs and summarize their evidence.",
                config=AgentConfig(
                    system_prompt="Read the supplied URLs and cite your sources.",
                    limits=Limits(max_iterations=5),
                ),
                tools=[fetch_url],
            ),
        ],
    ) as orchestrator:
        result = await orchestrator.run("Read https://example.com and summarize it.")
        if result.status != "completed":
            raise RuntimeError(str(result.error or result.pending))
        print(result.output)

asyncio.run(main())
```

### Definition and routing

`SubAgent` is a definition, not a running agent. Required fields:

- `name`: 1–55 letters, digits, underscores, or hyphens, starting with a letter or underscore.
- `description`: tells the orchestrator when this specialist is useful.
- `config`: an explicit `AgentConfig`; its system prompt instructs the specialist.

Optional fields:

- `tools`: the same functions, built-in names, definitions, or registry formats accepted by `Agent`.
- `skills`: a list of skill paths, loaded separately for each task.
- `permission`: permission to delegate; omitted means inherit the parent's manager.
- `timeout_seconds`: whole-task timeout, default 300 seconds, including child model and tool calls.

The SDK exposes `delegate_<name>(task: str)` to the model internally. The schema
and description tell the orchestrator how to request a specialist. Do not also
register a tool with that name; collisions and duplicate specialist names fail
during construction. No separate orchestrator class is necessary.

The model decides whether to delegate. Naming specialists does not force a fixed
sequence; write workflow code when you need a mandatory sequence. Use
`examples/multi_agent.py` for research, code review, and scoped file analysis.

### Context, resources, and permissions

Each delegated task gets fresh memory, configuration, and a tool registry. It sees
only its own prompt and the assigned task. Include the code, question, URLs, paths,
or other context it needs in the task. The child's model/provider comes from its
explicit config; it does not inherit the parent's model or injected provider.

Handler functions remain shared application objects: separate conversations do
not isolate files, databases, closures, or arbitrary Python code. Child-owned
providers and memory close on completion, failure, and cancellation. Skills may
add their own tool and prompt catalog; do not also register a `Skill` tool.

The parent's permission manager is shared, including its defaults, name overrides,
session grants, and approval callback. Generated delegation permission is separate
from each child tool's permission. Approving delegation does not approve its tools.
The same tool-name override applies in parent and child. An ordinary manager defaults
to ALLOW; use explicit permissions or another manager to change that policy.

Parent tools, hooks, middleware, extensions, MCP connections, and sandboxes are not
automatically copied. The generated call uses the parent's normal tool pipeline;
the child uses its own ordinary engine. Use the supported manual-wrapper pattern
for advanced child dependencies and their lifecycle management. It remains useful
for a specialist with custom tracing extensions or an injected provider.

### Results, streaming, and limits

The child returns its completed output to the orchestrator, which continues and
synthesizes a response. There is no conversation handoff. Non-completed child
statuses and child failures become delegation errors; partial output is not
reported as successful specialist work.

Generated delegation tools are non-concurrent and use manual replay policy. The
current engine serializes a tool batch containing such a delegation. Both
`run` and `run_stream` support the same behavior. Streaming reports ordinary
`tool_call_start`, `tool_call_complete`, and `tool_result` events for delegation;
child text does not appear as parent answer text.

Each specialist has its own iteration and cost limits. Parent `RunResult.usage`
and guardrail totals remain parent-only, not a combined team budget. Set child
limits explicitly. A timeout cancels waiting and closes child resources, but
cannot reverse external effects or forcibly stop a synchronous tool thread.

### Durable execution boundary

`AgentRuntime` persists the outer delegation as an ordinary manual-replay tool
call. Committed results are reused on recovery. An interrupted delegation with an
unknown outcome stops for manual recovery rather than automatically starting a
new specialist. Live model reruns are not deterministic.

The child uses an ordinary, non-durable run. Child progress, internal approvals,
intermediate artifacts, and conversations are not independently persisted by the
parent runtime. A child ASK tool uses the approval callback and fails closed
without one; it does not create a resumable approval in the parent runtime.
Treat output referring to child-local artifacts accordingly; durable child-artifact
transfer is not supplied by this API.

Rebind the same `subagents` definitions when restoring sessions or constructing
versioned runtime factories, just as you rebind tools. Session snapshots do not
serialize Python handlers. Keep definition changes under the application's normal
versioned deployment controls.

Independent child recovery, persistent specialist conversations, nested declarative
teams, handoffs, and child-token streaming are outside this initial API.

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
from harnessx import Agent, AgentConfig

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
from harnessx import SkillManager

manager = SkillManager.from_paths(["./skills/commit-message"])
agent = Agent(config=AgentConfig(system_prompt="..."), skills=manager)

for s in agent.skills.list():
    print(s.name, "-", s.description)
```

### Reacting to skill invocation

A `SKILL_INVOKED` hook fires when the model loads a skill:

```python
from harnessx import HookContext, HookEvent

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
from harnessx import AgentMemory, PermissionLevel

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
from harnessx import LongTermMemory

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

Tools use `ALLOW` (execute), `ASK` (request approval), or `DENY` (reject).
Omitting the permission inherits the manager's default, which is `ALLOW`.
This applies to constructor tools, later registrations, and direct
`ToolRegistry.execute()` calls without a custom manager. Built-in helpers also inherit the manager default; file writes, shell commands,
and URL fetching use ALLOW under a default agent. Select ASK explicitly when
approval is required. Explicit MCP server policies remain in effect.
Per-tool overrides take precedence over explicit registration permissions and
the manager default. `DENY` also overrides a previous session grant.

```python
from harnessx import PermissionLevel

# At registration
@agent.tools.register(permission=PermissionLevel.ALLOW)
def add(a: int, b: int) -> int: ...

# Or override later
agent.permissions.set_permission("run_bash", PermissionLevel.ASK)
agent.permissions.grant_session("read_file")   # pre-approve for this session
```

Direct execution handles `ASK` through a sync or async callback. Without a
callback it denies the call; the SDK never implicitly reads stdin.

```python
from harnessx import Agent, PermissionLevel, PermissionManager, ToolCall, ToolDefinition

async def approve(call: ToolCall, definition: ToolDefinition) -> bool:
    return await approval_ui.request(call.name, call.input)

agent = Agent(permissions=PermissionManager(
    default_level=PermissionLevel.ASK,
    approval_callback=approve,
))
```

Setting `default_level=PermissionLevel.ASK` requests approval for tools without
an explicit permission. Supplying only `approval_callback` leaves the default
at `ALLOW`; the callback handles tools explicitly configured as `ASK`.

Terminal apps can explicitly use `Agent(permissions=CliPermissionManager())`
with `CliPermissionManager` imported from `harnessx`; this opt-in manager defaults
to `ASK`. Durable runtimes persist
approval requests: `await runtime.approve(pending, resume=True)` records the
decision and finishes the run, and `runtime.decline(pending)` refuses it.

### Guardrails

```python
from harnessx import Agent, AgentConfig, Limits

agent = Agent(config=AgentConfig(limits=Limits(
    max_iterations=20,
    max_context_tokens=100_000,
    max_cost_dollars=0.50,
    input_cost_per_m=3.00,   # application-supplied prices, not a live price lookup
    output_cost_per_m=15.00,
)))
agent.guardrails.usage_summary
```

Limits are validated at construction. A reached guardrail returns a failed
`RunResult` with error details. `Limits(max_iterations=0)` means unlimited.

A reply cut off at the token budget ends the run with status `completed` and
stop reason `max_tokens`; `result.truncated` is true, `result.ok` is false, and
`raise_for_status()` raises `RunTruncated`. Raise `max_tokens` or ask for
shorter output.

### Errors

Every exception the SDK raises derives from `HarnessError`. Invalid
configuration raises `ConfigurationError` (a `ValueError`); using a closed or
busy agent raises `RuntimeStateError` (a `RuntimeError`); an unknown execution
key raises `UnknownExecutionKey` (a `KeyError`). Runs report their outcome as
data: `result.ok`, `result.failed`, and `result.needs_input` inspect the
status, and `result.raise_for_status()` turns a run that did not complete into
`RunFailed`, `RunAwaitingInput`, or `RunCancelled`, each carrying `.result`.
`PermissionLevel`, like every other enum here, is a `str` enum. A tool handler
that raises `TransientToolError` asks for a retry rather than reporting a
failure; see [Retrying a flaky tool](#retrying-a-flaky-tool).

### Retries

There is exactly one retry loop, in the engine. It clears buffered stream
events, emits `ATTEMPT_RESET`, journals every attempt, and marks usage
incomplete, so nothing outside the engine can stand in for it. What you
configure are the seams it consults.

| Seam | Where |
|---|---|
| How many attempts, how long to wait | `AgentConfig.retry` for the model, `ToolRetry` per tool |
| Which failures are worth retrying | `LLMProvider.is_transient(exc)`, overridable per provider |
| How long the server asked us to wait | `LLMProvider.retry_after(exc)`, reading `Retry-After` |
| What to call when one vendor is down | `AgentConfig.fallbacks` |
| What happened | the `RETRY` hook, and the journal |

Clients the harness builds itself pass `max_retries=0` to the vendor SDK, so a
503 costs the attempts you configured and no more. An SDK client you construct
and inject keeps whatever you set on it. Because the engine owns the count, it
also recognizes the SDK's own connection errors, including the case where an
egress proxy refuses a destination on policy grounds, which is a decision rather
than a blip and is never retried.

Middleware cannot retry: it transforms a request or a result and never re-issues
a call. Put normalization there, and retry policy here.

#### Failing over to another provider

Name the primary in `provider`, and the providers to try after it in
`fallbacks`:

```python
from harnessx import Agent, AgentConfig, Fallback, RetryPolicy

agent = Agent(config=AgentConfig(
    model="claude-sonnet-4-6",
    provider="anthropic",
    fallbacks=[
        Fallback("openrouter", model="anthropic/claude-sonnet-4.6"),  # same model, other vendor
        Fallback("openai", model="gpt-5", max_tokens=16_000),         # last resort
    ],
    retry=RetryPolicy(switch_after=2, cooldown_seconds=60),
))

async with agent:          # the agent built the chain, so the agent closes it
    ...
```

Each fallback names its own model id, because the same model is spelled
differently on different vendors; leave `model` unset to reuse
`AgentConfig.model`. A transient failure adds a strike to the provider in use,
and at `switch_after` the next one is tried inside the same call. A
deterministic error such as a bad request raises at once, and a stream fails
over only before its first chunk. When every provider fails, `attempts` decides
whether to walk the chain again. The `LLM_RESPONSE` and `RETRY` hooks report
which provider actually served the call, and so does LangSmith.

Because the chain is configuration, it is serializable: durable runs and
Temporal workers carry it like any other config, and there is nothing extra for
you to close.

#### When a provider has to be a live object

A pre-configured SDK client cannot be named in config. Build the chain yourself
and inject it, which also makes it yours to close:

```python
from harnessx import Agent, AgentConfig, Fallback, FallbackProvider, OpenAIProvider

chain = FallbackProvider(
    OpenAIProvider(client=my_client),
    fallbacks=[Fallback("anthropic")],
    switch_after=2,
)
async with Agent(config=AgentConfig(model="gpt-5", provider="openai"), provider=chain) as agent:
    ...
await chain.aclose()
```

Providers given as names are built and closed by the chain; instances you pass
in stay yours. Setting both `fallbacks` in config and `provider=` is a
`ConfigurationError`, since they would describe the same thing twice.

---

## 8. Middleware, hooks, and extensions

| Interface | Use it for |
| --- | --- |
| Middleware | Transform model requests/responses or tool calls/results |
| Hooks | Observe lifecycle events for logging or diagnostics |
| Extensions | Package middleware, hooks, tools, configuration, state, and cleanup |

An extension can install middleware with `ctx.add_middleware(...)`. Use standalone
middleware for one focused transformation; use an extension to reuse a complete
feature or manage its resources and saved state.

```python
from dataclasses import replace
from harnessx import Agent, Extension, ExtensionContext, Middleware, ToolCall

class NormalizeSearch(Middleware):
    async def before_tool_execution(self, call: ToolCall) -> ToolCall:
        query = call.input.get("query")
        if call.name == "search" and isinstance(query, str):
            return replace(call, input={**call.input, "query": query.strip()})
        return call

class SearchNormalization(Extension):
    name = "search_normalization"

    def install(self, ctx: ExtensionContext) -> None:
        ctx.add_middleware(NormalizeSearch())

agent = Agent(extensions=[SearchNormalization()])
# Register your search tool separately, then run inside `async with agent:`.
```

Read the detailed guides:

- [Middleware and hooks](https://harnessx-site.vercel.app/docs/hooks/): all four stages, ordering, complete
  examples, streaming behavior, errors, retries, and observation boundaries.
- [Extensions](https://harnessx-site.vercel.app/docs/extensions/): when to use an extension, composing middleware
  with tools and hooks, lifecycle callbacks, ownership, persistence, and built-ins.

Hook data is event specific, and the keys each built-in event carries are
declared as TypedDicts in `harnessx.hooks` (`HOOK_PAYLOADS` maps an event to
its shape). `LLM_REQUEST` carries the request as sent: the rendered `system`
prompt, `model`, `max_tokens`, `temperature`, `stream`, `message_count`,
`tool_count`, and `prefix_key`, the prompt-cache key for that request (`None`
when caching is disabled). `AGENT_END` carries the final `RunResult`; `SANDBOX_EXEC` reports
every sandbox execution; `CHECKPOINT` marks every persisted phase of a durable
run. `hooks.on(...)`, `before_tool(...)`, `after_tool(...)`, and `on_error(...)`
return a `Registration` whose `close()` unhooks the callback.

Mandatory permissions and effect checks belong in the execution path. Hook
failures are suppressed, and terminal extension observers are best effort.
Result middleware can change what the model receives without erasing raw data
already retained by durable execution or recording.

---

## 9. MCP servers

`MCPManager` connects to Model Context Protocol servers (stdio subprocess or
remote SSE). `Agent(mcp=manager)` bridges the discovered tools into the agent's
registry at construction, so MCP tools look exactly like native tools to the
model; their names are listed on `agent.mcp_tools`.

```python
from harnessx import Agent, AgentConfig, MCPManager, MCPServerConfig

async with MCPManager() as mcp:
    await mcp.connect(MCPServerConfig.stdio(
        "files", "npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    ))
    # or a remote server, streamable HTTP with an SSE fallback:
    # await mcp.connect(MCPServerConfig.http("remote", "http://localhost:8000/mcp",
    #                                        headers={"Authorization": "Bearer ..."}))

    async with Agent(config=AgentConfig(system_prompt="..."), mcp=mcp) as agent:
        print(agent.mcp_tools)                 # ('files_read_file', ...)
        print((await agent.run("What's in /tmp?")).output)
# leaving the manager's block disconnects every server
```

`MCPServerConfig.stdio(...)` and `.http(...)` pick the transport by
constructor; each takes `permission=` for the server's tools (default `ASK`),
plus `replay_policy=` and `retry=`. A server whose tools only read is worth
declaring `replay_policy="safe"`: bridged tools then retry a throttled call
three times on their own, including the common case of a server that reports a
429 inside an otherwise successful result.
`connect()` also accepts a name with `command=` or `url=` keywords. The name is
yours to choose: it keys the connection and prefixes every bridged tool. The
manager is caller-owned: closing the agent leaves it connected, and one manager
can serve several agents. Connecting a server name twice raises `ValueError`.

See `examples/mcp_agent.py` for the interactive version.

---

## 10. Session persistence

Save and restore an agent's conversation:

```python
session_id = await agent.save_session()            # -> writes .agent_sessions/

# later, or in another process:
from harnessx import Agent
async with await Agent.load_session(session_id) as restored:
    print((await restored.run("Where were we?")).output)
```

Snapshots include configuration, messages, usage, metadata, extension state, and
copied spill artifacts. Supply the same extension names on load; an explicitly
supplied configuration must match the snapshot. Saving is atomic and rejects
unsupported values instead of stringifying them. Saved artifacts survive agent
closure until `PersistentMemory(directory).delete_session(session_id)`.

For managed multi-session services (checkpointing, pause/resume, expiry),
use `AgentRuntime` with a backend; see the [durable runtime guide](https://harnessx-site.vercel.app/docs/durable-agent-runs/).

```python
from harnessx import Agent, AgentRuntime, SQLiteBackend

async with await SQLiteBackend.connect("runtime.db") as backend:
    async with AgentRuntime(Agent(), backend=backend) as runtime:
        result = await runtime.run("write the report")
        if result.needs_input:                      # a tool is waiting for approval
            result = await runtime.approve(result.pending[0], resume=True)
        async for text in runtime.stream_text("summarize it"):
            print(text, end="")
        print((await runtime.status())["run"].status)
```

`run()`, `run_stream()`, and `stream_text()` mirror the `Agent` methods;
`submit()` returns a `RunHandle` when the caller wants to detach. `execute()`,
`execute_stream()`, and `get_status()` remain as deprecated aliases until 0.5.

The default backend is SQLite. PostgreSQL automatically provisions its schema from
a connection string (`await PostgresBackend.connect(dsn)`); Temporal adds
distributed workflow recovery and Redis events. Every backend is an async
context manager.
For PostgreSQL, pass your configured agent directly:
`runtime = AgentRuntime(agent, backend=backend)`. In a replacement process,
construct a fresh compatible `Agent` and call `await runtime.resume(session_id)`.
No agent registry is needed for this pattern. `AgentRef` and `AgentRegistry` remain
available for named definitions and Temporal workers.

Try the [PostgreSQL crash-and-resume walkthrough](https://harnessx-site.vercel.app/docs/postgres-durability/): write
a report, let the process exit abruptly, and finish the same run in another process
with the tool invocation count still at one. It uses scripted model responses,
so only PostgreSQL and the `postgres` extra are required, not a model API key.
Connection details can come from `DATABASE_URL` or `--config
examples/postgres.config.json`; the [config template](https://github.com/datagol/agent-harness-x/blob/main/examples/postgres.config.example.json)
supports automatic password URL encoding. Use the `check` command to connect and
prepare the runtime schema before starting a run.

For incident debugging, opt in with `AgentRuntime(..., recording=True)`. The
[flight recorder](https://harnessx-site.vercel.app/docs/flight-recorder/) preserves model/tool attempt boundaries,
exports portable incident bundles, and verifies/plays them back offline. Export
payloads and artifacts are opt-in. SQLite is locally tested; PostgreSQL requires
service qualification, and Temporal recording is not implemented. Try
`python -m examples.flight_recorder --output /tmp/invoice-incident.hx` without an API key.

---

## 11. Shipping a web app

The harness is transport-agnostic: a web server runs the same `Agent` and
translates its run events to whatever the browser speaks. The pattern is

1. **Stream the run** with `agent.run_stream()` and forward each event as a
   Server-Sent Event or WebSocket frame:

```python
import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from harnessx import Agent, RunEventType

@asynccontextmanager
async def lifespan(app):
    async with Agent() as agent:
        app.state.agent = agent
        app.state.lock = asyncio.Lock()  # one agent, one active run at a time
        yield

app = FastAPI(lifespan=lifespan)

@app.post("/api/stream")
async def stream(req: dict):
    async def event_generator():
        async with app.state.lock:
            async with app.state.agent.run_stream(req["message"]) as events:
                async for event in events:
                    if event.type == RunEventType.TEXT_DELTA:
                        yield f"data: {json.dumps({'type': 'text_delta', 'content': event.data})}\n\n"
                    elif event.type == RunEventType.ATTEMPT_RESET:
                        yield 'data: {"type": "attempt_reset"}\n\n'
                    elif event.type == RunEventType.ERROR:
                        yield f"data: {json.dumps({'type': 'error', 'content': str(event.data)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

2. **Decide permissions server-side.** Pre-approve tools with
   `PermissionLevel.ALLOW`, or give the `PermissionManager` an approval
   callback that asks the browser; never fall back to stdin prompts.
3. **One agent per conversation.** An `Agent` runs one turn at a time, so a
   multi-user service keeps a distinct agent, or a durable `AgentRuntime`
   session, per conversation.

The repository's complete web application is [harness-web](https://github.com/datagol/agent-harness-x/tree/main/harness-web):
a FastAPI backend and a Vite frontend that run every example in this
repository, stream their output, present approvals in the browser, and host a
general chat with per-conversation skills and MCP servers. From the checkout:

```bash
npm --prefix harness-web ci && npm --prefix harness-web run build
uv run python harness-web/run.py        # http://127.0.0.1:8765
```

---

## 12. Evaluations with LangSmith

The harness includes first-class evaluation capabilities powered by the
**LangSmith evaluation framework** (`evaluate` / `aevaluate`). You can benchmark
tool selection, skill routing, multi-agent delegation, and guardrails either
locally without uploading results or in the LangSmith Web UI. Local evaluation
still invokes the supplied agent and may incur model/tool costs. The scripted
`python -m examples.run_evals --offline` example needs no model API key.

### Running an evaluation

```python
from harnessx import Agent, AgentConfig, PermissionLevel
from examples._calculator import calculate  # Shared helper when running from a checkout.
from harnessx.evals import (
    build_example,
    default_evaluators,
    evaluate_agent,
    tool_selection_evaluator,
    contains_evaluator,
)

# 1. Define your agent factory or instance
def make_agent(inputs):
    agent = Agent(config=AgentConfig(model="claude-sonnet-4-6"))
    agent.tools.register_tool(calculate, permission=PermissionLevel.ALLOW, replay_policy="safe")
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

# 3. Evaluate against the live model, keeping results local.
summary = evaluate_agent(
    agent=make_agent,
    dataset=dataset,
    evaluators=[tool_selection_evaluator, contains_evaluator],
    experiment_prefix="math-agent-benchmark",
    offline=True,   # Disables LangSmith upload; the agent still calls its model.
)

print(f"Pass rate: {summary.pass_rate * 100:.1f}%")
```

From a running event loop, `await evaluate_agent_async(...)` takes the same
arguments and runs the evaluation in a worker thread; pass a factory rather than
an agent bound to your loop. Experiments are named `harnessx-eval-...` by default.

### Running evals from the CLI

```bash
# Scripted integration demonstration (langsmith extra, no API keys required)
python -m examples.run_evals --offline

# Run live model benchmarks without upload (model API key required)
python -m harnessx.evals.cli --suite tool_calling --offline

# Run against Anthropic and upload results + traces to LangSmith
export LANGSMITH_API_KEY="lsv2_pt_..."
python -m harnessx.evals.cli --suite skills --model claude-sonnet-4-6 --upload

# Run all benchmark suites with concurrency
python -m harnessx.evals.cli --suite all --concurrency 2
```

When uploaded, the CLI prints a clickable LangSmith URL to inspect row-level scores,
side-by-side prompt diffs, and the complete nested execution tree for every turn.

---

## API reference (quick)

| Class / function | Module | Purpose |
|---|---|---|
| `Agent` | `harnessx` | Core agentic loop (`await agent.run(msg)`) |
| `RunResult` / `RunStream` | `harnessx` | Completion result (`ok`, `failed`, `needs_input`, `raise_for_status()`) and event stream (`text()`) |
| `PendingTool` | `harnessx` | A tool the run stopped on: `execution_key`, `call`, `status` |
| `AgentConfig` | `harnessx` | Model, provider, prompt, plus the `limits`, `retry`, `prompt_cache`, and `tools` sub-policies |
| `Limits` / `RetryPolicy` / `ToolPolicy` | `harnessx` | Budgets, transient-failure handling, registry-wide tool options |
| `HarnessError` and subclasses | `harnessx` | `ConfigurationError`, `RuntimeStateError`, `UnknownExecutionKey`, `RunFailed`, `RunAwaitingInput`, ... |
| `register_provider` | `harnessx` | Add a provider name that `AgentConfig` accepts |
| `PromptCachePolicy` / `PromptCacheHint` | `harnessx` | Prompt-cache policy on the config; the per-request hint providers receive |
| `ToolRegistry` | `harnessx` | `register`, `register_with_schema`, `execute` |
| `SubAgent` | `harnessx` | Declare isolated specialists with `Agent(subagents=[...])` |
| `PermissionLevel` | `harnessx` | `ALLOW` / `ASK` / `DENY` |
| `PermissionManager` | `harnessx` | Per-tool overrides, session grants |
| `CliPermissionManager` | `harnessx` | Explicit terminal approval prompts |
| `Message` / `ContentBlock` | `harnessx` | Validated conversation representation |
| `GuardrailsEngine` | `harnessx` | Iteration/cost limits, usage stats |
| `ConversationMemory` | `harnessx` | Message list with auto-trimming |
| `AgentMemory` / `LongTermMemory` | `harnessx` | Disk-backed notes / facts |
| `PersistentMemory` | `harnessx` | Session save/load |
| `HookManager` / `HookEvent` | `harnessx` | Lifecycle hooks |
| `Middleware` / `MiddlewarePipeline` | `harnessx` | Request/result transforms |
| `SkillManager` | `harnessx` | Lazy skill loading |
| `MCPManager` | `harnessx` | MCP server connections |
| `Sandbox` | `harnessx` | Sandboxed code execution |
| `AgentRuntime` | `harnessx` | Durable sessions: `run`, `run_stream`, `stream_text`, `approve(..., resume=True)`, `decline`, `status` |
| `SQLiteBackend` / `PostgresBackend` / `TemporalBackend` | `harnessx` | Runtime storage; `await Backend.connect(...)`, `async with backend` |
| `harnessx.durable` | module | Runtime, backends, recorder, and their errors in one namespace |
| `IncidentRecorder` / `ExportPolicy` | `harnessx` | Offline incident playback, verification, and export disclosure |
| `Extension` / `LangSmithExtension` | `harnessx` | Pluggable runtime extensions / LangSmith tracing |
| `evaluate_agent` / `evaluate_agent_async` | `harnessx.evals` | LangSmith evaluation runner, sync and from a running loop |
| `AgentTarget` | `harnessx.evals` | Target adapter with telemetry & trace linking |
| `default_evaluators` | `harnessx.evals` | Standard suite of evaluators |
| `register_all_tools` | `harnessx.builtin` | Filesystem, bash, web, memory tools |

## Runnable examples

Every example is a module in the repository's `examples/` directory, run from a
checkout with `uv run python -m examples.<name>`. Prerequisites and test coverage
are listed in [examples/README.md](https://github.com/datagol/agent-harness-x/blob/main/examples/README.md).
Examples marked "no services" use scripted model responses and run without keys.

| Example | Shows | Needs |
|---|---|---|
| `examples.simple_chat` | Streaming interactive chat, a calculator tool, workspace reads | Anthropic key |
| `examples.provider_chat --provider P --model M` | The same agent across Anthropic, OpenAI, Gemini, OpenRouter, and Azure; streaming or ordinary runs; usage with cache counters | The provider's extra and key |
| `examples.prompt_caching` | A stable prompt prefix, the cache hint a provider receives, and cache write/read counters across a two-step loop | No services; `--live` uses Anthropic |
| `examples.coding_agent` | File and shell tools, an audit middleware, terminal approvals | Anthropic key |
| `examples.memory_agent` | Markdown notes and structured long-term facts as tools | Anthropic key |
| `examples.multi_agent` | Constructor-declared specialists with isolated tools and permissions | Anthropic key; URL research asks for approval |
| `examples.skills_agent` | Lazy skill loading in a live conversation, with a hook showing invocations | Anthropic key by default |
| `examples.skills_demo` | Skill discovery, invocation, and hook ordering | No services |
| `examples.runtime_approvals` | A persisted ASK approval and an explicit resume on SQLite | No services; terminal input |
| `examples.flight_recorder --output incident.hx` | A retried model call, middleware boundaries, and offline playback of the exported incident | No services; unused output path |
| `examples.sandboxed_coder` | Process resource limits with a durable runtime | Anthropic key; POSIX host |
| `examples.postgres_runtime` | Live streaming on the PostgreSQL runtime | `postgres` extra, `DATABASE_URL` or `--config`, Anthropic key |
| `examples.postgres_runtime check` / `crash` / `status` / `resume` | Storage check, a report, a hard process exit, and recovery of the same run without repeating the tool | `postgres` extra and a database; no model key |
| `examples.jev_routing --min-confidence 0.8` | A Choice decision selects one agent factory, with a general fallback | No services by default; `--live` needs the `jev` extra |
| `examples.jev_classification` | A Choice decision classifies a document into five categories | No services by default |
| `examples.jev_answer_review` | Score reviews coverage and Noul checks evidence after a completed run | No services by default |
| `examples.langsmith_tracing` | LangSmith lifecycle tracing with a nested specialist span | `langsmith` extra, Anthropic and LangSmith keys |
| `examples.run_evals --offline` | Three scripted evaluations; drop `--offline` for a live model and uploads | `langsmith` extra |
| `examples.mcp_agent --server NAME --command CMD` | MCP tools alongside native tools | `mcp` extra, an MCP server, Anthropic key |
| `python harness-web/run.py` | The web workspace that runs every example above and hosts a general chat | `server` extra, `npm --prefix harness-web run build`; keys per example |
