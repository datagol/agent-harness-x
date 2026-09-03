# Extension System Design

**Status:** Proposal
**Date:** 2026-09-03
**Context:** Comparison of `datagol_agent_harness` against HuggingFace's [`tau`](https://github.com/huggingface/tau) (a Python port of Pi's minimalist coding agent), and a phased design for evolving our extension system. Style decision (confirmed with stakeholders): **curated facade** — extensions receive a stable `ExtensionAPI` object rather than the raw `Agent`.

---

## 1. tau architecture overview

tau is a terminal coding agent organized as three packages with one-directional dependencies:

```
tau_coding  →  tau_agent  →  tau_ai
   (CLI/TUI, tools,    (portable brain:    (providers →
    skills, sessions)   loop, events,       provider-neutral
                        tools, harness)      event stream)
```

Core principles:

- **Events are the contract.** Providers, renderers, the TUI, and custom frontends all meet at a typed, provider-neutral event stream (`turn_start`, `message_update`, `tool_execution_*`, compaction/retry events, …). Frontends never see raw provider chunks.
- **The core stays portable.** `tau_agent` never imports CLI, Rich, Textual, or resource-loading code.
- **Tools are ordinary typed functions** — a JSON schema plus an async `execute_fn(tool_call_id, arguments, signal, on_update)` with built-in **cancellation** (`signal`) and **progress streaming** (`on_update`).

### tau's extension system

- **Discovery & distribution:** extensions are plain Python modules loaded from `~/.tau/extensions/` (default), `<project>/.tau/extensions/` (requires trust approval *and* `--project-extensions`), or explicit `-e PATH`. `tau install git:github.com/owner/repo[@ref]` clones into the user dir. A folder with `extension.py` loads as a package (relative imports work); a repo with a `src/` layout declares entries via `[tool.tau] extensions = [...]` in `pyproject.toml`.
- **Entry point:** every module defines a synchronous `setup(tau)` receiving a **curated API**:
  - Registration: `register_tool`, `register_provider`, `register_command`, `add_prompt_guideline`, `add_prompt_section`, `register_message_renderer`, `on(event)`.
  - Actions (post-bind): `send_user_message`, `send_custom_message`, `append_entry` (durable session entries), `notify`.
  - Read-only context: `cwd`, `model`, `provider_name`, `session_id`, `system_prompt`, `is_running`, `transcript`, UI dialogs (`select`/`confirm`/`input`), sidebar sections, Textual component slots.
- **Intercepting hooks** (handler return values matter):
  - `input` → `InputHookResult(action, text, message)` — transform the prompt text or consume it without an agent run.
  - `tool_call` → `ToolCallHookResult(block, reason, arguments)` — block execution (reason reported to the model) or rewrite arguments. A crashing handler **blocks the tool** (fail-safe).
  - `tool_result` → `ToolResultHookResult(content, details)` — rewrite the result before the model sees it.
- **Lifecycle / generations:** `/reload` awaits `session_shutdown(reason="reload")`, re-imports every extension, re-runs `setup`, fires `session_start`. Registrations are **source-owned**: failed setup rolls back everything that source registered; retiring a generation removes its contributions and cancels its background work.

---

## 2. Our architecture recap

`datagol_agent_harness` is a library-first harness ("the 9 pillars of LLM agent infrastructure"):

- **Agent loops:** `Agent` (`core.py`, non-streaming, provider-agnostic via `LLMProvider`) and `StreamingAgent` (`streaming.py`, streaming, Anthropic-only) — two parallel implementations with duplicated logic.
- **Providers:** `providers/base.py` (`LLMProvider.create`/`count_tokens`, canonical Anthropic-shaped `ProviderResponse`), `AnthropicProvider`, `OpenAIProvider` (non-streaming; full format translation), `make_provider(name)` factory.
- **Tools:** `ToolRegistry` with decorator registration and schema auto-generation from type hints; `register_with_schema` for explicit schemas; execution never raises (errors → `ToolResult(is_error=True)`); results stringified. No cancellation or progress callbacks.
- **Safety:** `PermissionLevel` (`ALLOW`/`ASK`/`DENY`), `PermissionManager` (interactive stdin prompts, session grants), `GuardrailsEngine` (iteration/cost limits), 3-tier `Sandbox` (process/docker/seatbelt).
- **Extension points today:**
  - `HookManager` + `HookEvent` — observe-only lifecycle events; exceptions swallowed.
  - `Middleware`/`MiddlewarePipeline` — transform messages, responses, tool calls, tool results.
  - `Extension` ABC (`extensions/base.py`) — `name` + `install(agent)` + `teardown()`; constructor-injected list; duplicate-name validation; reverse-order teardown. Extensions compose the primitives above (tools, hooks, middleware, system-prompt mutation). Reference implementation: `ResultSpillExtension` (spills oversized tool results to disk, registers a `read_result` tool).
- **Ecosystem:** `SkillManager` (lazy SKILL.md catalog), `MCPManager` (stdio/SSE/HTTP MCP servers → tool registry), 4-layer memory (conversation/persistent/long-term/agent + vector store), `AgentRuntime` (sessions, checkpoints), FastAPI apps (`apps/server`, `apps/grocery_buddy`, `apps/dca`).

---

## 3. Comparison

| Dimension | tau | datagol_agent_harness |
|---|---|---|
| Agent loop | One loop, streaming-first, typed event stream | Two parallel loops (`Agent` / `StreamingAgent`), duplicated logic |
| Providers | Provider-neutral **stream**; catalog.toml + dynamic registration | Non-streaming `create()`; canonical shape is Anthropic's; streaming bypasses the abstraction |
| Event model | Typed events are *the* public contract | `HookEvent` (observe-only); `StreamEvent` only on the Anthropic streaming path |
| Tools | Schema + `execute_fn(id, args, signal, on_update)` — cancellation + progress | Decorator w/ schema-from-type-hints (better DX); no cancellation, no progress; stringified results |
| Extensions | Filesystem-discovered modules, git install, trust model, curated `setup(tau)` API | `Extension` ABC, constructor-injected only, no discovery, raw-agent access |
| Interception | `input` / `tool_call` / `tool_result` with block & rewrite, fail-safe | Middleware transforms; no blocking contract; no user-input hook |
| Lifecycle | Generations, `/reload`, per-source rollback | Install at construction, reverse-order teardown; no reload, no rollback |
| Sessions | Append-only JSONL, resume, branching, compaction events | `PersistentMemory` snapshots + `AgentRuntime` checkpoints |
| Permissions/safety | Trust model; permission gate is an extension | Built-in `PermissionManager`, `GuardrailsEngine`, `Sandbox` — richer |
| Ecosystem | Skills, prompt templates, themes, slash commands | Skills (lazy catalog), **MCP**, 4-layer memory, vector store, FastAPI apps |

### Where we win

Permissions, guardrails, and sandboxing are first-class; MCP support; memory layers; runtime checkpointing; production FastAPI/SSE server apps; friendlier tool registration DX.

### Where tau wins (ranked by impact for us)

1. **Curated extension API** — extensions code against a stable facade, not raw internals.
2. **Intercepting hooks** — block/rewrite `tool_call`, rewrite `tool_result`, transform/consume user `input`.
3. **Extension discovery & distribution** — filesystem loading, project trust, package-shaped extensions.
4. **Unified event-driven loop** with provider-neutral streaming (ours is split and Anthropic-only).
5. **Tool progress & cancellation** (`on_update`, `signal`).

---

## 4. Design proposal (curated facade)

Four phases. Phases 1–3 are the extension-system work; Phase 4 is an optional loop refactor. Deliberate **non-goals**: tau's TUI-specific seams (Textual widgets, sidebar, slash commands, themes). Our frontends are FastAPI/SSE; the analog (custom SSE message types/renderers) is an app-level concern to revisit later.

### Phase 1 — `ExtensionAPI` facade

New module `datagol_agent_harness/extensions/api.py`:

```python
@dataclass(frozen=True)
class ExtensionContext:
    model: str
    provider_name: str
    session_id: str | None
    system_prompt: str
    cwd: str

class ExtensionAPI:
    """Stable surface handed to extensions. Backed by, but decoupled from, Agent."""

    # registration
    def register_tool(self, name, description, input_schema, handler,
                      permission=PermissionLevel.ASK) -> None: ...
    def add_prompt_section(self, title: str | None, body: str) -> None: ...
    def add_prompt_guideline(self, text: str) -> None: ...
    def on(self, event: HookEvent | str, handler) -> None: ...   # also usable as decorator

    # actions
    def notify(self, message: str, level: str = "info") -> None: ...

    # read-only context
    @property
    def context(self) -> ExtensionContext: ...
```

Changes to `Extension` (`extensions/base.py`):

```python
class Extension(ABC):
    name: str = ""

    def install(self, api: ExtensionAPI) -> None: ...   # new contract
    # Back-compat: if a subclass overrides `install_agent(self, agent)` (or the
    # legacy `install(agent)` signature is detected), the base class adapts it
    # through a shim that exposes the raw agent while emitting a DeprecationWarning.

    async def teardown(self) -> None: ...
```

Notes:

- `add_prompt_section` / `add_prompt_guideline` append structured, de-duplicated content to `agent.config.system_prompt` through a single choke point (a `SystemPromptBuilder` the agent owns), instead of extensions concatenating strings.
- `on(event, handler)` delegates to `agent.hooks` but accepts string event names to keep the facade decoupled from the enum.
- Port `ResultSpillExtension` to the new API as the reference implementation; it should no longer touch `agent.middleware` directly — the API gains `add_result_transform(fn)` (see Phase 2) or keeps an explicit `middleware` escape hatch documented as advanced usage.

### Phase 2 — Intercepting hooks

New typed results in `hooks.py` (or `extensions/hooks.py`):

```python
@dataclass
class InputHookResult:
    action: Literal["pass", "transform", "handled"] = "pass"
    text: str | None = None        # for "transform"
    message: str | None = None     # for "handled" (shown to user, no agent run)

@dataclass
class ToolCallHookResult:
    block: bool = False
    reason: str | None = None      # reported to the model as the tool result
    arguments: dict | None = None  # rewritten args if provided

@dataclass
class ToolResultHookResult:
    content: str | None = None     # rewritten result content
```

Wiring:

- `Agent.run(user_message)` — new `HookEvent.INPUT` (or a dedicated dispatch) runs before the message enters memory; `transform` rewrites, `handled` short-circuits.
- `Agent._handle_tool_calls` — `TOOL_CALL` interception before permission check: `block=True` → `ToolResult(reason, is_error=False)` fed to the model without executing; `arguments` replaces `tool_call.input`. **Fail-safe:** a crashing `tool_call` handler blocks the tool (matching tau); all other hook failures are logged and swallowed.
- Tool-result rewriting composes with (and is implemented on top of) the existing middleware pipeline — Phase 2 largely *formalizes* what middleware can already do, giving extensions a named, typed contract plus the blocking semantics middleware lacks.
- Mirror the same wiring in `StreamingAgent` (until Phase 4 unifies the loops).

### Phase 3 — Discovery & loading

New `ExtensionRuntime` in `extensions/runtime.py`:

```python
class ExtensionRuntime:
    def load(self, search_paths: list[Path] | None = None,
             *, trust_project: bool = False,
             extra_paths: tuple[Path, ...] = ()) -> None: ...
    @property
    def extensions(self) -> list[Extension]: ...   # ready for Agent(extensions=[...])
    async def unload(self) -> None: ...
```

- **Search paths, in precedence order:** built-in extensions shipped with the package → `~/.datagol/extensions/` → `<project>/.datagol/extensions/` (only with `trust_project=True`) → explicit `extra_paths`. First registration wins on name conflicts.
- **Module shapes:** a `*.py` file, or a directory containing `extension.py` (imported as a synthetic package so `from . import helper` works; `sys.path` is never touched). Names starting with `_` skipped.
- **Source-owned registrations:** the runtime tracks which tools/prompt-sections/hook-handlers each source registered via a recording `ExtensionAPI`; a failed `install()` rolls back everything that source registered, and `unload()` removes contributions in reverse load order.
- **Security posture** (documented, matching tau): extensions execute arbitrary Python with user permissions; project extensions are off by default and require explicit opt-in.

### Phase 4 — (optional) Loop unification

- Make streaming the single loop: providers emit provider-neutral chunk events; `Agent.run_stream()` becomes the primary API; `run()` consumes the stream.
- `OpenAIProvider` gains streaming via `chat.completions.create(stream=True)` translation.
- Tool executor signature gains optional `on_update` (progress) and `signal` (`asyncio.Event` cancellation); `ToolRegistry.execute` passes them through.
- Larger effort; only schedule if streaming-OpenAI or unified-loop maintenance pain justifies it.

---

## 5. Appendix

**tau references**

- Architecture: https://twotimespi.dev/internals/architecture/
- Agent loop & events: https://twotimespi.dev/internals/agent-loop/
- Extensions guide: https://twotimespi.dev/guides/extensions/
- Example extensions: https://github.com/huggingface/tau/tree/main/examples/extensions and https://github.com/rian-dolphin/tau-subagents

**Our cross-references**

- `datagol_agent_harness/extensions/base.py` — current `Extension` ABC, `install_extensions`, `close_extensions`
- `datagol_agent_harness/extensions/result_spill.py` — reference extension (port of pi-dca's result-spill)
- `datagol_agent_harness/core.py` — `Agent._agentic_loop`, `_handle_tool_calls` (Phase 2 wiring points)
- `datagol_agent_harness/streaming.py` — `StreamingAgent` (mirrored wiring; Phase 4 merge target)
- `datagol_agent_harness/hooks.py` — `HookEvent`, `HookManager`, `Middleware`, `MiddlewarePipeline`
- `apps/dca/server/pi/session_registry.py` — production consumer of `ResultSpillExtension`
