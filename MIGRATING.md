# Migrating to harnessx 0.4

0.4 tidies the public API without touching the engine: one vocabulary for
running an agent, grouped configuration, a common error base, typed extension
points, and fewer ways to do the same thing. Most 0.3 code keeps working and
emits a `DeprecationWarning` naming the replacement. Everything marked
**deprecated** below is removed in 0.5; everything marked **removed** is gone in
0.4 and listed with its replacement.

Run your suite once with warnings promoted to errors to find every deprecated
use:

```bash
python -W error::DeprecationWarning -m pytest
```

Persisted state is forward compatible: sessions, run snapshots, and incident
bundles written by 0.3 load in 0.4. State written by 0.4 does not load in 0.3.

## Configuration

`AgentConfig` groups its fifteen flat fields into four sub-policies. `model`,
`provider`, `max_tokens`, `system_prompt`, and `temperature` stay where they were.

| 0.3 | 0.4 |
|---|---|
| `max_iterations`, `max_context_tokens`, `max_result_chars`, `max_cost_dollars`, `input_cost_per_m`, `output_cost_per_m` | `limits=Limits(...)` |
| `llm_max_attempts`, `llm_retry_backoff_seconds`, `model_timeout_seconds` | `retry=RetryPolicy(attempts, backoff_seconds, call_timeout_seconds)` |
| `prompt_cache` | unchanged: `prompt_cache=PromptCachePolicy()` or `None` |
| `ToolRegistry(default_timeout_seconds=, dedupe_calls=)` only | also `tools=ToolPolicy(default_timeout_seconds, dedupe_calls)` on the config |

```python
# 0.3
AgentConfig(max_iterations=20, max_cost_dollars=1.0, llm_max_attempts=3, model_timeout_seconds=60)

# 0.4
AgentConfig(
    limits=Limits(max_iterations=20, max_cost_dollars=1.0),
    retry=RetryPolicy(attempts=3, call_timeout_seconds=60),
)
```

The flat names are **deprecated**, not removed: they still work as keyword
arguments and as attributes (`config.max_iterations` reads and writes
`config.limits.max_iterations`) and warn on every use. The sub-policies are
frozen dataclasses; change one with `dataclasses.replace`.

`AgentConfig` construction beyond the third positional argument is keyword-only.
Only `model`, `provider`, and `max_tokens` may be positional. **Breaking** if you
passed `system_prompt` positionally.

`AgentConfig.from_dict()` reads both the 0.3 flat shape and the 0.4 nested shape
without warnings. `Agent.load_session`, `AgentRuntime.resume`, and the Temporal
worker use it, so saved sessions load unchanged.

`ToolPolicy` reaches registries that had no way to receive registry-wide options
before: `Agent(config=AgentConfig(tools=ToolPolicy(default_timeout_seconds=60)), tools=[fn])`
applies to the list. Precedence, highest first: a value given to the
`ToolRegistry(...)` constructor, the agent's `config.tools`, the built-in
default. Tools registered with their own `timeout_seconds` are never changed.
`registry.policy` shows the effective values.

## Running an agent

| 0.3 | 0.4 | Status |
|---|---|---|
| `agent.run(msg)` | same | |
| `agent.run_stream(msg)` | same, yields `RunEvent`s | |
| iterate `run_stream` and filter `TEXT_DELTA` | `agent.stream_text(msg, on_reset=...)` yields `str` | new |
| `runtime.execute(msg)` | `runtime.run(msg)` | deprecated alias |
| `runtime.execute_stream(msg)` | `runtime.run_stream(msg)` | deprecated alias |
| | `runtime.stream_text(msg)` | new |
| `runtime.get_status()` | `runtime.status()` | deprecated alias |
| `runtime.stop()` | same, plus `runtime.aclose()` | |
| `agent._closed`, `agent._busy` | `agent.closed`, `agent.busy` | new properties |

`stream_text()` raises `RunFailed` (or another `RunError`) if the run does not
complete, so a consumer that only wants text never has to inspect the result.
The `on_reset` callback fires on `ATTEMPT_RESET`, when a retried model call
restarts the answer; without it the answer simply starts over in the stream.
Breaking out of `stream_text()` early leaves the run to be cancelled when the
generator closes; wrap it in `contextlib.aclosing()` to make that immediate.

`RunStream.text(on_reset=...)` is the same text view over any event stream.

## Results

`RunResult` gains three predicates and a raiser:

```python
result = await agent.run("...")
if result.ok: ...            # status is COMPLETED
if result.failed: ...        # status is FAILED; result.error has type and message
if result.needs_input: ...   # status is AWAITING_INPUT; result.pending lists the tools

result.raise_for_status()    # returns result, or raises RunFailed / RunAwaitingInput / RunCancelled
```

`result.pending` holds `PendingTool` objects instead of dicts. Read attributes:
`pending.execution_key`, `pending.call.name`, `pending.call.input`,
`pending.status` (`"approval"` or `"uncertain"`), `pending.attempt`,
`pending.policy`, `pending.concurrent`, `pending.timeout_seconds`. Indexing
(`pending["execution_key"]`) still works and is **deprecated**. The persisted
shape is unchanged (`to_dict()` / `from_dict()`).

## Durable approvals

Approving no longer takes four calls.

```python
# 0.3
result = await runtime.execute("write the report")
if result.status == "awaiting_input":
    await runtime.approve(result.pending[0]["execution_key"])
    handle = await runtime.resume(runtime.session_id)
    result = await handle.result()

# 0.4
result = await runtime.run("write the report")
if result.needs_input:
    result = await runtime.approve(result.pending[0], resume=True)
```

`approve(target, *, allow=True, resume=False)` and the new
`decline(target, *, resume=False)` accept a `PendingTool` or an execution key.
With `resume=True` they return the finished `RunResult`. A run with several
pending tools takes one decision per tool and a single `resume()` at the end,
exactly as before. `resolve_tool()` also accepts a `PendingTool`.

`RunHandle.backend` is **deprecated**; runtimes own their backend.

## Errors

Every exception the SDK raises now derives from `harnessx.HarnessError`:

| Error | Also a | Raised for |
|---|---|---|
| `ConfigurationError` | `ValueError` | invalid `AgentConfig`, `Limits`, `RuntimeConfig`, `SandboxConfig`, mismatched sessions |
| `RuntimeStateError` | `RuntimeError` | closed or busy agent, runtime not started, run not awaiting a resolution |
| `ResolutionError` | `ValueError` | an approval or recovery decision that does not fit the tool's state |
| `UnknownExecutionKey` | `KeyError` | `approve`/`resolve_tool` with a key the run does not have |
| `RunError`, `RunFailed`, `RunAwaitingInput`, `RunCancelled` | | `raise_for_status()` and `stream_text()`; carry `.result` |
| `StorageError` and subclasses, `IncidentError`, `RecordingError`, `MaxIterationsError`, `CostLimitError`, `ToolNotFoundError`, `ToolApprovalRequired` | their 0.3 bases | unchanged sites |

Code that caught `ValueError` or `RuntimeError` keeps working; the old bases are
retained. Code that inspected `StopIteration` from an unknown execution key (a
0.3 bug) must catch `UnknownExecutionKey`.

The permission-denied tool result is now the single string `"Permission denied"`.

## Enums and constants

- `PermissionLevel` is a `str` enum like the others: `PermissionLevel.ASK == "ask"`
  and it serialises with `json.dumps` directly.
- `ReplayPolicy` moved to `harnessx.types` (still importable from
  `harnessx.execution`) and is accepted everywhere `replay_policy` is, alongside
  the plain strings. `ToolDefinition.replay_policy` stores the string.
- `DEFAULT_TIMEOUT_SECONDS = 300.0` replaces the literal `300` defaults on tools,
  sub-agents, and the model call timeout.

## Streaming

`harnessx.streaming` and `StreamingAgent` are **removed**; they had been a shim
over `Agent.run_stream()` since 0.3. `RunEvent(data=...)` is required (the
`None` default was rejected by its own validator).

## Hooks, middleware, sandbox

- `HookContext.data` stays a `dict`, but the keys every built-in event carries are
  now declared in `harnessx.hooks` as TypedDicts (`ToolCallStartData`,
  `LLMRequestData`, ...). `HOOK_PAYLOADS` maps each `HookEvent` to its shape;
  cast `ctx.data` to it for typed access.
- `hooks.before_tool(...)`, `after_tool(...)`, and `on_error(...)` return the
  `Registration` handle that removes them, like `hooks.on(...)`. They previously
  returned the callback, so `@hooks.before_tool` still works as a decorator but
  the decorated name is now the registration.
- `HookEvent.SANDBOX_EXEC` is emitted. `Sandbox(config, hooks=...)` reports every
  `execute` and `execute_command` with `kind`, `tier`, `exit_code`, `timed_out`,
  and `execution_time_ms`; `Agent(sandbox=...)` supplies its own hooks when the
  sandbox has none.
- `HookEvent.CHECKPOINT` is emitted by `AgentRuntime` at every persisted phase
  boundary with `session_id`, `run_id`, `status`, and `phase`.

## Agent construction: owned resources

- `Agent(client=...)` is **removed**. Wrap the SDK client in a provider:
  `Agent(provider=AnthropicProvider(client=client))`. Passing `client=` raises a
  `TypeError` that says so.
- `Agent(mcp=manager)` now does something: the manager's discovered tools are
  bridged into `agent.tools` at construction, and their names are listed on
  `agent.mcp_tools`. Remove any explicit `manager.register_tools(agent.tools)`
  call that follows; it is a no-op for the same manager, and an error if another
  tool already holds the name. `Agent.aclose()` does not disconnect the manager;
  it is caller-owned, like `provider=` and `memory=`.
- `MCPManager` is an async context manager: `async with MCPManager() as mcp:`
  disconnects every server on exit. Connecting a name twice raises `ValueError`.
- `Agent(sandbox=sandbox)` binds the sandbox as the bash built-in's engine:
  `Agent(tools=["bash"], sandbox=sandbox)` runs shell commands inside it. The
  registry exposes it as `agent.tools.sandbox`.

## Providers

- `register_provider(name, factory, *, replace=False)` adds a provider name that
  `AgentConfig(provider=name)` accepts and `make_provider(name)` builds.
  `registered_providers()` and `unregister_provider()` complete the set.
- The injected-provider check is relaxed. Only two *different built-in* names
  contradict each other (`Agent(provider=OpenAIProvider(), config=AgentConfig(provider="anthropic"))`).
  A custom provider with any `name` is accepted with any config label.
- `make_provider("azure-openai")` is **removed**; use `"azure"`, the only name
  `AgentConfig` ever accepted.
- `LLMProvider.count_tokens(system: str | None)` matches `create()`. Custom
  providers should accept `None`; the Anthropic provider omits the parameter.
- `LLMProvider.format_tools()` is **removed**; nothing called it.
- `prompt_cache.accepts_cache()` is true only for a provider whose `create()`
  declares a `cache` parameter. A `**kwargs` signature no longer counts.
- Transient-failure classification is unified in `harnessx.providers.retry.is_transient`
  and applied by one engine-level retry loop bounded by `RetryPolicy.attempts`.
  **Breaking:** the tool loop no longer retries arbitrary `OSError` subclasses,
  and vendor "overloaded" errors are now retried for tools as well.

## Backends

- `await SQLiteBackend.connect(path)` and `await PostgresBackend.connect(dsn, **kwargs)`
  construct and initialise in one call, matching `TemporalBackend.connect()`.
- Every backend is an async context manager: `async with backend:` initialises on
  entry and closes on exit.
- `TemporalBackend.aclose()` closes the Temporal client when `connect()` created
  it, and leaves an injected client open.
- `RuntimeConfig.checkpoint_interval` and `max_checkpoints` are **removed**; the
  runtime checkpoints at every phase boundary and had ignored them since 0.3.

## Evals

- Defaults renamed: `experiment_prefix="harnessx-eval"`, synced datasets are
  prefixed `harnessx-`, and `LangSmithExtension` defaults to the `harnessx-agents`
  project.
- `evaluate_agent_async(...)` awaits an evaluation from a running event loop. It
  runs the synchronous runner in a worker thread, so pass a factory rather than
  an `Agent` bound to your loop; it cannot be cancelled once started and must not
  be nested inside another evaluation.

## Removed names

| Removed | Use instead |
|---|---|
| `Agent(client=...)` | `Agent(provider=AnthropicProvider(client=...))` |
| `harnessx.streaming`, `StreamingAgent` | `Agent.run_stream()`, `Agent.stream_text()` |
| `Role` | the `"user"` / `"assistant"` strings |
| `StopReason.TOOL_CALLS` | `StopReason.TOOL_USE` |
| `RuntimeConfig.checkpoint_interval`, `max_checkpoints` | nothing; every phase is checkpointed |
| `LLMProvider.format_tools` | nothing |
| `make_provider("azure-openai")` | `make_provider("azure")` |

## New exports

`Limits`, `RetryPolicy`, `ToolPolicy`, `DEFAULT_TIMEOUT_SECONDS`, `HarnessError`,
`ConfigurationError`, `RuntimeStateError`, `ResolutionError`,
`UnknownExecutionKey`, `RunError`, `RunFailed`, `RunAwaitingInput`,
`RunCancelled`, `ToolApprovalRequired`, `ToolExecutionContext`, `Registration`,
`register_provider`, `evaluate_agent_async`, and `harnessx.durable`, a module
that groups the runtime, backends, recorder, and their errors under one import.
