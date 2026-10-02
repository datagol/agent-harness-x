# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.4.4] - 2026-10-02

Loop reliability. Read four other agent harnesses -- OpenCode, Pi, Hermes and
LangChain Deep Agents -- and fixed what the comparison exposed in ours. Nothing
here needs configuring; the defaults change behavior for the better and each one
can be switched off.

### Added

- `LoopGuard`, on by default, in `Limits(loop_guard=...)`. Watches for a
  repeating cycle of tool calls up to `max_period` long, `threshold` times
  running with the same arguments *and* the same results. A trip annotates the
  tool result the model reads and fires `HookEvent.REPETITION`; it never fails
  the run, which is what `Limits.max_iterations` is for.
  `LoopGuard(enabled=False)` opts out.
- A task list the model keeps and keeps seeing: `write_todos` and `read_todos`,
  registered by default, with `AgentConfig(planning=False)` to opt out. The list
  is rebuilt from the agent and merged into the tail of each request rather than
  written into the system prompt, so ticking a task off never invalidates the
  prompt cache, and condensing cannot summarize the list away. OpenCode persists
  todos and never re-injects them, which is how an agent ends up believing in a
  plan it can no longer read.
- `edit_file`, `delete`, `glob` and `grep` in the filesystem bundle, bringing it
  to parity with what Deep Agents, OpenCode and Hermes give a model. `edit_file`
  refuses an ambiguous match rather than editing the wrong occurrence, which is
  what makes it safe to prefer over rewriting a file whole. All four go through
  the existing no-symlink-follow containment layer.
- A `compact` phase. Condensing is now a real summary made by the agent's own
  model, as its own command, rather than character-level truncation; the lossy
  truncation remains as the fallback. Capped at `MAX_CONDENSATIONS` per turn, so
  a summary that fails to shrink the history cannot re-trigger forever --
  OpenCode has exactly that open as issue 15533.

### Fixed

- Tool calls are honored whatever the stop reason says. Providers return
  `end_turn` or `stop` while still carrying tool calls; ending the turn there
  dropped them silently and the run looked like the model had ignored its tools.
- A reply cut off at `max_tokens` never executes its tool calls, whose arguments
  may have been truncated mid-JSON and can parse while being incomplete.
- No path leaves a `tool_use` block unanswered in the transcript. Anthropic
  rejects the next request outright and the session is wedged; a fake provider
  accepts it, which is why this went unnoticed. Cancellation already did this
  correctly, so the block is now shared by the truncation guard and the generic
  failure handler as `close_open_tool_calls`.
- The loop guard watches every tool. It cleared its signature history on any
  successful call to a tool whose replay policy was not `safe`, and the default
  policy is `manual` -- so an agent stuck on `bash` or `write_file` was
  invisible to it.
- Condensing could never fire in an agentic conversation. The trim boundary
  demanded a user message with no `tool_result` preceded by an assistant message
  with no `tool_use`, which never occurs mid-tool-use; a six-round tool
  conversation offered 0 safe boundaries where it should offer 9.
- The summary call is accounted for and retried like any other model call. It
  never called `track_usage`, so its tokens were invisible to
  `Limits.max_cost_dollars`; and it was wrapped in its own try/except inside the
  command, so the driver's retry never fired once.
- The task reminder no longer produces two consecutive user turns on the wire.
- Model retries now carry jitter, so concurrent agents that hit one rate limit
  stop walking into the next one in lockstep. Permanently failing requests are
  no longer retried.
- Near-miss tool names are repaired rather than failed, and invalid arguments
  come back to the model as one actionable line instead of stalling the run.

## [0.4.3] - 2026-09-29

Retry seams. The engine has always had one retry loop; 0.4.3 keeps it there and
gives it the seams an application needs, extends it to tool calls that report a
transient failure, and stops the vendor SDKs from retrying underneath it.
Everything here is optional: unconfigured agents behave as they did.

### Added

- `ToolRetry(attempts, backoff_seconds, max_backoff_seconds, retry_error_results)`
  and `TransientToolError`. A handler raises the error to say the call failed
  for a reason that may clear and did not take effect; the engine runs it again
  under the tool's policy. Set it per tool (`register(..., retry=...)` on every
  registration route) or registry-wide with `ToolPolicy(retry=...)`. Retry
  applies only to `safe` and `idempotent` tools: a `manual` tool still stops for
  recovery on an unknown outcome, and a declared transient failure there goes to
  the model instead of being repeated.
- `ToolDefinition.retry_if_result`, and `ToolRetry.retry_error_results`, for an
  API that reports throttling inside an otherwise successful result.
- `LLMProvider.is_transient(exc)` and `LLMProvider.retry_after(exc)`: the engine
  now asks the provider how to classify a failure and how long the server asked
  it to wait. `Retry-After` (seconds or HTTP-date) is honored.
- `RetryPolicy.max_backoff_seconds` (default 30) caps one wait, including one
  the server asked for, and `RetryPolicy.wait_for()` exposes the calculation.
- `AgentConfig.fallbacks`, a tuple of `Fallback(provider, model=None,
  max_tokens=None)`: providers to try, in order, when the primary fails
  transiently. Each names its own model id, since the same model is spelled
  differently on different vendors. The Agent builds the chain and owns it, so
  there is nothing extra to close, and because it is configuration it is
  serializable: durable runs and Temporal workers carry it like any other
  config. `RetryPolicy.switch_after` and `.cooldown_seconds` govern how long a
  provider is given before the chain moves on and how long it then sits out.
  Deterministic errors never fail over, and a stream fails over only before its
  first chunk. `FallbackProvider` remains for the case where a member has to be
  a live object, such as a provider holding a pre-configured SDK client.
- `HookEvent.RETRY` with a `RetryData` payload (`kind`, `name`, `attempt`,
  `next_attempt`, `wait_seconds`, `error`, `provider`), and `provider` on the
  `LLM_RESPONSE` payload.
- `MCPServerConfig.replay_policy` and `.retry`. Bridged tools on a `safe` or
  `idempotent` server retry a throttled call three times by default, including
  when the server reports a 429 or 5xx inside a successful result.

### Fixed

- Connection loss is retried again. The SDKs raise `APIConnectionError` for a
  dropped connection, which they retry themselves and which the engine's
  classifier did not recognize, so turning the SDK's retries off left it with no
  retry at all. A proxy refusing the tunnel with a 4xx stays final: that is a
  policy decision, not a blip, and retrying it only spends the budget to be
  refused again.
- LangSmith labels a run with the provider that actually served it. After a
  failover it reported the configured primary, misattributing exactly the runs
  an operator is most likely to be looking at.

### Changed

- Provider clients the harness constructs pass `max_retries=0` to the vendor
  SDK (Anthropic, OpenAI, Azure, OpenRouter). Retries were previously doubled:
  the SDK's two on top of the engine's, invisible to the journal and to
  tracing. An SDK client you construct and inject keeps its own setting.
- A tool attempt is bounded by its `ToolRetry` instead of a hard-coded three,
  waits with capped exponential backoff instead of a fixed ramp, and is carried
  on the run state so the Temporal path uses the same bound. Run states written
  before this release default to the standard policy.
- `engine.retryable()` takes the agent and asks its provider; `PendingTool`
  gained a `retry` field.

### Removed

- `providers.retry.call_with_retry` and `stream_with_retry`, helpers nothing
  called. The engine's loop is the retry path.

## [0.4.2] - 2026-09-29

A patch release for one silent regression in 0.4.1.

### Fixed

- Gemini tool schemas no longer carry JSON Schema keywords Gemini does not
  define (`additionalProperties`, `$schema`), at any depth. `generateContent`
  tolerates them; `cachedContents` rejects the request outright, so a single
  such tool costs the caller explicit prompt caching with nothing but a
  warning line to show for it. The tool in question is this package's own
  `read_tool_result`, whose schema is generated from its signature and which
  0.4.1 registers on **every** agent — so every Gemini agent on 0.4.1 lost
  explicit caching. Found downstream on a workload where 99.96% of the prompt
  prefix had been cached, the moment it upgraded.

## [0.4.1] - 2026-09-28

Long replies. An agent that writes whole files hits the reply token budget,
and 0.4.0 handled that badly: the default budget was small, a truncated turn
reported `completed`, a bigger budget tripped the Anthropic SDK's guard on
non-streaming calls, and the per-call timeout was sized for short replies.

### Changed

- `AgentConfig.max_tokens` defaults to `None`, meaning the provider chooses a
  budget for the model: 20,000 tokens for Claude 4 models, 8,192 elsewhere
  (`LLMProvider.default_max_tokens(model)`, overridable). An explicit value
  still applies as before. The resolved budget is what `LLM_REQUEST` hooks and
  LangSmith spans record.
- `RetryPolicy.call_timeout_seconds` defaults to `None`, meaning the timeout
  for one model call is sized for the reply budget (`call_timeout_for`), never
  below the previous 300 seconds. An explicit value still applies as before.
- `RunResult.ok` is false when the model stopped at the token budget.

### Added

- `RunResult.truncated` and `RunTruncated`, raised by `raise_for_status()` and
  `stream_text()` when a reply or a tool call was cut off at `max_tokens`.
- The Anthropic provider streams a non-streaming request itself when the SDK
  refuses it as too long, instead of surfacing "Streaming is required".
- `read_tool_result` caps each read at `max_chars` (default 6,000) and pages a
  single long line with `char_offset`, so a minified JSON result cannot flood
  the context in one call.

## [0.4.0] - 2026-09-27

An API refinement release. The engine is unchanged; the public surface gets one
vocabulary for running an agent, grouped configuration, a common error base,
typed extension points, and fewer duplicate routes. `MIGRATING.md` lists every
change with before/after code. Deprecated names warn and are removed in 0.5.

### Breaking

- `Agent(client=...)` is removed; wrap the SDK client in a provider
  (`Agent(provider=AnthropicProvider(client=...))`). Passing `client=` raises a
  `TypeError` that says so.
- `harnessx.streaming` and `StreamingAgent` are removed (use `Agent.run_stream()`
  or `Agent.stream_text()`); `Role`, `StopReason.TOOL_CALLS`,
  `RuntimeConfig.checkpoint_interval`/`max_checkpoints`, `LLMProvider.format_tools`,
  and the `make_provider("azure-openai")` alias are removed.
- `AgentConfig` accepts only `model`, `provider`, and `max_tokens` positionally.
- `RunResult.pending` holds `PendingTool` dataclasses instead of dicts. Indexing
  still works with a `DeprecationWarning`; the persisted shape is unchanged.
- `Agent(mcp=manager)` now bridges the manager's tools into the registry at
  construction. A following `manager.register_tools(agent.tools)` is a no-op for
  the same manager and a `ValueError` if another tool holds the name.
- `hooks.before_tool/after_tool/on_error` return the `Registration` handle
  instead of the callback.
- Retry classification is unified: one engine-level loop bounded by
  `RetryPolicy.attempts` using `providers.retry.is_transient`. The tool loop no
  longer retries arbitrary `OSError` subclasses; vendor "overloaded" errors are
  retried for tools too.
- `prompt_cache.accepts_cache()` is true only for an explicit `cache` parameter.
- `LLMProvider.count_tokens(system: str | None)`; custom providers must accept `None`.
- The permission-denied tool result is the single string `"Permission denied"`.

### Deprecated (removed in 0.5)

- `AgentConfig` flat fields `max_iterations`, `max_context_tokens`,
  `max_result_chars`, `max_cost_dollars`, `input_cost_per_m`, `output_cost_per_m`
  (now `limits=Limits(...)`) and `llm_max_attempts`, `llm_retry_backoff_seconds`,
  `model_timeout_seconds` (now `retry=RetryPolicy(attempts, backoff_seconds,
  call_timeout_seconds)`). They still work as keyword arguments and attributes.
- `AgentRuntime.execute` → `run`, `execute_stream` → `run_stream`,
  `get_status` → `status`; `RunHandle.backend`.
- `PendingTool.__getitem__` / `.get`.

### Added

- `Limits`, `RetryPolicy`, `ToolPolicy` sub-policies; `AgentConfig.from_dict()`
  reads 0.3 flat and 0.4 nested snapshots; `AgentConfig(tools=ToolPolicy(...))`
  reaches list-built registries (`ToolRegistry.policy`, `adopt_policy()`).
- `harnessx.errors`: `HarnessError`, `ConfigurationError`, `RuntimeStateError`,
  `ResolutionError`, `UnknownExecutionKey`, `RunError`, `RunFailed`,
  `RunAwaitingInput`, `RunCancelled`; existing errors rebased on `HarnessError`
  while keeping their old bases.
- `RunResult.ok`, `failed`, `needs_input`, `raise_for_status()`;
  `RunStream.text(on_reset=)`; `Agent.stream_text()`, `Agent.closed`, `Agent.busy`.
- `AgentRuntime.run`, `run_stream`, `stream_text`, `status()`, `decline()`,
  `approve(target, allow=True, resume=False)` returning the finished `RunResult`
  when resumed, `aclose()`; `approve`/`decline`/`resolve_tool` accept a `PendingTool`.
- `PendingTool` dataclass with `to_dict()`/`from_dict()`.
- Typed hook payloads (`harnessx.hooks.*Data`, `HOOK_PAYLOADS`); `SANDBOX_EXEC`
  emitted by `Sandbox(config, hooks=)`; `CHECKPOINT` emitted by the runtime.
- `Agent(sandbox=)` binds the sandbox as the bash built-in's engine
  (`ToolRegistry(sandbox=)`); `agent.mcp_tools`.
- `MCPManager` as an async context manager; duplicate `connect` raises.
- `MCPServerConfig.stdio(...)` and `MCPServerConfig.http(...)` typed server
  definitions, accepted by `MCPManager.connect()` alongside the keyword form;
  the config validates that exactly one transport is given, and
  `list_servers()` reports the negotiated transport.
- `register_provider()`, `registered_providers()`, `unregister_provider()`;
  `AgentConfig.provider` accepts registered names; the injected-provider check
  only rejects two different built-in names.
- `SQLiteBackend.connect()`, `PostgresBackend.connect()`; every backend is an
  async context manager; `TemporalBackend.aclose()` closes a client it created.
- `PermissionLevel` is a `str` enum; `ReplayPolicy` lives in `harnessx.types`
  and is accepted wherever `replay_policy` is; `DEFAULT_TIMEOUT_SECONDS`.
- `evaluate_agent_async()`; eval defaults renamed to `harnessx-eval`,
  `harnessx-` dataset prefix, `harnessx-agents` project.
- `harnessx.durable` groups the runtime, backends, recorder, and their errors.
- `LLM_REQUEST` hooks carry the request as sent: the rendered `system` prompt,
  `model`, `max_tokens`, `temperature`, and `stream`, beside the counts and
  `prefix_key`. LangSmith model spans record all of it, plus `ls_model_name`
  and `ls_provider` so LangSmith can price the run, and usage with cache-read,
  cache-creation, and reasoning token details.
- `examples/prompt_caching.py`, a self-contained example showing the stable
  prefix key, the cache hint a provider receives, and cache counters.
- `GuardrailsEngine.usage_summary` now includes `cache_creation_input_tokens`,
  `cache_read_input_tokens`, and `thinking_tokens`.
- `examples/provider_chat.py` accepts `--provider azure`.

### Changed

- `TokenUsage.input_tokens` counts every prompt token on every provider. The
  Anthropic provider reported it net of cache reads and writes, so a cached
  prompt showed up as a handful of input tokens in `usage_summary` and traces.
- `RunEvent(data=...)` is required.
- `register_with_schema` takes its metadata (`permission`, `concurrent`,
  `replay_policy`, `timeout_seconds`, `replace`) as keyword arguments.
- Installation docs use `uv add harnessx` from PyPI; repository work uses
  `uv sync --all-extras` and `uv run`.
- The web example no longer installs wildcard CORS, and the coding agent
  example bounds filesystem tools to the working directory.

### Fixed

- Oversized tool results are read back through a built-in `read_tool_result`
  tool, registered on every agent, instead of a `read_file` call on a temp
  path that a sandboxed registry could not open. The eviction notice names
  the tool-use id to pass, and the reader survives session save and load.
- Tool errors reach the model as one line (`Tool execution error:
  PermissionError: ...`); the traceback goes to the log instead of the context.
- The tool dedupe cache is cleared at the start of every run; it used to leak
  results across turns.
- An unknown execution key passed to `approve`/`resolve_tool` raises
  `UnknownExecutionKey` instead of leaking `StopIteration`.
- Tools registered as `ToolDefinition`s or through `ExtensionContext.register_tool`
  now inherit the registry's default timeout.
- MCP tool-to-server lookup uses one naming helper for prefixed and unprefixed
  registration.

## [0.3.0] - 2026-09-26

First release under the new name. Version 0.2.2 was bumped in the repository but
never published; this release supersedes it.

### Changed

- **Breaking:** the distribution is renamed from `datagol-agent-harness` to
  `harnessx`, and the import package from `datagol_agent_harness` to `harnessx`.
  There is no compatibility alias; see the Setup section of the README.
- `jsonschema` is now a core dependency (tool input validation).
- The `all` extra now composes the other extras instead of duplicating their pins.
- Releases are published to PyPI from `v*` tags via Trusted Publishing.

### Added

- Azure OpenAI provider (`provider="azure"`, new `azure` extra) with deployment
  routing and token-based auth passthrough.
- Retry for transient provider failures (429, 5xx, timeouts), on by default via
  `AgentConfig.llm_max_attempts` and `llm_retry_backoff_seconds`. Streams are
  only retried before their first chunk.
- `TokenUsage.thinking_tokens` carries reasoning tokens reported by Gemini and
  OpenAI.
- Per-tool timeouts with a `ToolRegistry(default_timeout_seconds=...)` fallback,
  and opt-in `ToolRegistry(dedupe_calls=True)` so an identical repeated call
  returns the first result.
- Harness-level prompt caching, on by default: `AgentConfig.prompt_cache`
  (`PromptCachePolicy`) turns into a per-request `PromptCacheHint` with a stable
  prefix key and breakpoints. Anthropic gets `cache_control` markers, OpenAI a
  `prompt_cache_key`, Gemini an explicit cache when a TTL is set; all fail open.
  `LLM_REQUEST` hooks carry `prefix_key`. `prompt_cache=None` disables it.
- Opt-in explicit prompt caching for the Gemini provider (now driven by the
  policy TTL, with the constructor and `GEMINI_PROMPT_CACHE_TTL` as fallbacks).
- Durable runtime (`AgentRuntime`, `RunHandle`) with SQLite, PostgreSQL, and
  Temporal backends, S3 artifact storage, and run recovery.
- Flight recorder: incident export, offline playback, and integrity verification.
- Typed decisions with the optional Jev SDK (`harnessx.decisions`).
- Sub-agent delegation, agent registry, and tool registry normalization.
- New extras: `jev`, `postgres`, `temporal`.
- `harnessx-evals` console script for the evaluation CLI (requires the
  `langsmith` extra).
- `harnessx.__version__`, read from the installed distribution metadata.
- `py.typed` marker and typing fixtures under `tests/typing/`.

## [0.2.1] - 2026-09-08

Last release as `datagol-agent-harness`. Gemini and OpenRouter providers, MIT
license, flexible tool registration.

[0.4.3]: https://github.com/datagol/agent-harness-x/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/datagol/agent-harness-x/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/datagol/agent-harness-x/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/datagol/agent-harness-x/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/datagol/agent-harness-x/compare/c382675...v0.3.0
[0.2.1]: https://pypi.org/project/datagol-agent-harness/0.2.1/
