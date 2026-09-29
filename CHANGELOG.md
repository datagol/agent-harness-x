# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[0.4.1]: https://github.com/datagol/agent-harness-x/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/datagol/agent-harness-x/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/datagol/agent-harness-x/compare/c382675...v0.3.0
[0.2.1]: https://pypi.org/project/datagol-agent-harness/0.2.1/
