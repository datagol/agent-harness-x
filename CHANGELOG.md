# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-09-26

First release under the new name. Version 0.2.2 was bumped in the repository but
never published; this release supersedes it.

### Changed

- **Breaking:** the distribution is renamed from `datagol-agent-harness` to
  `harnessx`, and the import package from `datagol_agent_harness` to `harnessx`.
  There is no compatibility alias. See `doc/harnessx_rename.md`.
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

[0.3.0]: https://github.com/datagol/harness-x/compare/7fe798a...v0.3.0
[0.2.1]: https://pypi.org/project/datagol-agent-harness/0.2.1/
