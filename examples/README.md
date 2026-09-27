# HarnessX examples

Run these examples as modules from a checkout of the repository. The `examples`
package loads the repository's `.env` without replacing variables already set in
your environment. Examples are not included in the installed SDK wheel.

To launch every example from a browser, use [harness-web](../harness-web/README.md).
It includes live output, input and approval controls, generated-file downloads,
and a separate general chat workspace.

```bash
uv sync --all-extras
uv run python -m examples.skills_demo
```

`uv sync` alone installs the SDK without integrations; add `--extra NAME` for only the extras
listed below. Most interactive examples use the SDK's default Anthropic provider
and require `ANTHROPIC_API_KEY`; they make billable model calls.

## Start without external services

```bash
python -m examples.skills_demo
python -m examples.prompt_caching
python -m examples.runtime_approvals
python -m examples.flight_recorder --output /tmp/invoice-incident.hx
python -m examples.run_evals --offline  # Requires the langsmith extra.
python -m examples.jev_routing --min-confidence 0.8
python -m examples.jev_classification
python -m examples.jev_answer_review
```

These examples use scripted `LLMProvider` responses while exercising the real SDK
engine, tools, hooks, approvals, recording, or evaluations. Their outcomes verify
integration behavior, not model quality. The approval example prompts before
writing a temporary note. The recorder's destination must not already exist;
`.hx` is a ZIP archive of JSON records and optional captured artifacts.

## All entry points

| Command | Demonstrates | Requirements |
|---|---|---|
| `python -m examples.simple_chat` | Streaming chat, bounded arithmetic, workspace reads | Anthropic key |
| `python -m examples.provider_chat --provider PROVIDER --model MODEL_ID` | Same calculator agent across five providers, streaming or ordinary runs | Selected provider's SDK extra and API key |
| `python -m examples.coding_agent` | Coding tools, middleware, terminal permissions | Anthropic key; writes and shell calls ask for approval |
| `python -m examples.memory_agent` | Persistent markdown notes and structured facts | Anthropic key; stores data in `.agent_memory/` |
| `python -m examples.multi_agent` | Constructor-defined specialists with isolated conversations | Anthropic key; URL research requests ask for approval |
| `python -m examples.skills_agent` | Interactive lazy skill loading | Anthropic key by default |
| `python -m examples.skills_demo` | Skill invocation and hook ordering | No external services |
| `python -m examples.prompt_caching` | Stable prompt prefix, the cache hint a provider receives, and cache read/write counters | No external services by default; `--live` uses the Anthropic key |
| `python -m examples.jev_routing --min-confidence 0.8` | Choice selects one fixed agent factory; general fallback | No external services by default |
| `python -m examples.jev_classification` | Choice classifies W2, Deposit, Insurance, Payroll, or unknown | No external services by default |
| `python -m examples.jev_answer_review` | Score checks coverage; Noul checks evidence support after a completed run | No external services by default |
| `python -m examples.sandboxed_coder` | Process resource limits with a durable runtime | Anthropic key; POSIX host |
| `python -m examples.runtime_approvals` | Persisted `ASK` approval and explicit resume | No external services; terminal input |
| `python -m examples.flight_recorder --output incident.hx` | Retry history, middleware boundaries, offline playback | No external services; unused output path |
| `python -m examples.postgres_runtime` | PostgreSQL runtime streaming and cleanup | `postgres` extra, `DATABASE_URL` or `--config PATH`, Anthropic key |
| `python -m examples.postgres_runtime check` / `crash` / `status` / `resume` | Check storage, write a report, crash, and finish the same run without repeating the tool | `postgres` extra and `DATABASE_URL` or `--config PATH`; no model key |
| `python -m examples.langsmith_tracing` | Explicit tracing and specialist spans | `langsmith` extra, Anthropic and LangSmith keys |
| `python -m examples.run_evals --offline` | Three scripted math/selection evaluations | `langsmith` extra; no keys or uploads |
| `python -m examples.run_evals` | The same cases against a live model | `langsmith` extra, Anthropic key; upload enabled when a LangSmith key exists |
| `python -m examples.mcp_agent --server NAME --command COMMAND` | Local MCP tools alongside native tools | `mcp` extra, Anthropic key, an installed MCP server |
| `uvicorn examples.web_app.server:app --host 127.0.0.1 --port 8000` | Web chat, SSE, MCP management, snapshot save/load | `server` extra, Anthropic key; `mcp` extra if connecting servers |

The files `_console.py`, `_calculator.py`, `_fixtures.py`, and `_decision_fixtures.py` are shared helpers.
The `.md` files under `skills/` are sample skill inputs.

Jev examples print fixed fixture outputs unless `--live` is supplied. They do not
measure model quality offline. For live decisions install `.[jev]` and set
`TYPESAFE_API_KEY`. Live routing additionally requires `--model MODEL_ID` and the
selected `--provider` credentials/extra. Live answer review sends a fixed answer
and evidence to Jev; the agent remains scripted. See the [decision SDK guide](../doc/decisions.md)
for the API, usage semantics, lifecycle, and examples. Routing requires an explicit
confidence threshold; `0.8` demonstrates configuration and is not a calibrated default.

`simple_chat` registers its calculator with `Agent(tools=[calculate])`; custom
tools default to `ALLOW`, including tools registered after agent construction.
Built-in helpers inherit the same policy. Examples that use `CliPermissionManager`
opt into an `ASK` default, so writes and shell calls still request approval in the
coding example. Explicit permissions and manager overrides take precedence.

`multi_agent` uses `Agent(subagents=[SubAgent(...)])` to declare specialists.
Each task receives a fresh conversation and returns its result to the orchestrator.
See [subagents](../doc/subagents.md) and [tool registration](../doc/tools.md) for
the API, permissions, and recovery boundary. `langsmith_tracing` retains a manual
wrapper to demonstrate a child with its own tracing extension.

`skills_agent` also accepts `AGENT_PROVIDER` and `AGENT_MODEL`. When choosing a
different provider, install its SDK extra, set its API key, and specify a compatible
model explicitly. `provider_chat` accepts the command-line selection described
below. The remaining live examples use Anthropic unless edited.

## Try PostgreSQL recovery

Follow the [PostgreSQL durability walkthrough](../doc/postgres_durability.md) for
installation, connection setup, expected output, and recovery limits. Once
the `postgres` extra is installed, use `examples/postgres.config.json` for local
credentials. Copy `postgres.config.example.json` to that name if needed; the local
file is Git-ignored. Fill in its connection details and password field, leaving
`{password}` in the URL. Password URL encoding is automatic.

```bash
python -m examples.postgres_runtime check --config examples/postgres.config.json
python -m examples.postgres_runtime crash --config examples/postgres.config.json  # Exit 42.
python -m examples.postgres_runtime status --config examples/postgres.config.json
# Wait at least 30 seconds after the crash for the worker lease to expire.
python -m examples.postgres_runtime resume --config examples/postgres.config.json
```

`check` connects and prepares the runtime schema without running an agent. To use
the live Anthropic model, run `python -m examples.postgres_runtime chat --config
examples/postgres.config.json` with `ANTHROPIC_API_KEY` set. An explicit config
overrides `DATABASE_URL`. If you prefer the environment variable, omit `--config`.
The harness-web launcher continues to use `DATABASE_URL` from its environment.

The default workspace is `.agent_sessions/postgres-demo`. Use the same checkout,
database, and workspace for every command. For another experiment, pass a fresh
`--workspace` to all three commands. The test uses fixed report data and scripted
model responses; no model service is called. Running the module without a command
still runs the original live-model streaming example, including in harness-web.

Both PostgreSQL examples construct an `Agent` and pass it directly to
`AgentRuntime(agent, backend=backend)`. The recovery process creates a fresh agent
with the same configuration and tools before calling `resume(session_id)`;
`AgentRegistry` and `AgentRef` are not required.

## Choose a provider

`Agent()` and `simple_chat` default to **Anthropic** with `claude-sonnet-4-6`.
The new `provider_chat` example uses that same default, and accepts an explicit
provider and model on the command line. Configure the key for your chosen provider
in the repository's `.env` or your environment:

| Provider | Add the extra | API key |
|---|---|---|
| `anthropic` | `uv add harnessx` | `ANTHROPIC_API_KEY` |
| `openai` | `uv add "harnessx[openai]"` | `OPENAI_API_KEY` |
| `gemini` | `uv add "harnessx[gemini]"` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `openrouter` | `uv add "harnessx[openrouter]"` | `OPENROUTER_API_KEY` |
| `azure` | `uv add "harnessx[azure]"` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`; `--model` is the deployment name |

```bash
python3 -m examples.provider_chat
python3 -m examples.provider_chat --provider openai --model YOUR_OPENAI_MODEL
python3 -m examples.provider_chat --provider gemini --model YOUR_GEMINI_MODEL
python3 -m examples.provider_chat --provider openrouter --model YOUR_OPENROUTER_MODEL

# One turn, without streaming (Anthropic by default):
python3 -m examples.provider_chat --no-stream --prompt "What is 40 * 10?"
```

Replace the `YOUR_..._MODEL` placeholders with model IDs available to your account
that support tool calling. Non-Anthropic providers require `--model`; the example
does not guess a compatible model or switch providers based on which key is set.
It uses `AgentConfig(provider=..., model=...)` and `Agent(tools=[calculate])`.
Both run modes use the same tools. Interactive mode keeps history across turns;
use `usage` for token counts and `quit` to close the agent. `--prompt` exits with
status 1 if the run fails. No files or shell commands are exposed by this example.

In `harness-web`, select **Provider chat** and choose the provider, model, and
response mode in its launch dialog. Credentials remain in the server environment.

## MCP connections

Arguments in `--args` use shell-style quoting; the command is launched by MCP, not
by a shell. Supply exactly one transport:

```bash
python -m examples.mcp_agent --server files --command npx \
  --args='-y @modelcontextprotocol/server-filesystem /tmp'

python -m examples.mcp_agent --server remote --url http://localhost:8000/sse
```

For local MCP servers, install the server's runtime separately. Tools default to
terminal approval; `--permission allow` or `--permission deny` changes that policy.

## Execution and persistence boundaries

The coding and web examples expose host tools. The web app is a trusted local,
single-conversation demo with automatic tool permissions and no authentication.
It serializes requests and session/tool changes. Changing MCP connections or
clearing the session starts a new conversation. Its save/load endpoints use
`Agent.save_session()` and `Agent.load_session()` with tools/skills rebound; snapshots
restore conversation state, not in-flight execution.

`sandboxed_coder` uses the **process** tier, which retains host filesystem and
network access. Its working directory is temporary and removed on exit; the
runtime's SQLite checkpoints do not make those temporary files durable. Docker
and Seatbelt require their own configuration and qualification.

The PostgreSQL example requires an existing database and permissions to provision
the HarnessX schema. It does not start a database server. Recorder support is
locally verified with SQLite; PostgreSQL recording needs live qualification and
Temporal recording is not implemented. The remaining roadmap work is deferred.

## Verification

```bash
python -m pytest tests/test_examples.py tests/test_anthropic_provider.py -q
python -m mypy
```

The smoke suite runs every Python example's main flow and the declared specialists.
Provider chat is exercised for all four provider selections in both run modes,
with scripted responses, real calculator execution, and cleanup checks.
Interactive cases invoke their featured tools, including actual process execution
of a fixed print statement in the sandbox example.
It exercises the web routes using an in-process ASGI client, including concurrent
tool-result correlation, SSE, snapshots, skills, and cleanup. Model, MCP, tracing,
and PostgreSQL services are replaced with controlled stand-ins; network connections
are blocked. PostgreSQL runtime flow uses SQLite in these smoke tests. The existing
service integration tests separately require explicitly configured services.

The Anthropic regression suite uses the installed SDK with in-memory HTTP and
streaming responses. It covers nested caller/citation metadata, tool execution,
follow-up request serialization, snapshot save/load, the simple chat calculator
across turns, and the coding agent's CLI flow with a file-write approval. Requests
are checked against the installed SDK's content-block parameter fields, including
exclusion of SDK-only `parsed_output` metadata. Previously saved history is also
checked through ordinary, streaming, and token-count requests. These tests do not
call the live Anthropic API.

The no-service demos also run as separate CLI processes with network access
blocked. Type checking includes the entire examples directory in the normal
project check, so future SDK changes are checked against these examples.

The general evaluation runner's `offline=True` and `harnessx.evals.cli --offline`
disable LangSmith upload; they still execute the supplied/live model. Only
`examples.run_evals --offline` selects the scripted provider. Use the general eval
CLI for broader benchmark suites; this example is intentionally limited to its
three custom cases.
