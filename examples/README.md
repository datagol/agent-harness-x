# HarnessX examples

Every example is one Python file you run by its path:

```bash
uv sync --all-extras
python examples/04-control/loop_guard.py
```

They are plain scripts, not a package. Each file imports only `harnessx`, the
standard library, `python-dotenv`, and the optional SDK it is about. Each loads
the repository's `.env` without replacing variables already set in your shell.
So you can copy any file into your own project after `pip install harnessx` and
it runs the same. Examples are not part of the installed wheel.

The folders go in reading order, from a first chat to isolation. Inside each
folder, the files that need no API key come first in the tables below.

To launch every example from a browser, with tool calls shown as they happen,
use [harness-web](../harness-web/README.md). It also has a workspace where you
build your own agent. In containers, with PostgreSQL, Temporal and Redis already
running and no Python or Node to install:

```bash
docker compose up                                          # the browser at localhost:8765
docker compose run --rm web python examples/02-tools/skills_lazy_loading.py
```

## No API key needed

These use a scripted model and run the real engine, tools, hooks, approvals,
recording or evaluations. They verify integration behaviour, not model quality.

```bash
python examples/01-basics/prompt_caching.py
python examples/01-basics/progress_and_waiting.py
python examples/02-tools/skills_lazy_loading.py
python examples/03-context/planning_todos.py
python examples/03-context/condensing_a_long_history.py
python examples/04-control/loop_guard.py
python examples/04-control/retries_and_fallback.py
python examples/05-durability/tool_approvals_and_resume.py   # asks one question
python examples/05-durability/session_snapshots.py
python examples/06-quality/decisions_routing.py --min-confidence 0.8
python examples/06-quality/decisions_answer_review.py
python examples/06-quality/flight_recorder.py --output /tmp/invoice-incident.hx
python examples/06-quality/evaluating_with_datasets.py --offline   # needs the langsmith extra
```

## 01-basics

| File | What it shows | Needs |
|---|---|---|
| `prompt_caching.py` | A stable prompt prefix, the cache hint a provider receives, cache read and write counters | Nothing; `--live` uses `ANTHROPIC_API_KEY` |
| `progress_and_waiting.py` | A long run's event stream, including the `waiting` events sent while a slow model call is quiet (`ProgressPolicy`) | Nothing |
| `streaming_chat.py` | Interactive streaming chat with a bounded calculator and workspace reads; `usage` prints token counts | `ANTHROPIC_API_KEY` |
| `switching_providers.py` | The same agent on Anthropic, OpenAI, Gemini or OpenRouter; streaming or not; one-shot with `--prompt` | The provider's key and extra (below) |

## 02-tools

| File | What it shows | Needs |
|---|---|---|
| `skills_lazy_loading.py` | The model sees a skill catalog and loads one body with the `Skill` tool; hook order | Nothing |
| `filesystem_tools_and_permissions.py` | The file and shell built-ins under `CliPermissionManager`: reads allowed, writes and commands asked first | `ANTHROPIC_API_KEY` |
| `mcp_servers.py` | An MCP server's tools next to native ones; approvals for MCP tools | `ANTHROPIC_API_KEY`, `harnessx[mcp]`, an MCP server |
| `skills_interactive.py` | A live model picking skills from `examples/skills/` when the task calls for them | `ANTHROPIC_API_KEY` (or `AGENT_PROVIDER` + `AGENT_MODEL`) |

## 03-context

| File | What it shows | Needs |
|---|---|---|
| `planning_todos.py` | `write_todos` and `read_todos` on a multi-step task, and the `TODOS_UPDATED` events | Nothing |
| `condensing_a_long_history.py` | Old tool output cleared first, then a summary, when the history outgrows `max_context_tokens` | Nothing |
| `knowledge_bundles_okf.py` | Search, concept reads and link traversal over the OKF bundle in `examples/knowledge/` | `ANTHROPIC_API_KEY` (or `AGENT_PROVIDER` + `AGENT_MODEL`) |
| `custom_memory_tools.py` | The built-in `save_memory` and `recall_memories`, plus two tools of your own over the same stores | `ANTHROPIC_API_KEY` |

## 04-control

| File | What it shows | Needs |
|---|---|---|
| `loop_guard.py` | `LoopGuard` catching the same call repeating; the `REPETITION` hook; the note the model reads | Nothing |
| `retries_and_fallback.py` | `RetryPolicy` retrying a 503, then `FallbackProvider` switching to a backup | Nothing |
| `delegating_to_subagents.py` | An orchestrator handing one task to declared specialists (`Agent(subagents=[...])`) | `ANTHROPIC_API_KEY` |
| `autonomous_task_runner.py` | One task from the command line: it writes the deliverable and runs it | `ANTHROPIC_API_KEY`; `TAVILY_API_KEY` adds web search, `LANGSMITH_API_KEY` tracing |

## 05-durability

| File | What it shows | Needs |
|---|---|---|
| `tool_approvals_and_resume.py` | A run paused on an `ASK` tool, persisted, then finished with `approve(pending, resume=True)` | Nothing; answers one question |
| `session_snapshots.py` | `save_session` and `Agent.load_session` into a fresh agent: the step before a durable runtime | Nothing |
| `durable_crash_recovery.py` | A run that survives its process crashing, on PostgreSQL, without repeating its tool | `harnessx[postgres]` and a database; `chat` also `ANTHROPIC_API_KEY` |

## 06-quality

| File | What it shows | Needs |
|---|---|---|
| `decisions_routing.py` | A decision model classifies a document, then routes a query to one of three agents | Nothing by default; `--live` needs `harnessx[jev]`, `TYPESAFE_API_KEY`, `--model` |
| `decisions_answer_review.py` | Score for coverage and Noul for evidence support after a completed run | Nothing by default; `--live` needs `harnessx[jev]` and `TYPESAFE_API_KEY` |
| `flight_recorder.py` | A run recorded, exported as an `.hx` bundle, and played back with zero model or tool calls | Nothing; the `--output` file must not exist |
| `evaluating_with_datasets.py` | A small dataset of tool-selection and arithmetic cases, scored | `harnessx[langsmith]`; live mode `ANTHROPIC_API_KEY` |
| `tracing_with_langsmith.py` | One run tree per turn in LangSmith, with spans for model and tool calls | `harnessx[langsmith]`, `ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY` |

## 07-sandboxes

| File | What it shows | Needs |
|---|---|---|
| `sandbox_isolation_tiers.py` | A coding agent running its Python in a `Sandbox`; `--tier process` (default, not an isolation boundary), `docker` or `seatbelt` | `ANTHROPIC_API_KEY`; Docker for `--tier docker` |

`skills/` holds sample skills and `knowledge/` a sample Open Knowledge Format
bundle; set `AGENT_KNOWLEDGE` to a folder or git URL to load another.

## Choose a provider

`Agent()` and the chat examples default to **Anthropic** with `claude-sonnet-4-6`.
`switching_providers.py` takes the provider and model on the command line:

| Provider | Add the extra | API key |
|---|---|---|
| `anthropic` | `uv add harnessx` | `ANTHROPIC_API_KEY` |
| `openai` | `uv add "harnessx[openai]"` | `OPENAI_API_KEY` |
| `gemini` | `uv add "harnessx[gemini]"` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `openrouter` | `uv add "harnessx[openrouter]"` | `OPENROUTER_API_KEY` |
| `azure` | `uv add "harnessx[azure]"` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`; `--model` is the deployment name |

```bash
python examples/01-basics/switching_providers.py
python examples/01-basics/switching_providers.py --provider openai --model YOUR_OPENAI_MODEL
python examples/01-basics/switching_providers.py --no-stream --prompt "What is 40 * 10?"
```

Non-Anthropic providers require `--model`; the example does not guess a model or
switch providers based on which key is set.

## MCP connections

`--args` uses shell-style quoting; MCP launches the command, not a shell. Give
exactly one transport:

```bash
python examples/02-tools/mcp_servers.py --server files --command npx \
  --args='-y @modelcontextprotocol/server-filesystem /tmp'
python examples/02-tools/mcp_servers.py --server remote --url http://localhost:8000/sse
```

Tools default to terminal approval; `--permission allow` or `--permission deny`
changes that.

## PostgreSQL recovery

Follow the [PostgreSQL durability walkthrough](../doc/postgres_durability.md).
Copy `05-durability/postgres.config.example.json` to `postgres.config.json` in
the same folder (it is Git-ignored), fill in the connection details and password,
and leave `{password}` in the URL:

```bash
python examples/05-durability/durable_crash_recovery.py check  --config examples/05-durability/postgres.config.json
python examples/05-durability/durable_crash_recovery.py crash  --config examples/05-durability/postgres.config.json  # exits 42
python examples/05-durability/durable_crash_recovery.py status --config examples/05-durability/postgres.config.json
# Wait at least 30 seconds after the crash for the worker lease to expire.
python examples/05-durability/durable_crash_recovery.py resume --config examples/05-durability/postgres.config.json
```

`check` prepares the schema without running an agent. With no command it runs
the live streaming example. `--config` overrides `DATABASE_URL`. Use the same
database and `--workspace` (default `.agent_sessions/postgres-demo`) for every
command.

## Boundaries

- The filesystem example exposes host tools. A working directory is not a sandbox.
- `sandbox_isolation_tiers.py` defaults to the **process** tier, which keeps host
  filesystem and network access. Docker and Seatbelt are the isolating tiers.
- Decision examples print fixed fixture outputs unless `--live` is given; offline
  they do not measure model quality. Routing needs an explicit confidence
  threshold; `0.8` demonstrates the setting and is not a calibrated default.

## Verification

```bash
python -m pytest tests/test_examples.py -q
python -m mypy
```

The suite checks that every file stands alone. It runs every no-key example by
path from a temporary directory, and drives the interactive ones through
`input()` with scripted models. Model, MCP, tracing and PostgreSQL services are
replaced with stand-ins, and network connections are blocked.
