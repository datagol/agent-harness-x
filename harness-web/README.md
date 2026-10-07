# harness-web

A local workspace with two halves:

- **Examples.** Every script in `examples/`, grouped by topic. Each runs
  unchanged; you see its output stream, every tool call its agents make (with
  arguments, result and status), its prompts and approvals in the browser, and
  the files it writes, ready to download.
- **Build your agent.** One conversation-scoped agent you shape yourself: write
  its system prompt, choose provider and model, switch tools on and off, upload
  skills, connect MCP servers, and add subagents it can delegate to.

`datagol-web` was inspected before building this app. It is a React/Vite marketing
site with Datagol colors and animated landing-page components. This app reuses its
React/Vite/Lucide stack, dependency versions, and core color palette. The marketing
site remains unchanged. The backend uses FastAPI and HarnessX streaming, with
separate chat sessions and explicit approvals.

## Run

From the repository root, using Python 3.11+ and Node 22.12+:

```bash
python3 -m pip install -e '.[server]'
npm --prefix harness-web ci
npm --prefix harness-web run build
python3 harness-web/run.py
```

Open **http://127.0.0.1:8765**. Use `--port 8766` to choose another port, and
`--host 0.0.0.0` to listen on every interface, as the container does. It binds
localhost by default because the app runs example code, and it answers only
requests addressed to `localhost` or `127.0.0.1` whatever it binds.
The frontend build is served by the Python server; no second server is needed.

## Run everything in containers

No Python, no Node, and every optional service provisioned:

```bash
cp .env.example .env    # optional: your model keys
docker compose up --build
```

That builds the frontend, installs the SDK with every extra, and starts
PostgreSQL, Temporal and Redis alongside the app on
**http://localhost:8765**. The durable and Temporal examples are runnable with
nothing further to configure.

Model API keys come from your own `.env` (copy `.env.example` first), read at run
time and never baked into an image, or from **Workspace setup → API keys** in
the app, kept in the `harness-web-data` volume. Without them the offline
examples still run and the live ones say what is missing.

To run the suite with those services reachable, which is the only way nothing
is skipped:

```bash
docker compose run --rm tests
```

## Credentials and scope

The app lives in this checkout and is not bundled in the SDK wheel.

There are two ways to give the app your model keys:

- **`.env`.** Run `cp .env.example .env` in the repository root and set
  `ANTHROPIC_API_KEY` (and any others) before starting. It is loaded without
  overriding variables already set; restart after changing it.
- **In the app.** Open **Workspace setup** (the HX button, top right) and paste a
  key under **API keys**. It is used at once by the next example run and the next
  conversation, saved in the app's data folder (`.harness-web/api-keys.json`,
  readable by you only; in Docker, the `harness-web-data` volume), and never sent
  back to the browser, which only shows whether a key is set and where it came
  from. A key saved in the app takes precedence over `.env`; removing it restores
  the `.env` value.

Other chat providers use `OPENAI_API_KEY`, `GEMINI_API_KEY`, or
`OPENROUTER_API_KEY` and their SDK extras. Enter the model ID when selecting a
provider without a default.

Without a key, start with **Flight recorder**, **Approvals and resume**, **Loop
guard**, or **Skills, loaded on demand**. **Build your agent** also offers an explicitly labelled **Local demo** that
uses scripted responses and the real calculator; it is not a language model.
Try `What is 48 * 12?`.

## Example coverage

The library lists every script in `examples/`, grouped by the same topics as the
folders. Each entry runs the file unchanged, as `python examples/<topic>/<file>.py`.

| Topic | UI entry | Script | Setup / behavior |
|---|---|---|---|
| Basics | Prompt caching | `01-basics/prompt_caching.py` | Offline |
| Basics | Progress and waiting | `01-basics/progress_and_waiting.py` | Offline; prints every event as it arrives |
| Basics | Streaming chat | `01-basics/streaming_chat.py` | Anthropic; send follow-up messages or `quit` |
| Basics | Switching providers | `01-basics/switching_providers.py` | Choose Anthropic, OpenAI, Gemini, or OpenRouter, a model ID, and streaming or ordinary responses |
| Basics | Image, PDF and audio input | `01-basics/media_input.py` | `gemini` extra and `GEMINI_API_KEY`; sends the 1x1 PNG held in the script |
| Tools | Filesystem tools and permissions | `02-tools/filesystem_tools_and_permissions.py` | Anthropic; writes and shell commands ask for approval |
| Tools | MCP servers | `02-tools/mcp_servers.py` | `mcp` extra, Anthropic key, stdio command or SSE URL |
| Skills | Skills, loaded on demand | `03-skills/skills_lazy_loading.py` | Offline; displays real skill and hook events |
| Skills | Skills with a live model | `03-skills/skills_interactive.py` | Configured `AGENT_PROVIDER` / `AGENT_MODEL`, Anthropic by default |
| Context | Knowledge bundles (OKF) | `04-context/knowledge_bundles_okf.py` | Retrieval over the sample OKF bundle; `AGENT_KNOWLEDGE` selects a folder or Git URL |
| Context | Memory tools | `04-context/custom_memory_tools.py` | Anthropic; notes belong to the run's working directory |
| Context | Planning with to-dos | `04-context/planning_todos.py` | Offline |
| Context | Condensing a long history | `04-context/condensing_a_long_history.py` | Offline |
| Control | Loop guard | `05-control/loop_guard.py` | Offline |
| Control | Retries and fallback | `05-control/retries_and_fallback.py` | Offline |
| Control | Delegating to subagents | `05-control/delegating_to_subagents.py` | Anthropic; constructor-defined specialists |
| Control | Autonomous task runner | `05-control/autonomous_task_runner.py` | Anthropic; one task, writes and runs the deliverable |
| Durability | Approvals and resume | `06-durability/tool_approvals_and_resume.py` | Offline; choose Allow once or Deny in the browser |
| Durability | Durable runs on PostgreSQL | `06-durability/durable_crash_recovery.py` | `postgres` extra, `DATABASE_URL`, Anthropic key |
| Durability | Session snapshots | `06-durability/session_snapshots.py` | Offline |
| Quality | Evaluating with datasets | `07-quality/evaluating_with_datasets.py` | `langsmith` extra; offline by default, optional live model mode |
| Quality | Decisions: routing and classification | `07-quality/decisions_routing.py` | Offline fixed decisions |
| Quality | Decisions: answer review | `07-quality/decisions_answer_review.py` | Offline fixed decisions |
| Quality | Tracing with LangSmith | `07-quality/tracing_with_langsmith.py` | `langsmith` extra and Anthropic / LangSmith keys; uploads traces |
| Quality | Flight recorder | `07-quality/flight_recorder.py` | Offline; download `incident.hx` after completion |
| Sandboxes | Sandbox isolation tiers | `08-sandboxes/sandbox_isolation_tiers.py` | Anthropic; process tier by default |
| Sandboxes | Tools in an OpenShell sandbox | `08-sandboxes/openshell_backend.py` | `openshell` extra, a running OpenShell gateway, Anthropic key |

Examples run as child processes, in unique working directories beneath
`.harness-web/runs/`. They retain their existing provider configuration and tool
behavior. The launcher runs the script with `runpy.run_path` and bridges its `input()` calls
through a separate control pipe, so ordinary tool output cannot be confused with a prompt.
It also adds tool hooks to every `Agent` the script constructs, so each tool call
and result reaches the browser as a card beside the output; the example file
itself is not changed.
An answer must match the exact pending prompt ID; duplicate and stale approvals
are rejected. Closing an output tab does not stop a run. Use **Stop run** to
terminate the managed process group. Deliberately detached child processes are
outside that cleanup guarantee.

## Build your agent

Each conversation owns a separate `Agent`, message history, and working directory.
Out of the box the agent can calculate, read and write workspace files, run shell
commands in that directory, and ask you a question when a request is ambiguous.
Writes and shell commands wait for an explicit browser decision. Tool calls and
their results are correlated by ID, and streamed failures are visible. A
conversation permits one active turn at a time; separate conversations can run
concurrently.

The settings button beside the agent's name opens **Build your agent**, before or
after the first turn:

- **System prompt.** The agent's instructions.
- **Tools.** Every tool the agent can call, grouped by where it comes from, with
  its permission; switch any off for the next turn.
- **Skills.** Upload Markdown skills; the agent loads one only when it calls the
  lazy `Skill` tool.
- **MCP servers.** Connect a local stdio or remote HTTP/SSE server. Its tools
  default to `ASK`; `ALLOW` or `DENY` can be chosen instead.
- **Subagents.** Add a specialist with a name, a description that tells your
  agent when to delegate, its own instructions, and the tools it may use (taken
  from this conversation, so file tools stay in its directory and approvals still
  come to the browser). Each becomes a `delegate_<name>` tool; every delegation
  runs a fresh child agent on the conversation's provider and model and returns
  its answer. With the local demo provider, subagents still run on Anthropic
  and need `ANTHROPIC_API_KEY`.

Setup changes retain conversation history and rebuild only that conversation's
agent. They cannot be made during a response. Uploaded skills live in that
conversation's workspace and are limited to 256 KB. MCP command arguments are
parsed into an argument vector and never passed through a shell. Connecting a
server authorizes a connection to the endpoint or local command entered by the
local user; its tools remain subject to the selected execution policy. Remote MCP
headers and credentials are intentionally not entered through the browser.

Conversations and run history survive browser navigation/reload while the server
is alive. **They do not survive a server restart.** Files remain on disk; the UI
does not claim durable web sessions. Stopping a run does not reverse completed
effects. The launcher is a development app for a trusted local user, not an
authenticated multi-tenant deployment or an OS sandbox. It binds to loopback by
default, answers only local host names, and rejects unrelated browser origins. Do not publish it through a public proxy.

The server retains up to 100 runs, 50 chat sessions, and four active runs. Each
run retains the latest 4,000 events / approximately 2 MB, with an explicit gap
notice on truncated playback. Generated directories are not automatically deleted.
Set `HARNESS_WEB_DATA_DIR` to change the storage location.

## Development

Run the backend and Vite in separate terminals:

```bash
python3 -m uvicorn harness_web.server:app --app-dir harness-web --host 127.0.0.1 --port 8765 --reload
```

```bash
npm --prefix harness-web run dev
```

Open http://127.0.0.1:5173. Vite proxies `/api` to port 8765.

## Verification

```bash
python3 -m pytest tests/test_harness_web.py tests/test_examples.py tests/test_anthropic_provider.py -q
npm --prefix harness-web test
npm --prefix harness-web run build
```

Backend tests run the offline examples as real child processes and cover approval
allow/deny, cancellation, event reconnection, artifact downloads, isolated chat
sessions, blocked duplicate turns, and engine-enforced chat file permissions.
Frontend state tests cover duplicate event replay, out-of-order tool results,
model retries, completed responses, approval removal, and bounded output.

Live model, PostgreSQL, MCP, and LangSmith services still require their own
qualification. Credential/package availability is a prerequisite check, not proof
that a remote service is reachable. DOM interaction checks with a mocked API
passed for library search/filtering, example launch, streamed output, download
links, approvals, and chat. Browser visual verification could not run in
the build environment because localhost binding and browser startup were blocked.
After starting locally, verify: run the recorder and download its bundle; approve
and deny the approval demo; chat with the local demo;
then select your configured provider for a live turn.
