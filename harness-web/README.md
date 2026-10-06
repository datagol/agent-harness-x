# harness-web

A local workspace for **every HarnessX example entry point**, plus a separate
general chat interface. It runs the repository's real examples, streams their
output, presents terminal input and approvals in the browser, and makes generated
files available to download.

`datagol-web` was inspected before building this app. It is a React/Vite marketing
site with Datagol colors and animated landing-page components. This app reuses its
React/Vite/Lucide stack, dependency versions, and core color palette. The marketing
site remains unchanged. The backend follows the existing web example's use of
FastAPI and HarnessX streaming, with separate chat sessions and explicit approvals.

## Run

From the repository root, using Python 3.11+ and Node 22.12+:

```bash
python3 -m pip install -e '.[server]'
npm --prefix harness-web ci
npm --prefix harness-web run build
python3 harness-web/run.py
```

Open **http://127.0.0.1:8765**. Use `--port 8766` to choose another port, and
`--host 0.0.0.0` to accept connections from outside this machine. It binds
localhost by default because the app runs example code.
The frontend build is served by the Python server; no second server is needed.

## Run everything in containers

No Python, no Node, and every optional service provisioned:

```bash
docker compose up
```

That builds the frontend, installs the SDK with every extra, and starts
PostgreSQL, Temporal and Redis alongside the app on
**http://localhost:8765**. The durable and Temporal examples are runnable with
nothing further to configure.

Model API keys come from your own `.env`, read at run time and never baked into
an image. Without them the offline examples still run and the live ones say
what is missing.

To run the suite with those services reachable, which is the only way nothing
is skipped:

```bash
docker compose run --rm tests
```

## Credentials and scope

The app lives in this checkout and is not bundled in the SDK wheel.

The existing repository `.env` is loaded without overriding environment variables.
Set `ANTHROPIC_API_KEY` to use the default live agents. Keys are never returned by
the configuration API or entered into a browser form. Other chat providers use
`OPENAI_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY` and their corresponding
SDK extras. Enter the model ID when selecting a provider without a default.
Restart the server after changing environment variables.

Without a key, start with **Flight recorder**, **Approvals and resume**, **Loop
guard**, or **Skills, loaded on demand**. General chat also offers an explicitly labelled **Local demo** that
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

Examples run as child processes, in unique working directories beneath
`.harness-web/runs/`. They retain their existing provider configuration and tool
behavior. The launcher runs the script with `runpy.run_path` and bridges its `input()` calls
through a separate control pipe, so ordinary tool output cannot be confused with a prompt.
An answer must match the exact pending prompt ID; duplicate and stale approvals
are rejected. Closing an output tab does not stop a run. Use **Stop run** to
terminate the managed process group. Deliberately detached child processes are
outside that cleanup guarantee.

The original web demo retains automatic permissions for its host tools; it opens
in another tab. It is independent of the new General chat interface.

## General chat

Each conversation owns a separate `Agent`, message history, and working directory.
The assistant can calculate, list/read workspace files, and request file changes.
File changes wait for the SDK's permission callback and an explicit browser
decision. There is no shell tool in general chat. Tool calls and their results are
correlated by ID, and streamed failures are visible. A conversation permits one
active turn at a time; separate conversations can run concurrently.

Use the **conversation setup** button beside the chat title to configure a chat
before or after its first turn. It can change the system prompt, upload Markdown
skills, and connect a local stdio or remote HTTP/SSE MCP server. Skills are made
available through the lazy `Skill` tool. MCP tool calls default to `ASK`; the setup
screen also permits an explicit `ALLOW` or `DENY` policy.

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
authenticated multi-tenant deployment or an OS sandbox. It binds only to loopback
and rejects unrelated browser origins. Do not publish it through a public proxy.

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
links, approvals, and general chat. Browser visual verification could not run in
the build environment because localhost binding and browser startup were blocked.
After starting locally, verify: run the recorder and download its bundle; approve
and deny the approval demo; open the original web demo; chat with the local demo;
then select your configured provider for a live turn.
