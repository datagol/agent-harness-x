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

Open **http://127.0.0.1:8765**. Use `--port 8766` to choose another port.
The frontend build is served by the Python server; no second server is needed.
The app lives in this checkout and is not bundled in the SDK wheel.

The existing repository `.env` is loaded without overriding environment variables.
Set `ANTHROPIC_API_KEY` to use the default live agents. Keys are never returned by
the configuration API or entered into a browser form. Other chat providers use
`OPENAI_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY` and their corresponding
SDK extras. Enter the model ID when selecting a provider without a default.
Restart the server after changing environment variables.

Without a key, start with **Flight recorder**, **Durable approvals**, or **Skill
discovery**. General chat also offers an explicitly labelled **Local demo** that
uses scripted responses and the real calculator; it is not a language model.
Try `What is 48 * 12?`.

## Example coverage

| UI entry | Python entry point | Setup / behavior |
|---|---|---|
| Flight recorder | `examples.flight_recorder` | Offline; download `incident.hx` after completion |
| Durable approvals | `examples.runtime_approvals` | Offline; choose Allow once or Deny in the browser |
| Skill discovery | `examples.skills_demo` | Offline; displays real skill and hook events |
| Simple chat | `examples.simple_chat` | Anthropic; send follow-up messages or `quit` |
| Provider chat | `examples.provider_chat` | Choose Anthropic, OpenAI, Gemini, or OpenRouter, a model ID, and streaming or ordinary responses |
| Coding agent | `examples.coding_agent` | Anthropic; writes and shell commands ask for approval |
| Memory agent | `examples.memory_agent` | Anthropic; notes belong to the example's working directory |
| Specialist agents | `examples.multi_agent` | Anthropic; constructor-defined specialists |
| Skills agent | `examples.skills_agent` | Configured `AGENT_PROVIDER` / `AGENT_MODEL`, Anthropic by default |
| Knowledge agent | `examples.knowledge_agent` | BM25F retrieval over the sample OKF bundle; `AGENT_KNOWLEDGE` selects a folder or Git URL; configured provider/model |
| Sandboxed coder | `examples.sandboxed_coder` | Anthropic; process resource limits, temporary execution directory |
| Workflow evaluations | `examples.run_evals` | `langsmith` extra; offline by default, optional live model mode |
| LangSmith tracing | `examples.langsmith_tracing` | `langsmith` extra and Anthropic / LangSmith keys; uploads traces |
| PostgreSQL runtime | `examples.postgres_runtime` | `postgres` extra, `DATABASE_URL`, Anthropic key |
| MCP agent | `examples.mcp_agent` | `mcp` extra, Anthropic key, stdio command or SSE URL |

Examples run as child processes, in unique working directories beneath
`.harness-web/runs/`. They retain their existing provider configuration and tool
behavior. The launcher bridges `input()` and the shared console prompt through a
separate control pipe, so ordinary tool output cannot be confused with a prompt.
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
