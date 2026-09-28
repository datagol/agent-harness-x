import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  ArrowUp,
  Check,
  ChevronDown,
  ChevronRight,
  Code2,
  Copy,
  Database,
  Download,
  ExternalLink,
  FileCode2,
  FlaskConical,
  FolderOpen,
  Globe,
  History,
  LayoutGrid,
  LoaderCircle,
  Menu,
  MessageSquare,
  Network,
  Play,
  Plug,
  Plus,
  Radio,
  Search,
  Settings2,
  ToggleLeft,
  ToggleRight,
  Wrench,
  ShieldCheck,
  Sparkles,
  Square,
  Terminal,
  Trash2,
  Upload,
  X,
  Zap,
  Brain,
  AlertCircle,
} from "lucide-react";
import { api, finished, initialFeed, reduceFeed } from "./api.js";

const ICONS = {
  record: Radio,
  shield: ShieldCheck,
  sparkles: Sparkles,
  message: MessageSquare,
  code: Code2,
  memory: Brain,
  network: Network,
  terminal: Terminal,
  flask: FlaskConical,
  activity: Activity,
  database: Database,
  plug: Plug,
  globe: Globe,
};
const CATEGORIES = [
  "All examples",
  "Agents",
  "Runtime",
  "Skills",
  "Evaluation",
  "Integrations",
];
const go = (path) => {
  window.location.hash = path;
};
const label = (status) =>
  ({
    starting: "Starting",
    running: "Running",
    waiting: "Needs input",
    completed: "Completed",
    failed: "Failed",
    cancelled: "Stopped",
  })[status] || status;
const date = (value) =>
  new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

function Mark({ small = false }) {
  // The Harness-X mark from the docs site, white on the accent tile.
  const size = small ? 18 : 24;
  return (
    <div className={`mark ${small ? "small" : ""}`} aria-hidden="true">
      <svg viewBox="0 0 64 64" width={size} height={size}>
        <path fill="#fff" d="M8 8H18V20L30 32L18 44V56H8V40L16 32L8 24Z" />
        <path fill="#fff" d="M56 8H46V20L34 32L46 44V56H56V40L48 32L56 24Z" />
      </svg>
    </div>
  );
}
function Badge({ status, children }) {
  return (
    <span className={`badge ${status || ""}`}>
      <span className="status-dot" />
      {children || label(status)}
    </span>
  );
}
function ErrorNotice({ message, onDismiss }) {
  if (!message) return null;
  return (
    <div className="error-notice" role="alert">
      <AlertCircle size={17} />
      <span>{message}</span>
      {onDismiss && (
        <button
          className="icon-button"
          onClick={onDismiss}
          aria-label="Dismiss error"
        >
          <X size={15} />
        </button>
      )}
    </div>
  );
}
function Modal({ title, children, onClose }) {
  const ref = useRef(null);
  useEffect(() => {
    ref.current.showModal();
  }, []);
  return (
    <dialog
      ref={ref}
      className="modal"
      onCancel={onClose}
      aria-label={title}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal-head">
        <h2>{title}</h2>
        <button
          className="icon-button"
          onClick={onClose}
          aria-label="Close dialog"
        >
          <X size={20} />
        </button>
      </div>
      {children}
    </dialog>
  );
}

function useFeed(runId, onFinished) {
  const [feed, setFeed] = useState(initialFeed);
  const [error, setError] = useState("");
  const doneRef = useRef(onFinished);
  doneRef.current = onFinished;
  useEffect(() => {
    setFeed(initialFeed());
    setError("");
    if (!runId) return;
    let source,
      disposed = false;
    api(`/runs/${runId}`)
      .then((run) => {
        if (disposed) return;
        setFeed((prev) => ({ ...prev, run }));
        source = new EventSource(`/api/runs/${runId}/events`);
        source.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data);
            setFeed((prev) => reduceFeed(prev, payload));
            setError("");
          } catch {
            setError("Could not read a run event. Reload to reconnect.");
          }
        };
        source.addEventListener("done", () => {
          source.close();
          doneRef.current?.();
        });
        source.onerror = () => {
          if (!disposed)
            setError(
              "Connection interrupted. Reconnecting to retained output…",
            );
        };
      })
      .catch((err) => {
        if (!disposed) setError(err.message);
      });
    return () => {
      disposed = true;
      source?.close();
    };
  }, [runId]);
  return { ...feed, connectionError: error };
}

function LaunchDialog({ example, providers, onClose, onLaunch }) {
  const [config, setConfig] = useState({
    example_id: example.id,
    prompt: example.prompt,
    mode: "offline",
    provider: "anthropic",
    model:
      providers.find((provider) => provider.id === "anthropic")?.model || "",
    streaming: true,
    server: "",
    command: "",
    args: "",
    url: "",
  });
  const [transport, setTransport] = useState("command");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const missing =
    example.id === "provider_chat"
      ? providers.find((provider) => provider.id === config.provider)
          ?.missing || []
      : config.mode === "live"
        ? example.live_missing
        : example.missing;
  const Icon = ICONS[example.icon];
  const change = (key, value) =>
    setConfig((prev) => ({ ...prev, [key]: value }));
  async function launch(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const run = await api("/runs", { method: "POST", body: config });
      onLaunch(run);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }
  return (
    <Modal title="Launch example" onClose={onClose}>
      <form onSubmit={launch}>
        <div className="launch-intro">
          <div className="example-icon">
            <Icon size={25} />
          </div>
          <div>
            <h3>{example.title}</h3>
            <p>{example.description}</p>
          </div>
        </div>
        <p className="detail-note">{example.detail}</p>
        {example.id === "provider_chat" && (
          <div className="form-group">
            <label>
              Provider
              <select
                value={config.provider}
                onChange={(e) =>
                  setConfig((prev) => ({
                    ...prev,
                    provider: e.target.value,
                    model:
                      providers.find(
                        (provider) => provider.id === e.target.value,
                      )?.model || "",
                  }))
                }
              >
                {providers
                  .filter((provider) => provider.id !== "demo")
                  .map((provider) => (
                    <option key={provider.id} value={provider.id}>
                      {provider.name}
                    </option>
                  ))}
              </select>
            </label>
            <label>
              Model ID
              <input
                required
                value={config.model}
                onChange={(e) => change("model", e.target.value)}
                placeholder="Enter a model available to your account"
              />
            </label>
            <label>
              Responses
              <select
                value={config.streaming ? "streaming" : "ordinary"}
                onChange={(e) =>
                  change("streaming", e.target.value === "streaming")
                }
              >
                <option value="streaming">Streaming</option>
                <option value="ordinary">Ordinary</option>
              </select>
            </label>
          </div>
        )}
        {example.id === "run_evals" && (
          <label>
            Evaluation mode
            <select
              value={config.mode}
              onChange={(e) => change("mode", e.target.value)}
            >
              <option value="offline">Scripted fixture · no model calls</option>
              <option value="live">Live model · API usage applies</option>
            </select>
          </label>
        )}
        {example.interactive && (
          <label>
            First message <span className="optional">optional</span>
            <textarea
              rows={3}
              value={config.prompt}
              onChange={(e) => change("prompt", e.target.value)}
              placeholder="You can also send a message after launch."
            />
          </label>
        )}
        {example.id === "mcp_agent" && (
          <div className="form-group">
            <label>
              Server name
              <input
                required
                value={config.server}
                onChange={(e) => change("server", e.target.value)}
                placeholder="my-server"
              />
            </label>
            <label>
              Connection
              <select
                value={transport}
                onChange={(e) => {
                  setTransport(e.target.value);
                  setConfig((prev) => ({
                    ...prev,
                    command: "",
                    args: "",
                    url: "",
                  }));
                }}
              >
                <option value="command">Local command (stdio)</option>
                <option value="url">Remote SSE URL</option>
              </select>
            </label>
            {transport === "command" ? (
              <>
                <label>
                  Command
                  <input
                    required
                    value={config.command}
                    onChange={(e) => change("command", e.target.value)}
                    placeholder="python3"
                  />
                </label>
                <label>
                  Arguments
                  <input
                    value={config.args}
                    onChange={(e) => change("args", e.target.value)}
                    placeholder="/absolute/path/to/server.py"
                  />
                </label>
              </>
            ) : (
              <label>
                SSE URL
                <input
                  required
                  type="url"
                  value={config.url}
                  onChange={(e) => change("url", e.target.value)}
                  placeholder="http://localhost:8000/sse"
                />
              </label>
            )}
          </div>
        )}
        {!!missing.length && (
          <div className="setup-note">
            <Settings2 size={18} />
            <div>
              <strong>Setup needed</strong>
              <p>
                Configure {missing.join(", ")} in the server environment, then
                restart the server. Keys stay on the server.
              </p>
            </div>
          </div>
        )}
        <ErrorNotice message={error} />
        <div className="modal-footer">
          <span className="muted">
            <Terminal size={14} /> {example.module}
          </span>
          <button
            className="button primary"
            type="submit"
            disabled={
              busy ||
              !!missing.length ||
              (example.id === "provider_chat" && !config.model.trim())
            }
          >
            {busy ? (
              <LoaderCircle size={16} className="spin" />
            ) : (
              <Play size={16} />
            )}{" "}
            Launch example
          </button>
        </div>
      </form>
    </Modal>
  );
}

function Library({ examples, loading, onSelect }) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("All examples");
  const [offlineOnly, setOfflineOnly] = useState(false);
  const visible = examples.filter(
    (item) =>
      (category === "All examples" || item.category === category) &&
      (!offlineOnly || item.offline) &&
      `${item.title} ${item.description}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  return (
    <div className="library page">
      <div className="eyebrow">
        <span /> THE HARNESS-X PLAYGROUND
      </div>
      <div className="page-heading">
        <div>
          <h1>Learn by running.</h1>
          <p>
            Real workflows. Visible tool calls. A closer look at how agents
            work.
          </p>
        </div>
        <span className="count-pill">{examples.length} examples</span>
      </div>
      <div className="featured">
        <div className="featured-copy">
          <div className="featured-label">
            <Radio size={16} /> START HERE <span>NO API KEY</span>
          </div>
          <h2>A complete run. Every detail.</h2>
          <p>
            Watch an invoice workflow recover from a failure, then explore
            <br className="desktop-break" /> its flight recording without making
            another model call.
          </p>
          <button
            className="button primary"
            onClick={() =>
              onSelect(examples.find((item) => item.id === "flight_recorder"))
            }
            disabled={!examples.length}
          >
            Try the flight recorder <ArrowRight size={16} />
          </button>
        </div>
        <div className="run-illustration" aria-hidden="true">
          <div className="illustration-top">
            <span className="live-dot" /> invoice-analysis <span>run_01</span>
          </div>
          {[
            ["01", "Model request", "Recovered"],
            ["02", "overdue_balance", "$1,500"],
            ["03", "Incident bundle", "Verified"],
          ].map(([num, text, result]) => (
            <div className="illustration-row" key={num}>
              <span className="step-num">{num}</span>
              <span>{text}</span>
              <span className="step-result">
                <Check size={12} />
                {result}
              </span>
            </div>
          ))}
          <div className="illustration-footer">
            <ShieldCheck size={13} /> Captured once. Inspected offline.
          </div>
        </div>
      </div>
      <div className="library-toolbar">
        <div className="search">
          <Search size={17} />
          <input
            aria-label="Search examples"
            placeholder="Search examples…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <kbd>/</kbd>
        </div>
        <label className="check-label">
          <input
            type="checkbox"
            checked={offlineOnly}
            onChange={(e) => setOfflineOnly(e.target.checked)}
          />{" "}
          No API key needed
        </label>
      </div>
      <div className="tabs" role="tablist" aria-label="Example categories">
        {CATEGORIES.map((item) => (
          <button
            role="tab"
            aria-selected={category === item}
            key={item}
            className={category === item ? "active" : ""}
            onClick={() => setCategory(item)}
          >
            {item}
            {item === "All examples" && <span>{examples.length}</span>}
          </button>
        ))}
      </div>
      {loading ? (
        <div className="empty">
          <LoaderCircle className="spin" /> Loading examples…
        </div>
      ) : !visible.length ? (
        <div className="empty">
          <Search size={28} />
          <h3>No matching examples</h3>
          <p>Try a different search or category.</p>
        </div>
      ) : (
        <div className="example-grid">
          {visible.map((example) => {
            const Icon = ICONS[example.icon];
            return (
              <button
                className="example-card"
                onClick={() => onSelect(example)}
                key={example.id}
              >
                <div className="card-top">
                  <div className="example-icon">
                    <Icon size={21} />
                  </div>
                  <span
                    className={`mode-label ${example.offline ? "offline" : ""}`}
                  >
                    {example.offline ? "No API key" : "Live model"}
                  </span>
                </div>
                <h3>{example.title}</h3>
                <p>{example.description}</p>
                <div className="card-footer">
                  <span>
                    {example.category}
                    {example.missing.length > 0 && (
                      <span className="setup-dot" title="Setup required" />
                    )}
                  </span>
                  <span className="card-action">
                    {example.missing.length ? "View setup" : "Explore"}
                    <ArrowRight size={15} />
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      )}
      <div className="library-footnote">
        <Code2 size={15} /> Every example runs the Python module in this
        repository.
      </div>
    </div>
  );
}

function Prompt({ run, onReply }) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    setValue("");
    setError("");
    setBusy(false);
  }, [run?.pending?.id]);
  if (!run?.pending) return null;
  async function send(answer) {
    setBusy(true);
    setError("");
    try {
      await api(`/runs/${run.id}/input`, {
        method: "POST",
        body: { prompt_id: run.pending.id, value: answer },
      });
      onReply?.();
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }
  const approval = run.pending.kind === "approval";
  return (
    <div className={`prompt-box ${approval ? "approval" : ""}`}>
      <div className="prompt-title">
        {approval ? <ShieldCheck size={18} /> : <MessageSquare size={18} />}
        <strong>
          {approval ? "Your approval is needed" : "Continue the conversation"}
        </strong>
      </div>
      {approval ? (
        <>
          <pre>
            {run.pending.tool
              ? JSON.stringify(run.pending.tool, null, 2)
              : run.pending.prompt}
          </pre>
          <div className="approval-actions">
            <button
              className="button"
              disabled={busy}
              onClick={() => send("n")}
            >
              Deny
            </button>
            <button
              className="button primary"
              disabled={busy}
              onClick={() => send("y")}
            >
              {busy && <LoaderCircle size={15} className="spin" />}Allow once
            </button>
          </div>
        </>
      ) : (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (value.trim()) send(value);
          }}
        >
          <label className="sr-only" htmlFor="run-reply">
            Reply to example
          </label>
          <input
            id="run-reply"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={
              run.pending.kind === "message"
                ? "Send a message, or type quit to finish…"
                : run.pending.prompt || "Enter a response…"
            }
            disabled={busy}
            autoComplete="off"
          />
          <button
            className="button primary"
            disabled={busy || !value.trim()}
            type="submit"
          >
            <ArrowUp size={18} />
            <span className="sr-only">Send reply</span>
          </button>
        </form>
      )}
      <ErrorNotice message={error} />
    </div>
  );
}

function RunView({ id, refresh, examples }) {
  const feed = useFeed(id, refresh);
  const [files, setFiles] = useState([]);
  const [filesLoading, setFilesLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [follow, setFollow] = useState(true);
  const [copied, setCopied] = useState(false);
  const logRef = useRef(null);
  const run = feed.run;
  useEffect(() => {
    if (follow && logRef.current)
      logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [feed.log, follow]);
  useEffect(() => {
    setFiles([]);
    setFilesLoading(true);
    setError("");
    setBusy(false);
  }, [id]);
  useEffect(() => {
    let disposed = false;
    const load = () =>
      api(`/runs/${id}/files`)
        .then((data) => {
          if (!disposed) setFiles(data);
        })
        .catch((err) => {
          if (!disposed) setError(err.message);
        })
        .finally(() => {
          if (!disposed) setFilesLoading(false);
        });
    load();
    const timer = setInterval(load, 5000);
    return () => {
      disposed = true;
      clearInterval(timer);
    };
  }, [id]);
  async function stop() {
    setBusy(true);
    try {
      await api(`/runs/${id}/stop`, { method: "POST" });
      refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }
  async function copy() {
    try {
      await navigator.clipboard.writeText(feed.log);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setError("Clipboard access is unavailable in this browser.");
    }
  }
  const example = examples.find((item) => item.id === run?.example_id);
  return (
    <div className="page run-page">
      <button className="back-link" onClick={() => go("examples")}>
        <LayoutGrid size={14} /> Example library <ChevronRight size={13} />
      </button>
      <div className="page-heading">
        <div>
          <h1>{run?.title || "Opening run…"}</h1>
          <p>
            {example?.description ||
              "Live output from your HarnessX workflow."}
          </p>
        </div>
        {run && (
          <div className="heading-actions">
            <Badge status={run.status} />
            {!finished(run.status) && (
              <button className="button" disabled={busy} onClick={stop}>
                <Square size={14} /> Stop run
              </button>
            )}
          </div>
        )}
      </div>
      <ErrorNotice message={error || feed.connectionError || feed.error} />
      {feed.notice && <p className="detail-note">{feed.notice}</p>}
      {run?.url && (
        <div className="setup-note">
          <Globe size={20} />
          <div>
            <strong>The original web demo is ready</strong>
            <p>Its server stays running until you stop this run.</p>
          </div>
          <a
            className="button primary"
            href={run.url}
            target="_blank"
            rel="noreferrer"
          >
            Open web demo <ExternalLink size={15} />
          </a>
        </div>
      )}
      <div className="run-columns">
        <div>
          <section className="terminal-panel">
            <div className="terminal-bar">
              <span>
                <Terminal size={16} /> Live output
              </span>
              <div>
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={follow}
                    onChange={(e) => setFollow(e.target.checked)}
                  />{" "}
                  Follow
                </label>
                <button
                  className="icon-button"
                  onClick={copy}
                  aria-label="Copy output"
                >
                  {copied ? <Check size={15} /> : <Copy size={15} />}
                </button>
              </div>
            </div>
            <pre
              ref={logRef}
              className="terminal-output"
              aria-label="Run output"
            >
              {feed.log || "Waiting for the example to start…"}
            </pre>
            <div className="terminal-bottom">
              <span className="status-dot" />{" "}
              {run ? label(run.status) : "Connecting"}
              <span>
                {run?.created_at ? `Started ${date(run.created_at)}` : ""}
              </span>
            </div>
          </section>
          <Prompt run={run} />
        </div>
        <aside className="run-details">
          <div className="detail-panel">
            <h3>
              <FileCode2 size={17} /> About this run
            </h3>
            <dl>
              <dt>Entry point</dt>
              <dd>{example?.module || run?.example_id}</dd>
              <dt>Run ID</dt>
              <dd className="mono">{id.slice(0, 12)}</dd>
              <dt>Execution</dt>
              <dd>
                {run?.mode === "offline"
                  ? "Scripted fixture · no model calls"
                  : "Live provider"}
              </dd>
            </dl>
            <p>{example?.detail}</p>
          </div>
          <div className="detail-panel">
            <h3>
              <FolderOpen size={17} /> Files{" "}
              <span className="count-pill">{files.length}</span>
            </h3>
            {filesLoading ? (
              <p>Loading files…</p>
            ) : files.length ? (
              files.map((file) => (
                <a
                  className="file-row"
                  key={file.name}
                  href={`/api/runs/${id}/files/${file.name.split("/").map(encodeURIComponent).join("/")}`}
                  download
                >
                  <FileCode2 size={17} />
                  <span>
                    {file.name}
                    <small>{(file.size / 1024).toFixed(1)} KB</small>
                  </span>
                  <Download size={16} />
                </a>
              ))
            ) : (
              <p>Files created in this run’s working directory appear here.</p>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

// Markdown links to files this app serves become real download links; every
// other link stays as text so model output never becomes executable HTML.
const APP_LINK = /\[([^\]]+)\]\((?:sandbox:)?(\/api\/(?:chats|runs)\/[^\s)]+)\)/g;

function renderProse(text, keyPrefix) {
  const nodes = [];
  let last = 0;
  for (const match of text.matchAll(APP_LINK)) {
    if (match.index > last) nodes.push(text.slice(last, match.index));
    nodes.push(
      <a key={`${keyPrefix}-${match.index}`} className="inline-download" href={match[2]} download>
        <Download size={13} /> {match[1]}
      </a>,
    );
    last = match.index + match[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

function MessageContent({ content }) {
  return (
    <div className="message-content">
      {content.split(/(```[\s\S]*?```)/g).map((part, index) => {
        if (part.startsWith("```")) {
          const code = part.slice(3, -3);
          const newline = code.indexOf("\n");
          return (
            <pre key={index}>
              <code>{newline >= 0 ? code.slice(newline + 1) : code}</code>
            </pre>
          );
        }
        return <span key={index}>{renderProse(part, index)}</span>;
      })}
    </div>
  );
}

function FileLinks({ files = [] }) {
  if (!files.length) return null;
  return (
    <div className="message-files">
      {files.map((file) => (
        <a key={file.url} className="file-row" href={file.url} download>
          <FileCode2 size={16} />
          <span>
            {file.name}
            <small>{file.kind === "download" ? "Generated download" : "Saved in the conversation workspace"}</small>
          </span>
          <Download size={16} />
        </a>
      ))}
    </div>
  );
}

function ToolList({ tools = [] }) {
  return tools.map((tool) => (
    <details className="tool-detail" key={tool.id}>
      <summary>
        <Code2 size={14} />
        <span>{tool.name}</span>
        <Badge status={tool.status} />
        <ChevronDown size={13} />
      </summary>
      <pre>{JSON.stringify(tool.input, null, 2)}</pre>
      {tool.content && <pre>{tool.content}</pre>}
    </details>
  ));
}

const SOURCE_LABEL = { builtin: "Built-in", skill: "Skills", mcp: "MCP" };

function groupTools(tools = []) {
  const groups = [];
  for (const tool of tools) {
    const key = tool.source === "mcp" ? `mcp:${tool.server || ""}` : tool.source;
    let group = groups.find((g) => g.key === key);
    if (!group) {
      group = {
        key,
        source: tool.source,
        title:
          tool.source === "mcp"
            ? `${tool.server || "MCP"} · MCP server`
            : SOURCE_LABEL[tool.source] || tool.source,
        tools: [],
      };
      groups.push(group);
    }
    group.tools.push(tool);
  }
  return groups;
}

function ToolChips({ tools = [], showServer = false }) {
  return (
    <div className="tool-chips">
      {tools.map((tool) => (
        <span
          key={tool.name}
          className={`tool-chip ${tool.source || ""} ${tool.permission || ""}`}
          title={`${tool.description || tool.name}\nPermission: ${tool.permission}`}
        >
          {showServer && tool.server ? <em>{tool.server}/</em> : null}
          {tool.tool || tool.name}
        </span>
      ))}
    </div>
  );
}

function ToolCatalog({ tools = [], onToggle, busy = false }) {
  const groups = groupTools(tools);
  if (!groups.length) return <p className="setup-empty">No tools registered.</p>;
  return (
    <div className="tool-catalog">
      {groups.map((group) => {
        const enabled = group.tools.filter((tool) => tool.enabled !== false).length;
        return (
          <section key={group.key} className="tool-group">
            <header>
              <strong>{group.title}</strong>
              <span>
                {enabled === group.tools.length
                  ? `${group.tools.length} tools`
                  : `${enabled} of ${group.tools.length} enabled`}
              </span>
            </header>
            <ul>
              {group.tools.map((tool) => {
                const on = tool.enabled !== false;
                const fixed = tool.source === "skill";
                return (
                  <li key={tool.name} className={on ? "" : "off"}>
                    <div>
                      <code>{tool.name}</code>
                      <span className={`perm ${tool.permission}`}>{tool.permission}</span>
                      {onToggle && !fixed ? (
                        <button
                          type="button"
                          className={`switch ${on ? "on" : ""}`}
                          role="switch"
                          aria-checked={on}
                          aria-label={`${on ? "Disable" : "Enable"} ${tool.name}`}
                          disabled={busy}
                          onClick={() => onToggle(tool.name, !on)}
                        >
                          {on ? <ToggleRight size={22} /> : <ToggleLeft size={22} />}
                        </button>
                      ) : null}
                    </div>
                    <p>{tool.description || "No description provided."}</p>
                  </li>
                );
              })}
            </ul>
          </section>
        );
      })}
    </div>
  );
}

function ConversationSetup({ chat, onClose, onUpdated }) {
  const [systemPrompt, setSystemPrompt] = useState(chat.system_prompt);
  const [transport, setTransport] = useState("command");
  const [server, setServer] = useState("");
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [url, setUrl] = useState("");
  const [permission, setPermission] = useState("ask");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef(null);
  const servers = Object.entries(chat.mcp_servers || {});

  async function apply(request) {
    setBusy(true);
    setError("");
    try {
      const updated = await request();
      onUpdated(updated);
      return updated;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setBusy(false);
    }
  }
  async function savePrompt(event) {
    event.preventDefault();
    await apply(() =>
      api(`/chats/${chat.id}/setup`, {
        method: "PATCH",
        body: { system_prompt: systemPrompt },
      }),
    );
  }
  async function uploadSkill(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".md")) {
      setError("Upload a Markdown (.md) skill file.");
      return;
    }
    if (file.size > 262144) {
      setError("Skills must be 256 KB or smaller.");
      return;
    }
    const content = await file.text();
    await apply(() =>
      api(`/chats/${chat.id}/skills`, {
        method: "POST",
        body: { name: file.name, content },
      }),
    );
  }
  async function removeSkill(file) {
    await apply(() =>
      api(`/chats/${chat.id}/skills/${encodeURIComponent(file)}`, {
        method: "DELETE",
      }),
    );
  }
  async function connect(event) {
    event.preventDefault();
    await apply(async () => {
      const updated = await api(`/chats/${chat.id}/mcp`, {
        method: "POST",
        body: {
          name: server,
          command: transport === "command" ? command : "",
          args: transport === "command" ? args : "",
          url: transport === "url" ? url : "",
          permission,
        },
      });
      setServer("");
      setCommand("");
      setArgs("");
      setUrl("");
      return updated;
    });
  }
  async function disconnect(name) {
    await apply(() =>
      api(`/chats/${chat.id}/mcp/${encodeURIComponent(name)}`, {
        method: "DELETE",
      }),
    );
  }
  async function switchTool(name, enabled) {
    await apply(() =>
      api(`/chats/${chat.id}/tools`, { method: "PATCH", body: { name, enabled } }),
    );
  }
  return (
    <Modal title="Conversation setup" onClose={onClose}>
      <p className="detail-note">
        Changes apply to this conversation’s next turn. Its history stays in
        place; active responses must finish or be stopped first.
      </p>
      <form onSubmit={savePrompt}>
        <label>
          System prompt
          <textarea
            aria-label="System prompt"
            rows={5}
            value={systemPrompt}
            onChange={(event) => setSystemPrompt(event.target.value)}
          />
        </label>
        <button className="button" disabled={busy || !systemPrompt.trim()}>
          Save prompt
        </button>
      </form>
      <section className="setup-section">
        <div className="setup-section-title">
          <div>
            <Wrench size={17} />
            <strong>Tools</strong>
            <span className="count">{(chat.tools || []).length}</span>
          </div>
        </div>
        <p className="detail-copy">
          Everything the model can call in this conversation, with the
          permission that applies when it does. Switch a tool off to hide it
          from the model on the next turn; MCP tools are listed under the
          server that provides them.
        </p>
        <ToolCatalog tools={chat.tools} onToggle={switchTool} busy={busy} />
      </section>
      <section className="setup-section">
        <div className="setup-section-title">
          <div>
            <Sparkles size={17} />
            <strong>Skills</strong>
          </div>
          <button
            type="button"
            className="button"
            disabled={busy}
            onClick={() => fileRef.current?.click()}
          >
            <Upload size={15} /> Upload skill
          </button>
          <input
            ref={fileRef}
            className="visually-hidden"
            type="file"
            accept=".md,text/markdown"
            onChange={uploadSkill}
          />
        </div>
        <p className="detail-copy">
          Upload a Markdown skill with optional <code>name</code> and{" "}
          <code>description</code> frontmatter. The agent loads its full
          instructions only when it calls the Skill tool.
        </p>
        {chat.skills?.length ? (
          <div className="setup-list">
            {chat.skills.map((skill) => (
              <div key={skill.file} className="setup-row">
                <div>
                  <strong>{skill.name}</strong>
                  <span>{skill.description}</span>
                  <small>
                    {skill.file} · {skill.body_chars} characters
                  </small>
                </div>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`Remove skill ${skill.name}`}
                  disabled={busy}
                  onClick={() => removeSkill(skill.file)}
                >
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p className="setup-empty">
            No uploaded skills in this conversation.
          </p>
        )}
      </section>
      <section className="setup-section">
        <div className="setup-section-title">
          <div>
            <Plug size={17} />
            <strong>MCP servers</strong>
          </div>
        </div>
        <p className="detail-copy">
          Connect a local stdio server or a remote HTTP MCP endpoint. MCP tools
          default to asking for approval at execution time.
        </p>
        {servers.length ? (
          <div className="setup-list">
            {servers.map(([name, serverInfo]) => (
              <div key={name} className="setup-row">
                <div>
                  <strong>{name}</strong>
                  <span>
                    {serverInfo.transport} · {serverInfo.tools.length} tools
                  </span>
                  {serverInfo.tools.length ? (
                    <ToolChips
                      tools={(chat.tools || []).filter(
                        (tool) => tool.source === "mcp" && tool.server === name && tool.enabled !== false,
                      )}
                    />
                  ) : (
                    <small>No tools discovered</small>
                  )}
                </div>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`Disconnect MCP server ${name}`}
                  disabled={busy}
                  onClick={() => disconnect(name)}
                >
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
          </div>
        ) : null}
        <form className="mcp-form" onSubmit={connect}>
          <label>
            Server name
            <input
              required
              value={server}
              onChange={(event) => setServer(event.target.value)}
              placeholder="filesystem"
            />
          </label>
          <label>
            Connection
            <select
              value={transport}
              onChange={(event) => setTransport(event.target.value)}
            >
              <option value="command">Local command (stdio)</option>
              <option value="url">Remote HTTP / SSE URL</option>
            </select>
          </label>
          {transport === "command" ? (
            <>
              <label>
                Command
                <input
                  required
                  value={command}
                  onChange={(event) => setCommand(event.target.value)}
                  placeholder="npx"
                />
              </label>
              <label>
                Arguments
                <input
                  value={args}
                  onChange={(event) => setArgs(event.target.value)}
                  placeholder="-y @scope/server /workspace"
                />
              </label>
            </>
          ) : (
            <label>
              Endpoint URL
              <input
                required
                type="url"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="https://example.test/mcp"
              />
            </label>
          )}
          <label>
            Tool permission
            <select
              value={permission}
              onChange={(event) => setPermission(event.target.value)}
            >
              <option value="ask">Ask before every call</option>
              <option value="allow">Allow calls</option>
              <option value="deny">Deny calls</option>
            </select>
          </label>
          <button className="button primary" disabled={busy}>
            <Plug size={15} /> Connect MCP
          </button>
        </form>
      </section>
      <ErrorNotice message={error} onDismiss={() => setError("")} />
    </Modal>
  );
}

function ChatView({ id, providers, refresh, onNew }) {
  const [chat, setChat] = useState(null);
  const [runId, setRunId] = useState(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const [sending, setSending] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState("");
  const [model, setModel] = useState("");
  const [files, setFiles] = useState([]);
  const [showFiles, setShowFiles] = useState(false);
  const [filesLoading, setFilesLoading] = useState(false);
  const [showSetup, setShowSetup] = useState(false);
  const scrollRef = useRef(null);
  const composerRef = useRef(null);
  const load = useCallback(async () => {
    if (id) {
      const current = await api(`/chats/${id}`);
      setChat(current);
      refresh();
    }
  }, [id, refresh]);
  const feed = useFeed(runId, () => {
    load().catch((err) => setError(err.message));
  });
  useEffect(() => {
    let disposed = false;
    setChat(null);
    setRunId(null);
    setDraft("");
    setError("");
    setSending(false);
    if (id)
      api(`/chats/${id}`)
        .then((current) => {
          if (!disposed) {
            setChat(current);
            setRunId(current.active_run);
          }
        })
        .catch((err) => {
          if (!disposed) setError(err.message);
        });
    return () => {
      disposed = true;
    };
  }, [id]);
  useEffect(() => {
    if (!selectedProvider && providers.length) {
      const first = providers.find((p) => !p.missing.length) || providers[0];
      setSelectedProvider(first.id);
      setModel(first.model);
    }
  }, [providers, selectedProvider]);
  const active =
    sending || (runId && (!feed.run || !finished(feed.run.status)));
  const fileRun = runId || chat?.latest_run;
  useEffect(() => {
    if (!fileRun || !showFiles) return;
    let disposed = false;
    setFilesLoading(true);
    api(`/runs/${fileRun}/files`)
      .then((data) => {
        if (!disposed) setFiles(data);
      })
      .catch((err) => {
        if (!disposed) setError(err.message);
      })
      .finally(() => {
        if (!disposed) setFilesLoading(false);
      });
    return () => {
      disposed = true;
    };
  }, [fileRun, showFiles, feed.run?.status]);
  useEffect(() => {
    if (scrollRef.current)
      scrollRef.current.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [feed.text, chat?.messages.length, feed.run?.pending]);
  async function send(event, suggestion) {
    event?.preventDefault();
    const text = (suggestion ?? draft).trim();
    if (!text || active) return;
    setSending(true);
    setError("");
    try {
      let target = chat;
      if (!target)
        target = await api("/chats", {
          method: "POST",
          body: { provider: selectedProvider, model },
        });
      const run = await api(`/chats/${target.id}/messages`, {
        method: "POST",
        body: { message: text },
      });
      setChat({
        ...target,
        messages: [
          ...target.messages,
          { role: "user", content: text },
          {
            role: "assistant",
            content: "",
            tools: [],
            run_id: run.id,
            status: "running",
          },
        ],
      });
      setRunId(run.id);
      setDraft("");
      refresh();
      if (!id) {
        go(`chat/${target.id}`);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setSending(false);
    }
  }
  async function stop() {
    if (!runId) return;
    try {
      await api(`/runs/${runId}/stop`, { method: "POST" });
      await load();
    } catch (err) {
      setError(err.message);
    }
  }
  async function remove() {
    try {
      await api(`/chats/${id}`, { method: "DELETE" });
      refresh();
      go("chat");
    } catch (err) {
      setError(err.message);
    }
  }
  async function openSetup() {
    setError("");
    try {
      let target = chat;
      if (!target) {
        target = await api("/chats", {
          method: "POST",
          body: { provider: selectedProvider, model },
        });
        setChat(target);
        refresh();
      }
      setShowSetup(true);
    } catch (err) {
      setError(err.message);
    }
  }
  function updateSetup(updated) {
    setChat(updated);
    refresh();
  }
  const provider = providers.find(
    (p) => p.id === (chat?.provider || selectedProvider),
  );
  const enabledTools = (chat?.tools || []).filter((tool) => tool.enabled !== false);
  const visibleMessages = (chat?.messages || []).map((message) =>
    message.run_id === runId && feed.seq > 0
      ? {
          ...message,
          content: feed.text,
          tools: feed.tools,
          files: feed.files?.length ? feed.files : message.files,
          error: feed.error,
          status: feed.run?.status,
        }
      : message,
  );
  return (
    <div className="chat-page">
      <div className="chat-toolbar">
        <div>
          <MessageSquare size={19} />
          <strong>General chat</strong>
          <span className="toolbar-divider" />{" "}
          <span>{chat?.model || model || "Choose a model"}</span>
          {(chat?.provider || selectedProvider) === "demo" && (
            <span className="mode-label offline">Local fixture</span>
          )}
          {chat?.tools?.length ? (
            <button
              type="button"
              className="toolbar-chip"
              title="Show the tools this conversation can use"
              disabled={!!provider?.missing.length || !model.trim() || active}
              onClick={openSetup}
            >
              <Wrench size={13} />
              {enabledTools.length} tools
              {enabledTools.some((tool) => tool.source === "mcp")
                ? ` · ${enabledTools.filter((tool) => tool.source === "mcp").length} via MCP`
                : ""}
              {chat.tools.length > enabledTools.length
                ? ` · ${chat.tools.length - enabledTools.length} off`
                : ""}
            </button>
          ) : null}
        </div>
        <div>
          {fileRun && (
            <button
              className="icon-button"
              aria-label="Show conversation files"
              aria-pressed={showFiles}
              onClick={() => setShowFiles(!showFiles)}
            >
              <FolderOpen size={16} />
            </button>
          )}
          <button
            className="icon-button"
            aria-label="Conversation setup"
            disabled={!!provider?.missing.length || !model.trim() || active}
            onClick={openSetup}
          >
            <Settings2 size={16} />
          </button>
          {id && (
            <button
              className="icon-button"
              aria-label="Delete conversation"
              onClick={remove}
            >
              <Trash2 size={16} />
            </button>
          )}
          <button className="button" onClick={onNew}>
            <Plus size={15} /> New chat
          </button>
        </div>
      </div>
      {showSetup && chat && (
        <ConversationSetup
          chat={chat}
          onClose={() => setShowSetup(false)}
          onUpdated={updateSetup}
        />
      )}
      <div className="chat-scroll">
        <div className="chat-content">
          {showFiles && (
            <div className="detail-panel chat-files">
              <h3>
                <FolderOpen size={17} /> Conversation files
              </h3>
              {filesLoading ? (
                <p>Loading files…</p>
              ) : files.length ? (
                files.map((file) => (
                  <a
                    key={file.name}
                    className="file-row"
                    download
                    href={`/api/runs/${fileRun}/files/${file.name.split("/").map(encodeURIComponent).join("/")}`}
                  >
                    <FileCode2 size={16} />
                    <span>{file.name}</span>
                    <Download size={16} />
                  </a>
                ))
              ) : (
                <p>Files created by this conversation appear here.</p>
              )}
            </div>
          )}
          {id && !chat && !error ? (
            <div className="empty">
              <LoaderCircle className="spin" size={22} /> Loading conversation…
            </div>
          ) : !visibleMessages.length ? (
            <div className="chat-empty">
              <div className="chat-orbit">
                <Mark />
              </div>
              <div className="eyebrow">YOUR HARNESS-X ASSISTANT</div>
              <h1>
                A little curiosity.
                <br />A lot of possibility.
              </h1>
              <p>
                Think through an idea, work with code, or ask a question.
                <br />
                Your conversation has its own tools and workspace.
              </p>
              <div className="suggestions">
                {[
                  ["Run a calculation", "What is 48 * 12?"],
                  [
                    "Understand the runtime",
                    "Explain how an agent decides when to call a tool.",
                  ],
                  [
                    "Think through a problem",
                    "Help me design a reliable data analysis workflow.",
                  ],
                ].map(([title, prompt]) => (
                  <button
                    key={title}
                    onClick={() => {
                      setDraft(prompt);
                      composerRef.current?.focus();
                    }}
                  >
                    <span>{title}</span>
                    <p>{prompt}</p>
                    <ArrowUp size={15} />
                  </button>
                ))}
              </div>
              {chat?.tools?.length ? (
                <div className="toolbox">
                  <div className="toolbox-title">
                    <Wrench size={13} /> Available tools
                    <span>
                      {enabledTools.length}
                      {chat.tools.some((tool) => tool.source === "mcp")
                        ? ` · ${Object.keys(chat.mcp_servers || {}).length} MCP server${Object.keys(chat.mcp_servers || {}).length === 1 ? "" : "s"}`
                        : ""}
                    </span>
                  </div>
                  <ToolChips tools={enabledTools} showServer />
                </div>
              ) : null}
            </div>
          ) : (
            visibleMessages.map((message, index) => (
              <div
                className={`chat-message ${message.role}`}
                key={message.run_id || `message-${index}`}
              >
                <div className="message-avatar">
                  {message.role === "assistant" ? <Mark small /> : "Y"}
                </div>
                <div className="message-body">
                  <div className="message-label">
                    {message.role === "assistant" ? "Harness" : "You"}
                    {message.role === "assistant" &&
                      message.status === "cancelled" && <span>Stopped</span>}
                  </div>
                  <MessageContent content={message.content || ""} />
                  <ToolList tools={message.tools} />
                  <FileLinks files={message.files} />
                  {message.role === "assistant" &&
                    !message.content &&
                    active &&
                    message.run_id === runId && (
                      <div className="thinking">
                        <span />
                        <span />
                        <span />
                      </div>
                    )}
                  {message.error && <ErrorNotice message={message.error} />}
                </div>
              </div>
            ))
          )}
          {feed.notice && <p className="detail-note">{feed.notice}</p>}
          <Prompt run={feed.run} />
          <div ref={scrollRef} />
        </div>
      </div>
      <div className="composer-wrap">
        <ErrorNotice
          message={error || feed.connectionError}
          onDismiss={() => setError("")}
        />
        {!id && (
          <div className="model-settings">
            <label>
              <Settings2 size={14} /> Provider
              <select
                aria-label="Chat provider"
                value={selectedProvider}
                onChange={(e) => {
                  setSelectedProvider(e.target.value);
                  setModel(
                    providers.find((p) => p.id === e.target.value)?.model || "",
                  );
                }}
              >
                {providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                    {p.missing.length ? " · setup needed" : ""}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Model
              <input
                aria-label="Model ID"
                value={model}
                disabled={selectedProvider === "demo"}
                onChange={(e) => setModel(e.target.value)}
                placeholder="Enter a model ID"
              />
            </label>
          </div>
        )}
        {!!provider?.missing.length && (
          <div className="setup-note compact">
            Configure {provider.missing.join(", ")} in the server environment,
            or choose Local demo.
          </div>
        )}
        <form className="composer" onSubmit={send}>
          <textarea
            aria-label="Message your agent"
            ref={composerRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Message your agent…"
            rows={2}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (!active) send(e);
              }
            }}
          />
          <div className="composer-bottom">
            <span>
              <ShieldCheck size={14} /> File changes ask for approval
            </span>
            {active ? (
              <button
                type="button"
                className="send-button stop"
                aria-label="Stop response"
                onClick={stop}
              >
                <Square size={17} />
              </button>
            ) : (
              <button
                className="send-button"
                type="submit"
                aria-label="Send message"
                disabled={
                  !draft.trim() ||
                  !selectedProvider ||
                  !!provider?.missing.length ||
                  !model.trim()
                }
              >
                <ArrowUp size={19} />
              </button>
            )}
          </div>
        </form>
        <div className="composer-caption">
          {(chat?.provider || selectedProvider) === "demo"
            ? "Local demo · scripted responses and a real calculator · no model calls"
            : "Powered by HarnessX · API usage applies"}
          <span>Enter to send · Shift + Enter for a new line</span>
        </div>
      </div>
    </div>
  );
}

function RunsList({ runs }) {
  return (
    <div className="page">
      <div className="eyebrow">WORKSPACE ACTIVITY</div>
      <div className="page-heading">
        <div>
          <h1>Recent runs</h1>
          <p>
            Return to an example, pick up an approval, or inspect the output.
          </p>
        </div>
      </div>
      {!runs.length ? (
        <div className="empty">
          <History size={32} />
          <h3>Your first run starts here</h3>
          <p>Launch an example to see it in your activity.</p>
          <button className="button primary" onClick={() => go("examples")}>
            Explore examples <ArrowRight size={15} />
          </button>
        </div>
      ) : (
        <div className="runs-table">
          {runs.map((run) => (
            <button
              className="run-row"
              key={run.id}
              onClick={() =>
                go(run.chat_id ? `chat/${run.chat_id}` : `runs/${run.id}`)
              }
            >
              <div className="example-icon">
                {run.chat_id ? (
                  <MessageSquare size={19} />
                ) : (
                  <Terminal size={19} />
                )}
              </div>
              <div>
                <strong>{run.title}</strong>
                <span>
                  {run.example_id === "chat"
                    ? "General chat"
                    : `examples.${run.example_id}`}{" "}
                  · {date(run.created_at)}
                </span>
              </div>
              <Badge status={run.status} />
              <ChevronRight size={17} />
            </button>
          ))}
        </div>
      )}
      <p className="retention-note">
        Run history and conversations are kept while this server is running.
        Generated files remain on disk.
      </p>
    </div>
  );
}

export default function App() {
  const [route, setRoute] = useState(
    window.location.hash.slice(1) || "examples",
  );
  const [examples, setExamples] = useState([]);
  const [health, setHealth] = useState(null);
  const [runs, setRuns] = useState([]);
  const [chats, setChats] = useState([]);
  const [selected, setSelected] = useState(null);
  const [settings, setSettings] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [page, id] = route.split("/");
  const refresh = useCallback(async () => {
    try {
      const [nextRuns, nextChats] = await Promise.all([
        api("/runs"),
        api("/chats"),
      ]);
      setRuns(nextRuns);
      setChats(nextChats);
    } catch (err) {
      setError(err.message);
    }
  }, []);
  const connect = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [catalog, status] = await Promise.all([
        api("/examples"),
        api("/health"),
      ]);
      setExamples(catalog);
      setHealth(status);
      await refresh();
    } catch (err) {
      setHealth(null);
      setError(`Cannot connect to harness-web. ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [refresh]);
  useEffect(() => {
    connect();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [connect, refresh]);
  useEffect(() => {
    const change = () => {
      setRoute(window.location.hash.slice(1) || "examples");
      setSidebarOpen(false);
    };
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  useEffect(() => {
    const search = (e) => {
      if (
        e.key === "/" &&
        !["INPUT", "TEXTAREA", "SELECT"].includes(
          document.activeElement.tagName,
        ) &&
        page === "examples" &&
        !selected &&
        !settings
      ) {
        e.preventDefault();
        document.querySelector('[aria-label="Search examples"]')?.focus();
      }
    };
    document.addEventListener("keydown", search);
    return () => document.removeEventListener("keydown", search);
  }, [page, selected, settings]);
  const activeCount = runs.filter((run) => !finished(run.status)).length;
  return (
    <div className="app-shell">
      <button
        className="mobile-toggle icon-button"
        aria-label="Toggle navigation"
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        <Menu size={22} />
      </button>
      {sidebarOpen && (
        <div className="sidebar-scrim" onClick={() => setSidebarOpen(false)} />
      )}
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <a className="brand" href="#examples" aria-label="harness-web home">
          <img className="brand-logo" src="/harnessx-logo.svg" width="148" height="42" alt="Harness-X" />
          <span className="brand-caption">harness-web · by DataGOL</span>
        </a>
        <div className="workspace-label">
          <span className="live-dot" /> Local workspace{" "}
          <ChevronDown size={13} />
        </div>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          <button
            className={page === "examples" ? "active" : ""}
            onClick={() => go("examples")}
          >
            <LayoutGrid size={18} /> Examples{" "}
            <span className="nav-count">{examples.length || "—"}</span>
          </button>
          <button
            className={page === "chat" ? "active" : ""}
            onClick={() => go("chat")}
          >
            <MessageSquare size={18} /> General chat{" "}
            <Plus size={14} className="nav-end" />
          </button>
          <button
            className={page === "runs" ? "active" : ""}
            onClick={() => go("runs")}
          >
            <History size={18} /> Recent runs{" "}
            {!!activeCount && (
              <span className="nav-count running-count">{activeCount}</span>
            )}
          </button>
        </nav>
        {!!chats.length && (
          <>
            <div className="nav-label section-label">CONVERSATIONS</div>
            <div className="conversation-nav">
              {chats.slice(0, 8).map((chat) => (
                <button
                  key={chat.id}
                  className={id === chat.id ? "active" : ""}
                  onClick={() => go(`chat/${chat.id}`)}
                >
                  <MessageSquare size={14} />
                  <span>{chat.title}</span>
                </button>
              ))}
            </div>
          </>
        )}
        <div className="sidebar-bottom">
          <div className="sdk-note">
            <Zap size={17} />
            <strong>One SDK. Many possibilities.</strong>
            <p>
              Explore the engine behind
              <br />
              your next agent workflow.
            </p>
          </div>
          <button className="settings-link" onClick={() => setSettings(true)}>
            <Settings2 size={17} /> Workspace setup
          </button>
          <div className="connection-status">
            <span className={`status-dot ${health ? "online" : ""}`} />
            <span>{health ? "Runtime connected" : "Runtime disconnected"}</span>
            <span className="version">LOCAL</span>
          </div>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div>
            <span>Workspace</span>
            <ChevronRight size={14} />
            <strong>
              {page === "chat"
                ? "General chat"
                : page === "runs"
                  ? "Recent runs"
                  : "Examples"}
            </strong>
          </div>
          <div className="topbar-right">
            <span className="sdk-tag">HarnessX SDK</span>
            <button
              className="avatar"
              onClick={() => setSettings(true)}
              aria-label="Workspace settings"
            >
              HX
            </button>
          </div>
        </header>
        {error && (
          <div className="global-error">
            <ErrorNotice message={error} onDismiss={() => setError("")} />
            {!health && (
              <button className="button" onClick={connect}>
                Retry connection
              </button>
            )}
          </div>
        )}
        {page === "chat" ? (
          <ChatView
            key={id || "new"}
            id={id}
            providers={health?.providers || []}
            refresh={refresh}
            onNew={() => go("chat")}
          />
        ) : page === "runs" && id ? (
          <RunView id={id} refresh={refresh} examples={examples} />
        ) : page === "runs" ? (
          <RunsList runs={runs} />
        ) : (
          <Library
            examples={examples}
            loading={loading}
            onSelect={setSelected}
          />
        )}
      </main>
      {selected && (
        <LaunchDialog
          example={selected}
          providers={health?.providers || []}
          onClose={() => setSelected(null)}
          onLaunch={(run) => {
            setSelected(null);
            refresh();
            go(`runs/${run.id}`);
          }}
        />
      )}
      {settings && (
        <Modal title="Workspace setup" onClose={() => setSettings(false)}>
          <p className="detail-note">
            Credentials are read from the repository’s <code>.env</code> file or
            the server environment. Restart the server after changing them.
          </p>
          <div className="provider-list">
            {(health?.providers || []).map((provider) => (
              <div key={provider.id}>
                <div>
                  <strong>{provider.name}</strong>
                  <span>
                    {provider.missing.length
                      ? provider.missing.join(", ")
                      : provider.id === "demo"
                        ? "Scripted responses · no credentials"
                        : "Credentials and SDK available"}
                  </span>
                </div>
                <Badge
                  status={provider.missing.length ? "waiting" : "completed"}
                >
                  {provider.missing.length ? "Setup needed" : "Ready"}
                </Badge>
              </div>
            ))}
          </div>
          <div className="setup-note">
            <ShieldCheck size={19} />
            <p>
              This is a local development workspace. Examples retain their own
              tool permissions; separate working directories do not sandbox host
              access.
            </p>
          </div>
          <div className="modal-footer">
            <span className="muted">
              Service readiness is checked when you run.
            </span>
            <button
              className="button primary"
              onClick={() => {
                connect();
                setSettings(false);
              }}
            >
              Refresh status
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
