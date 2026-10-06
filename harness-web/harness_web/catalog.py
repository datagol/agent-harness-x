"""The explicit launch allowlist. Each entry runs an example script by its path."""

from dataclasses import asdict, dataclass
from importlib.util import find_spec
import os
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[2]


EXAMPLES_DIR = ROOT / "examples"

# Folder -> the topic the library groups by, in reading order.
TOPICS = {
    "01-basics": "Basics",
    "02-tools": "Tools",
    "03-context": "Context",
    "04-control": "Control",
    "05-durability": "Durability",
    "06-quality": "Quality",
    "07-sandboxes": "Sandboxes",
}


@dataclass(frozen=True)
class Example:
    id: str
    path: str  # relative to examples/; every example is a script run by path
    title: str
    description: str
    icon: str
    interactive: bool = False
    offline: bool = False
    extra: str = ""
    prompt: str = ""
    detail: str = ""

    @property
    def file(self):
        return EXAMPLES_DIR / self.path

    @property
    def category(self):
        return TOPICS[self.path.split("/", 1)[0]]

    @property
    def command(self):
        return f"python examples/{self.path}"


EXAMPLES = [
    # ── Basics ──
    Example(
        "streaming_chat",
        "01-basics/streaming_chat.py",
        "Streaming chat",
        "A streaming assistant with arithmetic and workspace inspection tools.",
        "message",
        interactive=True,
        prompt="What is (48 * 12) / 6?",
        detail="Read-only filesystem tools are scoped to this run's working directory. Type usage to see token statistics.",
    ),
    Example(
        "switching_providers",
        "01-basics/switching_providers.py",
        "Switching providers",
        "Run the same agent with Anthropic, OpenAI, Gemini, or OpenRouter.",
        "message",
        interactive=True,
        prompt="What is 40 * 10?",
        detail="Choose a provider and a model available to your account. The same tool and agent loop, with streaming or ordinary responses.",
    ),
    Example(
        "prompt_caching",
        "01-basics/prompt_caching.py",
        "Prompt caching",
        "Keep the prompt prefix stable and watch cache reads arrive across iterations.",
        "activity",
        offline=True,
        detail="Runs a two-iteration loop with a scripted provider that reports cache writes and reads; no model calls.",
    ),
    Example(
        "progress_and_waiting",
        "01-basics/progress_and_waiting.py",
        "Progress and waiting",
        "Follow a long run's event stream, including the events sent while a slow tool works.",
        "activity",
        offline=True,
        detail="A scripted model calls a deliberately slow tool; every event is printed as it arrives. No model calls.",
    ),
    # ── Tools ──
    Example(
        "filesystem_tools_and_permissions",
        "02-tools/filesystem_tools_and_permissions.py",
        "Filesystem tools and permissions",
        "Write and inspect code with the file and shell tools, each write approved first.",
        "code",
        interactive=True,
        prompt="Write a Python program to calculate Fibonacci numbers.",
        detail="Writes and shell commands ask for approval in the browser. The working directory is separate, but host tools are not an OS sandbox.",
    ),
    Example(
        "mcp_servers",
        "02-tools/mcp_servers.py",
        "MCP servers",
        "Connect a local or remote MCP server and use its tools in a conversation.",
        "plug",
        interactive=True,
        extra="mcp",
        prompt="List the tools available to you.",
        detail="Configure a stdio command or an SSE URL below. Tool calls keep terminal approvals, presented in the browser.",
    ),
    Example(
        "skills_lazy_loading",
        "02-tools/skills_lazy_loading.py",
        "Skills, loaded on demand",
        "Watch an agent find and load a code-review skill, with every hook visible.",
        "sparkles",
        offline=True,
        detail="A scripted provider loads the real code-review skill. No API key is needed.",
    ),
    Example(
        "skills_interactive",
        "02-tools/skills_interactive.py",
        "Skills with a live model",
        "Let a live model select engineering skills when the task calls for them.",
        "sparkles",
        interactive=True,
        prompt="Review this code: def divide(a, b): return a / b",
        detail="Loads the repository's review, commit-message, bug-triage, and SQL skills. Uses the configured default provider.",
    ),
    # ── Context ──
    Example(
        "knowledge_bundles_okf",
        "03-context/knowledge_bundles_okf.py",
        "Knowledge bundles (OKF)",
        "Answer questions from an Open Knowledge Format bundle with search, concept reads, and link traversal.",
        "memory",
        interactive=True,
        prompt="How are active users defined, and which table do they come from?",
        detail="Loads examples/knowledge, an OKF v0.2 bundle of four linked concepts; set AGENT_KNOWLEDGE to a folder or git URL to use another. Uses the configured default provider.",
    ),
    Example(
        "custom_memory_tools",
        "03-context/custom_memory_tools.py",
        "Memory tools",
        "Save notes with the built-in memory tools, and add a tool of your own beside them.",
        "memory",
        interactive=True,
        prompt="Remember that this project's name is HarnessX.",
        detail="Notes and facts are stored in this run's working directory.",
    ),
    Example(
        "planning_todos",
        "03-context/planning_todos.py",
        "Planning with to-dos",
        "Watch an agent write a task list, work through it, and mark items done.",
        "list",
        offline=True,
        detail="A scripted model drives write_todos and read_todos; every list change is printed. No model calls.",
    ),
    Example(
        "condensing_a_long_history",
        "03-context/condensing_a_long_history.py",
        "Condensing a long history",
        "See what survives when a conversation outgrows its context window.",
        "layers",
        offline=True,
        detail="A small context limit forces old tool output to be cleared, then a summary. No model calls.",
    ),
    # ── Control ──
    Example(
        "loop_guard",
        "04-control/loop_guard.py",
        "Loop guard",
        "Catch an agent repeating the same tool call and tell it so, without failing the run.",
        "repeat",
        offline=True,
        detail="A scripted model repeats one call; the repetition hook fires and the model reads the note. No model calls.",
    ),
    Example(
        "retries_and_fallback",
        "04-control/retries_and_fallback.py",
        "Retries and fallback",
        "Retry a failing model, then hand the conversation to a fallback provider.",
        "refresh",
        offline=True,
        detail="Two scripted providers: the first fails with a transient error, the second answers. No model calls.",
    ),
    Example(
        "delegating_to_subagents",
        "04-control/delegating_to_subagents.py",
        "Delegating to subagents",
        "Delegate research, code review, and file analysis to focused specialists.",
        "network",
        prompt="Review this Python code: def add(a, b): return a - b",
        detail="Constructor-defined specialists with a fresh conversation per task. Delegation can make several model calls.",
    ),
    Example(
        "autonomous_task_runner",
        "04-control/autonomous_task_runner.py",
        "Autonomous task runner",
        "Hand it one task on the command line; it writes the deliverable and runs it.",
        "terminal",
        prompt="Write a Python script that prints the first 10 Fibonacci numbers, run it, and confirm the output.",
        detail="One-shot rather than a conversation. It carries the file and shell tools. Web search and tracing switch on only when their keys are set.",
    ),
    # ── Durability ──
    Example(
        "tool_approvals_and_resume",
        "05-durability/tool_approvals_and_resume.py",
        "Approvals and resume",
        "Pause a run on a tool approval, then resume its persisted execution.",
        "shield",
        offline=True,
        detail="Approve or deny a note creation in the browser. Uses a temporary SQLite database.",
    ),
    Example(
        "durable_crash_recovery",
        "05-durability/durable_crash_recovery.py",
        "Durable runs on PostgreSQL",
        "Stream a model response with execution state stored in PostgreSQL.",
        "database",
        extra="psycopg",
        detail="Requires DATABASE_URL for an existing database. The example provisions HarnessX tables.",
    ),
    Example(
        "session_snapshots",
        "05-durability/session_snapshots.py",
        "Session snapshots",
        "Save a conversation, load it into a fresh agent, and carry on.",
        "save",
        offline=True,
        detail="The step before a durable runtime: a snapshot restores the conversation, not a run in flight. No model calls.",
    ),
    # ── Quality ──
    Example(
        "evaluating_with_datasets",
        "06-quality/evaluating_with_datasets.py",
        "Evaluating with datasets",
        "Run arithmetic and tool-selection cases and inspect their scores.",
        "flask",
        offline=True,
        extra="langsmith",
        detail="Choose the scripted, offline fixture or the live model. Live mode uploads results if a LangSmith key is configured.",
    ),
    Example(
        "decisions_routing",
        "06-quality/decisions_routing.py",
        "Decisions: routing and classification",
        "Use a Choice decision to pick an agent, and to classify a document.",
        "network",
        offline=True,
        detail="Fixed offline decisions and scripted agent responses. Demonstrates a 0.8 confidence threshold; applications must calibrate their own.",
    ),
    Example(
        "decisions_answer_review",
        "06-quality/decisions_answer_review.py",
        "Decisions: answer review",
        "Review answer coverage with Score and evidence support with Noul.",
        "flask",
        offline=True,
        detail="Fixed review results after a completed scripted agent run. The review leaves the original answer and run status unchanged.",
    ),
    Example(
        "tracing_with_langsmith",
        "06-quality/tracing_with_langsmith.py",
        "Tracing with LangSmith",
        "Trace an agent's model calls and tool call as a connected run tree.",
        "activity",
        extra="langsmith",
        detail="Requires Anthropic and LangSmith credentials. This example sends run data to your configured LangSmith project.",
    ),
    Example(
        "flight_recorder",
        "06-quality/flight_recorder.py",
        "Flight recorder",
        "Follow an invoice analysis through a failure, recovery, and offline playback.",
        "record",
        offline=True,
        detail="Exports a downloadable .hx incident bundle. Uses synthetic invoices and a scripted provider; no model calls.",
    ),
    # ── Sandboxes ──
    Example(
        "sandbox_isolation_tiers",
        "07-sandboxes/sandbox_isolation_tiers.py",
        "Sandbox isolation tiers",
        "Execute Python under process resource limits with a durable runtime.",
        "terminal",
        interactive=True,
        prompt="Use Python to calculate the first 10 Fibonacci numbers.",
        detail="Uses the process tier: resource limits, with host filesystem and network access. Docker and Seatbelt are the isolating tiers.",
    ),
]
CATALOG = {example.id: example for example in EXAMPLES}
PROVIDERS = {
    "anthropic": ("Anthropic", "ANTHROPIC_API_KEY", "anthropic", "claude-sonnet-4-6"),
    "openai": ("OpenAI", "OPENAI_API_KEY", "openai", ""),
    "gemini": ("Gemini", "GEMINI_API_KEY", "google.genai", ""),
    "openrouter": ("OpenRouter", "OPENROUTER_API_KEY", "openai", ""),
}


def installed(module):
    try:
        return find_spec(module) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def provider_missing(provider):
    if provider == "demo":
        return []
    if provider not in PROVIDERS:
        raise ValueError("Unknown provider")
    _, key, module, _ = PROVIDERS[provider]
    missing = []
    if not (
        os.environ.get(key)
        or (provider == "gemini" and os.environ.get("GOOGLE_API_KEY"))
    ):
        missing.append(key)
    if not installed(module):
        missing.append(f"Python package: {module}")
    return missing


def requirements(example, mode="offline", *, provider="anthropic"):
    missing = []
    if example.extra and not installed(example.extra):
        missing.append(f"Python package: {example.extra}")
    if not example.offline or (example.id == "evaluating_with_datasets" and mode == "live"):
        if example.id in ("skills_interactive", "knowledge_bundles_okf"):
            provider = os.getenv("AGENT_PROVIDER", "anthropic")
        elif example.id != "switching_providers":
            provider = "anthropic"
        missing.extend(provider_missing(provider))
        if (
            example.id in ("skills_interactive", "knowledge_bundles_okf")
            and provider != "anthropic"
            and not os.getenv("AGENT_MODEL")
        ):
            missing.append("AGENT_MODEL")
    if example.id == "durable_crash_recovery" and not os.getenv("DATABASE_URL"):
        missing.append("DATABASE_URL")
    if example.id == "tracing_with_langsmith" and not (
        os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    ):
        missing.append("LANGSMITH_API_KEY")
    return missing


def public_catalog():
    return [
        {
            **asdict(item),
            "category": item.category,
            "command": item.command,
            "missing": requirements(item),
            "live_missing": requirements(item, "live"),
        }
        for item in EXAMPLES
    ]


def arguments(example, config, workdir):
    """Build argument vectors, never shell strings supplied by the browser."""
    if example.id == "decisions_routing":
        # Fixed offline demonstration; live routing requires explicit CLI configuration.
        return ["--min-confidence", "0.8"]
    if example.id == "switching_providers":
        provider = config.get("provider", "anthropic")
        if provider not in PROVIDERS:
            raise ValueError("Unknown provider")
        model = config.get("model", "").strip()
        if provider != "anthropic" and not model:
            raise ValueError("Enter a model ID for the selected provider")
        args = ["--provider", provider]
        if model:
            args += ["--model=" + model]
        if not config.get("streaming", True):
            args += ["--no-stream"]
        return args
    if example.id == "flight_recorder":
        return ["--output", str(workdir / "incident.hx")]
    if example.id == "evaluating_with_datasets":
        if config.get("mode", "offline") not in ("offline", "live"):
            raise ValueError("Evaluation mode must be offline or live")
        return ["--offline"] if config.get("mode", "offline") == "offline" else []
    if example.id == "mcp_servers":
        name = config.get("server", "").strip()
        command, url = config.get("command", "").strip(), config.get("url", "").strip()
        if not name or bool(command) == bool(url):
            raise ValueError(
                "MCP needs a server name and exactly one command or SSE URL"
            )
        if url and not url.startswith(("http://", "https://")):
            raise ValueError("MCP URL must use http or https")
        args = ["--server", name, "--permission", "ask"]
        if command:
            shlex.split(config.get("args", ""))  # Validate quoting before launch.
            args += ["--command", command, "--args=" + config.get("args", "")]
        else:
            args += ["--url", url]
        return args
    return []
