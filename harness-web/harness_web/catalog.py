"""The explicit launch allowlist. Each entry invokes an existing example."""

from dataclasses import asdict, dataclass
from importlib.util import find_spec
import os
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Example:
    id: str
    title: str
    description: str
    category: str
    icon: str
    interactive: bool = False
    offline: bool = False
    extra: str = ""
    prompt: str = ""
    detail: str = ""

    @property
    def module(self):
        return f"examples.{self.id}"


EXAMPLES = [
    Example(
        "jev_routing",
        "Jev routing",
        "Use a Choice decision to select one SQL, research, or general agent.",
        "Evaluation",
        "network",
        offline=True,
        detail="Fixed offline decisions and scripted agent responses. Demonstrates a 0.8 confidence threshold; applications must calibrate their own.",
    ),
    Example(
        "jev_classification",
        "Jev document classification",
        "Inspect a Choice result for W2, Deposit, Insurance, Payroll, or unknown.",
        "Evaluation",
        "flask",
        offline=True,
        detail="Uses a fixed W2 fixture without API keys or network calls. Live classification is available through the command-line example.",
    ),
    Example(
        "jev_answer_review",
        "Jev answer review",
        "Review answer coverage with Score and evidence support with Noul.",
        "Evaluation",
        "flask",
        offline=True,
        detail="Fixed review results after a completed scripted agent run. The review leaves the original answer and run status unchanged.",
    ),
    Example(
        "flight_recorder",
        "Flight recorder",
        "Follow an invoice analysis through a failure, recovery, and offline playback.",
        "Runtime",
        "record",
        offline=True,
        detail="Exports a downloadable .hx incident bundle. Uses synthetic invoices and a scripted provider; no model calls.",
    ),
    Example(
        "prompt_caching",
        "Prompt caching",
        "Keep the prompt prefix stable and watch cache reads arrive across iterations.",
        "Agents",
        "activity",
        offline=True,
        detail="Runs a two-iteration loop with a scripted provider that reports cache writes and reads; no model calls.",
    ),
    Example(
        "runtime_approvals",
        "Durable approvals",
        "Pause a workflow, review a tool action, and resume its persisted execution.",
        "Runtime",
        "shield",
        offline=True,
        detail="Approve or deny a note creation in the browser. The original example uses a temporary SQLite database.",
    ),
    Example(
        "skills_demo",
        "Skill discovery",
        "Watch an agent find and load a code-review skill, with every hook visible.",
        "Skills",
        "sparkles",
        offline=True,
        detail="A scripted provider loads the real code-review skill. No API key is needed.",
    ),
    Example(
        "simple_chat",
        "Simple chat",
        "A streaming assistant with arithmetic and workspace inspection tools.",
        "Agents",
        "message",
        interactive=True,
        prompt="What is (48 * 12) / 6?",
        detail="Read-only filesystem tools are scoped to this run's working directory. Type usage to see token statistics.",
    ),
    Example(
        "provider_chat",
        "Provider chat",
        "Run the same calculator agent with Anthropic, OpenAI, Gemini, or OpenRouter.",
        "Agents",
        "message",
        interactive=True,
        prompt="What is 40 * 10?",
        detail="Choose a provider and a model available to your account. Uses the same calculator tool and agent loop with streaming or ordinary responses.",
    ),
    Example(
        "coding_agent",
        "Coding agent",
        "Write and inspect code with streaming output and a tool audit trail.",
        "Agents",
        "code",
        interactive=True,
        prompt="Write a Python program to calculate Fibonacci numbers.",
        detail="Writes and shell commands retain the original example's approval prompts. The working directory is separate, but host tools are not an OS sandbox.",
    ),
    Example(
        "memory_agent",
        "Memory agent",
        "Save notes and structured facts, then recall them within a conversation.",
        "Agents",
        "memory",
        interactive=True,
        prompt="Remember that this project's name is HarnessX.",
        detail="Notes and facts are stored in this run's working directory. Type memories to inspect them.",
    ),
    Example(
        "multi_agent",
        "Specialist agents",
        "Delegate research, code review, and file analysis to focused assistants.",
        "Agents",
        "network",
        interactive=True,
        prompt="Review this Python code: def add(a, b): return a - b",
        detail="Uses constructor-defined specialists with fresh conversations per task. Delegation can make several model calls per turn.",
    ),
    Example(
        "skills_agent",
        "Skills agent",
        "Let a live model select engineering skills when the task calls for them.",
        "Skills",
        "sparkles",
        interactive=True,
        prompt="Review this code: def divide(a, b): return a / b",
        detail="Loads the repository's review, commit-message, bug-triage, and SQL skills. Uses the configured default provider.",
    ),
    Example(
        "sandboxed_coder",
        "Sandboxed coder",
        "Execute Python with process resource limits and a durable runtime.",
        "Runtime",
        "terminal",
        interactive=True,
        prompt="Use Python to calculate the first 10 Fibonacci numbers.",
        detail="Uses the process tier: resource limits, with host filesystem and network access. Temporary code files are removed by the example.",
    ),
    Example(
        "run_evals",
        "Workflow evaluations",
        "Run three arithmetic and tool-selection cases and inspect their scores.",
        "Evaluation",
        "flask",
        offline=True,
        extra="langsmith",
        detail="Choose the scripted, offline fixture or the live model. Live mode uploads results if a LangSmith key is configured.",
    ),
    Example(
        "langsmith_tracing",
        "LangSmith tracing",
        "Trace a tool call and a delegated specialist as a connected run tree.",
        "Integrations",
        "activity",
        extra="langsmith",
        detail="Requires Anthropic and LangSmith credentials. This example sends run data to your configured LangSmith project.",
    ),
    Example(
        "postgres_runtime",
        "PostgreSQL runtime",
        "Stream a model response with execution state stored in PostgreSQL.",
        "Integrations",
        "database",
        extra="psycopg",
        detail="Requires DATABASE_URL for an existing database. The original example provisions HarnessX tables.",
    ),
    Example(
        "task_agent",
        "Task agent",
        "Hand it one task on the command line; it writes the deliverable and runs it.",
        "Agents",
        "terminal",
        prompt="Write a Python script that prints the first 10 Fibonacci numbers, run it, and confirm the output.",
        detail="One-shot rather than a conversation. It carries the file and shell tools, so it can produce any format without a pre-written renderer. Web search and tracing switch on only when their keys are set.",
    ),
    Example(
        "mcp_agent",
        "MCP agent",
        "Connect a local or remote MCP server and use its tools in a conversation.",
        "Integrations",
        "plug",
        interactive=True,
        extra="mcp",
        prompt="List the tools available to you.",
        detail="Configure a stdio command or an SSE URL below. Tool calls keep terminal approvals, presented in the browser.",
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
    if not example.offline or (example.id == "run_evals" and mode == "live"):
        if example.id == "skills_agent":
            provider = os.getenv("AGENT_PROVIDER", "anthropic")
        elif example.id != "provider_chat":
            provider = "anthropic"
        missing.extend(provider_missing(provider))
        if (
            example.id == "skills_agent"
            and provider != "anthropic"
            and not os.getenv("AGENT_MODEL")
        ):
            missing.append("AGENT_MODEL")
    if example.id == "postgres_runtime" and not os.getenv("DATABASE_URL"):
        missing.append("DATABASE_URL")
    if example.id == "langsmith_tracing" and not (
        os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    ):
        missing.append("LANGSMITH_API_KEY")
    return missing


def public_catalog():
    return [
        {
            **asdict(item),
            "module": item.module,
            "missing": requirements(item),
            "live_missing": requirements(item, "live"),
        }
        for item in EXAMPLES
    ]


def arguments(example, config, workdir):
    """Build argument vectors, never shell strings supplied by the browser."""
    if example.id == "jev_routing":
        # Fixed offline demonstration; live routing requires explicit CLI configuration.
        return ["--min-confidence", "0.8"]
    if example.id == "provider_chat":
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
    if example.id == "run_evals":
        if config.get("mode", "offline") not in ("offline", "live"):
            raise ValueError("Evaluation mode must be offline or live")
        return ["--offline"] if config.get("mode", "offline") == "offline" else []
    if example.id == "mcp_agent":
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
