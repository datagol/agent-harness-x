"""Run every example's main flow without model charges or external services.

Examples are standalone scripts, so they are loaded by path (``example_loader``)
and driven the way a person drives them: ``input()`` for what they type, stdout
for what they read.
"""

import ast
import builtins
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from example_loader import EXAMPLES, example_scripts, load_example
from harnessx import (
    Agent,
    Extension,
    PermissionLevel,
    ProviderResponse,
    SQLiteBackend,
    ToolCall,
)
from harnessx.providers import LLMProvider
from scripted_provider import ScriptedProvider


class TextProvider(LLMProvider):
    async def create(self, **kwargs):
        return ProviderResponse(text="Fixture response")

    async def count_tokens(self, **kwargs):
        return 0


@pytest.fixture
def example_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("ANTHROPIC_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "MCP_SERVERS", "TAVILY_API_KEY"):
        monkeypatch.setenv(name, "")
    for name in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2"):
        monkeypatch.setenv(name, "false")
    monkeypatch.setenv("AGENT_PROVIDER", "anthropic")
    monkeypatch.setenv("AGENT_MODEL", "fixture")
    monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path / "output"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Example smoke tests must not contact live services")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("harnessx.core.make_provider", lambda *args, **kwargs: TextProvider())
    created = []
    original = Agent.__init__

    def track(agent, *args, **kwargs):
        original(agent, *args, **kwargs)
        created.append(agent)

    monkeypatch.setattr(Agent, "__init__", track)
    yield created
    assert all(agent.closed and not agent.busy for agent in created)


def typed(monkeypatch, *lines):
    """Answer the example's input() calls with ``lines``, then end of input."""
    answers = iter(lines)

    def answer(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", answer)


# ── every file follows the rules ─────────────────────────────────────────────


def test_examples_live_in_topic_folders_and_nowhere_else():
    assert [path.name for path in EXAMPLES.glob("*.py")] == []
    assert len(example_scripts()) == 25


@pytest.mark.parametrize("path", example_scripts(), ids=lambda p: p.relative_to(EXAMPLES).as_posix())
def test_every_example_is_a_standalone_script(path):
    source = path.read_text()
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    relative = path.relative_to(EXAMPLES).as_posix()
    assert f"python examples/{relative}" in docstring, "the docstring says how to run it by path"
    assert "Needs:" in docstring
    imported = {
        (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    local = {name for name in imported if name == "examples" or name.startswith("examples.")
             or (name.startswith("_") and name != "__future__")}
    assert not local, f"imports a shared helper instead of standing alone: {local}"
    assert "python -m" not in source
    assert "load_dotenv()" in source
    assert 'if __name__ == "__main__":' in source


OFFLINE = [
    ("01-basics/prompt_caching.py", [], "prefix stable across iterations: yes"),
    ("01-basics/progress_and_waiting.py", [], "waiting"),
    ("03-skills/skills_lazy_loading.py", [], "SKILL_INVOKED"),
    ("04-context/planning_todos.py", [], "TODOS_UPDATED [3/3]"),
    ("04-context/condensing_a_long_history.py", [], "summarized=True"),
    ("05-control/loop_guard.py", [], "REPETITION:"),
    ("05-control/retries_and_fallback.py", [], "served by backup"),
    ("06-durability/session_snapshots.py", [], "History after continuing"),
    ("07-quality/decisions_routing.py", ["--min-confidence", "0.8"], '"route": "sql"'),
    ("07-quality/decisions_answer_review.py", [], '"probability": 0.96'),
    ("07-quality/flight_recorder.py", ["--output", "incident.hx"], "Playback made zero model/tool calls"),
]


@pytest.mark.parametrize("relative, args, expected", OFFLINE, ids=[case[0] for case in OFFLINE])
def test_offline_examples_run_by_path_from_anywhere(relative, args, expected, tmp_path):
    """The way a reader runs them: `python examples/...` from another directory."""
    environment = {**os.environ, "ANTHROPIC_API_KEY": "", "LANGSMITH_TRACING": "false", "PYTHONUNBUFFERED": "1"}
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / relative), *args], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert expected in done.stdout


# ── interactive examples ─────────────────────────────────────────────────────


INTERACTIVE = {
    "01-basics/streaming_chat.py": [ToolCall("math", "calculate", {"expression": "6 * 7"})],
    "02-tools/filesystem_tools_and_permissions.py": [ToolCall("read", "read_file", {"path": "fixture.txt"})],
    "04-context/custom_memory_tools.py": [
        ToolCall("note", "save_memory", {"title": "Fixture", "content": "Example note"}),
        ToolCall("fact", "save_fact", {"content": "Example fact"}),
    ],
    "03-skills/skills_interactive.py": [ToolCall("skill", "Skill", {"skill": "code-review"})],
    "04-context/knowledge_bundles_okf.py": [ToolCall("kb", "search_concepts", {"query": "active users"})],
    "08-sandboxes/sandbox_isolation_tiers.py": [ToolCall("code", "run_python", {"code": "print('fixture execution')"})],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("relative", list(INTERACTIVE))
async def test_interactive_examples_complete_and_close(relative, example_environment, monkeypatch, tmp_path, capsys):
    (tmp_path / "fixture.txt").write_text("Fixture file")
    calls = INTERACTIVE[relative]
    providers = []

    def provider_factory(*args, **kwargs):
        provider = TextProvider() if providers else ScriptedProvider([
            ProviderResponse(tool_calls=calls, stop_reason="tool_use"),
            ProviderResponse(text="Fixture response"),
        ])
        providers.append(provider)
        return provider

    monkeypatch.setattr("harnessx.core.make_provider", provider_factory)
    module = load_example(relative)
    typed(monkeypatch, "hello", "quit")
    await module.main()
    assert "Error:" not in capsys.readouterr().out
    assert example_environment and all(agent.memory.get_messages() for agent in example_environment)
    outcomes = [
        block
        for message in example_environment[0].memory.get_messages()
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(outcomes) == len(calls)
    assert not any(block.get("is_error") for block in outcomes), outcomes
    if "sandbox" in relative:
        assert "fixture execution" in outcomes[0]["content"]


@pytest.mark.asyncio
async def test_declared_specialists_close_their_agents(example_environment, tmp_path):
    specialist_definitions = load_example("05-control/delegating_to_subagents.py").specialist_definitions

    async with Agent(subagents=specialist_definitions(str(tmp_path))) as parent:
        parent.permissions.set_permission("delegate_research", PermissionLevel.ALLOW)
        for name, task in (
            ("code_review", "Review print(1)"),
            ("research", "Read https://example.test"),
            ("file_analysis", "Describe the workspace"),
        ):
            result = await parent.tools.execute(
                ToolCall(name, f"delegate_{name}", {"task": task}), permissions=parent.permissions,
            )
            assert not result.is_error and result.content == "Fixture response"


@pytest.mark.asyncio
async def test_delegation_example_runs_one_task(example_environment, capsys):
    example = load_example("05-control/delegating_to_subagents.py")
    await example.main("Review print(1)")
    assert "Error:" not in capsys.readouterr().out


# ── switching providers ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["anthropic", "openai", "gemini", "openrouter"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_switching_providers_selection_tool_execution_and_cleanup(
    example_environment, monkeypatch, capsys, provider_name, streaming,
):
    example = load_example("01-basics/switching_providers.py")
    selected, calls = [], []

    class Provider(ScriptedProvider):
        async def create(self, **kwargs):
            calls.append(kwargs)
            return await super().create(**kwargs)

    def factory(name):
        selected.append(name)
        provider = Provider([
            ProviderResponse(tool_calls=[ToolCall("math", "calculate", {"expression": "40*10"})], stop_reason="tool_use"),
            ProviderResponse(text="40 * 10 = 400"),
        ])
        provider.name = name
        return provider

    monkeypatch.setattr("harnessx.core.make_provider", factory)
    monkeypatch.setattr(builtins, "input", lambda prompt="": pytest.fail("One-shot run read stdin"))
    args = ["--provider", provider_name, "--model", "fixture-model", "--prompt", "What is 40*10?"]
    if not streaming:
        args.append("--no-stream")
    assert await example.main(args) == 0
    assert selected == [provider_name]
    assert len(calls) == 2 and all(call["model"] == "fixture-model" for call in calls)
    result = calls[1]["messages"][2]["content"][0]
    assert result["content"] == "400" and not result.get("is_error")
    assert "40 * 10 = 400" in capsys.readouterr().out
    assert example_environment[0].closed


def test_switching_providers_defaults_and_model_validation(capsys):
    from harnessx import AgentConfig

    parse_args = load_example("01-basics/switching_providers.py").parse_args
    args = parse_args([])
    assert (args.provider, args.model) == (AgentConfig().provider, AgentConfig().model)
    for args in (["--provider", "openai"], ["--provider", "gemini"], ["--provider", "openrouter"],
                 ["--model", " "], ["--prompt", " "]):
        with pytest.raises(SystemExit) as error:
            parse_args(args)
        assert error.value.code == 2
    assert "--model is required" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_switching_providers_interactive_followups(example_environment, monkeypatch, capsys):
    example = load_example("01-basics/switching_providers.py")
    typed(monkeypatch, "", "hello", "usage", "follow up", "quit")
    assert await example.main([]) == 0
    agent = example_environment[0]
    assert len(agent.memory.get_messages()) == 4
    assert "cache_read_input_tokens" in capsys.readouterr().out
    assert agent.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_switching_providers_one_shot_reports_failed_run(example_environment, monkeypatch, capsys, streaming):
    example = load_example("01-basics/switching_providers.py")

    class BrokenProvider(TextProvider):
        async def create(self, **kwargs):
            raise RuntimeError("Provider fixture unavailable")

    monkeypatch.setattr("harnessx.core.make_provider", lambda *args: BrokenProvider())
    args = ["--prompt", "hello"] + ([] if streaming else ["--no-stream"])
    assert await example.main(args) == 1
    out = capsys.readouterr().out
    assert out.count("Error:") == 1 and "Provider fixture unavailable" in out
    assert example_environment[0].closed


# ── offline recipes, in process ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skills_lazy_loading_uses_the_real_skill_tool(example_environment, capsys):
    await load_example("03-skills/skills_lazy_loading.py").main()
    assert "SKILL_INVOKED  skill=code-review" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_prompt_caching_example_shows_a_stable_prefix(example_environment, capsys):
    main = load_example("01-basics/prompt_caching.py").main
    assert await main() == 0
    out = capsys.readouterr().out
    assert "prefix stable across iterations: yes" in out
    assert "cache reads (tokens)  1150" in out and "cache writes (tokens) 1150" in out

    assert await main(disabled=True) == 0
    out = capsys.readouterr().out
    assert "Caching: off" in out and "enabled=False" in out and "cache reads (tokens)  0" in out


@pytest.mark.asyncio
async def test_flight_recorder_example_exports_after_runtime_closes(example_environment, tmp_path):
    from harnessx import IncidentRecorder

    path = tmp_path / "incident.hx"
    await load_example("07-quality/flight_recorder.py").main(path)
    assert (await IncidentRecorder().verify(path)).complete


@pytest.mark.asyncio
@pytest.mark.parametrize("answer, exists", [("y", True), ("n", False)])
async def test_approval_example_both_decisions(example_environment, monkeypatch, capsys, answer, exists):
    monkeypatch.setattr(builtins, "input", lambda prompt="": answer)
    await load_example("06-durability/tool_approvals_and_resume.py").main()
    output = capsys.readouterr().out
    assert "Before approval: awaiting_input | Note exists: False" in output
    assert f"After approval: completed | Note exists: {exists}" in output


@pytest.mark.asyncio
async def test_new_recipes_show_their_feature_firing(example_environment, capsys):
    for relative, expected in [
        ("05-control/loop_guard.py", "REPETITION: cycle of 1 call(s), 3 laps"),
        ("05-control/retries_and_fallback.py", "LLM_RESPONSE: served by backup"),
        ("04-context/planning_todos.py", "TODOS_UPDATED [3/3]"),
        ("04-context/condensing_a_long_history.py", "summarized=True"),
        ("06-durability/session_snapshots.py", "History after continuing"),
    ]:
        await load_example(relative).main()
        assert expected in capsys.readouterr().out, relative


# ── examples that talk to services ───────────────────────────────────────────


class FakeMCP:
    def __init__(self):
        self.connected = False
        self.closed = False
        self.args = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.disconnect_all()

    async def connect(self, name, **options):
        self.connected = True
        # the example passes an MCPServerConfig; the keyword form is still accepted
        self.args = list(getattr(name, "args", options.get("args", [])))
        return self.list_tools()

    async def disconnect_all(self):
        self.closed = True
        self.connected = False

    async def disconnect(self, name):
        self.connected = False

    def list_servers(self):
        return {"fixture": {}} if self.connected else {}

    def list_tools(self):
        return [SimpleNamespace(tool_name="inspect", description="Fixture tool", server_name="fixture")] \
            if self.connected else []

    @property
    def tool_count(self):
        return int(self.connected)

    def register_tools(self, registry, **kwargs):
        if not self.connected:
            return []
        registry.register(name="fixture_inspect", permission=PermissionLevel.ALLOW)(lambda: "fixture")
        return ["fixture_inspect"]


@pytest.mark.asyncio
async def test_mcp_example_setup_chat_and_teardown(example_environment, monkeypatch, capsys):
    example = load_example("02-tools/mcp_servers.py")
    manager = FakeMCP()
    monkeypatch.setattr(example, "MCPManager", lambda: manager)
    typed(monkeypatch, "tools", "hello", "quit")
    await example.main(["--server", "fixture", "--command", "fixture", "--args", "'directory with spaces'"])
    assert manager.closed and manager.args == ["directory with spaces"]
    assert "Error:" not in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("connection_fails", [False, True])
async def test_postgres_example_runtime_flow_with_local_store(example_environment, monkeypatch, tmp_path, connection_fails):
    example = load_example("06-durability/durable_crash_recovery.py")
    # The example's runtime ownership/stream API is tested here; real PostgreSQL
    # remains covered by the separately configured service integration suite.
    backend = SQLiteBackend(str(tmp_path / "runtime.db"))
    monkeypatch.setenv("DATABASE_URL", "fixture")
    monkeypatch.setattr(example, "PostgresBackend", lambda dsn: backend)
    if connection_fails:
        async def unavailable():
            raise RuntimeError("Database unavailable")

        monkeypatch.setattr(backend, "initialize", unavailable)
        with pytest.raises(RuntimeError, match="Database unavailable"):
            await example.main([])
    else:
        await example.main([])
    assert backend.conn is None


@pytest.mark.asyncio
async def test_tracing_example_closes_its_agent(example_environment, monkeypatch, capsys):
    example = load_example("07-quality/tracing_with_langsmith.py")

    class LocalTrace(Extension):
        name = "local-trace"

        def __init__(self, **kwargs):
            pass

        def install(self, context):
            pass

    monkeypatch.setattr(example, "LangSmithExtension", LocalTrace)
    await example.main()
    assert "Error:" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_offline_evaluation_example_runs_without_model_or_upload(example_environment, monkeypatch, capsys):
    pytest.importorskip("langsmith")
    example = load_example("07-quality/evaluating_with_datasets.py")

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline example constructed a live model")

    monkeypatch.setattr("harnessx.core.make_provider", forbidden)
    await example.main(["--offline"])
    output = capsys.readouterr().out
    assert "SCRIPTED FIXTURE" in output and "100.0%" in output
    assert len(example_environment) == 3


def test_examples_folder_has_only_topic_folders_and_data():
    entries = {path.name for path in EXAMPLES.iterdir() if path.name != "__pycache__"}
    assert entries == {
        "01-basics", "02-tools", "03-skills", "04-context", "05-control", "06-durability", "07-quality", "08-sandboxes",
        "skills", "knowledge", "README.md",
    }, entries
    assert Path(EXAMPLES / "06-durability" / "postgres.config.example.json").is_file()
