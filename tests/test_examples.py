"""Run example application flows without model charges or external services."""

import asyncio
import importlib
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from harnessx import (
    Agent,
    Extension,
    PermissionLevel,
    ProviderResponse,
    SQLiteBackend,
    ToolCall,
)
from harnessx.providers import LLMProvider


class TextProvider(LLMProvider):
    async def create(self, **kwargs):
        return ProviderResponse(text="Fixture response")

    async def count_tokens(self, **kwargs):
        return 0


@pytest.fixture
def example_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in (
        "ANTHROPIC_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "MCP_SERVERS",
    ):
        monkeypatch.setenv(name, "")
    for name in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2"):
        monkeypatch.setenv(name, "false")
    monkeypatch.setenv("AGENT_PROVIDER", "anthropic")
    monkeypatch.setenv("AGENT_MODEL", "fixture")
    monkeypatch.setenv("AGENT_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv(
        "AGENT_SKILLS",
        str(Path(__file__).resolve().parents[1] / "examples/skills/code-review"),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Example smoke tests must not contact live services")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(
        "harnessx.core.make_provider", lambda *args, **kwargs: TextProvider()
    )
    created = []
    original = Agent.__init__

    def track(agent, *args, **kwargs):
        original(agent, *args, **kwargs)
        created.append(agent)

    monkeypatch.setattr(Agent, "__init__", track)
    yield created
    assert all(agent.closed and not agent.busy for agent in created)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "simple_chat",
        "coding_agent",
        "memory_agent",
        "multi_agent",
        "skills_agent",
        "sandboxed_coder",
    ],
)
async def test_interactive_examples_complete_and_close(
    name, example_environment, monkeypatch, tmp_path
):
    from examples._fixtures import ScriptedProvider

    (tmp_path / "fixture.txt").write_text("Fixture file")
    featured_calls = {
        "simple_chat": [ToolCall("math", "calculate", {"expression": "6 * 7"})],
        "coding_agent": [ToolCall("read", "read_file", {"path": "fixture.txt"})],
        "memory_agent": [
            ToolCall(
                "note", "save_memory", {"title": "Fixture", "content": "Example note"}
            ),
            ToolCall("fact", "save_fact", {"content": "Example fact"}),
        ],
        "multi_agent": [
            ToolCall("review", "delegate_code_review", {"task": "Review print(1)"})
        ],
        "skills_agent": [ToolCall("skill", "Skill", {"skill": "code-review"})],
        "sandboxed_coder": [
            ToolCall("code", "run_python", {"code": "print('fixture execution')"})
        ],
    }
    providers = []

    def provider_factory(*args, **kwargs):
        provider = (
            TextProvider()
            if providers
            else ScriptedProvider(
                [
                    ProviderResponse(
                        tool_calls=featured_calls[name], stop_reason="tool_use"
                    ),
                    ProviderResponse(text="Fixture response"),
                ]
            )
        )
        providers.append(provider)
        return provider

    monkeypatch.setattr("harnessx.core.make_provider", provider_factory)
    module = importlib.import_module(f"examples.{name}")
    answers = iter(["hello", "quit"])
    monkeypatch.setattr(module, "get_user_input", lambda: next(answers))
    errors = []
    monkeypatch.setattr(module, "print_error", errors.append)
    await module.main()
    assert not errors
    assert example_environment and all(
        agent.memory.get_messages() for agent in example_environment
    )
    outcomes = [
        block
        for message in example_environment[0].memory.get_messages()
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(outcomes) == len(featured_calls[name])
    assert not any(block.get("is_error") for block in outcomes), outcomes
    if name == "sandboxed_coder":
        assert "fixture execution" in outcomes[0]["content"]


@pytest.mark.asyncio
async def test_declared_specialists_close_their_agents(example_environment, tmp_path):
    from examples.multi_agent import specialist_definitions

    async with Agent(subagents=specialist_definitions(str(tmp_path))) as parent:
        parent.permissions.set_permission("delegate_research", PermissionLevel.ALLOW)
        for name, task in (
            ("code_review", "Review print(1)"),
            ("research", "Read https://example.test"),
            ("file_analysis", "Describe the workspace"),
        ):
            result = await parent.tools.execute(
                ToolCall(name, f"delegate_{name}", {"task": task}),
                permissions=parent.permissions,
            )
            assert not result.is_error and result.content == "Fixture response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_name", ["anthropic", "openai", "gemini", "openrouter"]
)
@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_chat_selection_tool_execution_and_cleanup(
    example_environment,
    monkeypatch,
    capsys,
    provider_name,
    streaming,
):
    from examples import provider_chat
    from examples._fixtures import ScriptedProvider

    selected, calls = [], []

    class Provider(ScriptedProvider):
        async def create(self, **kwargs):
            calls.append(kwargs)
            return await super().create(**kwargs)

    def factory(name):
        selected.append(name)
        provider = Provider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall("math", "calculate", {"expression": "40*10"})],
                    stop_reason="tool_use",
                ),
                ProviderResponse(text="40 * 10 = 400"),
            ]
        )
        provider.name = name
        return provider

    monkeypatch.setattr("harnessx.core.make_provider", factory)
    monkeypatch.setattr(
        provider_chat,
        "get_user_input",
        lambda: pytest.fail("One-shot run requested stdin"),
    )
    args = [
        "--provider",
        provider_name,
        "--model",
        "fixture-model",
        "--prompt",
        "What is 40*10?",
    ]
    if not streaming:
        args.append("--no-stream")
    assert await provider_chat.main(args) == 0
    assert selected == [provider_name]
    assert len(calls) == 2 and all(call["model"] == "fixture-model" for call in calls)
    result = calls[1]["messages"][2]["content"][0]
    assert result["content"] == "400" and not result.get("is_error")
    assert "40 * 10 = 400" in capsys.readouterr().out
    assert example_environment[0].closed


def test_provider_chat_defaults_and_model_validation(capsys):
    from examples.provider_chat import parse_args
    from harnessx import AgentConfig

    args = parse_args([])
    assert (args.provider, args.model) == (AgentConfig().provider, AgentConfig().model)
    for args in (
        ["--provider", "openai"],
        ["--provider", "gemini"],
        ["--provider", "openrouter"],
        ["--model", " "],
        ["--prompt", " "],
    ):
        with pytest.raises(SystemExit) as error:
            parse_args(args)
        assert error.value.code == 2
    assert "--model is required" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_provider_chat_interactive_followups(
    example_environment, monkeypatch, capsys
):
    from examples import provider_chat

    answers = iter(["", "hello", "usage", "follow up", "quit"])
    monkeypatch.setattr(provider_chat, "get_user_input", lambda: next(answers))
    assert await provider_chat.main([]) == 0
    agent = example_environment[0]
    assert len(agent.memory.get_messages()) == 4
    assert "cache_read_input_tokens" in capsys.readouterr().out
    assert agent.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_chat_one_shot_reports_failed_run(
    example_environment, monkeypatch, streaming
):
    from examples import provider_chat

    class BrokenProvider(TextProvider):
        async def create(self, **kwargs):
            raise RuntimeError("Provider fixture unavailable")

    monkeypatch.setattr("harnessx.core.make_provider", lambda *args: BrokenProvider())
    errors = []
    monkeypatch.setattr(provider_chat, "print_error", errors.append)
    args = ["--prompt", "hello"] + ([] if streaming else ["--no-stream"])
    assert await provider_chat.main(args) == 1
    assert len(errors) == 1 and "Provider fixture unavailable" in str(errors[0])
    assert example_environment[0].closed


@pytest.mark.asyncio
async def test_skills_demo_uses_real_skill_tool(example_environment, capsys):
    from examples import skills_demo

    await skills_demo.main()
    assert "SKILL_INVOKED  skill=code-review" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_prompt_caching_example_shows_a_stable_prefix(example_environment, capsys):
    from examples.prompt_caching import main

    assert await main() == 0
    out = capsys.readouterr().out
    assert "prefix stable across iterations: yes" in out
    assert "cache reads (tokens)  1150" in out and "cache writes (tokens) 1150" in out

    assert await main(disabled=True) == 0
    out = capsys.readouterr().out
    assert "Caching: off" in out and "enabled=False" in out and "cache reads (tokens)  0" in out


@pytest.mark.asyncio
async def test_flight_recorder_example_exports_after_runtime_closes(
    example_environment, tmp_path
):
    from examples.flight_recorder import main
    from harnessx import IncidentRecorder

    path = tmp_path / "incident.hx"
    await main(path)
    assert (await IncidentRecorder().verify(path)).complete


@pytest.mark.asyncio
@pytest.mark.parametrize("answer, exists", [("y", True), ("n", False)])
async def test_approval_example_both_decisions(
    example_environment, monkeypatch, capsys, answer, exists
):
    from examples import runtime_approvals

    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    await runtime_approvals.main()
    output = capsys.readouterr().out
    assert "Before approval: awaiting_input | Note exists: False" in output
    assert f"After approval: completed | Note exists: {exists}" in output


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
        self.args = options.get("args", [])
        return self.list_tools()

    async def disconnect_all(self):
        self.closed = True
        self.connected = False

    async def disconnect(self, name):
        self.connected = False

    def list_servers(self):
        return {"fixture": {}} if self.connected else {}

    def list_tools(self):
        return (
            [
                SimpleNamespace(
                    tool_name="inspect",
                    description="Fixture tool",
                    server_name="fixture",
                )
            ]
            if self.connected
            else []
        )

    @property
    def tool_count(self):
        return int(self.connected)

    def register_tools(self, registry, **kwargs):
        if not self.connected:
            return []
        registry.register(name="fixture_inspect", permission=PermissionLevel.ALLOW)(
            lambda: "fixture"
        )
        return ["fixture_inspect"]


@pytest.mark.asyncio
async def test_mcp_example_setup_chat_and_teardown(example_environment, monkeypatch):
    from examples import mcp_agent

    manager = FakeMCP()
    monkeypatch.setattr(mcp_agent, "MCPManager", lambda: manager)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mcp_agent",
            "--server",
            "fixture",
            "--command",
            "fixture",
            "--args",
            "'directory with spaces'",
        ],
    )
    answers = iter(["tools", "hello", "quit"])
    monkeypatch.setattr(mcp_agent, "get_user_input", lambda: next(answers))
    errors = []
    monkeypatch.setattr(mcp_agent, "print_error", errors.append)
    await mcp_agent.main()
    assert manager.closed and manager.args == ["directory with spaces"] and not errors


@pytest.mark.asyncio
@pytest.mark.parametrize("connection_fails", [False, True])
async def test_postgres_example_runtime_flow_with_local_store(
    example_environment, monkeypatch, tmp_path, connection_fails
):
    from examples import postgres_runtime

    # The example's runtime ownership/stream API is tested here; real PostgreSQL
    # remains covered by the separately configured service integration suite.
    backend = SQLiteBackend(str(tmp_path / "runtime.db"))
    monkeypatch.setenv("DATABASE_URL", "fixture")
    monkeypatch.setattr(postgres_runtime, "PostgresBackend", lambda dsn: backend)
    if connection_fails:
        async def unavailable():
            raise RuntimeError("Database unavailable")

        monkeypatch.setattr(backend, "initialize", unavailable)
        with pytest.raises(RuntimeError, match="Database unavailable"):
            await postgres_runtime.main([])
    else:
        await postgres_runtime.main([])
    assert backend.conn is None


@pytest.mark.asyncio
async def test_tracing_example_closes_parent_and_specialist(
    example_environment, monkeypatch
):
    from examples import langsmith_tracing

    class LocalTrace(Extension):
        name = "local-trace"

        def __init__(self, **kwargs):
            pass

        def install(self, context):
            pass

    monkeypatch.setattr(langsmith_tracing, "LangSmithExtension", LocalTrace)
    assert await langsmith_tracing.run_specialist("fixture") == "Fixture response"
    await langsmith_tracing.main()


def test_offline_evaluation_example_runs_without_model_or_upload(
    example_environment, monkeypatch, capsys
):
    pytest.importorskip("langsmith")
    from examples import run_evals

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline example constructed a live model")

    monkeypatch.setattr("harnessx.core.make_provider", forbidden)
    monkeypatch.setattr(sys, "argv", ["run_evals", "--offline"])
    run_evals.main()
    output = capsys.readouterr().out
    assert "SCRIPTED FIXTURE" in output and "100.0%" in output
    assert len(example_environment) == 3


@pytest.mark.asyncio
async def test_web_example_chat_stream_snapshots_and_cleanup(
    example_environment, monkeypatch
):
    pytest.importorskip("fastapi")
    import httpx
    from examples.web_app import server

    responses = iter(
        [
            ProviderResponse(
                tool_calls=[ToolCall("a", "first", {}), ToolCall("b", "second", {})],
                stop_reason="tool_use",
            ),
            ProviderResponse(text="Chat completed"),
            ProviderResponse(
                tool_calls=[ToolCall("c", "second", {})], stop_reason="tool_use"
            ),
            ProviderResponse(text="Stream completed"),
        ]
    )

    class WebProvider(TextProvider):
        async def create(self, **kwargs):
            return next(responses, ProviderResponse(text="Restored conversation"))

    monkeypatch.setattr("harnessx.core.make_provider", lambda *a, **kw: WebProvider())
    monkeypatch.setattr(server, "MCPManager", FakeMCP)
    monkeypatch.setattr(server, "agent_lock", asyncio.Lock())
    bindings = server.agent_bindings

    def with_test_tools():
        options = bindings()

        @options["tools"].register(permission=PermissionLevel.ALLOW)
        async def first():
            await asyncio.sleep(0.02)
            return "first result"

        @options["tools"].register(permission=PermissionLevel.ALLOW)
        async def second():
            return "second result"

        return options

    monkeypatch.setattr(server, "agent_bindings", with_test_tools)
    async with server.lifespan(server.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://example.test",
        ) as client:
            assert (await client.get("/")).status_code == 200
            response = await client.post("/api/chat", json={"message": "hello"})
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["response"] == "Chat completed"
            assert {item["id"]: item["result"] for item in body["tool_calls"]} == {
                "a": "first result",
                "b": "second result",
            }
            streamed = await client.post("/api/stream", json={"message": "stream"})
            assert (
                '"tool_call_id": "c"' in streamed.text
                and "Stream completed" in streamed.text
            )
            assert "data: [DONE]" in streamed.text
            snapshot = (await client.post("/api/session/save")).json()["session_id"]
            saved_prompt = server.streaming_agent.config.system_prompt
            saved_usage = server.streaming_agent.guardrails.total_usage
            assert (await client.post("/api/session/clear")).status_code == 200
            assert (
                await client.post(f"/api/session/load/{snapshot}")
            ).status_code == 200
            assert server.streaming_agent.config.system_prompt == saved_prompt
            assert server.streaming_agent.guardrails.total_usage == saved_usage
            assert (
                server.streaming_agent.config.system_prompt.count("<available-skills>")
                == 1
            )
            assert (await client.get("/api/status")).json()["session_id"] == snapshot
            assert (
                await client.post("/api/chat", json={"message": "continue"})
            ).json()["response"] == "Restored conversation"
            assert (await client.get("/api/sessions")).status_code == 200
            assert (await client.get("/api/skills")).json()["skills"]
            assert (
                await client.post(
                    "/api/mcp/connect", json={"name": "fixture", "command": "fixture"}
                )
            ).status_code == 200
            assert (await client.post("/api/mcp/disconnect/fixture")).status_code == 200
    assert server.streaming_agent is None and server.mcp_manager is None
