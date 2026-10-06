"""Web API contracts and real offline example processes; no service credentials."""

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness-web"))

from harness_web.catalog import CATALOG, ROOT, arguments
from harness_web.chat import create_chat
from harness_web.files import open_workspace_file
from harness_web.runs import RunManager
from harness_web.server import create_app
from harnessx import PermissionLevel, ProviderResponse, ToolCall
from example_loader import example_scripts
from scripted_provider import ScriptedProvider


@pytest.fixture(autouse=True)
def no_live_services(monkeypatch):
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "DATABASE_URL",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("AGENT_PROVIDER", "anthropic")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "false")

    def forbidden(*args, **kwargs):
        raise AssertionError("Web tests must not contact live services")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


@asynccontextmanager
async def web(tmp_path, **kwargs):
    app = create_app(data_dir=tmp_path, load_env=False, **kwargs)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://testserver"
        ) as client:
            yield app, client


async def start(client, example_id, **config):
    response = await client.post("/api/runs", json={"example_id": example_id, **config})
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def wait_status(run, status):
    async with asyncio.timeout(20):
        while run.status != status:
            assert run.status not in ("completed", "failed", "cancelled"), run.events
            async with run.condition:
                await run.condition.wait()


def output(run):
    return "".join(event.get("content", "") for event in run.events)


def test_catalog_covers_every_example_entry_point(tmp_path):
    expected = {path.relative_to(ROOT / "examples").as_posix() for path in example_scripts()}
    assert {item.path for item in CATALOG.values()} == expected
    for item in CATALOG.values():
        assert item.file.is_file() and item.id == item.file.stem
        assert item.command == f"python examples/{item.path}"
        config = {
            "server": "fixture",
            "command": "python3",
            "args": "'path with spaces.py'",
        }
        args = arguments(item, config, tmp_path)
        assert isinstance(args, list) and all(isinstance(arg, str) for arg in args)
    assert arguments(CATALOG["evaluating_with_datasets"], {}, tmp_path) == ["--offline"]
    assert arguments(CATALOG["evaluating_with_datasets"], {"mode": "live"}, tmp_path) == []
    assert arguments(CATALOG["flight_recorder"], {}, tmp_path)[-1] == str(
        tmp_path / "incident.hx"
    )


@pytest.mark.asyncio
async def test_health_missing_credentials_and_local_request_boundary(tmp_path):
    async with web(tmp_path) as (app, client):
        health = (await client.get("/api/health")).json()
        assert health["app"] == "harness-web"
        assert health["providers"][0]["missing"] == ["ANTHROPIC_API_KEY"]
        assert len((await client.get("/api/examples")).json()) == len(CATALOG)
        response = await client.post("/api/runs", json={"example_id": "filesystem_tools_and_permissions"})
        assert response.status_code == 422 and "ANTHROPIC_API_KEY" in response.text
        assert not app.state.manager.runs
        assert (
            await client.post("/api/runs", json={"example_id": "os"})
        ).status_code == 404
        assert (
            await client.post(
                "/api/runs", json={"example_id": "skills_lazy_loading", "command_line": "bad"}
            )
        ).status_code == 422
        assert (
            await client.post(
                "/api/runs",
                json={"example_id": "skills_lazy_loading"},
                headers={"origin": "https://unrelated.example"},
            )
        ).status_code == 403
        assert (
            await client.get("/api/health", headers={"host": "unrelated.example"})
        ).status_code == 400


@pytest.mark.asyncio
async def test_provider_example_checks_selected_credentials_and_builds_arguments(
    tmp_path, monkeypatch
):
    captured = []

    async def launch(run, example, args, prompt):
        captured.append((example.id, args, prompt))
        await run.state("completed")

    monkeypatch.setattr("harness_web.server.run_example", launch)
    monkeypatch.setattr("harness_web.catalog.installed", lambda module: True)
    async with web(tmp_path) as (app, client):
        body = {
            "example_id": "switching_providers",
            "provider": "openai",
            "model": "fixture-model",
            "streaming": False,
            "prompt": "40 * 10",
        }
        denied = await client.post("/api/runs", json=body)
        assert denied.status_code == 422 and "OPENAI_API_KEY" in denied.text
        assert "ANTHROPIC_API_KEY" not in denied.text
        monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
        missing_model = await client.post("/api/runs", json={**body, "model": ""})
        assert missing_model.status_code == 422 and "model ID" in missing_model.text
        assert not app.state.manager.runs
        response = await client.post("/api/runs", json=body)
        assert response.status_code == 201, response.text
        run = app.state.manager.runs[response.json()["id"]]
        await run.task
        assert captured == [
            (
                "switching_providers",
                ["--provider", "openai", "--model=fixture-model", "--no-stream"],
                "40 * 10",
            )
        ]
        assert run.mode == "live"


@pytest.mark.asyncio
async def test_an_example_run_reports_every_tool_call_as_a_structured_event(tmp_path):
    async with web(tmp_path) as (app, client):
        run_id = await start(client, "loop_guard")
        run = app.state.manager.runs[run_id]
        await asyncio.wait_for(run.task, 30)
        assert run.status == "completed", output(run)
        events = [event["event"] for event in run.events if event["type"] == "agent"]
        starts = [e["data"] for e in events if e["type"] == "tool_call_start"]
        results = [e["data"] for e in events if e["type"] == "tool_result"]
        assert [call["name"] for call in starts] == ["check_build"] * 3
        assert starts[0]["input"] == {"job": "1234"} and starts[0]["agent"] == 1
        # Results pair with their calls and carry what the tool returned.
        assert [r["tool_call_id"] for r in results] == [call["id"] for call in starts]
        assert results[-1]["content"] == "build 1234: queued (position 4)"
        assert not any(r["is_error"] for r in results)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "example_id, expected",
    [
        ("skills_lazy_loading", "SKILL_INVOKED"),
        ("flight_recorder", "Playback made zero model/tool calls"),
        ("evaluating_with_datasets", "SCRIPTED FIXTURE"),
        ("decisions_routing", '"route": "sql"'),
        ("decisions_answer_review", '"probability": 0.96'),
    ],
)
async def test_real_offline_example_and_event_replay(tmp_path, example_id, expected):
    if example_id == "evaluating_with_datasets":
        pytest.importorskip("langsmith")
    async with web(tmp_path) as (app, client):
        run_id = await start(client, example_id)
        run = app.state.manager.runs[run_id]
        await asyncio.wait_for(run.task, 30)
        assert run.status == "completed", output(run)
        assert expected in output(run)
        first = await client.get(f"/api/runs/{run_id}/events")
        records = [
            json.loads(line[6:])
            for line in first.text.splitlines()
            if line.startswith("data: ") and line != "data: {}"
        ]
        assert records[-1]["status"] == "completed"
        assert len({record["seq"] for record in records}) == len(records)
        after = records[-2]["seq"]
        replay = await client.get(
            f"/api/runs/{run_id}/events", headers={"last-event-id": str(after)}
        )
        assert f"id: {after}\n" not in replay.text
        assert f"id: {records[-1]['seq']}\n" in replay.text
        if example_id == "flight_recorder":
            files = (await client.get(f"/api/runs/{run_id}/files")).json()
            assert files == [
                {
                    "name": "incident.hx",
                    "size": (run.workdir / "incident.hx").stat().st_size,
                }
            ]
            bundle = await client.get(f"/api/runs/{run_id}/files/incident.hx")
            assert bundle.content.startswith(b"PK")
            assert "attachment" in bundle.headers["content-disposition"]


@pytest.mark.asyncio
@pytest.mark.parametrize("decision, exists", [("y", True), ("n", False)])
async def test_original_persisted_approval_waits_for_exact_browser_prompt(
    tmp_path, decision, exists
):
    async with web(tmp_path) as (app, client):
        run_id = await start(client, "tool_approvals_and_resume")
        run = app.state.manager.runs[run_id]
        await wait_status(run, "waiting")
        assert "Note exists: False" in output(run)
        assert run.pending["kind"] == "approval"
        assert not run.task.done()
        bad = await client.post(
            f"/api/runs/{run_id}/input", json={"prompt_id": "stale", "value": "y"}
        )
        assert bad.status_code == 409
        prompt = run.pending["id"]
        rejected = await client.post(
            f"/api/runs/{run_id}/input", json={"prompt_id": prompt, "value": "a"}
        )
        assert rejected.status_code == 409
        answer = {"prompt_id": prompt, "value": decision}
        responses = await asyncio.gather(
            *(client.post(f"/api/runs/{run_id}/input", json=answer) for _ in range(2))
        )
        assert sorted(response.status_code for response in responses) == [200, 409]
        await asyncio.wait_for(run.task, 10)
        assert run.status == "completed", output(run)
        assert f"After approval: completed | Note exists: {exists}" in output(run)


@pytest.mark.asyncio
async def test_stop_pending_process_and_reject_late_approval(tmp_path):
    async with web(tmp_path) as (app, client):
        run_id = await start(client, "tool_approvals_and_resume")
        run = app.state.manager.runs[run_id]
        await wait_status(run, "waiting")
        prompt = run.pending["id"]
        response = await client.post(f"/api/runs/{run_id}/stop")
        assert response.json()["status"] == "cancelled"
        assert run.task.done() and run.process.returncode is not None
        assert run.pending is None
        assert (
            await client.post(
                f"/api/runs/{run_id}/input", json={"prompt_id": prompt, "value": "y"}
            )
        ).status_code == 409


@pytest.mark.asyncio
async def test_downloads_reject_symlinks_and_other_run_files(tmp_path):
    async with web(tmp_path) as (app, client):
        run = app.state.manager.create("Files", "fixture")
        secret = tmp_path / "secret.txt"
        secret.write_text("private")
        (run.workdir / "escape.txt").symlink_to(secret)
        (run.workdir / "folder").symlink_to(tmp_path, target_is_directory=True)
        (run.workdir / ".env").write_text("private")
        (run.workdir / "safe.txt").write_text("safe")
        files = (await client.get(f"/api/runs/{run.id}/files")).json()
        assert [item["name"] for item in files] == ["safe.txt"]
        for name in ("escape.txt", "folder/secret.txt", ".env", "%2e%2e%2fsecret.txt"):
            assert (
                await client.get(f"/api/runs/{run.id}/files/{name}")
            ).status_code == 404


@pytest.mark.asyncio
async def test_demo_chat_tools_history_and_session_isolation(tmp_path):
    async with web(tmp_path) as (app, client):
        first = (await client.post("/api/chats", json={"provider": "demo"})).json()
        second = (await client.post("/api/chats", json={"provider": "demo"})).json()
        response = await client.post(
            f"/api/chats/{first['id']}/messages", json={"message": "What is 48 * 12?"}
        )
        assert response.status_code == 201, response.text
        run = app.state.manager.runs[response.json()["id"]]
        await asyncio.wait_for(run.task, 5)
        chat = (await client.get(f"/api/chats/{first['id']}")).json()
        assert chat["messages"][-1]["tools"][0]["content"] == "576"
        assert [s["type"] for s in chat["messages"][-1]["segments"]] == ["tool", "text"], "shown in the order it happened"
        assert "576" in chat["messages"][-1]["content"]
        assert "no model calls" in chat["messages"][-1]["content"]
        assert (await client.get(f"/api/chats/{second['id']}")).json()["messages"] == []
        agent = app.state.chats[first["id"]].agent
        assert (await client.delete(f"/api/chats/{first['id']}")).status_code == 200
        assert agent.closed


@pytest.mark.asyncio
async def test_generated_files_get_download_links(tmp_path):
    def factory(provider, model, directory):
        scripted = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[ToolCall("gen", "generate_file", {"path": "report.html", "content": "<h1>Hi</h1>"})],
                    stop_reason="tool_use",
                ),
                ProviderResponse(text="Your report is ready: report.html"),
            ]
        )
        return create_chat(provider, model, directory, provider_instance=scripted)

    async with web(tmp_path, chat_factory=factory) as (app, client):
        chat_id = (await client.post("/api/chats", json={"provider": "demo"})).json()["id"]
        response = await client.post(f"/api/chats/{chat_id}/messages", json={"message": "Make me a report"})
        run = app.state.manager.runs[response.json()["id"]]
        await wait_status(run, "waiting")
        await client.post(f"/api/runs/{run.id}/input", json={"prompt_id": run.pending["id"], "value": "y"})
        await asyncio.wait_for(run.task, 5)

        message = (await client.get(f"/api/chats/{chat_id}")).json()["messages"][-1]
        assert "Download link: /api/chats/" in message["tools"][0]["content"]
        assert "__FILE__" not in message["tools"][0]["content"]
        kinds = {item["kind"]: item for item in message["files"]}
        assert kinds["workspace"]["url"] == f"/api/runs/{run.id}/files/report.html"
        assert kinds["download"]["url"].startswith(f"/api/chats/{chat_id}/downloads/")
        for item in message["files"]:
            served = await client.get(item["url"])
            assert served.status_code == 200 and served.content == b"<h1>Hi</h1>"
        assert any(event["type"] == "chat_files" for event in run.events)
        assert (await client.get(f"/api/chats/{chat_id}/downloads/zz/report.html")).status_code == 404
        listed = [item["name"] for item in (await client.get(f"/api/runs/{run.id}/files")).json()]
        assert listed == ["report.html"], "the hidden downloads copy stays out of the workspace listing"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision, exists", [("y", True), ("n", False), ("stop", False)]
)
async def test_chat_write_permission_is_enforced_in_engine(tmp_path, decision, exists):
    def factory(provider, model, directory):
        scripted = ScriptedProvider(
            [
                ProviderResponse(
                    tool_calls=[
                        ToolCall(
                            "write",
                            "write_file",
                            {"path": "review.txt", "content": "Approved"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(text="Finished."),
            ]
        )
        return create_chat(provider, model, directory, provider_instance=scripted)

    async with web(tmp_path, chat_factory=factory) as (app, client):
        chat_id = (await client.post("/api/chats", json={"provider": "demo"})).json()[
            "id"
        ]
        response = await client.post(
            f"/api/chats/{chat_id}/messages", json={"message": "Write a review"}
        )
        run = app.state.manager.runs[response.json()["id"]]
        await wait_status(run, "waiting")
        assert not (run.workdir / "review.txt").exists()
        assert run.pending["tool"]["name"] == "write_file"
        overlap = await client.post(
            f"/api/chats/{chat_id}/messages", json={"message": "Another turn"}
        )
        assert overlap.status_code == 409
        if decision == "stop":
            await client.post(f"/api/runs/{run.id}/stop")
        else:
            await client.post(
                f"/api/runs/{run.id}/input",
                json={"prompt_id": run.pending["id"], "value": decision},
            )
            await asyncio.wait_for(run.task, 5)
        assert (run.workdir / "review.txt").exists() is exists
        assert not app.state.chats[chat_id].agent.busy


@pytest.mark.asyncio
async def test_chat_setup_rebuilds_prompt_skills_and_mcp_without_losing_history(
    tmp_path,
):
    class FakeMCP:
        def __init__(self):
            self.servers = {}
            self.connect_calls = []
            self.closed = False

        @property
        def tool_count(self):
            return sum(len(info["tools"]) for info in self.servers.values())

        async def connect(self, name, **kwargs):
            self.connect_calls.append((name, kwargs))
            tools = [
                SimpleNamespace(
                    server_name=name,
                    tool_name="inspect",
                    description="Inspect a configured fixture",
                    input_schema={"type": "object", "properties": {}},
                )
            ]
            self.servers[name] = {
                "transport": "stdio" if kwargs["command"] else "sse",
                "tools": tools,
                "permission": kwargs["permission"],
            }
            return tools

        async def disconnect(self, name):
            self.servers.pop(name, None)

        async def disconnect_all(self):
            self.closed = True
            self.servers.clear()

        def list_servers(self):
            return {
                name: {
                    "connected": True,
                    "transport": info["transport"],
                    "tools": [tool.tool_name for tool in info["tools"]],
                }
                for name, info in self.servers.items()
            }

        def register_tools(self, registry):
            names = []
            for name, info in self.servers.items():
                for tool in info["tools"]:
                    tool_name = f"{name}_{tool.tool_name}"
                    registry.register_with_schema(
                        tool_name,
                        tool.description,
                        tool.input_schema,
                        lambda: "fixture",
                        permission=info["permission"],
                    )
                    names.append(tool_name)
            return names

    async with web(tmp_path) as (app, client):
        chat_id = (await client.post("/api/chats", json={"provider": "demo"})).json()["id"]
        chat = app.state.chats[chat_id]
        fake_mcp = FakeMCP()
        chat.mcp = fake_mcp
        chat.agent.memory.add_user_message("Remember this history")
        original_agent = chat.agent

        updated = await client.patch(
            f"/api/chats/{chat_id}/setup",
            json={"system_prompt": "You are a precise data analyst."},
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["system_prompt"] == "You are a precise data analyst."
        assert chat.agent is not original_agent and original_agent.closed
        assert chat.agent.memory.get_messages() == [{"role": "user", "content": "Remember this history"}]

        skill = "---\nname: audit-check\ndescription: Check business-data assumptions\n---\n\nVerify source dates before advising.\n"
        uploaded = await client.post(
            f"/api/chats/{chat_id}/skills",
            json={"name": "audit.md", "content": skill},
        )
        assert uploaded.status_code == 201, uploaded.text
        assert uploaded.json()["skills"] == [
            {
                "name": "audit-check",
                "description": "Check business-data assumptions",
                "file": "audit.md",
                "body_chars": len("Verify source dates before advising."),
            }
        ]
        assert "<available-skills>" in chat.agent.config.system_prompt
        assert "Today's date is" in chat.agent._build_system_prompt(), "the model is told the current date"
        duplicate = await client.post(
            f"/api/chats/{chat_id}/skills",
            json={"name": "same-name.md", "content": skill},
        )
        assert duplicate.status_code == 422 and not (chat.directory / "skills" / "same-name.md").exists()
        invalid_name = await client.post(
            f"/api/chats/{chat_id}/skills",
            json={"name": "../escape.md", "content": skill},
        )
        assert invalid_name.status_code == 422

        connected = await client.post(
            f"/api/chats/{chat_id}/mcp",
            json={"name": "fixture", "command": "fixture-server", "args": "--safe 'path with spaces'"},
        )
        assert connected.status_code == 201, connected.text
        assert fake_mcp.connect_calls == [
            ("fixture", {"command": "fixture-server", "args": ["--safe", "path with spaces"], "url": None, "permission": PermissionLevel.ASK})
        ]
        assert connected.json()["mcp_servers"]["fixture"]["tools"] == ["inspect"]
        tools = {tool["name"]: tool for tool in connected.json()["tools"]}
        assert tools["fixture_inspect"] == {
            "name": "fixture_inspect", "tool": "inspect", "description": "Inspect a configured fixture",
            "source": "mcp", "server": "fixture", "permission": "ask", "enabled": True,
        }
        assert tools["calculate"]["source"] == "builtin" and tools["calculate"]["permission"] == "allow"
        assert tools["write_file"]["permission"] == "ask" and tools["Skill"]["source"] == "skill"
        assert tools["run_bash"]["source"] == "builtin" and tools["run_bash"]["permission"] == "ask"
        assert tools["ask_user"]["permission"] == "allow", "a question needs no approval to be asked"
        assert all(tool["enabled"] for tool in tools.values())

        # Switch the MCP tool off: it leaves the model's registry but stays listed.
        switched = await client.patch(f"/api/chats/{chat_id}/tools", json={"name": "fixture_inspect", "enabled": False})
        assert switched.status_code == 200, switched.text
        listed = {tool["name"]: tool for tool in switched.json()["tools"]}
        assert listed["fixture_inspect"]["enabled"] is False and not chat.agent.tools.has_tool("fixture_inspect")
        assert chat.agent.tools.has_tool("calculate")
        assert (await client.patch(f"/api/chats/{chat_id}/tools", json={"name": "Skill", "enabled": False})).status_code == 422
        assert (await client.patch(f"/api/chats/{chat_id}/tools", json={"name": "nope", "enabled": False})).status_code == 404
        restored = await client.patch(f"/api/chats/{chat_id}/tools", json={"name": "fixture_inspect", "enabled": True})
        assert restored.status_code == 200 and chat.agent.tools.has_tool("fixture_inspect")
        definition = chat.agent.tools.get_tool("fixture_inspect")
        assert definition.permission_level == PermissionLevel.ASK
        denied = await chat.agent.tools.execute(ToolCall("mcp", "fixture_inspect", {}), permissions=chat.agent.permissions)
        assert denied.is_error

        disconnected = await client.delete(f"/api/chats/{chat_id}/mcp/fixture")
        assert disconnected.status_code == 200
        assert disconnected.json()["mcp_servers"] == {}
        assert not [tool for tool in disconnected.json()["tools"] if tool["source"] == "mcp"]
        assert not chat.agent.tools.has_tool("fixture_inspect")
        removed = await client.delete(f"/api/chats/{chat_id}/skills/audit.md")
        assert removed.status_code == 200 and removed.json()["skills"] == []
        assert not (chat.directory / "skills" / "audit.md").exists()
        await client.delete(f"/api/chats/{chat_id}")
        assert fake_mcp.closed


@pytest.mark.asyncio
async def test_chat_setup_rejects_active_turns_and_invalid_mcp_configuration(tmp_path):
    async with web(tmp_path) as (app, client):
        chat_id = (await client.post("/api/chats", json={"provider": "demo"})).json()["id"]
        chat = app.state.chats[chat_id]
        run = app.state.manager.create("active", "chat", workdir=chat.directory)
        run.task = asyncio.create_task(asyncio.sleep(60))
        chat.active = run
        blocked = await client.patch(
            f"/api/chats/{chat_id}/setup", json={"system_prompt": "Changed"}
        )
        assert blocked.status_code == 409
        await run.stop()
        for body in (
            {"name": "both", "command": "tool", "url": "https://example.test/mcp"},
            {"name": "bad", "url": "file:///tmp/server"},
            {"name": "bad/name", "command": "tool"},
            {"name": "quoted", "command": "tool", "args": "'unterminated"},
        ):
            response = await client.post(f"/api/chats/{chat_id}/mcp", json=body)
            assert response.status_code == 422


@pytest.mark.asyncio
async def test_bounded_history_reports_gap_and_early_cancel_is_terminal(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("harness_web.runs.MAX_EVENTS", 3)
    manager = RunManager(tmp_path)
    run = manager.create("bounded", "fixture")
    for _ in range(8):
        await run.emit("output", content="line")
    await run.state("completed")
    events = [event async for event in run.subscribe()]
    assert events[0]["type"] == "gap" and len(events) == 4
    pending = manager.create("cancel early", "fixture")
    pending.task = asyncio.create_task(asyncio.sleep(60))
    await pending.stop()
    assert pending.status == "cancelled" and pending.task.cancelled()


def test_mcp_arguments_validate_transports_and_keep_shell_characters_literal(tmp_path):
    item = CATALOG["mcp_servers"]
    with pytest.raises(ValueError):
        arguments(
            item,
            {"server": "both", "command": "python", "url": "http://example.test"},
            tmp_path,
        )
    with pytest.raises(ValueError):
        arguments(item, {"server": "invalid", "url": "file:///etc/passwd"}, tmp_path)
    args = arguments(
        item,
        {"server": "literal", "command": "python3", "args": "'a b.py' '; echo no'"},
        tmp_path,
    )
    assert args[-1] == "--args='a b.py' '; echo no'"


def test_download_pins_the_file_before_a_concurrent_path_swap(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    report = workspace / "report.txt"
    report.write_text("intended download")
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    with open_workspace_file(workspace, "report.txt") as stream:
        report.unlink()
        report.symlink_to(outside)
        assert stream.read() == b"intended download"
    with pytest.raises(OSError):
        open_workspace_file(workspace, "report.txt")


@pytest.mark.asyncio
async def test_build_your_agent_adds_subagents_that_delegate_with_only_their_tools(tmp_path, monkeypatch):
    async with web(tmp_path) as (app, client):
        chat_id = (await client.post("/api/chats", json={"provider": "demo"})).json()["id"]
        chat = app.state.chats[chat_id]
        (chat.directory / "notes.txt").write_text("ship on friday")
        reviewer = {
            "name": "reviewer", "description": "Reviews a file when asked for a second opinion.",
            "instructions": "You review files and answer in one sentence.", "tools": ["read_file", "list_directory"],
        }
        updated = await client.put(f"/api/chats/{chat_id}/subagents", json={"subagents": [reviewer]})
        assert updated.status_code == 200, updated.text
        body = updated.json()
        assert body["subagents"] == [{**reviewer, "tools": ["list_directory", "read_file"]}]
        tools = {tool["name"]: tool for tool in body["tools"]}
        assert tools["delegate_reviewer"]["source"] == "subagent"

        # Bad definitions are refused and leave the agent as it was.
        for bad, why in [
            ({**reviewer, "tools": ["no_such_tool"]}, "does not have"),
            ({**reviewer, "name": "two words"}, "use 1-55"),
        ]:
            refused = await client.put(f"/api/chats/{chat_id}/subagents", json={"subagents": [bad]})
            assert refused.status_code == 422 and why in refused.text
        twice = await client.put(f"/api/chats/{chat_id}/subagents", json={"subagents": [reviewer, reviewer]})
        assert twice.status_code == 422 and "Two subagents" in twice.text
        assert chat.agent.tools.has_tool("delegate_reviewer")

        # A delegation runs a fresh child with the subagent's prompt and tools.
        seen = []

        def child_provider(*args, **kwargs):
            class Child(ScriptedProvider):
                async def create(self, **request):
                    seen.append(request)
                    return await super().create(**request)

            return Child([
                ProviderResponse(tool_calls=[ToolCall("r", "read_file", {"path": "notes.txt"})], stop_reason="tool_use"),
                ProviderResponse(text="The note says to ship on Friday."),
            ])

        monkeypatch.setattr("harnessx.core.make_provider", child_provider)
        result = await chat.agent.tools.execute(
            ToolCall("d", "delegate_reviewer", {"task": "What does notes.txt say?"}), permissions=chat.agent.permissions,
        )
        assert not result.is_error and result.content == "The note says to ship on Friday."
        assert seen[0]["system"].startswith("You review files")
        # Its own tools, plus read_tool_result, which every agent has for spilled results.
        assert sorted(tool["name"] for tool in seen[0]["tools"]) == ["list_directory", "read_file", "read_tool_result"]

        cleared = await client.put(f"/api/chats/{chat_id}/subagents", json={"subagents": []})
        assert cleared.json()["subagents"] == [] and not chat.agent.tools.has_tool("delegate_reviewer")
