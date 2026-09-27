"""Execute complete documentation examples without contacting live services."""

import asyncio
from pathlib import Path
import re
import socket

import pytest

from harnessx import Agent, ProviderResponse, SQLiteBackend, ToolCall
from examples._fixtures import ScriptedProvider


@pytest.mark.parametrize("document", ["tools", "subagents", "middleware", "extensions"])
def test_documentation_examples(document, monkeypatch, tmp_path):
    content = (Path(__file__).resolve().parents[1] / "doc" / f"{document}.md").read_text()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket.socket, "connect", lambda *_: pytest.fail("Documentation contacted a service"))
    monkeypatch.setattr("harnessx.core.make_provider", lambda _: ScriptedProvider([ProviderResponse(text="Fixture response")]))
    created = []
    original = Agent.__init__

    def track(agent, *args, **kwargs):
        original(agent, *args, **kwargs)
        created.append(agent)

    monkeypatch.setattr(Agent, "__init__", track)

    async def close():
        for agent in created:
            if not agent._closed:
                await agent.aclose()

    try:
        for snippet in re.findall(r"^```python\n(.*?)^```", content, flags=re.MULTILINE | re.DOTALL):
            exec(compile(snippet, f"doc/{document}.md", "exec"), {})
    finally:
        asyncio.run(close())
    assert created and all(agent._closed for agent in created)


@pytest.mark.asyncio
async def test_documented_postgres_start_and_resume_with_fresh_agent(monkeypatch, tmp_path):
    content = (Path(__file__).resolve().parents[1] / "doc/postgres_durability.md").read_text()
    snippets = re.findall(r"^```python\n(.*?)^```", content, flags=re.MULTILINE | re.DOTALL)
    monkeypatch.setattr(socket.socket, "connect", lambda *_: pytest.fail("Documentation contacted a service"))
    stores, providers, created, requests = [], [], [], []

    def backend(dsn):
        store = SQLiteBackend(dsn)
        stores.append(store)
        return store

    class RecoveringProvider(ScriptedProvider):
        async def create(self, **kwargs):
            requests.append(kwargs["messages"])
            # Down for the whole first process: the default in-process retry
            # (two attempts) is exhausted, so the run fails and must be resumed.
            if len(requests) <= 2:
                raise RuntimeError("Model temporarily unavailable")
            return await super().create(**kwargs)

    def provider(_):
        instance = RecoveringProvider([ProviderResponse(text="Recovered answer")])
        providers.append(instance)
        return instance

    original = Agent.__init__

    def track(agent, *args, **kwargs):
        original(agent, *args, **kwargs)
        created.append(agent)

    monkeypatch.setattr("harnessx.PostgresBackend", backend)
    monkeypatch.setattr("harnessx.core.make_provider", provider)
    monkeypatch.setattr(Agent, "__init__", track)
    namespace = {}
    for snippet in snippets:
        exec(compile(snippet, "doc/postgres_durability.md", "exec"), namespace)

    dsn = str(tmp_path / "documented-runtime.db")
    session_id, interrupted = await namespace["start_session"](dsn)
    assert interrupted.status == "failed"
    recovered = await namespace["resume_session"](dsn, session_id)
    assert recovered.status == "completed" and recovered.output == "Recovered answer"
    assert recovered.run_id == interrupted.run_id
    assert len(requests) == 3 and requests[1] == requests[0] and requests[2] == requests[0]
    assert await namespace["resume_session"](dsn, session_id) is None
    assert len(created) == len(providers) == len(stores) == 3
    assert len({id(agent) for agent in created}) == 3
    assert all(agent._closed for agent in created)
    assert all(store.conn is None for store in stores)


@pytest.mark.asyncio
async def test_documented_extension_transforms_persists_and_uninstalls(monkeypatch, tmp_path):
    text = (Path(__file__).resolve().parents[1] / "doc/extensions.md").read_text()
    snippet = re.findall(r"^```python\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL)[0]
    monkeypatch.setattr("harnessx.core.make_provider", lambda _: ScriptedProvider([]))
    namespace = {}
    exec(compile(snippet, "doc/extensions.md", "exec"), namespace)
    await namespace["agent"].aclose()
    extension = namespace["SearchExtension"]("fixture_docs")
    provider = ScriptedProvider([
        ProviderResponse(tool_calls=[ToolCall("search", "search", {"query": "  runtime  "})], stop_reason="tool_use"),
        ProviderResponse(text="Found the reference"),
    ])
    async with Agent(provider=provider, extensions=[extension]) as agent:
        result = await agent.run("Search for runtime")
        assert result.status == "completed" and extension.completed_searches == 1
        results = [
            block for message in agent.memory.get_messages() for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert results[0]["content"] == "fixture_docs: runtime"
        sid = await agent.save_session(str(tmp_path))
    assert not agent.tools.has_tool("search")
    assert not agent.middleware._middleware and not agent.prompt_providers
    restored_extension = namespace["SearchExtension"]("fixture_docs")
    restored = await Agent.load_session(
        sid, str(tmp_path), provider=ScriptedProvider([]), extensions=[restored_extension],
    )
    async with restored:
        assert restored_extension.completed_searches == 1
        assert restored.tools.has_tool("search")
