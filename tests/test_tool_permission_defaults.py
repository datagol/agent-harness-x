"""Registration syntax must not change permission behavior."""

import httpx
import pytest

from harnessx import Agent, PermissionLevel, PermissionManager, ToolCall, ToolRegistry
from harnessx.builtin import register_all_tools, register_filesystem_tools, register_web_tools
from harnessx.builtin.web import fetch_url
from scripted_provider import ScriptedProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["function", "string", "bundle", "helper", "later", "load", "all", "definition", "decorator"])
async def test_fetch_permissions_are_independent_of_registration(route, monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, text="fixture page")

    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs))
    registry = ToolRegistry()
    definition = registry.register_tool(fetch_url)
    initial = {
        "function": [fetch_url], "string": ["fetch_url"], "bundle": ["web"],
        "definition": [definition],
    }.get(route)
    async with Agent(provider=ScriptedProvider([]), tools=initial) as agent:
        if route == "helper":
            register_web_tools(agent.tools)
        elif route == "later":
            agent.tools.register_tool(fetch_url)
        elif route == "load":
            agent.tools.load_builtin("fetch_url")
        elif route == "all":
            register_all_tools(agent.tools, include=["fetch_url"])
        elif route == "decorator":
            agent.tools.register()(fetch_url)
        call = ToolCall("fetch", "fetch_url", {"url": "https://example.test"})
        result = await agent.tools.execute(call)
        assert not result.is_error and "fixture page" in result.content
        assert len(requests) == 1
        for level in (PermissionLevel.ASK, PermissionLevel.DENY):
            manager = PermissionManager(level)
            denied = await agent.tools.execute(call, permissions=manager)
            assert denied.is_error and len(requests) == 1
        approvals = []
        manager = PermissionManager(PermissionLevel.ASK, approval_callback=lambda call, _: approvals.append(call.name) or True)
        assert not (await agent.tools.execute(call, permissions=manager)).is_error
        assert approvals == ["fetch_url"] and len(requests) == 2
        manager.set_permission("fetch_url", PermissionLevel.DENY)
        manager.grant_session("fetch_url")
        assert (await agent.tools.execute(call, permissions=manager)).is_error
        assert len(requests) == 2


@pytest.mark.parametrize("bundle", ["filesystem", "web", "bash", "memory", "all"])
def test_bundles_preserve_explicit_policy_and_inherit_when_omitted(bundle, tmp_path):
    for explicit in (None, PermissionLevel.ALLOW, PermissionLevel.ASK, PermissionLevel.DENY):
        registry = ToolRegistry()
        options = {"base_path": str(tmp_path)} if bundle in ("filesystem", "all") else {}
        registry.load_builtin(bundle, permission=explicit, **options)
        for tool in registry.get_tools():
            assert tool.permission_level == explicit
            for default in PermissionLevel:
                manager = PermissionManager(default)
                assert manager.get_effective_permission(tool.name, tool) == (explicit or default)
                manager.set_permission(tool.name, PermissionLevel.DENY)
                assert manager.get_effective_permission(tool.name, tool) == PermissionLevel.DENY
        for name in ("write_file", "generate_file"):
            if registry.has_tool(name):
                assert not registry.get_tool(name).concurrent


@pytest.mark.asyncio
async def test_filesystem_allow_default_and_explicit_ask(tmp_path):
    registry = ToolRegistry()
    register_filesystem_tools(registry, base_path=str(tmp_path))
    call = ToolCall("write", "write_file", {"path": "report.txt", "content": "first"})
    assert not (await registry.execute(call)).is_error
    assert (tmp_path / "report.txt").read_text() == "first"
    register_filesystem_tools(registry, base_path=str(tmp_path), permission=PermissionLevel.ASK, replace=True)
    call.input.update(content="second", overwrite=True)
    assert (await registry.execute(call)).is_error
    assert (tmp_path / "report.txt").read_text() == "first"
    manager = PermissionManager(approval_callback=lambda *_: True)
    assert not (await registry.execute(call, permissions=manager)).is_error
    assert (tmp_path / "report.txt").read_text() == "second"
