"""Tests for LensExtension in harnessx."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
import unittest
from unittest.mock import MagicMock

import httpx

from harnessx import (
    Agent,
    AgentConfig,
    AnthropicProvider,
    LensExtension,
    PermissionLevel,
    TokenUsage,
)
from harnessx.providers.base import LLMProvider


# ── Mock Response Structures ────────────────────────────────────────────────


@dataclass
class MockTextBlock:
    text: str
    type: str = "text"


@dataclass
class MockToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class MockResponse:
    content: list[Any]
    stop_reason: str = "end_turn"
    usage: TokenUsage = field(
        default_factory=lambda: TokenUsage(input_tokens=10, output_tokens=5)
    )


class MockProvider(LLMProvider):
    """Predictable mock provider for offline testing."""

    def __init__(self, responses: list[MockResponse]) -> None:
        self._responses = list(responses)

    async def create(self, **kwargs: Any) -> MockResponse:
        if self._responses:
            return self._responses.pop(0)
        return MockResponse(content=[MockTextBlock(text="Default done")], stop_reason="end_turn")

    async def count_tokens(self, **kwargs: Any) -> int:
        return 10


class LensServer:
    """Records what the extension posts; answers with ``status`` or raises ``fail``."""

    def __init__(self, status: int = 200, fail: Exception | None = None, delay: float = 0.0) -> None:
        self.requests: list[httpx.Request] = []
        self.status, self.fail, self.delay = status, fail, delay

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        return httpx.Response(self.status, json={"id": "trace-1"})

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    @property
    def traces(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def lens(server: LensServer, **kwargs: Any) -> LensExtension:
    options: dict[str, Any] = {
        "enabled": True,
        "api_url": "http://lens.test/",
        "api_key": "lens-key",
        "project_id": "proj-1",
        "http_client": server.client(),
    }
    options.update(kwargs)
    return LensExtension(**options)


# ── Test Suite ──────────────────────────────────────────────────────────────


class TestLensExtension(unittest.IsolatedAsyncioTestCase):

    async def test_not_configured_is_a_no_op(self):
        """Without a key, a project id, or the enable flag, nothing is installed or sent."""
        server = LensServer()
        for ext in (
            lens(server, api_key=""),
            lens(server, project_id=""),
            lens(server, enabled=False),
        ):
            self.assertFalse(ext.configured)
            agent = Agent(provider=MockProvider([]), extensions=[ext])
            self.assertFalse(any(type(m).__name__ == "_LensMiddleware" for m in agent.middleware._middleware))
            async with agent:
                await agent.run("hello")
        self.assertEqual(server.requests, [])

    def test_configuration_falls_back_to_the_environment(self):
        env = {
            "DATAGOL_LENS_ENABLED": "true",
            "DATAGOL_LENS_API_URL": "http://env-lens:3014",
            "DATAGOL_LENS_API_KEY": "env-key",
            "DATAGOL_LENS_PROJECT_ID": "env-proj",
            "DATAGOL_LENS_MAX_FIELD_CHARS": "123",
        }
        with unittest.mock.patch.dict("os.environ", env):
            ext = LensExtension()
        self.assertTrue(ext.configured)
        self.assertEqual((ext.api_url, ext.project_id, ext.max_field_chars), ("http://env-lens:3014", "env-proj", 123))
        with unittest.mock.patch.dict("os.environ", {**env, "DATAGOL_LENS_ENABLED": "false"}):
            self.assertFalse(LensExtension().configured)

    async def test_single_turn_trace(self):
        """One POST per turn, with the project URL, the bearer key, and an llm span."""
        server = LensServer()
        ext = lens(server, run_name="Main", session_id="thread-1", user_id="u@example.com",
                   tags=["unit"], metadata={"env": "test"})
        agent = Agent(
            config=AgentConfig(model="claude-sonnet-4-6", provider="anthropic"),
            provider=MockProvider([MockResponse(content=[MockTextBlock(text="42")])]),
            extensions=[ext],
        )
        async with agent:
            self.assertEqual((await agent.run("What is 17 + 25?")).output, "42")

        self.assertEqual(len(server.requests), 1)
        request = server.requests[0]
        self.assertEqual(str(request.url), "http://lens.test/api/v1/projects/proj-1/traces")
        self.assertEqual(request.headers["authorization"], "Bearer lens-key")

        trace = server.traces[0]
        self.assertEqual((trace["name"], trace["agent_name"], trace["id"]), ("Main", "Main", ext.trace_id))
        self.assertEqual(trace["input"], {"query": "What is 17 + 25?"})
        self.assertEqual(trace["output"], {"content": "42"})
        self.assertEqual(trace["status"], "success")
        self.assertIsNone(trace["error"])
        self.assertEqual((trace["session_id"], trace["external_user_id"]), ("thread-1", "u@example.com"))
        self.assertEqual(trace["tags"], ["unit"])
        self.assertEqual(trace["metadata"]["env"], "test")
        self.assertEqual(trace["model"], "claude-sonnet-4-6")
        self.assertEqual((trace["prompt_tokens"], trace["completion_tokens"], trace["total_tokens"]), (10, 5, 15))
        self.assertNotIn("lens-key", request.content.decode())

        [span] = trace["spans"]
        self.assertEqual((span["name"], span["span_type"], span["model"]), ("llm_call", "llm", "claude-sonnet-4-6"))
        self.assertEqual(span["input"]["system"], agent._build_system_prompt())
        self.assertEqual(span["input"]["messages"][0]["role"], "user")
        self.assertEqual(span["input"]["tools"], ["read_tool_result", "write_todos", "read_todos"])
        self.assertEqual(span["output"]["content"], "42")
        self.assertEqual(span["total_tokens"], 15)
        self.assertEqual(span["metadata"]["provider"], "anthropic")
        self.assertGreater(span["cost"], 0)
        self.assertAlmostEqual(trace["total_cost"], span["cost"])

    async def test_cost_and_cache_details_come_from_the_harness_pricing(self):
        server = LensServer()
        usage = TokenUsage(input_tokens=1000, output_tokens=60, cache_read_input_tokens=900,
                           cache_creation_input_tokens=50, thinking_tokens=20)
        agent = Agent(config=AgentConfig(model="m"),
                      provider=MockProvider([MockResponse(content=[MockTextBlock(text="ok")], usage=usage)]),
                      extensions=[lens(server)])
        async with agent:
            await agent.run("hi")
            expected = agent.guardrails.cost_of(usage)

        span = server.traces[0]["spans"][0]
        self.assertAlmostEqual(span["cost"], expected)
        self.assertEqual(
            {k: span[k] for k in ("prompt_tokens", "completion_tokens", "cache_read_tokens",
                                  "cache_write_tokens", "reasoning_tokens", "cache_tokens_in_prompt")},
            {"prompt_tokens": 1000, "completion_tokens": 60, "cache_read_tokens": 900,
             "cache_write_tokens": 50, "reasoning_tokens": 20, "cache_tokens_in_prompt": True},
        )
        trace = server.traces[0]
        self.assertEqual((trace["cache_read_tokens"], trace["cache_write_tokens"]), (900, 50))

    async def test_tool_spans_are_keyed_by_call_id(self):
        """Two calls in one reply each get their own span with their own result."""
        server = LensServer()
        provider = MockProvider([
            MockResponse(
                content=[
                    MockToolUseBlock(id="call_1", name="add", input={"a": 1, "b": 2}),
                    MockToolUseBlock(id="call_2", name="add", input={"a": 10, "b": 20}),
                ],
                stop_reason="tool_use",
            ),
            MockResponse(content=[MockTextBlock(text="3 and 30")]),
        ])
        agent = Agent(config=AgentConfig(model="m"), provider=provider, extensions=[lens(server)])

        @agent.tools.register(permission=PermissionLevel.ALLOW)
        async def add(a: int, b: int) -> int:
            await asyncio.sleep(0)
            return a + b

        async with agent:
            await agent.run("add twice")

        spans = server.traces[0]["spans"]
        self.assertEqual([s["span_type"] for s in spans].count("llm"), 2)
        tools = {json.dumps(s["input"], sort_keys=True): s for s in spans if s["span_type"] == "tool"}
        self.assertEqual(tools[json.dumps({"a": 1, "b": 2}, sort_keys=True)]["output"], {"result": "3"})
        self.assertEqual(tools[json.dumps({"a": 10, "b": 20}, sort_keys=True)]["output"], {"result": "30"})
        self.assertTrue(all(s["name"] == "add" for s in tools.values()))
        first_llm = next(s for s in spans if s["span_type"] == "llm")
        self.assertEqual([c["id"] for c in first_llm["output"]["tool_calls"]], ["call_1", "call_2"])

    async def test_tool_error_is_marked(self):
        server = LensServer()
        provider = MockProvider([
            MockResponse(content=[MockToolUseBlock(id="c1", name="boom", input={})], stop_reason="tool_use"),
            MockResponse(content=[MockTextBlock(text="it failed")]),
        ])
        agent = Agent(config=AgentConfig(model="m"), provider=provider, extensions=[lens(server)])

        @agent.tools.register(permission=PermissionLevel.ALLOW)
        def boom() -> str:
            raise ValueError("nope")

        async with agent:
            await agent.run("go")
        tool = next(s for s in server.traces[0]["spans"] if s["span_type"] == "tool")
        self.assertEqual(tool["status"], "error")
        self.assertIn("nope", tool["output"]["result"])
        self.assertIn("nope", tool["error"])

    async def test_sub_agent_joins_the_parent_trace(self):
        """A sub-agent run from a tool adds its spans to the parent's trace; no second trace."""
        server = LensServer()
        sub = Agent(
            config=AgentConfig(model="sub-model"),
            provider=MockProvider([MockResponse(content=[MockTextBlock(text="Specialist answer")])]),
            extensions=[lens(server, run_name="Specialist")],
        )
        orchestrator = Agent(
            config=AgentConfig(model="orch-model"),
            provider=MockProvider([
                MockResponse(content=[MockToolUseBlock(id="call_sub", name="delegate", input={"q": "deep"})],
                             stop_reason="tool_use"),
                MockResponse(content=[MockTextBlock(text="Synthesized answer")]),
            ]),
            extensions=[lens(server, run_name="Orchestrator")],
        )

        @orchestrator.tools.register(permission=PermissionLevel.ALLOW)
        async def delegate(q: str) -> str:
            return (await sub.run(q)).output

        async with orchestrator, sub:
            self.assertEqual((await orchestrator.run("Solve it")).output, "Synthesized answer")

        self.assertEqual(len(server.traces), 1)
        trace = server.traces[0]
        self.assertEqual(trace["name"], "Orchestrator")
        nested = [s for s in trace["spans"] if s["metadata"].get("agent") == "Specialist"]
        self.assertEqual(len(nested), 1)
        self.assertEqual((nested[0]["model"], nested[0]["metadata"]["parent_tool"]), ("sub-model", "delegate"))
        # The sub-agent's spans nest under the tool span that ran it; the parent's are top level.
        tool_span = next(s for s in trace["spans"] if s["name"] == "delegate")
        self.assertEqual(nested[0]["parent_id"], tool_span["id"])
        self.assertIsNone(tool_span["parent_id"])
        self.assertEqual(len({s["id"] for s in trace["spans"]}), len(trace["spans"]))
        # Tokens from both agents count toward the trace.
        self.assertEqual(trace["total_tokens"], 45)

    async def test_failed_run_has_error_status(self):
        server = LensServer()

        class FailingProvider(LLMProvider):
            async def create(self, **kwargs: Any) -> Any:
                raise RuntimeError("API failure")

            async def count_tokens(self, **kwargs: Any) -> int:
                return 0

        agent = Agent(config=AgentConfig(model="m"), provider=FailingProvider(), extensions=[lens(server)])
        async with agent:
            result = await agent.run("Fail now")
        self.assertEqual(result.status, "failed")

        trace = server.traces[0]
        self.assertEqual(trace["status"], "error")
        self.assertIn("API failure", trace["error"])
        llm = [s for s in trace["spans"] if s["span_type"] == "llm"]
        self.assertTrue(llm and all(s["status"] == "error" and s["error"] for s in llm))

    async def test_streaming_turn(self):
        """run_stream sends the same trace, and the extension does not buffer the text."""
        server = LensServer()

        class MockStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args: Any):
                pass

            @property
            def text_stream(self):
                async def _gen():
                    yield "Hello "
                    yield "world!"
                return _gen()

            async def get_final_message(self):
                return MockResponse(content=[MockTextBlock(text="Hello world!")])

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = MockStream()
        agent = Agent(config=AgentConfig(model="claude-sonnet-4-6"),
                      provider=AnthropicProvider(client=mock_client), extensions=[lens(server)])
        self.assertFalse(any(
            type(m).after_llm_call is not type(m).__mro__[-2].after_llm_call
            for m in agent.middleware._middleware if type(m).__name__ == "_LensMiddleware"
        ))
        async with agent:
            async for _ in agent.run_stream("Hi streaming"):
                pass

        trace = server.traces[0]
        self.assertEqual(trace["input"], {"query": "Hi streaming"})
        self.assertEqual(trace["output"], {"content": "Hello world!"})
        self.assertEqual([s["span_type"] for s in trace["spans"]], ["llm"])

    async def test_held_trace_takes_the_application_output(self):
        """With auto_flush=False the application can replace the answer before the send."""
        server = LensServer()
        ext = lens(server, auto_flush=False)
        agent = Agent(config=AgentConfig(model="m"),
                      provider=MockProvider([MockResponse(content=[MockTextBlock(text="raw [1]")])]),
                      extensions=[ext])
        async with agent:
            await agent.run("q")
            await asyncio.sleep(0)
            self.assertEqual(server.requests, [])
            ext.set_output({"content": "clean"})
            ext.add_metadata(sources=["a.pdf"])
            await ext.flush()
            self.assertEqual(len(server.requests), 1)
        trace = server.traces[0]
        self.assertEqual(trace["output"], {"content": "clean"})
        self.assertEqual(trace["metadata"]["sources"], ["a.pdf"])

    async def test_send_failures_never_reach_the_run(self):
        """A refused, broken, or slow Lens is logged and dropped; the run still succeeds."""
        for server in (
            LensServer(status=500),
            LensServer(fail=httpx.ConnectError("down")),
            LensServer(delay=1.0),
        ):
            agent = Agent(config=AgentConfig(model="m"),
                          provider=MockProvider([MockResponse(content=[MockTextBlock(text="ok")])]),
                          extensions=[lens(server, flush_timeout_seconds=0.05)])
            with self.assertLogs("harnessx.extensions.lens", level="WARNING"):
                async with agent:
                    self.assertEqual((await agent.run("hi")).output, "ok")
            self.assertEqual(len(server.requests), 1)

    async def test_long_fields_are_truncated_and_images_left_out(self):
        server = LensServer()
        agent = Agent(config=AgentConfig(model="m"),
                      provider=MockProvider([MockResponse(content=[MockTextBlock(text="y" * 500)])]),
                      extensions=[lens(server, max_field_chars=50)])
        async with agent:
            await agent.run("x" * 500)
        trace = server.traces[0]
        self.assertTrue(trace["input"]["query"].endswith("…[truncated]"))
        self.assertLessEqual(len(trace["input"]["query"]), 50 + len("…[truncated]"))
        self.assertTrue(trace["spans"][0]["output"]["content"].endswith("…[truncated]"))

        from harnessx.extensions.lens import _jsonable
        image = {"type": "image", "source": {"type": "base64", "data": "A" * 10_000}}
        self.assertEqual(_jsonable([image], 0), [{"type": "image", "source": "[omitted]"}])

    async def test_teardown_sends_an_unfinished_turn_as_stopped(self):
        server = LensServer()
        ext = lens(server)
        agent = Agent(config=AgentConfig(model="m"), provider=MockProvider([]), extensions=[ext])
        from harnessx.hooks import HookContext, HookEvent
        await ext._on_agent_start(HookContext(event=HookEvent.AGENT_START, agent=agent, data={"message": "q"}))
        await agent.aclose()
        self.assertEqual(server.traces[0]["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
