"""Tests for LangSmithExtension in harnessx."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

from harnessx import (
    AgentConfig,
    AnthropicProvider,
    LangSmithExtension,
    PermissionLevel,
    Agent,
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
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> MockResponse:
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return MockResponse(content=[MockTextBlock(text="Default done")], stop_reason="end_turn")

    async def count_tokens(self, **kwargs: Any) -> int:
        return 10


# ── Test Suite ──────────────────────────────────────────────────────────────


class TestLangSmithExtension(unittest.IsolatedAsyncioTestCase):

    def test_extension_name_and_missing_dependency(self):
        """Verify extension metadata and graceful missing package handling."""
        self.assertEqual(LangSmithExtension.name, "langsmith")

        with patch("harnessx.extensions.langsmith._LANGSMITH_AVAILABLE", False):
            with self.assertRaises(ImportError) as ctx:
                LangSmithExtension()
            self.assertIn("The 'langsmith' package is required", str(ctx.exception))

    async def test_trace_single_turn_lifecycle(self):
        """Verify root chain run and child llm run are created and completed."""
        posted_runs: list[Any] = []
        patched_runs: list[Any] = []

        mock_provider = MockProvider(
            [MockResponse(content=[MockTextBlock(text="42")], stop_reason="end_turn")]
        )

        with patch("harnessx.extensions.langsmith._safe_post", side_effect=posted_runs.append), \
             patch("harnessx.extensions.langsmith._safe_patch", side_effect=patched_runs.append):

            ext = LangSmithExtension(
                project_name="test-project",
                tags=["unit-test"],
                metadata={"env": "testing"},
            )

            agent = Agent(
                config=AgentConfig(model="claude-sonnet-4-6", provider="anthropic"),
                provider=mock_provider,
                extensions=[ext],
            )

            result = (await agent.run("What is 17 + 25?")).output
            self.assertEqual(result, "42")

        # Verify posted runs: 1 root chain run + 1 child llm run
        self.assertEqual(len(posted_runs), 2)
        root_run = posted_runs[0]
        llm_run = posted_runs[1]

        self.assertEqual(root_run.run_type, "chain")
        self.assertEqual(root_run.inputs, {"input": "What is 17 + 25?"})
        self.assertEqual(root_run.session_name, "test-project")
        self.assertIn("unit-test", root_run.tags)
        self.assertEqual(root_run.extra["metadata"]["env"], "testing")
        self.assertEqual(root_run.extra["metadata"]["model"], "claude-sonnet-4-6")

        self.assertEqual(llm_run.run_type, "llm")
        self.assertEqual(llm_run.parent_run_id, root_run.id)
        # The span records the request as sent, not just the messages.
        self.assertEqual(llm_run.inputs["system"], agent._build_system_prompt())
        self.assertEqual(llm_run.inputs["messages"][0]["role"], "user")
        self.assertEqual(
            [t["name"] for t in llm_run.inputs["tools"]],
            ["read_tool_result", "write_todos", "read_todos"],
        )
        self.assertEqual(llm_run.extra["metadata"]["ls_model_name"], "claude-sonnet-4-6")
        self.assertEqual(llm_run.extra["metadata"]["ls_provider"], "anthropic")
        self.assertIn("prefix_key", llm_run.extra["metadata"])
        params = llm_run.extra["invocation_params"]
        self.assertEqual((params["model"], params["max_tokens"], params["temperature"], params["stream"]),
                         ("claude-sonnet-4-6", 8192, None, False))
        self.assertEqual(llm_run.extra["metadata"]["usage_metadata"],
                         {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

        # Verify patched (completed) runs
        self.assertEqual(len(patched_runs), 2)
        # LLM run patched first, then root run
        self.assertEqual(patched_runs[0].id, llm_run.id)
        self.assertEqual(patched_runs[1].id, root_run.id)
        self.assertEqual(root_run.outputs, {"output": "42"})

    async def test_usage_details_include_cache_and_reasoning_tokens(self):
        """Cached and reasoning tokens reach LangSmith in its token-details shape."""
        posted_runs: list[Any] = []
        patched_runs: list[Any] = []
        usage = TokenUsage(input_tokens=1000, output_tokens=60, cache_read_input_tokens=900,
                           cache_creation_input_tokens=50, thinking_tokens=20)
        provider = MockProvider([MockResponse(content=[MockTextBlock(text="ok")], usage=usage)])

        with patch("harnessx.extensions.langsmith._safe_post", side_effect=posted_runs.append), \
             patch("harnessx.extensions.langsmith._safe_patch", side_effect=patched_runs.append):
            agent = Agent(config=AgentConfig(model="m"), provider=provider,
                          extensions=[LangSmithExtension(project_name="p")])
            await agent.run("hi")

        llm_run = next(run for run in posted_runs if run.run_type == "llm")
        self.assertEqual(llm_run.extra["metadata"]["usage_metadata"], {
            "input_tokens": 1000, "output_tokens": 60, "total_tokens": 1060,
            "input_token_details": {"cache_read": 900, "cache_creation": 50},
            "output_token_details": {"reasoning": 20},
        })

    async def test_trace_tool_execution(self):
        """Verify tool calls create child spans with inputs, outputs, and errors."""
        posted_runs: list[Any] = []
        patched_runs: list[Any] = []

        mock_provider = MockProvider(
            [
                # First turn: call tool 'add'
                MockResponse(
                    content=[MockToolUseBlock(id="call_1", name="add", input={"a": 10, "b": 20})],
                    stop_reason="tool_use",
                ),
                # Second turn: final response
                MockResponse(
                    content=[MockTextBlock(text="The sum is 30")],
                    stop_reason="end_turn",
                ),
            ]
        )

        with patch("harnessx.extensions.langsmith._safe_post", side_effect=posted_runs.append), \
             patch("harnessx.extensions.langsmith._safe_patch", side_effect=patched_runs.append):

            ext = LangSmithExtension(project_name="tool-tests")
            agent = Agent(
                config=AgentConfig(model="claude-sonnet-4-6"),
                provider=mock_provider,
                extensions=[ext],
            )

            @agent.tools.register(permission=PermissionLevel.ALLOW)
            def add(a: int, b: int) -> int:
                return a + b

            result = (await agent.run("Compute 10 + 20")).output
            self.assertEqual(result, "The sum is 30")

        # Runs posted:
        # 1. Root chain run
        # 2. LLM call 1 (returns tool_use)
        # 3. Tool run ('add')
        # 4. LLM call 2 (returns text)
        self.assertEqual(len(posted_runs), 4)
        root_run = posted_runs[0]
        tool_run = posted_runs[2]

        self.assertEqual(tool_run.run_type, "tool")
        self.assertEqual(tool_run.name, "add")
        self.assertEqual(tool_run.inputs, {"a": 10, "b": 20})
        self.assertEqual(tool_run.parent_run_id, root_run.id)
        self.assertEqual(tool_run.outputs, {"output": "30"})
        self.assertIsNone(tool_run.error)

    async def test_trace_nested_sub_agent(self):
        """Verify sub-agents invoked from tools automatically nest under the parent tool run."""
        posted_runs: list[Any] = []
        patched_runs: list[Any] = []

        sub_provider = MockProvider(
            [MockResponse(content=[MockTextBlock(text="Specialist answer")], stop_reason="end_turn")]
        )
        orchestrator_provider = MockProvider(
            [
                MockResponse(
                    content=[MockToolUseBlock(id="call_sub", name="delegate", input={"q": "deep question"})],
                    stop_reason="tool_use",
                ),
                MockResponse(
                    content=[MockTextBlock(text="Synthesized answer")],
                    stop_reason="end_turn",
                ),
            ]
        )

        with patch("harnessx.extensions.langsmith._safe_post", side_effect=posted_runs.append), \
             patch("harnessx.extensions.langsmith._safe_patch", side_effect=patched_runs.append):

            ext_orch = LangSmithExtension(project_name="multi-agent", run_name="Orchestrator")
            orchestrator = Agent(
                config=AgentConfig(model="claude-sonnet-4-6"),
                provider=orchestrator_provider,
                extensions=[ext_orch],
            )

            ext_sub = LangSmithExtension(project_name="multi-agent", run_name="Specialist")
            sub_agent = Agent(
                config=AgentConfig(model="claude-sonnet-4-6"),
                provider=sub_provider,
                extensions=[ext_sub],
            )

            @orchestrator.tools.register(permission=PermissionLevel.ALLOW)
            async def delegate(q: str) -> str:
                return (await sub_agent.run(q)).output

            res = (await orchestrator.run("Solve complex problem")).output
            self.assertEqual(res, "Synthesized answer")

        # Let's inspect the hierarchy:
        # 1. Orchestrator root (chain)
        # 2. Orchestrator LLM 1 (llm)
        # 3. Tool run: delegate (tool)
        # 4. Specialist root (chain) -> parent MUST BE tool run 'delegate'!
        # 5. Specialist LLM (llm) -> parent MUST BE Specialist root!
        # 6. Orchestrator LLM 2 (llm)
        self.assertEqual(len(posted_runs), 6)
        orch_root = posted_runs[0]
        delegate_tool = posted_runs[2]
        specialist_root = posted_runs[3]
        specialist_llm = posted_runs[4]

        self.assertEqual(orch_root.name, "Orchestrator")
        self.assertEqual(delegate_tool.name, "delegate")
        self.assertEqual(delegate_tool.parent_run_id, orch_root.id)

        # Specialist nested under the delegate tool run:
        self.assertEqual(specialist_root.name, "Specialist")
        self.assertEqual(specialist_root.parent_run_id, delegate_tool.id)

        # Specialist LLM nested under Specialist root:
        self.assertEqual(specialist_llm.parent_run_id, specialist_root.id)

    async def test_error_handling(self):
        """Verify error is reported on active spans if an exception occurs."""
        posted_runs: list[Any] = []
        patched_runs: list[Any] = []

        class FailingProvider(LLMProvider):
            async def create(self, **kwargs: Any) -> Any:
                raise RuntimeError("API failure")

            async def count_tokens(self, **kwargs: Any) -> int:
                return 0

        with patch("harnessx.extensions.langsmith._safe_post", side_effect=posted_runs.append), \
             patch("harnessx.extensions.langsmith._safe_patch", side_effect=patched_runs.append):

            ext = LangSmithExtension(project_name="error-test")
            agent = Agent(
                config=AgentConfig(model="claude-sonnet-4-6"),
                provider=FailingProvider(),
                extensions=[ext],
            )

            result = await agent.run("Fail now")
            self.assertEqual(result.status, "failed")
            self.assertIn("API failure", result.error["message"])

        # Root run and LLM run should have ended with error
        self.assertTrue(any("API failure" in str(r.error) for r in patched_runs))

    async def test_disabled_extension(self):
        """Verify nothing is traced when enabled=False."""
        posted_runs: list[Any] = []
        mock_provider = MockProvider(
            [MockResponse(content=[MockTextBlock(text="ok")], stop_reason="end_turn")]
        )

        with patch("harnessx.extensions.langsmith._safe_post", side_effect=posted_runs.append):
            ext = LangSmithExtension(enabled=False)
            agent = Agent(provider=mock_provider, extensions=[ext])
            (await agent.run("hello")).output

        self.assertEqual(len(posted_runs), 0)

    async def test_streaming_agent_lifecycle(self):
        """Verify Agent turn events and LLM spans are traced."""
        posted_runs: list[Any] = []
        patched_runs: list[Any] = []

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
                return MockResponse(
                    content=[MockTextBlock(text="Hello world!")],
                    stop_reason="end_turn",
                )

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = MockStream()

        with patch("harnessx.extensions.langsmith._safe_post", side_effect=posted_runs.append), \
             patch("harnessx.extensions.langsmith._safe_patch", side_effect=patched_runs.append):

            ext = LangSmithExtension(project_name="stream-test")
            streaming_agent = Agent(
                config=AgentConfig(model="claude-sonnet-4-6"),
                provider=AnthropicProvider(client=mock_client),
                extensions=[ext],
            )

            events = []
            async for ev in streaming_agent.run_stream("Hi streaming"):
                events.append(ev)

        self.assertEqual(len(posted_runs), 2)  # root chain + child llm
        self.assertEqual(posted_runs[0].inputs, {"input": "Hi streaming"})
        self.assertEqual(patched_runs[1].outputs, {"output": "Hello world!"})


if __name__ == "__main__":
    unittest.main()


class TestProviderLabel(unittest.IsolatedAsyncioTestCase):
    """A failed-over call must be labelled with the vendor that answered.

    LangSmith prices and filters runs by ls_provider, so reporting the
    configured primary after a failover misattributes exactly the runs an
    operator is most likely to be looking at.
    """

    async def test_failover_is_labelled_with_the_member_that_served(self):
        from harnessx.extensions.langsmith import _serving_provider

        agent = Agent(
            config=AgentConfig(model="m", provider="anthropic"),
            provider=MockProvider([MockResponse(content=[MockTextBlock(text="hi")])]),
        )
        chain = agent.provider
        chain.name = "fallback"          # it is a chain, not a vendor
        chain.last_served = None
        self.assertEqual(_serving_provider(agent), "anthropic", "falls back to the config")

        chain.last_served = "openai"     # a call was served by the fallback member
        self.assertEqual(_serving_provider(agent), "openai")
        await agent.aclose()

    async def test_a_plain_provider_reports_its_own_name(self):
        from harnessx.extensions.langsmith import _serving_provider

        provider = MockProvider([MockResponse(content=[MockTextBlock(text="hi")])])
        provider.name = "openrouter"
        agent = Agent(config=AgentConfig(model="m", provider="openrouter"), provider=provider)
        self.assertEqual(_serving_provider(agent), "openrouter")
        await agent.aclose()
