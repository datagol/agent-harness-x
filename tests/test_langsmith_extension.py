"""Tests for LangSmithExtension in datagol-agent-harness."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

from datagol_agent_harness import (
    Agent,
    AgentConfig,
    LangSmithExtension,
    PermissionLevel,
    StreamingAgent,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from datagol_agent_harness.providers.base import LLMProvider


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

        with patch("datagol_agent_harness.extensions.langsmith._LANGSMITH_AVAILABLE", False):
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

        with patch("datagol_agent_harness.extensions.langsmith._safe_post", side_effect=posted_runs.append), \
             patch("datagol_agent_harness.extensions.langsmith._safe_patch", side_effect=patched_runs.append):

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

            result = await agent.run("What is 17 + 25?")
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

        # Verify patched (completed) runs
        self.assertEqual(len(patched_runs), 2)
        # LLM run patched first, then root run
        self.assertEqual(patched_runs[0].id, llm_run.id)
        self.assertEqual(patched_runs[1].id, root_run.id)
        self.assertEqual(root_run.outputs, {"output": "42"})

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

        with patch("datagol_agent_harness.extensions.langsmith._safe_post", side_effect=posted_runs.append), \
             patch("datagol_agent_harness.extensions.langsmith._safe_patch", side_effect=patched_runs.append):

            ext = LangSmithExtension(project_name="tool-tests")
            agent = Agent(
                config=AgentConfig(model="claude-sonnet-4-6"),
                provider=mock_provider,
                extensions=[ext],
            )

            @agent.tools.register(permission=PermissionLevel.ALLOW)
            def add(a: int, b: int) -> int:
                return a + b

            result = await agent.run("Compute 10 + 20")
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

        with patch("datagol_agent_harness.extensions.langsmith._safe_post", side_effect=posted_runs.append), \
             patch("datagol_agent_harness.extensions.langsmith._safe_patch", side_effect=patched_runs.append):

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
                return await sub_agent.run(q)

            res = await orchestrator.run("Solve complex problem")
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

        with patch("datagol_agent_harness.extensions.langsmith._safe_post", side_effect=posted_runs.append), \
             patch("datagol_agent_harness.extensions.langsmith._safe_patch", side_effect=patched_runs.append):

            ext = LangSmithExtension(project_name="error-test")
            agent = Agent(
                config=AgentConfig(model="claude-sonnet-4-6"),
                provider=FailingProvider(),
                extensions=[ext],
            )

            with self.assertRaises(RuntimeError):
                await agent.run("Fail now")

        # Root run and LLM run should have ended with error
        self.assertTrue(any("API failure" in str(r.error) for r in patched_runs))

    async def test_disabled_extension(self):
        """Verify nothing is traced when enabled=False."""
        posted_runs: list[Any] = []
        mock_provider = MockProvider(
            [MockResponse(content=[MockTextBlock(text="ok")], stop_reason="end_turn")]
        )

        with patch("datagol_agent_harness.extensions.langsmith._safe_post", side_effect=posted_runs.append):
            ext = LangSmithExtension(enabled=False)
            agent = Agent(provider=mock_provider, extensions=[ext])
            await agent.run("hello")

        self.assertEqual(len(posted_runs), 0)

    async def test_streaming_agent_lifecycle(self):
        """Verify StreamingAgent turn events and LLM spans are traced."""
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

        with patch("datagol_agent_harness.extensions.langsmith._safe_post", side_effect=posted_runs.append), \
             patch("datagol_agent_harness.extensions.langsmith._safe_patch", side_effect=patched_runs.append):

            ext = LangSmithExtension(project_name="stream-test")
            streaming_agent = StreamingAgent(
                config=AgentConfig(model="claude-sonnet-4-6"),
                client=mock_client,
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
