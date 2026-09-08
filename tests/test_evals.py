"""Comprehensive offline unit tests for the DataGOL Agent Harness LangSmith eval framework."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
import unittest
from unittest.mock import MagicMock, patch
import uuid

from langsmith.schemas import Example, Run

from datagol_agent_harness import (
    Agent,
    AgentConfig,
    PermissionLevel,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from datagol_agent_harness.evals import (
    AgentTarget,
    build_example,
    contains_evaluator,
    default_evaluators,
    evaluate_agent,
    exact_match_evaluator,
    get_dataset,
    iteration_budget_evaluator,
    json_valid_evaluator,
    list_datasets,
    no_agent_errors_evaluator,
    no_tool_errors_evaluator,
    regex_evaluator,
    skill_invoked_evaluator,
    step_sequence_evaluator,
    subagent_delegated_evaluator,
    tool_args_evaluator,
    tool_call_count_evaluator,
    tool_selection_evaluator,
)
from datagol_agent_harness.providers.base import LLMProvider


# ── Mock Response & Provider ────────────────────────────────────────────────


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
        default_factory=lambda: TokenUsage(input_tokens=15, output_tokens=10)
    )


class MockProvider(LLMProvider):
    """Predictable mock provider for deterministic offline testing."""

    def __init__(self, responses: list[MockResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> MockResponse:
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return MockResponse(
            content=[MockTextBlock(text="Default mock complete")],
            stop_reason="end_turn",
        )

    async def count_tokens(self, **kwargs: Any) -> int:
        return 15


# ── Test Suite ──────────────────────────────────────────────────────────────


class TestEvalFramework(unittest.IsolatedAsyncioTestCase):

    def _make_dummy_run(self, outputs: dict[str, Any]) -> Run:
        """Create a lightweight Run schema for testing evaluators directly."""
        run = MagicMock(spec=Run)
        run.outputs = outputs
        run.inputs = {}
        return run

    # ── 1. Target Adapter & Telemetry Tests ─────────────────────────────────

    async def test_agent_target_telemetry_capture(self):
        """Verify AgentTarget captures tool executions, token usage, iterations, and cost."""
        provider = MockProvider(
            [
                MockResponse(
                    content=[
                        MockToolUseBlock(id="call_add", name="calculate", input={"expr": "10 + 20"})
                    ],
                    stop_reason="tool_use",
                ),
                MockResponse(
                    content=[MockTextBlock(text="The result is 30")],
                    stop_reason="end_turn",
                ),
            ]
        )

        agent = Agent(
            config=AgentConfig(model="mock-agent"),
            provider=provider,
        )

        @agent.tools.register(permission=PermissionLevel.ALLOW)
        def calculate(expr: str) -> str:
            return "30"

        target = AgentTarget(agent, attach_langsmith=False)
        result = await target.run_async({"prompt": "Calculate 10 + 20"})

        self.assertEqual(result["output"], "The result is 30")
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["tools_used"], ["calculate"])
        self.assertEqual(len(result["tool_calls"]), 1)

        tc = result["tool_calls"][0]
        self.assertEqual(tc["name"], "calculate")
        self.assertEqual(tc["input"], {"expr": "10 + 20"})
        self.assertEqual(tc["output"], "30")
        self.assertFalse(tc["is_error"])
        self.assertGreaterEqual(tc["duration_ms"], 0.0)

        # Token stats
        self.assertGreater(result["token_usage"]["total_tokens"], 0)
        self.assertIsNone(result["error"])

    async def test_agent_target_error_containment(self):
        """Verify errors during agent execution are caught in telemetry without raising."""
        class BrokenProvider(LLMProvider):
            async def create(self, **kwargs: Any) -> Any:
                raise RuntimeError("Simulated API failure")

            async def count_tokens(self, **kwargs: Any) -> int:
                return 0

        agent = Agent(provider=BrokenProvider())
        target = AgentTarget(agent, attach_langsmith=False)

        result = await target.run_async({"prompt": "Trigger crash"})
        self.assertIn("Simulated API failure", result["output"])
        self.assertIn("Simulated API failure", result["error"] or "")

    # ── 2. Evaluator Unit Tests ─────────────────────────────────────────────

    def test_tool_selection_evaluator(self):
        # Case 1: Expected tools called
        run = self._make_dummy_run({"tools_used": ["calculate", "read_file"]})
        ex = build_example("test", outputs={"expected_tools": ["calculate"]})
        res = tool_selection_evaluator(run, ex)
        self.assertEqual(res["score"], 1.0)

        # Case 2: Missing expected tool
        ex_missing = build_example("test", outputs={"expected_tools": ["write_file"]})
        res_missing = tool_selection_evaluator(run, ex_missing)
        self.assertEqual(res_missing["score"], 0.0)

        # Case 3: Forbidden tool called
        ex_forbid = build_example("test", outputs={"forbidden_tools": ["read_file"]})
        res_forbid = tool_selection_evaluator(run, ex_forbid)
        self.assertLess(res_forbid["score"], 1.0)

    def test_no_tool_errors_evaluator(self):
        # Case 1: All tools succeed
        run_ok = self._make_dummy_run({
            "tool_calls": [{"name": "add", "is_error": False}]
        })
        ex = build_example("test")
        self.assertEqual(no_tool_errors_evaluator(run_ok, ex)["score"], 1.0)

        # Case 2: A tool failed
        run_err = self._make_dummy_run({
            "tool_calls": [{"name": "add", "is_error": True}]
        })
        self.assertEqual(no_tool_errors_evaluator(run_err, ex)["score"], 0.0)

    def test_tool_args_evaluator(self):
        run = self._make_dummy_run({
            "tool_calls": [{"name": "add", "input": {"a": 10, "b": 20}}]
        })
        # Matching args
        ex_match = build_example("test", outputs={"expected_args": {"add": {"a": 10, "b": 20}}})
        self.assertEqual(tool_args_evaluator(run, ex_match)["score"], 1.0)

        # Mismatched args
        ex_mismatch = build_example("test", outputs={"expected_args": {"add": {"a": 99}}})
        self.assertEqual(tool_args_evaluator(run, ex_mismatch)["score"], 0.0)

    def test_correctness_evaluators(self):
        run = self._make_dummy_run({"output": "The capital of France is Paris."})

        # Contains
        ex_contains = build_example("test", outputs={"contains_all": ["Paris", "France"]})
        self.assertEqual(contains_evaluator(run, ex_contains)["score"], 1.0)

        ex_missing = build_example("test", outputs={"contains_all": ["London"]})
        self.assertEqual(contains_evaluator(run, ex_missing)["score"], 0.0)

        # Exact match
        ex_exact = build_example("test", outputs={"expected_output": "The capital of France is Paris."})
        self.assertEqual(exact_match_evaluator(run, ex_exact)["score"], 1.0)

        # Regex
        ex_regex = build_example("test", outputs={"regex_pattern": r"capital.*Paris"})
        self.assertEqual(regex_evaluator(run, ex_regex)["score"], 1.0)

    def test_json_valid_evaluator(self):
        run_valid = self._make_dummy_run({"output": '```json\n{"name": "Alice", "age": 30}\n```'})
        ex = build_example("test", outputs={"json_required_keys": ["name", "age"]})
        self.assertEqual(json_valid_evaluator(run_valid, ex)["score"], 1.0)

        run_invalid = self._make_dummy_run({"output": "Not a json response"})
        self.assertEqual(json_valid_evaluator(run_invalid, ex)["score"], 0.0)

    def test_trajectory_evaluators(self):
        run = self._make_dummy_run({
            "iterations": 3,
            "tool_calls": [{"name": "read_file"}, {"name": "calculate"}],
        })

        # Budget ok
        ex_budget = build_example("test", outputs={"max_allowed_iterations": 5})
        self.assertEqual(iteration_budget_evaluator(run, ex_budget)["score"], 1.0)

        # Budget exceeded
        ex_exceeded = build_example("test", outputs={"max_allowed_iterations": 2})
        self.assertEqual(iteration_budget_evaluator(run, ex_exceeded)["score"], 0.0)

        # Step sequence
        ex_seq = build_example("test", outputs={"expected_tool_sequence": ["read_file", "calculate"]})
        self.assertEqual(step_sequence_evaluator(run, ex_seq)["score"], 1.0)

        ex_bad_seq = build_example("test", outputs={"expected_tool_sequence": ["calculate", "read_file"]})
        self.assertEqual(step_sequence_evaluator(run, ex_bad_seq)["score"], 0.0)

    def test_skills_and_delegation_evaluators(self):
        run = self._make_dummy_run({
            "skills_invoked": ["commit-message"],
            "tools_used": ["delegate_research"],
        })

        # Skill invoked
        ex_skill = build_example("test", outputs={"expected_skill": "commit-message"})
        self.assertEqual(skill_invoked_evaluator(run, ex_skill)["score"], 1.0)

        # Subagent delegated
        ex_deleg = build_example("test", outputs={"expected_delegation": "delegate_research"})
        self.assertEqual(subagent_delegated_evaluator(run, ex_deleg)["score"], 1.0)

    # ── 3. Dataset Registry Tests ───────────────────────────────────────────

    def test_dataset_registry(self):
        suites = list_datasets()
        self.assertIn("tool_calling", suites)
        self.assertIn("skills", suites)
        self.assertIn("multi_agent", suites)
        self.assertIn("guardrails", suites)
        self.assertIn("memory", suites)

        tools_data = get_dataset("tool_calling")
        self.assertGreater(len(tools_data), 0)
        self.assertIsInstance(tools_data[0], Example)

        all_data = get_dataset("all")
        self.assertGreater(len(all_data), len(tools_data))

    # ── 4. End-to-End Offline Evaluation Run ────────────────────────────────

    def test_end_to_end_evaluate_agent_offline(self):
        """Run full evaluation suite with mock provider in offline mode."""
        def agent_factory(inputs: dict[str, Any]) -> Agent:
            provider = MockProvider(
                [
                    MockResponse(
                        content=[
                            MockToolUseBlock(id="c1", name="calculate", input={"expr": "12 * 12"})
                        ],
                        stop_reason="tool_use",
                    ),
                    MockResponse(
                        content=[MockTextBlock(text="The answer is 144")],
                        stop_reason="end_turn",
                    ),
                ]
            )
            agent = Agent(provider=provider)

            @agent.tools.register(permission=PermissionLevel.ALLOW)
            def calculate(expr: str) -> str:
                return "144"

            return agent

        custom_examples = [
            build_example(
                inputs={"prompt": "What is 12 * 12?"},
                outputs={
                    "expected_tools": ["calculate"],
                    "contains_all": ["144"],
                    "max_allowed_iterations": 3,
                },
            )
        ]

        summary = evaluate_agent(
            agent=agent_factory,
            dataset=custom_examples,
            evaluators=[
                tool_selection_evaluator,
                contains_evaluator,
                no_tool_errors_evaluator,
            ],
            experiment_prefix="test-offline-eval",
            offline=True,
            print_summary=False,
        )

        self.assertEqual(summary.total_examples, 1)
        self.assertEqual(summary.pass_rate, 1.0)
        self.assertEqual(summary.scores.get("tool_selection"), 1.0)
        self.assertEqual(summary.scores.get("text_contains"), 1.0)
        self.assertEqual(summary.scores.get("no_tool_errors"), 1.0)
        self.assertEqual(len(summary.results), 1)


if __name__ == "__main__":
    unittest.main()
