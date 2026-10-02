"""Comprehensive tests for architecture improvements, bug fixes, and extensions in harnessx."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

from harnessx import (
    Limits,
    AgentConfig,
    ConversationMemory,
    Extension,
    ExtensionContext,
    LLMProvider,
    PermissionLevel,
    ProviderResponse,
    StopReason,
    StreamChunk,
    RunEvent,
    RunEventType,
    Agent,
    ToolCall,
    ToolResult,
)


class MockTestProvider(LLMProvider):
    """Configurable mock provider for testing loop behaviors and streaming."""

    def __init__(self, responses: list[ProviderResponse] | None = None) -> None:
        self.responses: list[ProviderResponse] = list(responses or [])
        self.recorded_calls: list[dict[str, Any]] = []

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> ProviderResponse:
        self.recorded_calls.append({
            "model": model,
            "messages": messages,
            "system": system,
            "tools": tools,
        })
        if self.responses:
            return self.responses.pop(0)
        return ProviderResponse(text="Default mock reply", stop_reason=StopReason.END_TURN)

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[StreamChunk]:
        resp = await self.create(
            model=model,
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if resp.text:
            yield StreamChunk(kind="text_delta", data=resp.text)
        yield StreamChunk(kind="response", data=resp)

    async def count_tokens(self, **kwargs: Any) -> int:
        return 10


class TestCanonicalTypes(unittest.TestCase):
    """Test canonical types, stop reasons, and backward-compatible aliases."""

    def test_tool_result_dual_alias(self):
        # Initialized with tool_call_id
        r1 = ToolResult(tool_call_id="call_1", content="ok")
        self.assertEqual(r1.tool_call_id, "call_1")
        self.assertEqual(r1.tool_use_id, "call_1")

        # Initialized with tool_use_id (legacy Anthropic style)
        r2 = ToolResult(tool_use_id="call_2", content="ok")
        self.assertEqual(r2.tool_call_id, "call_2")
        self.assertEqual(r2.tool_use_id, "call_2")

    def test_stop_reason_string_equality(self):
        # StopReason inherits from str, Enum
        self.assertEqual(StopReason.END_TURN, "end_turn")
        self.assertEqual(StopReason.TOOL_USE, "tool_use")
        self.assertEqual(StopReason.MAX_TOKENS, "max_tokens")

    def test_provider_response_post_init(self):
        # Initialized with text and tool_calls
        tc = ToolCall(id="t1", name="search", input={"q": "test"})
        resp = ProviderResponse(text="Found results", tool_calls=[tc])
        self.assertEqual(len(resp.content), 2)
        self.assertEqual(resp.content[0].type, "text")
        self.assertEqual(resp.content[1].type, "tool_use")


class TestAgentLoopFixes(unittest.IsolatedAsyncioTestCase):
    """Test agentic loop edge cases: stop reasons, empty tools, multi-turn iterations."""

    async def test_custom_stop_sequence_does_not_infinite_loop(self):
        """Unknown or custom stop reasons should terminate gracefully without looping forever."""
        provider = MockTestProvider([
            ProviderResponse(text="Custom stop reached", stop_reason=StopReason.STOP_SEQUENCE)
        ])
        agent = Agent(provider=provider)
        result = (await agent.run("Run custom stop")).output
        self.assertEqual(result, "Custom stop reached")
        self.assertEqual(len(provider.recorded_calls), 1)

    async def test_empty_tool_calls_terminate_safely(self):
        """Model signaling tool_use without valid tool blocks terminates safely."""
        provider = MockTestProvider([
            ProviderResponse(text="I wanted to use tools", tool_calls=[], stop_reason=StopReason.TOOL_USE)
        ])
        agent = Agent(provider=provider)
        result = (await agent.run("Hello")).output
        self.assertIn("wanted to use tools", result)
        self.assertEqual(len(provider.recorded_calls), 1)

    async def test_multi_turn_iteration_reset(self):
        """Turn-level iteration limit resets per turn, preventing premature MaxIterationsError."""
        # Provider returns 3 tool turns then end_turn for each user prompt
        def make_turn_responses():
            return [
                ProviderResponse(
                    text="",
                    tool_calls=[ToolCall(id="c1", name="dummy", input={})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                ProviderResponse(
                    text="",
                    tool_calls=[ToolCall(id="c2", name="dummy", input={})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                ProviderResponse(text="Turn complete", stop_reason=StopReason.END_TURN),
            ]

        # max_iterations = 4: turn 1 takes 3 iterations, turn 2 takes 3 iterations.
        # Cumulative = 6. With old bug, turn 2 would crash because 6 >= 4.
        provider = MockTestProvider(make_turn_responses() + make_turn_responses())
        agent = Agent(config=AgentConfig(limits=Limits(max_iterations=4)), provider=provider)

        @agent.tools.register(permission=PermissionLevel.ALLOW)
        def dummy() -> str:
            return "done"

        turn1 = (await agent.run("First prompt")).output
        self.assertEqual(turn1, "Turn complete")

        turn2 = (await agent.run("Second prompt")).output
        self.assertEqual(turn2, "Turn complete")
        self.assertEqual(agent.guardrails.lifetime_iterations, 6)


class TestParallelToolExecution(unittest.IsolatedAsyncioTestCase):
    """Test parallel tool execution for concurrent-safe tools."""

    async def test_concurrent_tools_execute_in_parallel(self):
        call_order: list[str] = []

        async def fetch_a() -> str:
            call_order.append("start_a")
            await asyncio.sleep(0.05)
            call_order.append("end_a")
            return "data_a"

        async def fetch_b() -> str:
            call_order.append("start_b")
            await asyncio.sleep(0.05)
            call_order.append("end_b")
            return "data_b"

        provider = MockTestProvider([
            ProviderResponse(
                text="",
                tool_calls=[
                    ToolCall(id="c_a", name="fetch_a", input={}),
                    ToolCall(id="c_b", name="fetch_b", input={}),
                ],
                stop_reason=StopReason.TOOL_USE,
            ),
            ProviderResponse(text="Both fetched", stop_reason=StopReason.END_TURN),
        ])

        agent = Agent(provider=provider)
        agent.tools.register(name="fetch_a", permission=PermissionLevel.ALLOW, concurrent=True)(fetch_a)
        agent.tools.register(name="fetch_b", permission=PermissionLevel.ALLOW, concurrent=True)(fetch_b)

        result = (await agent.run("Fetch both")).output
        self.assertEqual(result, "Both fetched")

        # Because they ran concurrently via asyncio.gather, both started before either finished:
        self.assertEqual(call_order[0:2], ["start_a", "start_b"])


class TestConversationMemorySafety(unittest.IsolatedAsyncioTestCase):
    """Test conversation trimming boundary safety."""

    async def test_trim_if_needed_preserves_tool_pairs_and_alternating_roles(self):
        memory = ConversationMemory()
        # Add a conversation with user -> assistant(tool_use) -> user(tool_result) -> assistant
        memory.add_user_message("Initial message 1")
        memory.add_assistant_message([{"type": "text", "text": "Reply 1"}])
        memory.add_user_message("Query with tools")
        memory.add_assistant_message([{"type": "tool_use", "id": "t1", "name": "dummy", "input": {}}])
        memory.add_tool_results([ToolResult(tool_call_id="t1", content="tool output")])
        memory.add_assistant_message([{"type": "text", "text": "Tool finished"}])
        memory.add_user_message("Follow-up user prompt")
        memory.add_assistant_message([{"type": "text", "text": "Final answer"}])

        mock_provider = MockTestProvider()
        # Force trimming with a low token limit
        trimmed = await memory.trim_if_needed(
            mock_provider, model="mock", system="", tools=[], max_context_tokens=10
        )
        self.assertTrue(trimmed)
        messages = memory.get_messages()

        # Check alternating roles: user -> assistant -> user ...
        for i in range(len(messages) - 1):
            self.assertNotEqual(
                messages[i]["role"],
                messages[i + 1]["role"],
                f"Consecutive roles found at index {i} and {i+1}: {messages[i]['role']}",
            )


class TestExtensionAPI(unittest.IsolatedAsyncioTestCase):
    """Test the modernized ExtensionContext API and persistence."""

    async def test_extension_context_dynamic_prompt_provider(self):
        class DynamicPromptExtension(Extension):
            name = "dynamic_prompt"

            def install(self, ctx: ExtensionContext) -> None:
                ctx.register_prompt_provider(lambda: "Dynamic Context: Current Date 2026-09-07")

        ext = DynamicPromptExtension()
        provider = MockTestProvider([
            ProviderResponse(text="Understood date", stop_reason=StopReason.END_TURN)
        ])
        agent = Agent(
            config=AgentConfig(system_prompt="Base System"),
            provider=provider,
            extensions=[ext],
        )

        res = (await agent.run("Hello")).output
        self.assertEqual(res, "Understood date")
        # Verify dynamic prompt was passed to provider
        self.assertIn("Dynamic Context: Current Date 2026-09-07", provider.recorded_calls[0]["system"])
        self.assertIn("Base System", provider.recorded_calls[0]["system"])

    async def test_extension_session_state_persistence(self):
        class StatefulExtension(Extension):
            name = "stateful"

            def __init__(self) -> None:
                self.records: dict[str, str] = {}

            def install(self, ctx: ExtensionContext) -> None:
                pass

            async def on_save_session(self, ctx: ExtensionContext) -> dict[str, Any]:
                return {"records": dict(self.records)}

            async def on_load_session(self, ctx: ExtensionContext, state: dict[str, Any]) -> None:
                self.records = dict(state.get("records", {}))

        ext1 = StatefulExtension()
        ext1.records["item_1"] = "val_1"

        agent1 = Agent(
            config=AgentConfig(system_prompt="Test"),
            provider=MockTestProvider(),
            extensions=[ext1],
        )
        session_id = await agent1.save_session(storage_dir="/tmp/test_agent_sessions")

        # Restore in a fresh agent instance with fresh extension
        ext2 = StatefulExtension()
        await Agent.load_session(
            session_id,
            storage_dir="/tmp/test_agent_sessions",
            provider=MockTestProvider(),
            extensions=[ext2],
        )

        self.assertEqual(ext2.records.get("item_1"), "val_1")


class TestProviderAgnosticStreaming(unittest.IsolatedAsyncioTestCase):
    """Test Agent streaming without vendor lock-in."""

    async def test_streaming_agent_with_mock_provider(self):
        provider = MockTestProvider([
            ProviderResponse(text="Streamed response", stop_reason=StopReason.END_TURN)
        ])
        streaming_agent = Agent(provider=provider)

        events: list[RunEvent] = []
        async for ev in streaming_agent.run_stream("Stream this"):
            events.append(ev)

        event_types = [e.type for e in events]
        self.assertIn(RunEventType.TEXT_DELTA, event_types)
        self.assertIn(RunEventType.TEXT_COMPLETE, event_types)
        self.assertIn(RunEventType.TURN_COMPLETE, event_types)

    async def test_streaming_agent_handles_unknown_tool_gracefully(self):
        """Model hallucinating a tool name yields tool_result with is_error=True instead of crashing."""
        provider = MockTestProvider([
            ProviderResponse(
                text="",
                tool_calls=[ToolCall(id="c_unknown", name="hallucinated_tool", input={})],
                stop_reason=StopReason.TOOL_USE,
            ),
            ProviderResponse(text="Recovered from error", stop_reason=StopReason.END_TURN),
        ])
        streaming_agent = Agent(provider=provider)

        events: list[RunEvent] = []
        async for ev in streaming_agent.run_stream("Use hallucinated tool"):
            events.append(ev)

        tool_results = [e.data for e in events if e.type == RunEventType.TOOL_RESULT]
        self.assertEqual(len(tool_results), 1)
        self.assertTrue(tool_results[0].is_error)
        self.assertIn("No tool named", tool_results[0].content)
        self.assertIn("Available tools", tool_results[0].content)

    async def test_anthropic_provider_temperature_fallback(self):
        """Verify AnthropicProvider handles SDK versions or models rejecting temperature."""
        from anthropic.types import TextBlock
        from harnessx.providers.anthropic import AnthropicProvider

        class MockMessages:
            def __init__(self):
                self.create_calls = []
                self.stream_calls = []

            async def create(self, **kwargs):
                self.create_calls.append(kwargs)
                if "temperature" in kwargs:
                    raise TypeError("AsyncMessages.create() got an unexpected keyword argument 'temperature'")
                # Return dummy response
                mock_resp = MagicMock()
                mock_resp.content = [TextBlock(type="text", text="ok")]
                mock_resp.stop_reason = "end_turn"
                mock_resp.usage = MagicMock(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0)
                return mock_resp

            def stream(self, **kwargs):
                self.stream_calls.append(kwargs)
                if "temperature" in kwargs:
                    raise TypeError("AsyncMessages.stream() got an unexpected keyword argument 'temperature'")

                class MockStreamContext:
                    async def __aenter__(self):
                        class StreamObj:
                            @property
                            def text_stream(self):
                                async def _gen():
                                    yield "streamed text"
                                return _gen()

                            async def get_final_message(self):
                                mock_resp = MagicMock()
                                mock_resp.content = [TextBlock(type="text", text="streamed text")]
                                mock_resp.stop_reason = "end_turn"
                                mock_resp.usage = MagicMock(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0)
                                return mock_resp
                        return StreamObj()

                    async def __aexit__(self, *args):
                        pass

                return MockStreamContext()

        mock_client = MagicMock()
        mock_messages = MockMessages()
        mock_client.messages = mock_messages

        provider = AnthropicProvider(client=mock_client)

        # 1. Test create() with temperature -> should catch TypeError and retry without temperature
        resp = await provider.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            system=None,
            tools=[],
            max_tokens=100,
            temperature=0.7,
        )
        self.assertEqual(resp.text, "ok")
        # First call had temperature, retry did not
        self.assertEqual(len(mock_messages.create_calls), 2)
        self.assertIn("temperature", mock_messages.create_calls[0])
        self.assertNotIn("temperature", mock_messages.create_calls[1])

        # 2. Test stream() with temperature -> should catch TypeError and retry without temperature
        chunks = []
        async for chunk in provider.stream(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            system=None,
            tools=[],
            max_tokens=100,
            temperature=0.7,
        ):
            chunks.append(chunk)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(len(mock_messages.stream_calls), 2)
        self.assertIn("temperature", mock_messages.stream_calls[0])
        self.assertNotIn("temperature", mock_messages.stream_calls[1])

        # 3. Test default temperature=None -> should omit temperature on first call
        mock_messages.create_calls.clear()
        await provider.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            system=None,
            tools=[],
            max_tokens=100,
            temperature=None,
        )
        self.assertEqual(len(mock_messages.create_calls), 1)
        self.assertNotIn("temperature", mock_messages.create_calls[0])

    def test_agent_config_default_temperature_is_none(self):
        config = AgentConfig()
        self.assertIsNone(config.temperature)


if __name__ == "__main__":
    unittest.main()
