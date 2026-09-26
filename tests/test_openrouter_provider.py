"""Tests for OpenRouterProvider: initialization, credentials, attribution headers,
reasoning token extraction, streaming, token counting, and Agent integration.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from harnessx.core import Agent
from harnessx.providers import make_provider
from harnessx.providers.openrouter import (
    DEFAULT_OPENROUTER_BASE_URL,
    OpenRouterProvider,
)
from harnessx import Agent
from harnessx.types import AgentConfig, StopReason, ToolCall


def _run(coro):
    return asyncio.run(coro)


TOOLS = [
    {
        "name": "calc",
        "description": "Calculate expression",
        "input_schema": {
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
    }
]


# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeCompletions:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("No more fake responses configured")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class FakeChat:
    def __init__(self, completions: FakeCompletions):
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, responses: list[Any]):
        self.completions = FakeCompletions(responses)
        self.chat = FakeChat(self.completions)


def _make_completion_response(
    content: str | None = "Hello from OpenRouter!",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    reasoning: str | None = None,
    prompt_tokens: int = 15,
    completion_tokens: int = 8,
    cached_tokens: int = 4,
) -> SimpleNamespace:
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        role="assistant",
        reasoning=reasoning,
        model_extra={"reasoning": reasoning} if reasoning else {},
    )
    choice = SimpleNamespace(
        message=msg,
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    return SimpleNamespace(
        choices=[choice],
        usage=usage,
    )


class FakeAsyncStream:
    def __init__(self, chunks: list[Any]):
        self.chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestOpenRouterInitialization(unittest.TestCase):
    def test_default_base_url_and_custom_headers(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test-key"}, clear=True):
            provider = OpenRouterProvider(
                site_url="https://myapp.ai",
                app_name="MyApp",
            )
            self.assertEqual(provider.name, "openrouter")
            self.assertEqual(str(provider.client.base_url).rstrip("/"), DEFAULT_OPENROUTER_BASE_URL)
            self.assertEqual(provider.client.api_key, "sk-or-test-key")
            self.assertEqual(provider.client.default_headers.get("HTTP-Referer"), "https://myapp.ai")
            self.assertEqual(provider.client.default_headers.get("X-Title"), "MyApp")

    def test_env_var_resolution_fallback(self):
        env = {
            "OPENAI_API_KEY": "sk-openai-fallback",
            "OPENROUTER_BASE_URL": "https://custom.openrouter.ai/api/v1",
            "OPENROUTER_SITE_URL": "https://custom-site.org",
            "OPENROUTER_APP_NAME": "CustomApp",
        }
        with patch.dict(os.environ, env, clear=True):
            provider = OpenRouterProvider()
            self.assertEqual(provider.client.api_key, "sk-openai-fallback")
            self.assertEqual(
                str(provider.client.base_url).rstrip("/"),
                "https://custom.openrouter.ai/api/v1",
            )
            self.assertEqual(
                provider.client.default_headers.get("HTTP-Referer"),
                "https://custom-site.org",
            )
            self.assertEqual(
                provider.client.default_headers.get("X-Title"),
                "CustomApp",
            )

    def test_make_provider_factory(self):
        fake_client = FakeOpenAIClient([])
        provider = make_provider("openrouter", client=fake_client)
        self.assertIsInstance(provider, OpenRouterProvider)
        self.assertEqual(provider.name, "openrouter")

    def test_make_provider_case_insensitive(self):
        fake_client = FakeOpenAIClient([])
        provider = make_provider("OpenRouter", client=fake_client)
        self.assertIsInstance(provider, OpenRouterProvider)


class TestOpenRouterCreate(unittest.TestCase):
    def test_create_translates_request_and_normalizes_response(self):
        fake_resp = _make_completion_response(
            content="Answer: 42",
            prompt_tokens=20,
            completion_tokens=5,
            cached_tokens=10,
        )
        client = FakeOpenAIClient([fake_resp])
        provider = OpenRouterProvider(client=client)

        resp = _run(
            provider.create(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": "What is 6 * 7?"}],
                system="You are an expert mathematician.",
                tools=TOOLS,
                max_tokens=512,
            )
        )

        self.assertEqual(resp.text, "Answer: 42")
        self.assertEqual(resp.stop_reason, StopReason.END_TURN)
        self.assertEqual(resp.usage.input_tokens, 20)
        self.assertEqual(resp.usage.output_tokens, 5)
        self.assertEqual(resp.usage.cache_read_input_tokens, 10)

        # Check call arguments passed to client
        call = client.completions.calls[0]
        self.assertEqual(call["model"], "deepseek/deepseek-chat")
        self.assertEqual(call["max_completion_tokens"], 512)
        self.assertEqual(
            call["messages"][0],
            {"role": "system", "content": "You are an expert mathematician."},
        )
        self.assertEqual(
            call["messages"][1],
            {"role": "user", "content": "What is 6 * 7?"},
        )
        self.assertEqual(
            call["tools"][0]["function"]["name"],
            "calc",
        )

    def test_create_extracts_reasoning(self):
        fake_resp = _make_completion_response(
            content="Final answer",
            reasoning="Thinking through the calculation...",
        )
        client = FakeOpenAIClient([fake_resp])
        provider = OpenRouterProvider(client=client)

        resp = _run(
            provider.create(
                model="deepseek/deepseek-r1",
                messages=[{"role": "user", "content": "Solve 2+2"}],
                system=None,
                tools=[],
                max_tokens=256,
            )
        )

        self.assertEqual(resp.text, "Final answer")
        self.assertEqual(resp.thinking, "Thinking through the calculation...")
        # Check back-compat content blocks contain thinking
        self.assertTrue(any(b.type == "thinking" for b in resp.content))

    def test_create_tool_calls(self):
        tool_call_mock = SimpleNamespace(
            id="call_abc123",
            function=SimpleNamespace(
                name="calc",
                arguments='{"expr": "100 / 4"}',
            ),
        )
        fake_resp = _make_completion_response(
            content=None,
            tool_calls=[tool_call_mock],
            finish_reason="tool_calls",
        )
        client = FakeOpenAIClient([fake_resp])
        provider = OpenRouterProvider(client=client)

        resp = _run(
            provider.create(
                model="anthropic/claude-3.7-sonnet",
                messages=[{"role": "user", "content": "calc 100/4"}],
                system=None,
                tools=TOOLS,
                max_tokens=256,
            )
        )

        self.assertEqual(resp.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].id, "call_abc123")
        self.assertEqual(resp.tool_calls[0].name, "calc")
        self.assertEqual(resp.tool_calls[0].input, {"expr": "100 / 4"})


class TestOpenRouterStream(unittest.TestCase):
    def test_streaming_emits_thinking_and_text_and_captures_usage(self):
        chunk1 = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning="I will ", content=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        chunk2 = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning="compute this.", content=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        chunk3 = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning=None, content="Result: "),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        chunk4 = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning=None, content="42"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        # OpenRouter returns usage in the final chunk
        chunk5 = SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=25,
                completion_tokens=12,
                prompt_tokens_details=SimpleNamespace(cached_tokens=5),
            ),
        )

        stream = FakeAsyncStream([chunk1, chunk2, chunk3, chunk4, chunk5])
        client = FakeOpenAIClient([stream])
        provider = OpenRouterProvider(client=client)

        async def collect():
            events = []
            async for ch in provider.stream(
                model="deepseek/deepseek-r1",
                messages=[{"role": "user", "content": "Compute"}],
                system=None,
                tools=[],
                max_tokens=100,
            ):
                events.append(ch)
            return events

        events = _run(collect())

        # Verify thinking deltas
        thinking_deltas = [e.data for e in events if e.kind == "thinking_delta"]
        self.assertEqual(thinking_deltas, ["I will ", "compute this."])

        # Verify text deltas
        text_deltas = [e.data for e in events if e.kind == "text_delta"]
        self.assertEqual(text_deltas, ["Result: ", "42"])

        # Verify final response chunk
        response_chunks = [e for e in events if e.kind == "response"]
        self.assertEqual(len(response_chunks), 1)
        final_resp = response_chunks[0].data
        self.assertEqual(final_resp.text, "Result: 42")
        self.assertEqual(final_resp.thinking, "I will compute this.")
        self.assertEqual(final_resp.stop_reason, StopReason.END_TURN)
        self.assertEqual(final_resp.usage.input_tokens, 25)
        self.assertEqual(final_resp.usage.output_tokens, 12)
        self.assertEqual(final_resp.usage.cache_read_input_tokens, 5)


class TestOpenRouterTokenCounting(unittest.TestCase):
    def test_count_tokens_fallback_heuristic(self):
        client = FakeOpenAIClient([])
        provider = OpenRouterProvider(client=client)

        messages = [
            {"role": "user", "content": "Hello world!"},
            {"role": "assistant", "content": "How can I help you today?"},
        ]
        system = "You are a helpful assistant."

        count = _run(
            provider.count_tokens(
                model="anthropic/claude-3.7-sonnet",
                messages=messages,
                system=system,
                tools=[],
            )
        )
        self.assertGreater(count, 0)

    def test_count_tokens_tiktoken_strip_prefix(self):
        client = FakeOpenAIClient([])
        provider = OpenRouterProvider(client=client)

        messages = [{"role": "user", "content": "Count me"}]
        count = _run(
            provider.count_tokens(
                model="openai/gpt-4o",
                messages=messages,
                system="system prompt",
                tools=[],
            )
        )
        self.assertGreater(count, 0)


class TestOpenRouterAgentIntegration(unittest.TestCase):
    def test_agent_run_with_openrouter(self):
        fake_resp = _make_completion_response(content="Integrated output!")
        client = FakeOpenAIClient([fake_resp])
        provider = OpenRouterProvider(client=client)

        config = AgentConfig(provider="openrouter", model="anthropic/claude-3.7-sonnet")
        agent = Agent(config=config, provider=provider)

        result = _run(agent.run("Hi")).output
        self.assertEqual(result, "Integrated output!")

    def test_streaming_agent_with_openrouter(self):
        chunk1 = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning=None, content="Streaming "),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        chunk2 = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning=None, content="live!"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        stream = FakeAsyncStream([chunk1, chunk2])
        client = FakeOpenAIClient([stream])
        provider = OpenRouterProvider(client=client)

        config = AgentConfig(provider="openrouter", model="google/gemini-2.5-pro")
        agent = Agent(config=config, provider=provider)

        async def run_streaming():
            deltas = []
            async for event in agent.run_stream("Stream test"):
                if event.type.value == "text_delta":
                    deltas.append(event.data)
            return deltas

        deltas = _run(run_streaming())
        self.assertEqual(deltas, ["Streaming ", "live!"])


    def test_missing_sdk_import_error_message(self):
        import importlib
        import sys

        import harnessx.providers.openrouter as openrouter_mod

        with patch.dict(sys.modules, {"openai": None}):
            with self.assertRaises(ImportError) as ctx:
                importlib.reload(openrouter_mod)
            self.assertIn("harnessx[openrouter]", str(ctx.exception))
        importlib.reload(openrouter_mod)


if __name__ == "__main__":
    unittest.main()
