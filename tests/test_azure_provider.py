"""Tests for AzureOpenAIProvider: factory, credential resolution, injected
clients, deployment routing, completion, streaming, token counting, and Agent
integration. All use injected fake clients or a patched SDK constructor — no
network and no real Azure credentials required.
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
from harnessx.providers.azure_openai import AzureOpenAIProvider
from harnessx.runtime import AgentRuntime
from harnessx.types import (
    AgentConfig,
    RuntimeConfig,
    StopReason,
)


def _run(coro):
    return asyncio.run(coro)


TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
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


class FakeAzureClient:
    """Stand-in for AsyncAzureOpenAI: same chat.completions.create surface."""

    def __init__(self, responses: list[Any] | None = None):
        self.completions = FakeCompletions(responses or [])
        self.chat = FakeChat(self.completions)


class FakeAsyncStream:
    def __init__(self, chunks: list[Any]):
        self.chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


def _completion_response(
    content: str | None = "hello",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> SimpleNamespace:
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, role="assistant")
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _text_chunk(content: str | None, finish_reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(reasoning=None, content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


class RecordingAzureSDK:
    """Patch target for AsyncAzureOpenAI that records constructor kwargs."""

    instances: list["RecordingAzureSDK"] = []

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        RecordingAzureSDK.instances.append(self)
        self.completions = FakeCompletions([])
        self.chat = FakeChat(self.completions)


# ── Factory ──────────────────────────────────────────────────────────────────


class TestAzureFactory(unittest.TestCase):
    def test_factory_builds_azure_provider_both_aliases(self):
        client = FakeAzureClient()
        self.assertIsInstance(make_provider("azure", client=client), AzureOpenAIProvider)
        self.assertIsInstance(
            make_provider("azure-openai", client=client), AzureOpenAIProvider
        )

    def test_factory_case_insensitive(self):
        client = FakeAzureClient()
        self.assertIsInstance(make_provider("Azure", client=client), AzureOpenAIProvider)

    def test_factory_rejects_unknown_name(self):
        with self.assertRaises(ValueError) as ctx:
            make_provider("bedrock")
        # Error message advertises azure as a supported provider.
        self.assertIn("azure", str(ctx.exception))


# ── Construction / credential resolution ─────────────────────────────────────


class TestAzureConstruction(unittest.TestCase):
    def setUp(self):
        RecordingAzureSDK.instances = []

    def test_injected_client_preserves_identity_and_skips_env(self):
        client = FakeAzureClient()
        with patch.dict(os.environ, {}, clear=True):
            provider = AzureOpenAIProvider(client=client)
        self.assertIs(provider.client, client)

    def test_explicit_args_forwarded_to_sdk(self):
        with patch(
            "harnessx.providers.azure_openai.AsyncAzureOpenAI",
            RecordingAzureSDK,
        ):
            with patch.dict(os.environ, {}, clear=True):
                AzureOpenAIProvider(
                    azure_endpoint="https://res.openai.azure.com",
                    api_key="secret-key",
                    api_version="2024-10-21",
                )
        kwargs = RecordingAzureSDK.instances[0].kwargs
        self.assertEqual(kwargs["azure_endpoint"], "https://res.openai.azure.com")
        self.assertEqual(kwargs["api_key"], "secret-key")
        self.assertEqual(kwargs["api_version"], "2024-10-21")

    def test_env_fallback(self):
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://env-res.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "env-key",
            "OPENAI_API_VERSION": "2024-06-01",
        }
        with patch(
            "harnessx.providers.azure_openai.AsyncAzureOpenAI",
            RecordingAzureSDK,
        ):
            with patch.dict(os.environ, env, clear=True):
                AzureOpenAIProvider()
        kwargs = RecordingAzureSDK.instances[0].kwargs
        self.assertEqual(kwargs["azure_endpoint"], "https://env-res.openai.azure.com")
        self.assertEqual(kwargs["api_key"], "env-key")
        self.assertEqual(kwargs["api_version"], "2024-06-01")

    def test_explicit_args_take_precedence_over_env(self):
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "env-key",
            "OPENAI_API_VERSION": "2024-06-01",
        }
        with patch(
            "harnessx.providers.azure_openai.AsyncAzureOpenAI",
            RecordingAzureSDK,
        ):
            with patch.dict(os.environ, env, clear=True):
                AzureOpenAIProvider(
                    azure_endpoint="https://explicit.openai.azure.com",
                    api_key="explicit-key",
                    api_version="2024-10-21",
                )
        kwargs = RecordingAzureSDK.instances[0].kwargs
        self.assertEqual(kwargs["azure_endpoint"], "https://explicit.openai.azure.com")
        self.assertEqual(kwargs["api_key"], "explicit-key")
        self.assertEqual(kwargs["api_version"], "2024-10-21")

    def test_missing_endpoint_raises_clear_error(self):
        with patch(
            "harnessx.providers.azure_openai.AsyncAzureOpenAI",
            RecordingAzureSDK,
        ):
            with patch.dict(os.environ, {"OPENAI_API_VERSION": "2024-10-21"}, clear=True):
                with self.assertRaises(ValueError) as ctx:
                    AzureOpenAIProvider(api_key="k")
        msg = str(ctx.exception)
        self.assertIn("azure_endpoint", msg)
        self.assertIn("AZURE_OPENAI_ENDPOINT", msg)

    def test_missing_api_version_raises_clear_error(self):
        with patch(
            "harnessx.providers.azure_openai.AsyncAzureOpenAI",
            RecordingAzureSDK,
        ):
            with patch.dict(
                os.environ,
                {"AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com"},
                clear=True,
            ):
                with self.assertRaises(ValueError) as ctx:
                    AzureOpenAIProvider(api_key="k")
        msg = str(ctx.exception)
        self.assertIn("api_version", msg)
        self.assertIn("OPENAI_API_VERSION", msg)

    def test_token_auth_without_api_key(self):
        """azure_ad_token_provider auth works and no dummy api_key is injected."""

        def token_provider() -> str:
            return "aad-token"

        with patch(
            "harnessx.providers.azure_openai.AsyncAzureOpenAI",
            RecordingAzureSDK,
        ):
            with patch.dict(os.environ, {}, clear=True):
                AzureOpenAIProvider(
                    azure_endpoint="https://res.openai.azure.com",
                    api_version="2024-10-21",
                    azure_ad_token_provider=token_provider,
                )
        kwargs = RecordingAzureSDK.instances[0].kwargs
        self.assertIs(kwargs["azure_ad_token_provider"], token_provider)
        self.assertNotIn("api_key", kwargs)

    def test_config_driven_construction_through_make_provider(self):
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://factory.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "factory-key",
            "OPENAI_API_VERSION": "2024-10-21",
        }
        with patch(
            "harnessx.providers.azure_openai.AsyncAzureOpenAI",
            RecordingAzureSDK,
        ):
            with patch.dict(os.environ, env, clear=True):
                provider = make_provider("azure")
        self.assertIsInstance(provider, AzureOpenAIProvider)
        kwargs = RecordingAzureSDK.instances[0].kwargs
        self.assertEqual(kwargs["azure_endpoint"], "https://factory.openai.azure.com")
        self.assertEqual(kwargs["api_key"], "factory-key")
        self.assertEqual(kwargs["api_version"], "2024-10-21")


# ── Completion ───────────────────────────────────────────────────────────────


class TestAzureCreate(unittest.TestCase):
    def test_model_passes_through_as_deployment(self):
        client = FakeAzureClient([_completion_response()])
        provider = AzureOpenAIProvider(client=client)
        _run(
            provider.create(
                model="my-gpt4o-prod",
                messages=[{"role": "user", "content": "hi"}],
                system=None,
                tools=[],
                max_tokens=100,
                temperature=0.0,
            )
        )
        self.assertEqual(client.completions.calls[0]["model"], "my-gpt4o-prod")

    def test_azure_deployment_overrides_model(self):
        client = FakeAzureClient([_completion_response()])
        provider = AzureOpenAIProvider(client=client, azure_deployment="pinned-deploy")
        _run(
            provider.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                system=None,
                tools=[],
                max_tokens=100,
                temperature=0.0,
            )
        )
        self.assertEqual(client.completions.calls[0]["model"], "pinned-deploy")

    def test_optional_temperature_omitted(self):
        client = FakeAzureClient([_completion_response()])
        provider = AzureOpenAIProvider(client=client)
        _run(
            provider.create(
                model="d",
                messages=[{"role": "user", "content": "hi"}],
                system=None,
                tools=[],
                max_tokens=100,
            )
        )
        self.assertNotIn("temperature", client.completions.calls[0])

    def test_text_and_usage_normalization(self):
        client = FakeAzureClient(
            [_completion_response(content="Answer", prompt_tokens=12, completion_tokens=7)]
        )
        provider = AzureOpenAIProvider(client=client)
        resp = _run(
            provider.create(
                model="d",
                messages=[{"role": "user", "content": "q"}],
                system=None,
                tools=[],
                max_tokens=100,
            )
        )
        self.assertEqual(resp.text, "Answer")
        self.assertEqual(resp.stop_reason, StopReason.END_TURN)
        self.assertEqual(resp.usage.input_tokens, 12)
        self.assertEqual(resp.usage.output_tokens, 7)

    def test_tool_call_response_round_trip(self):
        tc = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="get_weather", arguments='{"city": "Paris"}'),
        )
        client = FakeAzureClient(
            [_completion_response(content=None, tool_calls=[tc], finish_reason="tool_calls")]
        )
        provider = AzureOpenAIProvider(client=client)
        resp = _run(
            provider.create(
                model="d",
                messages=[{"role": "user", "content": "weather?"}],
                system=None,
                tools=TOOLS,
                max_tokens=100,
            )
        )
        self.assertEqual(resp.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].name, "get_weather")
        self.assertEqual(resp.tool_calls[0].input, {"city": "Paris"})

        # The tool schema is translated to OpenAI's function format on the wire.
        call = client.completions.calls[0]
        self.assertEqual(call["tools"][0]["function"]["name"], "get_weather")


# ── Streaming ────────────────────────────────────────────────────────────────


class TestAzureStream(unittest.TestCase):
    def _collect(self, provider, **kwargs):
        async def collect():
            events = []
            async for ch in provider.stream(**kwargs):
                events.append(ch)
            return events

        return _run(collect())

    def test_stream_text_deltas_and_final_response(self):
        stream = FakeAsyncStream([_text_chunk("Hello "), _text_chunk("world", "stop")])
        client = FakeAzureClient([stream])
        provider = AzureOpenAIProvider(client=client)

        events = self._collect(
            provider,
            model="my-deploy",
            messages=[{"role": "user", "content": "hi"}],
            system=None,
            tools=[],
            max_tokens=100,
        )

        text_deltas = [e.data for e in events if e.kind == "text_delta"]
        self.assertEqual(text_deltas, ["Hello ", "world"])

        response_chunks = [e for e in events if e.kind == "response"]
        self.assertEqual(len(response_chunks), 1)
        self.assertEqual(response_chunks[0].data.text, "Hello world")
        self.assertEqual(response_chunks[0].data.stop_reason, StopReason.END_TURN)

        # Deployment name (model) passed straight through in streaming too.
        self.assertEqual(client.completions.calls[0]["model"], "my-deploy")

    def test_stream_applies_deployment_override(self):
        stream = FakeAsyncStream([_text_chunk("x", "stop")])
        client = FakeAzureClient([stream])
        provider = AzureOpenAIProvider(client=client, azure_deployment="pinned")

        self._collect(
            provider,
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            system=None,
            tools=[],
            max_tokens=100,
        )
        self.assertEqual(client.completions.calls[0]["model"], "pinned")

    def test_stream_assembles_tool_calls_and_usage(self):
        tc_delta = SimpleNamespace(
            index=0,
            id="call_1",
            function=SimpleNamespace(name="get_weather", arguments='{"city": "Paris"}'),
        )
        chunk_tool = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning=None, content=None, tool_calls=[tc_delta]),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        )
        chunk_usage = SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=6),
        )
        stream = FakeAsyncStream([chunk_tool, chunk_usage])
        client = FakeAzureClient([stream])
        provider = AzureOpenAIProvider(client=client)

        events = self._collect(
            provider,
            model="d",
            messages=[{"role": "user", "content": "weather?"}],
            system=None,
            tools=TOOLS,
            max_tokens=100,
        )
        final = [e for e in events if e.kind == "response"][0].data
        self.assertEqual(final.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(len(final.tool_calls), 1)
        self.assertEqual(final.tool_calls[0].name, "get_weather")
        self.assertEqual(final.tool_calls[0].input, {"city": "Paris"})
        self.assertEqual(final.usage.input_tokens, 20)
        self.assertEqual(final.usage.output_tokens, 6)


# ── Token counting ───────────────────────────────────────────────────────────


class TestAzureTokenCounting(unittest.TestCase):
    def test_count_tokens_handles_unknown_deployment_names(self):
        client = FakeAzureClient()
        provider = AzureOpenAIProvider(client=client)
        n = _run(
            provider.count_tokens(
                model="my-weird-deployment",
                messages=[{"role": "user", "content": "hello world"}],
                system="be brief",
                tools=[],
            )
        )
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_count_tokens_falls_back_when_tiktoken_unavailable(self):
        import sys

        client = FakeAzureClient()
        provider = AzureOpenAIProvider(client=client)
        # Force the tiktoken import inside count_tokens to fail -> char heuristic.
        with patch.dict(sys.modules, {"tiktoken": None}):
            n = _run(
                provider.count_tokens(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hello world"}],
                    system="system",
                    tools=[],
                )
            )
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)


# ── Agent / runtime integration ──────────────────────────────────────────────


class TestAzureIntegration(unittest.TestCase):
    def test_agent_run_with_azure(self):
        client = FakeAzureClient([_completion_response(content="Integrated!")])
        provider = AzureOpenAIProvider(client=client)
        agent = Agent(
            config=AgentConfig(provider="azure", model="my-deploy"),
            provider=provider,
        )
        result = _run(agent.run("Hi"))
        self.assertEqual(result.output, "Integrated!")

    def test_streaming_agent_with_azure(self):
        stream = FakeAsyncStream([_text_chunk("Streaming "), _text_chunk("live!", "stop")])
        client = FakeAzureClient([stream])
        provider = AzureOpenAIProvider(client=client)
        agent = Agent(
            config=AgentConfig(provider="azure", model="my-deploy"),
            provider=provider,
        )

        async def run_streaming():
            deltas = []
            async with agent.run_stream("Stream test") as stream:
                async for event in stream:
                    if event.type.value == "text_delta":
                        deltas.append(event.data)
            return deltas

        self.assertEqual(_run(run_streaming()), ["Streaming ", "live!"])

    def test_runtime_uses_injected_provider(self):
        import tempfile

        client = FakeAzureClient([_completion_response()])
        provider = AzureOpenAIProvider(client=client)
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(
                config=AgentConfig(provider="azure", model="my-deploy"),
                provider=provider,
            )
            runtime = AgentRuntime(agent, runtime_config=RuntimeConfig(storage_dir=str(tmp)))

            async def exercise():
                await runtime.start()
                try:
                    self.assertIs(runtime._agent.provider, provider)
                    result = await runtime.execute("Hi")
                    self.assertEqual(result.output, "hello")
                finally:
                    await runtime.stop()

            _run(exercise())


# ── Optional-dependency behavior ─────────────────────────────────────────────


class TestAzureImportGuard(unittest.TestCase):
    def test_missing_sdk_import_error_message(self):
        import importlib
        import sys

        import harnessx.providers.azure_openai as azure_mod

        with patch.dict(sys.modules, {"openai": None}):
            with self.assertRaises(ImportError) as ctx:
                importlib.reload(azure_mod)
            self.assertIn("harnessx[azure]", str(ctx.exception))
        importlib.reload(azure_mod)


if __name__ == "__main__":
    unittest.main()
