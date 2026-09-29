"""Tests for GeminiProvider: canonical translation, thought_signature
round-tripping, streaming chunk contract, token counting, and factory wiring."""

from __future__ import annotations

import asyncio
import base64
import unittest
from types import SimpleNamespace
from typing import Any

from harnessx.providers import make_provider
from harnessx.providers.gemini import (
    THOUGHT_SIGNATURE_KEY,
    GeminiProvider,
    _to_gemini_contents,
    _to_gemini_tools,
)
from harnessx.types import StopReason


# ── genai response fakes ─────────────────────────────────────────────────────


def _text_part(text: str, thought: bool = False) -> SimpleNamespace:
    return SimpleNamespace(text=text, thought=thought, function_call=None)


def _call_part(
    name: str, args: dict, sig: bytes | None = None, call_id: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        text=None,
        thought=False,
        function_call=SimpleNamespace(id=call_id, name=name, args=args),
        thought_signature=sig,
    )


def _response(parts: list, finish_reason: str = "STOP") -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=parts), finish_reason=finish_reason
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            cached_content_token_count=0,
        ),
    )


class FakeAioModels:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def generate_content_stream(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp

        async def gen():
            # Emit each part as its own chunk, like the real stream.
            for part in resp.candidates[0].content.parts:
                yield SimpleNamespace(
                    candidates=[
                        SimpleNamespace(
                            content=SimpleNamespace(parts=[part]),
                            finish_reason=resp.candidates[0].finish_reason,
                        )
                    ],
                    usage_metadata=resp.usage_metadata,
                )

        return gen()

    async def count_tokens(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(total_tokens=42)


class FakeAioCaches:
    def __init__(self, fail: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail
        self._n = 0

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail
        self._n += 1
        return SimpleNamespace(name=f"cachedContents/fake-{self._n}")


class FakeClient:
    def __init__(
        self, responses: list[Any], caches: FakeAioCaches | None = None
    ) -> None:
        self.models = FakeAioModels(responses)
        self.caches = caches or FakeAioCaches()
        self.aio = SimpleNamespace(models=self.models, caches=self.caches)


def _run(coro):
    return asyncio.run(coro)


TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]


class TestGeminiCreate(unittest.TestCase):
    def test_create_translates_request_and_returns_canonical_blocks(self):
        client = FakeClient([_response([_text_part("Hello!")])])
        provider = GeminiProvider(client=client)

        resp = _run(
            provider.create(
                model="gemini-test",
                messages=[{"role": "user", "content": "Hi"}],
                system="Be nice",
                tools=TOOLS,
                max_tokens=256,
            )
        )

        self.assertEqual(resp.text, "Hello!")
        self.assertEqual(resp.stop_reason, StopReason.END_TURN)
        self.assertEqual(resp.usage.input_tokens, 10)
        self.assertEqual(resp.usage.output_tokens, 5)
        self.assertEqual(resp.content, [{"type": "text", "text": "Hello!"}])

        call = client.models.calls[0]
        self.assertEqual(call["model"], "gemini-test")
        self.assertEqual(
            call["contents"], [{"role": "user", "parts": [{"text": "Hi"}]}]
        )
        config = call["config"]
        self.assertEqual(config["system_instruction"], "Be nice")
        self.assertEqual(config["max_output_tokens"], 256)
        self.assertEqual(config["automatic_function_calling"], {"disable": True})
        self.assertEqual(
            config["tools"][0]["function_declarations"],
            [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": TOOLS[0]["input_schema"],
                }
            ],
        )

    def test_tool_use_mapping_and_thought_signature_stash(self):
        sig = b"opaque-signature-bytes"
        client = FakeClient(
            [_response([_call_part("get_weather", {"city": "Paris"}, sig=sig)])]
        )
        provider = GeminiProvider(client=client)

        resp = _run(
            provider.create(
                model="gemini-test",
                messages=[{"role": "user", "content": "Weather?"}],
                system=None,
                tools=TOOLS,
                max_tokens=256,
            )
        )

        self.assertEqual(resp.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(len(resp.tool_calls), 1)
        tc = resp.tool_calls[0]
        self.assertEqual(tc.name, "get_weather")
        self.assertEqual(tc.input, {"city": "Paris"})
        self.assertTrue(tc.id)

        block = resp.content[0]
        self.assertIsInstance(block, dict)  # dicts survive ConversationMemory verbatim
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], tc.id)
        self.assertEqual(
            base64.b64decode(block[THOUGHT_SIGNATURE_KEY]), sig
        )

    def test_max_tokens_finish_reason(self):
        client = FakeClient([_response([_text_part("partial")], finish_reason="MAX_TOKENS")])
        provider = GeminiProvider(client=client)
        resp = _run(
            provider.create(
                model="gemini-test",
                messages=[{"role": "user", "content": "Hi"}],
                system=None,
                tools=[],
                max_tokens=8,
            )
        )
        self.assertEqual(resp.stop_reason, StopReason.MAX_TOKENS)


class TestHistoryTranslation(unittest.TestCase):
    def test_thought_signature_restored_into_contents(self):
        sig = b"sig-bytes"
        b64 = base64.b64encode(sig).decode()
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                        THOUGHT_SIGNATURE_KEY: b64,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "Sunny, 22C",
                    }
                ],
            },
        ]
        contents = _to_gemini_contents(messages)

        self.assertEqual(len(contents), 3)
        model_turn = contents[1]
        self.assertEqual(model_turn["role"], "model")
        part = model_turn["parts"][0]
        self.assertEqual(part["function_call"]["name"], "get_weather")
        self.assertEqual(part["function_call"]["args"], {"city": "Paris"})
        self.assertEqual(part["thought_signature"], sig)

        # tool_result maps back to a function_response addressed by name
        fr = contents[2]["parts"][0]["function_response"]
        self.assertEqual(fr["name"], "get_weather")
        self.assertEqual(fr["response"]["output"], "Sunny, 22C")

    def test_round_trip_through_conversation_memory(self):
        """Provider response blocks -> memory -> back to Gemini contents keeps the signature."""
        from harnessx.memory import ConversationMemory

        sig = b"round-trip-sig"
        client = FakeClient(
            [_response([_call_part("get_weather", {"city": "Oslo"}, sig=sig)])]
        )
        provider = GeminiProvider(client=client)
        resp = _run(
            provider.create(
                model="gemini-test",
                messages=[{"role": "user", "content": "Weather?"}],
                system=None,
                tools=TOOLS,
                max_tokens=64,
            )
        )

        memory = ConversationMemory()
        memory.add_user_message("Weather?")
        memory.add_assistant_message(resp.content)

        contents = _to_gemini_contents(memory.get_messages())
        part = contents[1]["parts"][0]
        self.assertEqual(part["thought_signature"], sig)


class TestGeminiStream(unittest.TestCase):
    def test_stream_yields_anthropic_style_chunks(self):
        sig = b"stream-sig"
        client = FakeClient(
            [
                _response(
                    [
                        _text_part("thinking about it", thought=True),
                        _text_part("Hello "),
                        _text_part("world"),
                        _call_part("get_weather", {"city": "Rome"}, sig=sig),
                    ]
                )
            ]
        )
        provider = GeminiProvider(client=client)

        async def collect():
            chunks = []
            async for chunk in provider.stream(
                model="gemini-test",
                messages=[{"role": "user", "content": "Hi"}],
                system=None,
                tools=TOOLS,
                max_tokens=64,
            ):
                chunks.append(chunk)
            return chunks

        chunks = _run(collect())
        kinds = [c.kind for c in chunks]
        self.assertEqual(kinds, ["thinking_delta", "text_delta", "text_delta", "response"])
        self.assertEqual(chunks[1].data, "Hello ")
        self.assertEqual(chunks[2].data, "world")

        final = chunks[-1].data
        self.assertEqual(final.text, "Hello world")
        self.assertEqual(final.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(final.tool_calls[0].name, "get_weather")
        tool_block = [b for b in final.content if b["type"] == "tool_use"][0]
        self.assertEqual(base64.b64decode(tool_block[THOUGHT_SIGNATURE_KEY]), sig)


class TestGeminiMisc(unittest.TestCase):
    def test_count_tokens(self):
        client = FakeClient([])
        provider = GeminiProvider(client=client)
        n = _run(
            provider.count_tokens(
                model="gemini-test",
                messages=[{"role": "user", "content": "Hi"}],
                system="",
                tools=[],
            )
        )
        self.assertEqual(n, 42)

    def test_make_provider_returns_gemini(self):
        client = FakeClient([])
        provider = make_provider("gemini", client=client)
        self.assertIsInstance(provider, GeminiProvider)
        self.assertEqual(provider.name, "gemini")

    def test_make_provider_unknown_name(self):
        with self.assertRaises(ValueError):
            make_provider("no-such-provider")

    def test_tool_declaration_translation(self):
        self.assertEqual(
            _to_gemini_tools(TOOLS),
            [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": TOOLS[0]["input_schema"],
                }
            ],
        )

    def test_thinking_level_config(self):
        client = FakeClient([_response([_text_part("ok")])])
        provider = GeminiProvider(client=client, thinking_level="low")
        _run(
            provider.create(
                model="gemini-test",
                messages=[{"role": "user", "content": "Hi"}],
                system=None,
                tools=[],
                max_tokens=8,
            )
        )
        config = client.models.calls[0]["config"]
        self.assertEqual(config["thinking_config"], {"thinking_level": "low"})

    def test_missing_sdk_import_error_message(self):
        import importlib
        import sys
        import unittest.mock as mock

        import harnessx.providers.gemini as gemini_mod

        with mock.patch.dict(sys.modules, {"google": None, "google.genai": None}):
            with self.assertRaises(ImportError) as ctx:
                importlib.reload(gemini_mod)
            self.assertIn("harnessx[gemini]", str(ctx.exception))
        importlib.reload(gemini_mod)


class TestGeminiPromptCache(unittest.TestCase):
    def _create(self, provider, **overrides):
        kwargs = dict(
            model="gemini-test",
            messages=[{"role": "user", "content": "Hi"}],
            system="Long static system prompt",
            tools=TOOLS,
            max_tokens=64,
        )
        kwargs.update(overrides)
        return _run(provider.create(**kwargs))

    def test_disabled_by_default_no_caches_calls(self):
        client = FakeClient([_response([_text_part("ok")])])
        provider = GeminiProvider(client=client)
        self._create(provider)
        self.assertEqual(client.caches.calls, [])
        config = client.models.calls[0]["config"]
        self.assertIn("system_instruction", config)
        self.assertNotIn("cached_content", config)

    def test_enabled_creates_cache_once_and_reuses(self):
        client = FakeClient(
            [_response([_text_part("one")]), _response([_text_part("two")])]
        )
        provider = GeminiProvider(client=client, prompt_cache_ttl=3600)
        self._create(provider)
        self._create(provider)

        # caches.create awaited exactly once across the two calls
        self.assertEqual(len(client.caches.calls), 1)
        cache_call = client.caches.calls[0]
        self.assertEqual(cache_call["model"], "gemini-test")
        self.assertEqual(
            cache_call["config"]["system_instruction"], "Long static system prompt"
        )
        self.assertEqual(cache_call["config"]["ttl"], "3600s")
        self.assertIn("tools", cache_call["config"])

        for call in client.models.calls:
            config = call["config"]
            self.assertEqual(config["cached_content"], "cachedContents/fake-1")
            # Gemini requires exclusivity: no system/tools alongside the cache
            self.assertNotIn("system_instruction", config)
            self.assertNotIn("tools", config)
            self.assertEqual(config["max_output_tokens"], 64)

    def test_cached_config_keeps_thinking_config(self):
        client = FakeClient([_response([_text_part("ok")])])
        provider = GeminiProvider(
            client=client, prompt_cache_ttl=3600, thinking_level="low"
        )
        self._create(provider)
        config = client.models.calls[0]["config"]
        self.assertEqual(config["thinking_config"], {"thinking_level": "low"})
        self.assertIn("cached_content", config)

    def test_env_var_enables_caching(self):
        import unittest.mock as mock

        client = FakeClient([_response([_text_part("ok")])])
        with mock.patch.dict("os.environ", {"GEMINI_PROMPT_CACHE_TTL": "120"}):
            provider = GeminiProvider(client=client)
        self.assertEqual(provider.prompt_cache_ttl, 120)
        self._create(provider)
        self.assertEqual(client.caches.calls[0]["config"]["ttl"], "120s")

    def test_create_failure_falls_back_uncached(self):
        client = FakeClient(
            [_response([_text_part("ok")])],
            caches=FakeAioCaches(fail=RuntimeError("Cached content is too small")),
        )
        provider = GeminiProvider(client=client, prompt_cache_ttl=3600)
        resp = self._create(provider)
        self.assertEqual(resp.text, "ok")
        config = client.models.calls[0]["config"]
        self.assertNotIn("cached_content", config)
        self.assertEqual(config["system_instruction"], "Long static system prompt")

    def test_stale_cache_error_drops_entry_and_retries_uncached(self):
        client = FakeClient(
            [
                RuntimeError("403 CachedContent not found: cachedContents/fake-1"),
                _response([_text_part("recovered")]),
            ]
        )
        provider = GeminiProvider(client=client, prompt_cache_ttl=3600)
        resp = self._create(provider)
        self.assertEqual(resp.text, "recovered")
        self.assertEqual(len(client.models.calls), 2)
        self.assertIn("cached_content", client.models.calls[0]["config"])
        retry_config = client.models.calls[1]["config"]
        self.assertNotIn("cached_content", retry_config)
        self.assertIn("system_instruction", retry_config)
        self.assertEqual(provider._prompt_caches, {})

    def test_unrelated_error_propagates(self):
        client = FakeClient([RuntimeError("429 rate limited")])
        provider = GeminiProvider(client=client, prompt_cache_ttl=3600)
        with self.assertRaises(RuntimeError):
            self._create(provider)
        self.assertEqual(len(client.models.calls), 1)

    def test_ttl_expiry_recreates_cache(self):
        import unittest.mock as mock

        client = FakeClient(
            [_response([_text_part("one")]), _response([_text_part("two")])]
        )
        provider = GeminiProvider(client=client, prompt_cache_ttl=3600)
        self._create(provider)
        # Jump past expires_at (ttl minus the 60s safety margin)
        key, (_, expires_at) = next(iter(provider._prompt_caches.items()))
        with mock.patch(
            "harnessx.providers.gemini.time.monotonic",
            return_value=expires_at + 1,
        ):
            self._create(provider)
        self.assertEqual(len(client.caches.calls), 2)
        self.assertEqual(
            client.models.calls[1]["config"]["cached_content"],
            "cachedContents/fake-2",
        )

    def test_stream_uses_cached_config(self):
        client = FakeClient(
            [_response([_text_part("hi")]), _response([_text_part("again")])]
        )
        provider = GeminiProvider(client=client, prompt_cache_ttl=3600)

        async def collect():
            return [
                c
                async for c in provider.stream(
                    model="gemini-test",
                    messages=[{"role": "user", "content": "Hi"}],
                    system="Long static system prompt",
                    tools=TOOLS,
                    max_tokens=64,
                )
            ]

        _run(collect())
        _run(collect())
        self.assertEqual(len(client.caches.calls), 1)
        for call in client.models.calls:
            self.assertIn("cached_content", call["config"])
            self.assertNotIn("system_instruction", call["config"])

    def test_stream_stale_cache_retries_uncached(self):
        client = FakeClient(
            [
                RuntimeError("CachedContent not found"),
                _response([_text_part("recovered")]),
            ]
        )
        provider = GeminiProvider(client=client, prompt_cache_ttl=3600)

        async def collect():
            return [
                c
                async for c in provider.stream(
                    model="gemini-test",
                    messages=[{"role": "user", "content": "Hi"}],
                    system="Long static system prompt",
                    tools=TOOLS,
                    max_tokens=64,
                )
            ]

        chunks = _run(collect())
        self.assertEqual(chunks[-1].data.text, "recovered")
        self.assertEqual(len(client.models.calls), 2)
        self.assertNotIn("cached_content", client.models.calls[1]["config"])

    def test_no_cache_when_nothing_to_cache(self):
        client = FakeClient([_response([_text_part("ok")])])
        provider = GeminiProvider(client=client, prompt_cache_ttl=3600)
        self._create(provider, system=None, tools=[])
        self.assertEqual(client.caches.calls, [])


if __name__ == "__main__":
    unittest.main()


def test_tool_schemas_drop_keywords_gemini_does_not_define():
    """A tool carrying `additionalProperties` costs the caller prompt caching.

    generateContent tolerates the unknown keyword; cachedContents rejects the
    whole request with "Unknown name \"additional_properties\"". The tool that
    triggered it in practice is this package's own `read_tool_result`, whose
    schema is generated from its signature and registered on every agent — so
    on 0.4 every Gemini agent lost explicit caching, with one warning line to
    show for it.
    """
    from harnessx.providers.gemini import _to_gemini_tools

    declarations = _to_gemini_tools([
        {
            "name": "read_tool_result",
            "description": "",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tool_use_id": {"type": "string"},
                    "nested": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
                "$schema": "https://json-schema.org/draft/2020-12/schema",
            },
        }
    ])
    parameters = declarations[0]["parameters"]
    assert "additionalProperties" not in parameters
    assert "$schema" not in parameters
    # Removed at every depth, not just the top level.
    assert "additionalProperties" not in parameters["properties"]["nested"]
    # Everything Gemini does define survives untouched.
    assert parameters["type"] == "object"
    assert parameters["properties"]["tool_use_id"] == {"type": "string"}
    assert parameters["properties"]["nested"]["properties"] == {"x": {"type": "string"}}


def test_the_built_in_read_tool_result_schema_is_gemini_safe():
    """The regression this fixes, caught at the source rather than in a mock."""
    from harnessx import Agent, AgentConfig

    agent = Agent(
        config=AgentConfig(model="gemini-3.6-flash", provider="gemini", system_prompt="x"),
        provider=GeminiProvider(client=SimpleNamespace()),
    )
    declarations = _to_gemini_tools(agent.tools.get_tool_params())
    assert declarations, "the built-in tool should be registered"
    for declaration in declarations:
        assert "additionalProperties" not in declaration["parameters"]
