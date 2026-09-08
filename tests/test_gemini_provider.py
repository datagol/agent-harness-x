"""Tests for GeminiProvider: canonical translation, thought_signature
round-tripping, streaming chunk contract, token counting, and factory wiring."""

from __future__ import annotations

import asyncio
import base64
import unittest
from types import SimpleNamespace
from typing import Any

from datagol_agent_harness.providers import make_provider
from datagol_agent_harness.providers.gemini import (
    THOUGHT_SIGNATURE_KEY,
    GeminiProvider,
    _to_gemini_contents,
    _to_gemini_tools,
)
from datagol_agent_harness.types import StopReason


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
        return self.responses.pop(0)

    async def generate_content_stream(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        resp = self.responses.pop(0)

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


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.models = FakeAioModels(responses)
        self.aio = SimpleNamespace(models=self.models)


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
        from datagol_agent_harness.memory import ConversationMemory

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

        import datagol_agent_harness.providers.gemini as gemini_mod

        with mock.patch.dict(sys.modules, {"google": None, "google.genai": None}):
            with self.assertRaises(ImportError) as ctx:
                importlib.reload(gemini_mod)
            self.assertIn("datagol-agent-harness[gemini]", str(ctx.exception))
        importlib.reload(gemini_mod)


if __name__ == "__main__":
    unittest.main()
