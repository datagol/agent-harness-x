"""Image, document and audio input: canonical block, and each provider's wire form.

The canonical format follows Anthropic's image/document blocks, which the rest
of this format already mirrors, with `audio` added for the providers that take
it. A provider that cannot carry a kind raises rather than dropping it: a
prompt that arrives without the photo it refers to is answered confidently
about nothing, which is worse than a failed call.
"""

from __future__ import annotations

import base64
import unittest

from harnessx.messages import MEDIA_KINDS, Message
from harnessx.providers.anthropic import _messages_for_request
from harnessx.providers.gemini import _to_gemini_contents
from harnessx.providers.openai import _to_openai_messages

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n fake pixels").decode()
WAV = base64.b64encode(b"RIFF....WAVEfmt fake samples").decode()
PDF = base64.b64encode(b"%PDF-1.7 fake document").decode()


def block(kind: str, media_type: str, data: str) -> dict:
    return {"type": kind, "source": {"type": "base64", "media_type": media_type, "data": data}}


def user(*blocks) -> dict:
    return {"role": "user", "content": list(blocks)}


class CanonicalBlock(unittest.TestCase):
    def test_every_kind_is_accepted_on_a_user_message(self):
        for kind, media_type, data in (
            ("image", "image/png", PNG),
            ("document", "application/pdf", PDF),
            ("audio", "audio/wav", WAV),
        ):
            msg = Message(role="user", content=[block(kind, media_type, data)])
            self.assertEqual(msg.content[0]["type"], kind)

    def test_text_and_media_travel_together(self):
        msg = Message(role="user", content=[
            {"type": "text", "text": "what is on this list?"},
            block("image", "image/jpeg", PNG),
        ])
        self.assertEqual([b["type"] for b in msg.content], ["text", "image"])

    def test_media_belongs_to_the_user(self):
        with self.assertRaises(ValueError) as caught:
            Message(role="assistant", content=[block("image", "image/png", PNG)])
        self.assertIn("user message", str(caught.exception))

    def test_an_unsupported_media_type_fails_at_construction(self):
        """A typo should fail here, not as a provider 400 several seconds later."""
        with self.assertRaises(ValueError) as caught:
            Message(role="user", content=[block("image", "image/tiff", PNG)])
        self.assertIn("image/tiff", str(caught.exception))
        self.assertIn("image/png", str(caught.exception))

    def test_a_kind_cannot_carry_another_kinds_media_type(self):
        with self.assertRaises(ValueError):
            Message(role="user", content=[block("audio", "image/png", PNG)])

    def test_the_source_must_be_base64(self):
        with self.assertRaises(ValueError) as caught:
            Message(role="user", content=[
                {"type": "image", "source": {"type": "url", "media_type": "image/png",
                                             "url": "https://example.com/a.png"}}])
        self.assertIn("base64", str(caught.exception))

    def test_empty_data_is_refused(self):
        with self.assertRaises(ValueError):
            Message(role="user", content=[block("image", "image/png", "")])

    def test_the_m4a_a_phone_records_is_accepted(self):
        """The format iOS writes by default; rejecting it would fail real voice notes."""
        self.assertIn("audio/x-m4a", MEDIA_KINDS["audio"])
        Message(role="user", content=[block("audio", "audio/x-m4a", WAV)])


class GeminiWire(unittest.TestCase):
    def test_media_becomes_inline_data_with_decoded_bytes(self):
        contents = _to_gemini_contents([user(block("image", "image/png", PNG))])
        part = contents[0]["parts"][0]
        self.assertEqual(part["inline_data"]["mime_type"], "image/png")
        self.assertEqual(part["inline_data"]["data"], base64.b64decode(PNG))

    def test_audio_and_documents_go_the_same_way(self):
        contents = _to_gemini_contents([user(
            block("audio", "audio/x-m4a", WAV), block("document", "application/pdf", PDF))])
        kinds = [p["inline_data"]["mime_type"] for p in contents[0]["parts"]]
        self.assertEqual(kinds, ["audio/x-m4a", "application/pdf"])

    def test_text_keeps_its_place_beside_the_media(self):
        contents = _to_gemini_contents([user(
            {"type": "text", "text": "read this"}, block("image", "image/jpeg", PNG))])
        parts = contents[0]["parts"]
        self.assertEqual(parts[0], {"text": "read this"})
        self.assertIn("inline_data", parts[1])


class OpenAIWire(unittest.TestCase):
    def test_an_image_becomes_a_data_uri(self):
        out = _to_openai_messages([user(block("image", "image/png", PNG))], system=None)
        part = out[-1]["content"][0]
        self.assertEqual(part["type"], "image_url")
        self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_audio_carries_a_container_name_not_a_media_type(self):
        out = _to_openai_messages([user(block("audio", "audio/wav", WAV))], system=None)
        part = out[-1]["content"][0]
        self.assertEqual(part["type"], "input_audio")
        self.assertEqual(part["input_audio"]["format"], "wav")

    def test_audio_it_cannot_take_is_named_in_the_error(self):
        with self.assertRaises(ValueError) as caught:
            _to_openai_messages([user(block("audio", "audio/x-m4a", WAV))], system=None)
        self.assertIn("audio/x-m4a", str(caught.exception))

    def test_text_and_media_become_one_parts_list(self):
        out = _to_openai_messages(
            [user({"type": "text", "text": "what is this?"}, block("image", "image/png", PNG))],
            system=None)
        content = out[-1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "what is this?"})
        self.assertEqual(content[1]["type"], "image_url")

    def test_a_text_only_turn_is_still_a_plain_string(self):
        """The shape almost every call uses must not change."""
        out = _to_openai_messages(
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}], system=None)
        self.assertEqual(out[-1]["content"], "hello")


class AnthropicWire(unittest.TestCase):
    def test_an_image_passes_through_in_its_canonical_shape(self):
        out = _messages_for_request([user(block("image", "image/png", PNG))])
        sent = out[-1]["content"][0]
        self.assertEqual(sent["type"], "image")
        self.assertEqual(sent["source"]["media_type"], "image/png")

    def test_a_pdf_passes_through(self):
        out = _messages_for_request([user(block("document", "application/pdf", PDF))])
        self.assertEqual(out[-1]["content"][0]["type"], "document")

    def test_audio_says_what_to_do_instead_of_dropping_it(self):
        with self.assertRaises(ValueError) as caught:
            _messages_for_request([user(block("audio", "audio/wav", WAV))])
        self.assertIn("transcribe", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
