"""Send an image, a PDF or an audio clip alongside the text of a turn.

Content blocks replace the usual string: a `text` block for the question and an
`image`, `document` or `audio` block for the file. With no argument this uses a
1x1 PNG held in the script, so it runs without a file on disk.

Run: python examples/01-basics/media_input.py
Run: python examples/01-basics/media_input.py path/to/file.jpg
Needs: GEMINI_API_KEY
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import sys

from dotenv import load_dotenv

from harnessx import Agent
from harnessx.types import AgentConfig

load_dotenv()

#: Which block kind carries which media type.
KIND_BY_PREFIX = {"image/": "image", "audio/": "audio", "application/pdf": "document"}


def kind_for(media_type: str) -> str:
    for prefix, kind in KIND_BY_PREFIX.items():
        if media_type.startswith(prefix):
            return kind
    raise SystemExit(f"Nothing carries {media_type!r}: send an image, a PDF or audio.")


def a_small_png() -> tuple[bytes, str]:
    """A 1x1 red pixel, so the example needs no file on disk."""
    return (
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        ),
        "image/png",
    )


async def main() -> None:
    if len(sys.argv) > 1:
        path = sys.argv[1]
        data = open(path, "rb").read()
        media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        question = "Describe what this contains, briefly."
    else:
        data, media_type = a_small_png()
        question = "What colour is this image?"

    agent = Agent(
        config=AgentConfig(
            provider="gemini",
            model=os.environ.get("HARNESSX_MODEL", "gemini-3.6-flash"),
            system_prompt="Answer in one short line.",
        )
    )
    try:
        result = await agent.run([
            {"type": "text", "text": question},
            {
                "type": kind_for(media_type),
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(data).decode(),
                },
            },
        ])
        print(result.output)
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
