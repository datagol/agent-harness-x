"""API keys entered in the browser, kept on the server and never sent back.

Keys normally come from the repository's ``.env`` (or the container's
environment). This adds a second way: paste a key into the app. It is saved in
the app's data directory (in Docker, a named volume on your machine, never the
image or the repository), applied to this process at once so the next example
run and the next conversation use it, and reported to the browser only as set
or not set.

A key saved here takes precedence over the environment; removing it restores
whatever the environment had when the server started.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# The keys the examples and the agent builder read, in the order they are shown.
KNOWN_KEYS: dict[str, str] = {
    "ANTHROPIC_API_KEY": "Anthropic",
    "OPENAI_API_KEY": "OpenAI",
    "GEMINI_API_KEY": "Gemini",
    "OPENROUTER_API_KEY": "OpenRouter",
    "LANGSMITH_API_KEY": "LangSmith",
    "TAVILY_API_KEY": "Tavily web search",
    "TYPESAFE_API_KEY": "Jev decisions",
}

_VALUE = re.compile(r"\S{8,4096}")


class KeyStore:
    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / "api-keys.json"
        # What the environment had before any key was saved here, to restore on removal.
        self._environment = {name: os.environ.get(name) for name in KNOWN_KEYS}
        self._saved: dict[str, str] = {}
        if self.path.is_file():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            self._saved = {k: v for k, v in loaded.items() if k in KNOWN_KEYS and isinstance(v, str)}
        for name, value in self._saved.items():
            os.environ[name] = value

    def records(self) -> list[dict[str, object]]:
        """Each key's state, without its value."""
        records = []
        for name, label in KNOWN_KEYS.items():
            if name in self._saved:
                source = "app"
            elif os.environ.get(name):
                source = "environment"
            else:
                source = None
            records.append({"name": name, "label": label, "set": source is not None, "source": source})
        return records

    def save(self, name: str, value: str) -> None:
        if name not in KNOWN_KEYS:
            raise KeyError(name)
        value = value.strip()
        if not _VALUE.fullmatch(value):
            raise ValueError("Paste the whole key: at least 8 characters, no spaces")
        self._saved[name] = value
        self._write()
        os.environ[name] = value

    def remove(self, name: str) -> None:
        if name not in KNOWN_KEYS:
            raise KeyError(name)
        self._saved.pop(name, None)
        self._write()
        original = self._environment.get(name)
        if original:
            os.environ[name] = original
        else:
            os.environ.pop(name, None)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staged = self.path.with_suffix(".tmp")
        # Readable by this user only, from the moment it exists.
        descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(self._saved, stream)
        os.replace(staged, self.path)
