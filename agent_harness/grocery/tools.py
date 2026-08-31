"""GroceryBuddy tools: the six functions the system prompt references.

Web semantics: `propose_items` only presents items (never mutates the
list — the user taps to add, and the client sends add_items back).
"""

from __future__ import annotations

import json

from agent_harness import PermissionLevel, StreamingAgent


def _coerce_items(raw) -> list[str]:
    """Accept a list of strings, or a JSON-encoded list that arrived as a string."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(i) for i in parsed]
        except (ValueError, TypeError):
            pass
        return [raw]
    if isinstance(raw, list):
        return [str(i) for i in raw]
    return []


def register_grocery_tools(agent: StreamingAgent, state: dict) -> None:
    """Register the six GroceryBuddy functions, closed over one session's state."""

    async def _propose_items(query: str = "", items=None, intro_text: str = "") -> str:
        if (query and items) or (not query and not items):
            return "Error: provide exactly one of `query` or `items`, never both or neither."
        resolved = [query] if query else _coerce_items(items)
        return f"Presented {len(resolved)} item(s) to the user."

    agent.tools.register_with_schema(
        name="propose_items",
        description=(
            "Turn-ender: propose items to add to the grocery list, via query (the user's own "
            "words, unchanged) or items (strings you wrote yourself). Exactly one of the two."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's own words, untouched. Use only when changing nothing at all.",
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Item strings you wrote yourself, one per line item.",
                },
                "intro_text": {
                    "type": "string",
                    "description": "One short, complete sentence said while handing the list over.",
                },
            },
        },
        handler=_propose_items,
        permission=PermissionLevel.ALLOW,
    )

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def present_choice(options: list, prompt: str, client_action: str = "") -> str:
        """Turn-ender: present a tappable choosing question with 2-4 options."""
        if not isinstance(options, list) or not (2 <= len(options) <= 4):
            return "Error: options must be a list of 2-4 entries."
        return f"Choice question presented: {prompt!r}"

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def respond_text(text: str, client_action: str = "") -> str:
        """Turn-ender: respond with plain text when there is nothing to choose or add."""
        return f"Response presented{': ' + client_action if client_action else ''}"

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def get_staples() -> str:
        """Fetch the user's most-bought items (their 'usuals'). Silent, never opens a screen."""
        if state["staples_status"] == "error":
            return "Error: staples unavailable right now"
        staples = state["staples"]
        return "No usuals saved yet." if not staples else f"Most-bought items: {', '.join(staples)}"

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def memory_write(key: str, value: str = "") -> str:
        """Save (lowercase dotted key + value) or forget (key with empty value) one fact."""
        if not value:
            existed = key in state["memory"]
            state["memory"].pop(key, None)
            return f"Forgot {key!r}" if existed else f"Nothing saved under {key!r}"
        state["memory"][key] = value
        return f"Saved {key!r}"

    @agent.tools.register(permission=PermissionLevel.ALLOW)
    async def fetch_recipe_url(url: str) -> str:
        """Fetch a pasted link and extract its ingredient list. Call first for any URL."""
        if state["fetch_mode"] == "mock_fail":
            return "Error: could not read that page"
        try:
            import re

            import httpx

            headers = {"User-Agent": "Mozilla/5.0 (compatible; GroceryBuddy/1.0)"}
            async with httpx.AsyncClient(follow_redirects=True, timeout=20.0, headers=headers) as client:
                resp = await client.get(url)
                html = resp.text

            schema_items = re.findall(
                r'"recipeIngredient"\s*:\s*\[(.*?)\]', html, flags=re.DOTALL | re.IGNORECASE
            )
            if schema_items:
                raw = re.findall(r'"((?:[^"\\]|\\.)*)"', schema_items[0])
                items = [s.replace('\\"', '"').replace("\\'", "'").strip() for s in raw if s.strip()]
                if items:
                    return "Ingredients on page: " + " | ".join(items[:50])

            soup = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", "\n", soup)
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            lowered = [ln.lower() for ln in lines]
            start = next((i for i, ln in enumerate(lowered) if "ingredient" in ln and len(ln) < 60), None)
            if start is not None:
                collected: list[str] = []
                for ln in lines[start + 1 :]:
                    low = ln.lower()
                    if any(k in low for k in ("instruction", "direction", "preparation", "method", "step")) and len(ln) < 60:
                        break
                    if len(ln) < 140 and not low.startswith(("copyright", "print", "share", "jump")):
                        collected.append(ln)
                    if len(collected) >= 40:
                        break
                if collected:
                    return "Ingredients on page: " + " | ".join(collected)

            return "Error: page loaded but no ingredient list was found on it."

        except Exception as exc:
            return f"Error: could not read that page ({exc})"
