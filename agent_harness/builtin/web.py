"""Built-in web tools: fetch URL content."""

from __future__ import annotations

from agent_harness.tools import ToolRegistry
from agent_harness.types import PermissionLevel


def register_web_tools(registry: ToolRegistry) -> None:
    """Register web tools onto a registry."""

    @registry.register(permission=PermissionLevel.ASK)
    async def fetch_url(url: str, max_length: int = 500_000) -> str:
        """Fetch the content of a URL.

        Args:
            url: The URL to fetch.
            max_length: Maximum characters to return (default 500000).
        """
        import httpx

        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            text = response.text

            if len(text) > max_length:
                text = text[:max_length] + f"\n\n[truncated at {max_length} chars]"

            return f"Status: {response.status_code}\nContent-Type: {content_type}\n\n{text}"
