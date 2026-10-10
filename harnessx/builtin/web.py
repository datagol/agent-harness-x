"""Built-in web tools: fetch URL content."""

from __future__ import annotations

from typing import TYPE_CHECKING

from harnessx.types import PermissionLevel

from ._registration import select_tools

if TYPE_CHECKING:
    from harnessx.tools import ToolRegistry

# A slow site is worth waiting for longer than a local tool, not five minutes.
FETCH_TIMEOUT_SECONDS = 60.0


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


def register_web_tools(
    registry: ToolRegistry,
    *,
    permission: PermissionLevel | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    max_length: int = 500_000,
    replace: bool = False,
) -> list[str]:
    """Register web tools onto a registry.

    Args:
        registry: Target tool registry.
        permission: Explicit policy; None inherits the manager default (normally ALLOW).
        include: Specific tool names to register ('fetch_url').
        exclude: Tool names to omit.
        max_length: Maximum response characters cap.

    Returns:
        List of registered tool names.
    """
    if not select_tools(registry, ["fetch_url"], include, exclude, replace=replace):
        return []

    if max_length != 500_000:
        async def custom_fetch_url(url: str) -> str:
            """Fetch the content of a URL.

            Args:
                url: The URL to fetch.
            """
            return await fetch_url(url, max_length=max_length)

        registry.register_tool(custom_fetch_url, name="fetch_url", permission=permission, replace=replace,
                               timeout_seconds=FETCH_TIMEOUT_SECONDS)
    else:
        registry.register_tool(fetch_url, name="fetch_url", permission=permission, replace=replace,
                               timeout_seconds=FETCH_TIMEOUT_SECONDS)

    return ["fetch_url"]
