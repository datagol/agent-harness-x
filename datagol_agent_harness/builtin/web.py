"""Built-in web tools: fetch URL content."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datagol_agent_harness.types import PermissionLevel

if TYPE_CHECKING:
    from datagol_agent_harness.tools import ToolRegistry


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
) -> list[str]:
    """Register web tools onto a registry.

    Args:
        registry: Target tool registry.
        permission: Override default permission level (default PermissionLevel.ASK).
        include: Specific tool names to register ('fetch_url').
        exclude: Tool names to omit.
        max_length: Maximum response characters cap.

    Returns:
        List of registered tool names.
    """
    if include is not None and "fetch_url" not in include:
        return []
    if exclude and "fetch_url" in exclude:
        return []

    perm = permission if permission is not None else PermissionLevel.ASK

    if max_length != 500_000:
        async def custom_fetch_url(url: str) -> str:
            """Fetch the content of a URL.

            Args:
                url: The URL to fetch.
            """
            return await fetch_url(url, max_length=max_length)

        registry.register_tool(custom_fetch_url, name="fetch_url", permission=perm)
    else:
        registry.register_tool(fetch_url, name="fetch_url", permission=perm)

    return ["fetch_url"]
