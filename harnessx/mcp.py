"""MCP (Model Context Protocol) integration.

Connects to MCP servers, discovers their tools, and registers them into
the existing ToolRegistry so the LLM uses MCP tools exactly like native tools.

Supports two transports:
  - stdio: launches server as a subprocess (local tools)
  - sse: connects to a remote HTTP server via Server-Sent Events

Usage:
    async with MCPManager() as mcp:
        await mcp.connect("filesystem", command="npx", args=[...])
        await mcp.connect("remote", url="http://localhost:8000/sse")
        agent = Agent(mcp=mcp)  # bridges the discovered tools into agent.tools
        ...
    # leaving the block disconnects every server
"""

from __future__ import annotations

import logging
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from .tools import ToolRegistry
from .types import PermissionLevel

logger = logging.getLogger(__name__)


def _sanitize_tool_name(name: str) -> str:
    """Sanitize tool name to match Anthropic's pattern: ^[a-zA-Z0-9_-]{1,128}$"""
    # Replace any character not in [a-zA-Z0-9_-] with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_-')
    # Truncate to 128 chars
    return sanitized[:128]


def _registered_name(server_name: str, tool_name: str, *, prefix: bool = True) -> str:
    """The registry name of an MCP tool: sanitized, server-prefixed unless already so."""
    san_server = _sanitize_tool_name(server_name)
    san_tool = _sanitize_tool_name(tool_name)
    if prefix and not san_tool.startswith(san_server):
        return _sanitize_tool_name(f"{san_server}_{san_tool}")
    return san_tool


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server."""

    name: str
    # stdio transport
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    # sse transport
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    # common
    permission: PermissionLevel = PermissionLevel.ASK


@dataclass
class MCPToolInfo:
    """Metadata about a tool discovered from an MCP server."""

    server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]


class MCPConnection:
    """A live connection to a single MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.session: Any = None  # mcp.ClientSession
        self._exit_stack: AsyncExitStack | None = None
        self._tools: list[MCPToolInfo] = []

    async def connect(self) -> list[MCPToolInfo]:
        """Connect to the MCP server and discover its tools."""
        try:
            from mcp import ClientSession, StdioServerParameters
        except ImportError:
            raise ImportError(
                "MCP support requires the 'mcp' package. "
                "Install it with: pip install 'mcp[cli]'"
            )

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        if self.config.command:
            # stdio transport
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env,
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
        elif self.config.url:
            # Try streamable HTTP first (modern transport), fall back to SSE
            connected = False
            try:
                try:
                    from mcp.client.streamable_http import streamable_http_client
                except ImportError:
                    from mcp.client.streamable_http import streamablehttp_client as streamable_http_client

                client_kwargs: dict[str, Any] = {}
                if self.config.headers:
                    try:
                        from mcp.client.streamable_http import httpx2
                        client_kwargs["http_client"] = httpx2.AsyncClient(headers=self.config.headers)
                    except Exception:
                        import httpx
                        client_kwargs["http_client"] = httpx.AsyncClient(headers=self.config.headers)

                result = await self._exit_stack.enter_async_context(
                    streamable_http_client(self.config.url, **client_kwargs)
                )
                read, write = result[0], result[1]
                connected = True
                logger.info(f"MCP '{self.config.name}': connected via streamable HTTP")
            except Exception as e:
                logger.info(f"MCP '{self.config.name}': streamable HTTP failed ({e}), trying SSE")

            if not connected:
                try:
                    from mcp.client.sse import sse_client

                    read, write = await self._exit_stack.enter_async_context(
                        sse_client(self.config.url, headers=self.config.headers)
                    )
                    logger.info(f"MCP '{self.config.name}': connected via SSE")
                except Exception as e:
                    raise ConnectionError(
                        f"Failed to connect to MCP server '{self.config.name}' at {self.config.url}. "
                        f"Tried streamable HTTP and SSE transports. Last error: {e}"
                    )
        else:
            raise ValueError(
                f"MCP server '{self.config.name}': must specify either 'command' (stdio) or 'url' (sse)"
            )

        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()

        # Discover tools
        tools_result = await self.session.list_tools()
        self._tools = []
        for tool in tools_result.tools:
            raw_schema = (
                getattr(tool, "input_schema", None)
                or getattr(tool, "inputSchema", None)
                or {}
            )
            if not isinstance(raw_schema, dict):
                raw_schema = dict(raw_schema) if hasattr(raw_schema, "__dict__") else {}
            if not raw_schema.get("type"):
                raw_schema = {**raw_schema, "type": "object"}
            if "properties" not in raw_schema:
                raw_schema = {**raw_schema, "properties": {}}

            self._tools.append(MCPToolInfo(
                server_name=self.config.name,
                tool_name=tool.name,
                description=tool.description or tool.name,
                input_schema=raw_schema,
            ))

        logger.info(
            f"MCP '{self.config.name}': connected, {len(self._tools)} tools discovered"
        )
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on this MCP server and return the text result."""
        if self.session is None:
            raise RuntimeError(f"MCP server '{self.config.name}' not connected")

        result = await self.session.call_tool(name=tool_name, arguments=arguments)

        # Extract text from result content blocks
        parts: list[str] = []
        if hasattr(result, "content"):
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif hasattr(block, "data"):
                    parts.append(str(block.data))
                else:
                    parts.append(str(block))

        text = "\n".join(parts) if parts else str(result)

        # Safety cap: prevent unbounded memory from MCP results
        MAX_MCP_RESULT_CHARS = 1_000_000
        if len(text) > MAX_MCP_RESULT_CHARS:
            text = text[:MAX_MCP_RESULT_CHARS] + f"\n\n[truncated MCP result at {MAX_MCP_RESULT_CHARS} chars]"

        # Check for errors
        is_error = getattr(result, "isError", False)
        if is_error:
            raise RuntimeError(f"MCP tool error: {text}")

        return text

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._exit_stack:
            try:
                await self._exit_stack.__aexit__(None, None, None)
            except Exception:
                pass
            self._exit_stack = None
            self.session = None
            self._tools = []

    @property
    def tools(self) -> list[MCPToolInfo]:
        return list(self._tools)

    @property
    def is_connected(self) -> bool:
        return self.session is not None


class MCPManager:
    """Manages connections to multiple MCP servers and bridges their tools
    into an agent's ToolRegistry.

    Usage:
        async with MCPManager() as mcp:
            # Connect to servers first; discovery happens here.
            await mcp.connect("files", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
            await mcp.connect("api", url="http://localhost:8000/sse")

            # Agent(mcp=...) bridges every discovered tool into agent.tools and
            # lists the registered names on agent.mcp_tools.
            async with Agent(mcp=mcp) as agent:
                ...  # the model uses MCP tools like native ones
        # leaving the block disconnects every server

    ``register_tools(registry)`` remains available for a registry that is not
    handed to an Agent; it is idempotent for the same manager. The manager is
    caller-owned: closing an agent never disconnects it.
    """

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._tool_to_server: dict[str, str] = {}  # registry tool name -> server name

    async def __aenter__(self) -> "MCPManager":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.disconnect_all()

    async def connect(
        self,
        name: str,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        permission: PermissionLevel = PermissionLevel.ASK,
    ) -> list[MCPToolInfo]:
        """Connect to an MCP server and discover its tools.

        For stdio transport: provide command (and optionally args, env).
        For SSE transport: provide url (and optionally headers).
        """
        if name in self._connections:
            raise ValueError(f"MCP server {name!r} is already connected; disconnect it first")
        config = MCPServerConfig(
            name=name,
            command=command,
            args=args or [],
            env=env,
            url=url,
            headers=headers or {},
            permission=permission,
        )

        conn = MCPConnection(config)
        tools = await conn.connect()

        self._connections[name] = conn

        # Map registry names to their server so lookups match register_tools().
        for tool in tools:
            self._tool_to_server[_registered_name(name, tool.tool_name)] = name

        return tools

    async def connect_from_config(
        self, servers: dict[str, dict[str, Any]]
    ) -> dict[str, list[MCPToolInfo]]:
        """Connect to multiple MCP servers from a config dict.

        Config format:
            {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                },
                "remote": {
                    "url": "http://localhost:8000/sse",
                    "headers": {"Authorization": "Bearer token"},
                },
            }
        """
        results: dict[str, list[MCPToolInfo]] = {}
        for name, config in servers.items():
            tools = await self.connect(
                name,
                command=config.get("command"),
                args=config.get("args", []),
                env=config.get("env"),
                url=config.get("url"),
                headers=config.get("headers", {}),
                permission=PermissionLevel(config.get("permission", "ask")),
            )
            results[name] = tools
        return results

    def register_tools(
        self,
        registry: ToolRegistry,
        *,
        prefix: bool = True,
        permission: PermissionLevel | None = None,
    ) -> list[str]:
        """Register all discovered MCP tools into a ToolRegistry.

        Args:
            registry: The tool registry to add tools to.
            prefix: If True, tool names are prefixed with server name (e.g., 'filesystem__read_file').
                    If False, uses raw tool names (may collide across servers).
            permission: Override permission level for all MCP tools. If None, uses per-server config.

        Returns:
            The MCP tool names present in the registry after the call. Calling
            this twice with the same manager is a no-op for tools it already
            bridged; a name held by anything else raises ValueError.
        """
        registered: list[str] = []

        for server_name, conn in self._connections.items():
            for tool_info in conn.tools:
                tool_name = _registered_name(server_name, tool_info.tool_name, prefix=prefix)
                perm = permission or conn.config.permission
                if registry.has_tool(tool_name):
                    existing = registry.get_tool(tool_name).handler
                    if (
                        getattr(existing, "__mcp_tool__", None) == (server_name, tool_info.tool_name)
                        and getattr(existing, "__mcp_manager__", None) is self
                    ):
                        registered.append(tool_name)
                        continue
                    raise ValueError(
                        f"Tool {tool_name!r} is already registered by something other than "
                        f"MCP server {server_name!r} on this manager"
                    )

                # Create a handler that routes to the MCP server
                handler = self._make_mcp_handler(server_name, tool_info.tool_name)

                registry.register_with_schema(
                    name=tool_name,
                    description=f"[MCP:{server_name}] {tool_info.description}",
                    input_schema=tool_info.input_schema,
                    handler=handler,
                    permission=perm,
                )

                self._tool_to_server[tool_name] = server_name
                registered.append(tool_name)

        return registered

    def _make_mcp_handler(self, server_name: str, tool_name: str):
        """Create an async handler that routes a tool call to the MCP server."""
        manager = self

        async def handler(**kwargs: Any) -> str:
            conn = manager._connections.get(server_name)
            if conn is None or not conn.is_connected:
                raise RuntimeError(f"MCP server '{server_name}' is not connected")
            return await conn.call_tool(tool_name, kwargs)

        handler.__name__ = f"mcp_{server_name}_{tool_name}"
        handler.__doc__ = f"MCP tool '{tool_name}' on server '{server_name}'"
        handler.__mcp_tool__ = (server_name, tool_name)  # type: ignore[attr-defined]
        handler.__mcp_manager__ = manager  # type: ignore[attr-defined]
        return handler

    async def disconnect(self, name: str) -> None:
        """Disconnect from a specific MCP server."""
        conn = self._connections.pop(name, None)
        if conn:
            await conn.disconnect()
            # Remove tool mappings
            self._tool_to_server = {
                k: v for k, v in self._tool_to_server.items() if v != name
            }

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for conn in self._connections.values():
            try:
                await conn.disconnect()
            except Exception:
                pass
        self._connections.clear()
        self._tool_to_server.clear()

    def list_servers(self) -> dict[str, dict[str, Any]]:
        """List all connected servers and their tools."""
        result: dict[str, dict[str, Any]] = {}
        for name, conn in self._connections.items():
            result[name] = {
                "connected": conn.is_connected,
                "transport": "stdio" if conn.config.command else "sse",
                "tools": [t.tool_name for t in conn.tools],
            }
        return result

    def list_tools(self) -> list[MCPToolInfo]:
        """List all tools across all connected servers."""
        all_tools: list[MCPToolInfo] = []
        for conn in self._connections.values():
            all_tools.extend(conn.tools)
        return all_tools

    @property
    def server_count(self) -> int:
        return len(self._connections)

    @property
    def tool_count(self) -> int:
        return sum(len(conn.tools) for conn in self._connections.values())
