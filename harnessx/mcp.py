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

import asyncio
import json
import logging
import re
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence, overload

from .tools import ToolRegistry
from .errors import ConfigurationError, TransientToolError
from .providers.retry import is_transient_text
from .types import DEFAULT_TIMEOUT_SECONDS, PermissionLevel, ReplayPolicy, ToolResult, ToolRetry

# Bridged tools re-run a throttled or failed call up to three times, once the
# server is declared safe or idempotent (a manual tool is never retried).
DEFAULT_MCP_RETRY = ToolRetry(attempts=3, backoff_seconds=1.0, retry_error_results=True)

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
    """How to reach one MCP server: a stdio command or an HTTP URL, never both.

    Build one with ``MCPServerConfig.stdio(...)`` or ``MCPServerConfig.http(...)``
    and hand it to ``MCPManager.connect``. ``name`` is yours to choose; it keys
    the connection and prefixes every bridged tool (``files_read_file``).
    """

    name: str
    # stdio transport
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    # HTTP transport: streamable HTTP first, then SSE on the same URL
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    # common
    permission: PermissionLevel = PermissionLevel.ASK
    # What durable execution and retry may do with this server's tools when an
    # outcome is lost or transient: "manual" (stop and ask), "safe" (repeat
    # freely), or "idempotent" (the server deduplicates repeats).
    replay_policy: ReplayPolicy | str = "manual"
    # Retry for this server's tools; None means DEFAULT_MCP_RETRY. Only acts on a
    # safe or idempotent replay policy.
    retry: ToolRetry | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError("MCP server name must be a nonempty string")
        if bool(self.command) == bool(self.url):
            raise ConfigurationError(
                f"MCP server {self.name!r}: give exactly one of command (stdio) or url (http)"
            )
        self.args = list(self.args)
        self.headers = dict(self.headers)
        self.permission = PermissionLevel(self.permission)
        try:
            self.replay_policy = ReplayPolicy(self.replay_policy).value
        except ValueError:
            raise ConfigurationError(f"MCP server {self.name!r}: invalid replay policy") from None
        if self.retry is not None and not isinstance(self.retry, ToolRetry):
            raise TypeError("MCP server retry must be a ToolRetry")

    @classmethod
    def stdio(
        cls,
        name: str,
        command: str,
        *,
        args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        permission: PermissionLevel | str = PermissionLevel.ASK,
        replay_policy: ReplayPolicy | str = "manual",
        retry: ToolRetry | None = None,
    ) -> MCPServerConfig:
        """A server launched as a subprocess and spoken to over stdio."""
        return cls(
            name=name, command=command, args=list(args),
            env=dict(env) if env is not None else None, permission=PermissionLevel(permission),
            replay_policy=replay_policy, retry=retry,
        )

    @classmethod
    def http(
        cls,
        name: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        permission: PermissionLevel | str = PermissionLevel.ASK,
        replay_policy: ReplayPolicy | str = "manual",
        retry: ToolRetry | None = None,
    ) -> MCPServerConfig:
        """A server reached over HTTP: streamable HTTP first, falling back to SSE."""
        return cls(
            name=name, url=url, headers=dict(headers or {}), permission=PermissionLevel(permission),
            replay_policy=replay_policy, retry=retry,
        )

    @property
    def transport(self) -> Literal["stdio", "http"]:
        return "stdio" if self.command else "http"


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
        self._task: asyncio.Task[None] | None = None
        self._closing: asyncio.Event | None = None
        self._ready: asyncio.Future[list[MCPToolInfo]] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._tools: list[MCPToolInfo] = []
        self.transport: str | None = None  # negotiated: stdio, streamable-http, or sse

    async def connect(self) -> list[MCPToolInfo]:
        """Connect to the MCP server and discover its tools.

        The transports are anyio context managers, so their cancel scopes bind
        to the task that enters them and must be exited from that same task.
        Entering them here would tie the connection to whoever called
        ``connect()``: inside a web request handler that task ends when the
        response is sent, anyio raises "Attempted to exit a cancel scope that
        isn't the current task's current cancel scope", the transport unwinds,
        and the next tool call fails with "Connection closed".

        So a dedicated task owns the whole lifecycle instead. It opens the
        transports, hands the discovered tools back, and then waits. The
        connection lives until ``disconnect()``, whatever happens to the caller.
        """
        async with self._lifecycle_lock:
            if self._task is None or self._task.done():
                loop = asyncio.get_running_loop()
                self._tools = []
                self._ready = loop.create_future()
                self._ready.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
                self._closing = asyncio.Event()
                self._task = loop.create_task(self._lifecycle(self._ready))
            ready = self._ready
        assert ready is not None
        # One cancelled request must not cancel discovery for other callers.
        return list(await asyncio.shield(ready))

    async def _lifecycle(self, ready: "asyncio.Future[list[MCPToolInfo]]") -> None:
        """Own the transports from open to close, in one task."""
        try:
            async with AsyncExitStack() as stack:
                tools = await self._open(stack)
                if ready.done():  # the caller was cancelled while we connected
                    return
                ready.set_result(tools)
                await self._closing.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(RuntimeError("MCP connection closed during initialization")
                                    if isinstance(exc, asyncio.CancelledError) else exc)
            elif not isinstance(exc, asyncio.CancelledError):
                logger.warning(
                    "MCP '%s': connection ended: %s", self.config.name, exc
                )
        finally:
            self.session = None

    async def _open(self, stack: AsyncExitStack) -> list[MCPToolInfo]:
        """Enter the transport contexts and discover tools.

        Runs inside ``_lifecycle``'s task, never the caller's, because the
        transports are anyio context managers whose cancel scopes belong to
        whichever task entered them.
        """
        try:
            from mcp import ClientSession, StdioServerParameters
        except ImportError:
            raise ImportError(
                "MCP support requires the 'mcp' package. "
                "Install it with: pip install 'mcp[cli]'"
            )

        if self.config.command:
            # stdio transport
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env,
            )
            read, write = await stack.enter_async_context(
                stdio_client(server_params)
            )
            self.transport = "stdio"
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

                result = await stack.enter_async_context(
                    streamable_http_client(self.config.url, **client_kwargs)
                )
                read, write = result[0], result[1]
                connected = True
                self.transport = "streamable-http"
                logger.info(f"MCP '{self.config.name}': connected via streamable HTTP")
            except Exception as e:
                logger.info(f"MCP '{self.config.name}': streamable HTTP failed ({e}), trying SSE")

            if not connected:
                try:
                    from mcp.client.sse import sse_client

                    read, write = await stack.enter_async_context(
                        sse_client(self.config.url, headers=self.config.headers)
                    )
                    self.transport = "sse"
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

        self.session = await stack.enter_async_context(
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
            if is_transient_text(text):
                raise TransientToolError(f"MCP tool error: {text}")
            raise RuntimeError(f"MCP tool error: {text}")

        return text

    async def disconnect(self) -> None:
        """Disconnect from the MCP server.

        Signals the owning task rather than exiting the stack here, so the
        contexts unwind in the task that entered them.
        """
        async with self._lifecycle_lock:
            task = self._task
            if self._closing is not None:
                self._closing.set()
            if task is not None and not task.done():
                if self._ready is not None and not self._ready.done():
                    task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=10)
                except (TimeoutError, asyncio.TimeoutError):
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task
                except asyncio.CancelledError:
                    if not task.cancelled():
                        raise
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(RuntimeError("MCP connection closed during initialization"))
            self._task = None
            self._closing = None
            self.session = None
            self._tools = []

    @property
    def tools(self) -> list[MCPToolInfo]:
        return list(self._tools)

    @property
    def is_connected(self) -> bool:
        return self.session is not None


def mcp_result_transient(result: ToolResult) -> bool:
    """A successful MCP result that carries a throttling or server failure in its body.

    Servers wrap upstream HTTP errors as ordinary text: a JSON object with an
    ``error`` and a ``status``/``code`` of 429 or 5xx, or a short message that
    reads as a rate limit. Only short bodies are inspected; real content is not.
    """
    text = result.content if isinstance(result.content, str) else ""
    if not text or len(text) > 2000:
        return False
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            body = json.loads(stripped)
        except ValueError:
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            if error is None and "status" not in body and "code" not in body:
                return False
            for key in ("status", "status_code", "code"):
                for holder in (body, error if isinstance(error, dict) else {}):
                    value = holder.get(key)
                    if isinstance(value, int) and (value == 429 or value >= 500):
                        return True
            message = error if isinstance(error, str) else json.dumps(error) if error is not None else ""
            return is_transient_text(message)
    return result.is_error and is_transient_text(stripped)


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

    @overload
    async def connect(self, name: MCPServerConfig, /) -> list[MCPToolInfo]: ...

    @overload
    async def connect(
        self,
        name: str,
        *,
        command: str | None = ...,
        args: list[str] | None = ...,
        env: dict[str, str] | None = ...,
        url: str | None = ...,
        headers: dict[str, str] | None = ...,
        permission: PermissionLevel = ...,
    ) -> list[MCPToolInfo]: ...

    async def connect(
        self,
        name: str | MCPServerConfig,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        permission: PermissionLevel = PermissionLevel.ASK,
    ) -> list[MCPToolInfo]:
        """Connect to an MCP server and discover its tools.

        Pass an ``MCPServerConfig`` (``MCPServerConfig.stdio(...)`` or
        ``MCPServerConfig.http(...)``), or a name with ``command=`` (stdio) or
        ``url=`` (HTTP) keywords.
        """
        if isinstance(name, MCPServerConfig):
            given = [k for k, v in (("command", command), ("args", args), ("env", env), ("url", url), ("headers", headers)) if v is not None]
            if given or permission is not PermissionLevel.ASK:
                raise TypeError(
                    f"connect(MCPServerConfig) takes no keyword arguments ({', '.join(given) or 'permission'} given); "
                    "put them on the config"
                )
            config = name
        else:
            config = MCPServerConfig(
                name=name, command=command, args=args or [], env=env, url=url,
                headers=headers or {}, permission=permission,
            )
        name = config.name
        if name in self._connections:
            raise ValueError(f"MCP server {name!r} is already connected; disconnect it first")

        conn = MCPConnection(config)
        tools = await conn.connect()

        self._connections[name] = conn

        # Map registry names to their server so lookups match register_tools().
        for tool in tools:
            self._tool_to_server[_registered_name(name, tool.tool_name)] = name

        return tools

    async def connect_all(
        self, servers: Sequence[MCPServerConfig], *, concurrent: bool = True,
    ) -> dict[str, list[MCPToolInfo]]:
        """Connect several servers and return the discovered tools by server name.

        Servers connect concurrently unless ``concurrent=False``. If any fail,
        the ones that connected stay connected and an ``ExceptionGroup`` names
        the failures, so a caller can report them or disconnect and retry.
        """
        configs = list(servers)
        for config in configs:
            if not isinstance(config, MCPServerConfig):
                raise TypeError("connect_all takes MCPServerConfig entries; use connect() for the keyword form")
            if config.name in self._connections:
                raise ValueError(f"MCP server {config.name!r} is already connected; disconnect it first")
        names = [config.name for config in configs]
        if len(set(names)) != len(names):
            raise ValueError("MCP server names must be unique within one connect_all call")
        if concurrent:
            outcomes = await asyncio.gather(*(self.connect(config) for config in configs), return_exceptions=True)
        else:
            outcomes = []
            for config in configs:
                try:
                    outcomes.append(await self.connect(config))
                except Exception as exc:  # keep going; the group below reports every failure
                    outcomes.append(exc)
        tools: dict[str, list[MCPToolInfo]] = {}
        failures: list[Exception] = []
        for config, outcome in zip(configs, outcomes):
            if isinstance(outcome, BaseException):
                failures.append(outcome if isinstance(outcome, Exception) else RuntimeError(str(outcome)))
                failures[-1].add_note(f"MCP server {config.name!r}")
            else:
                tools[config.name] = outcome
        if failures:
            connected = ", ".join(tools) or "none"
            raise ExceptionGroup(
                f"{len(failures)} of {len(configs)} MCP servers failed to connect (connected: {connected})", failures,
            )
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
                    replay_policy=conn.config.replay_policy,
                    retry=conn.config.retry or DEFAULT_MCP_RETRY,
                    retry_if_result=mcp_result_transient,
                    # A remote server, often doing real work (search, queries):
                    # not held to the default for local tools.
                    timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
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
                "transport": conn.transport or conn.config.transport,
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
