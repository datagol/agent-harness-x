"""Tool registry and execution engine.

Handles the two-sided contract: converting Python functions into JSON schemas
the LLM can understand, and dispatching LLM tool calls back to those functions.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import traceback
from typing import Any, Callable, get_type_hints

from .types import PermissionLevel, ToolCall, ToolDefinition, ToolResult


# Python type → JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _extract_docstring_description(docstring: str | None) -> str:
    """Extract complete description from a docstring before Args:, Returns:, etc."""
    if not docstring:
        return ""
    desc_lines: list[str] = []
    for line in docstring.splitlines():
        stripped = line.strip()
        if re.match(r"^(args|parameters|returns|raises|yields|examples?|note|notes):", stripped, re.IGNORECASE):
            break
        desc_lines.append(line)
    return "\n".join(desc_lines).strip()


def _parse_docstring_args(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from an Args: section in a docstring."""
    if not docstring:
        return {}

    descriptions: dict[str, str] = {}
    in_args = False
    current_param: str | None = None
    current_desc_lines: list[str] = []

    for line in docstring.splitlines():
        stripped = line.strip()

        if stripped.lower().startswith("args:"):
            in_args = True
            continue

        if in_args:
            # A new section header ends Args
            if stripped and not stripped.startswith("-") and ":" not in stripped and current_param is None:
                break
            if re.match(r"^(returns|raises|yields|examples|note):", stripped, re.IGNORECASE):
                break

            # Try to match "param_name: description" or "param_name (type): description"
            param_match = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
            if param_match:
                # Save previous param
                if current_param:
                    descriptions[current_param] = " ".join(current_desc_lines).strip()
                current_param = param_match.group(1)
                current_desc_lines = [param_match.group(2)] if param_match.group(2) else []
            elif current_param and stripped:
                current_desc_lines.append(stripped)
            elif not stripped and current_param:
                # Blank line might end the param description
                pass

    if current_param:
        descriptions[current_param] = " ".join(current_desc_lines).strip()

    return descriptions


def _get_json_schema_type(python_type: Any) -> dict[str, Any]:
    """Convert a Python type hint to a JSON Schema type object."""
    # Handle None / NoneType
    if python_type is type(None):
        return {"type": "string"}

    # Direct mapping
    if python_type in _TYPE_MAP:
        return {"type": _TYPE_MAP[python_type]}

    # Handle list[X]
    origin = getattr(python_type, "__origin__", None)
    if origin is list:
        args = getattr(python_type, "__args__", ())
        if args:
            return {"type": "array", "items": _get_json_schema_type(args[0])}
        return {"type": "array"}

    if origin is dict:
        return {"type": "object"}

    # Fallback
    return {"type": "string"}


def _is_optional(python_type: Any) -> tuple[bool, Any]:
    """Check if a type is Optional[X] (Union[X, None]). Returns (is_optional, inner_type)."""
    origin = getattr(python_type, "__origin__", None)
    if origin is not None:
        import types as _types

        if origin is _types.UnionType or (hasattr(origin, "__name__") and origin.__name__ == "Union"):
            args = python_type.__args__
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1 and len(args) == 2:
                return True, non_none[0]
    # Also check X | None syntax (Python 3.10+)
    if hasattr(python_type, "__args__") and hasattr(python_type, "__origin__"):
        pass  # Already handled above
    return False, python_type


def _generate_input_schema(func: Callable) -> tuple[dict[str, Any], list[str]]:
    """Generate JSON Schema input_schema from function signature and type hints.

    Returns (schema_dict, required_params_list).
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    doc_descriptions = _parse_docstring_args(func.__doc__)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        hint = hints.get(name, str)
        is_opt, inner_type = _is_optional(hint)

        prop = _get_json_schema_type(inner_type if is_opt else hint)

        if name in doc_descriptions:
            prop["description"] = doc_descriptions[name]

        properties[name] = prop

        # Required if no default and not Optional
        if param.default is inspect.Parameter.empty and not is_opt:
            required.append(name)

    schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema, required


class ToolNotFoundError(Exception):
    pass


class ToolRegistry:
    """Registry for tools. Handles registration, schema generation, and execution."""

    def __init__(
        self,
        *,
        default_timeout_seconds: float | None = None,
        dedupe_calls: bool = False,
    ) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        # Applied to any tool that does not set its own timeout_seconds.
        self._default_timeout_seconds = default_timeout_seconds
        # When on, an identical repeated call (same name AND same arguments)
        # returns the first result instead of running again. A model that
        # re-emits a call would otherwise do the work twice - two identical
        # side effects and double the cost. Off by default: a caller whose
        # tools are meant to be called repeatedly with the same arguments
        # must opt in, not be surprised.
        self._dedupe_calls = dedupe_calls
        self._call_results: dict[str, ToolResult] = {}

    def reset_call_cache(self) -> None:
        """Forget deduped results. Call between turns; results are per-turn."""
        self._call_results.clear()

    @staticmethod
    def _call_key(tool_call: ToolCall) -> str | None:
        try:
            return f"{tool_call.name}:{json.dumps(tool_call.input, sort_keys=True, default=str)}"
        except Exception:
            return None  # unserialisable arguments: never dedupe

    def register(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        permission: PermissionLevel = PermissionLevel.ASK,
        concurrent: bool = True,
    ) -> Callable:
        """Decorator to register a function as a tool.

        Auto-generates JSON Schema from type hints and docstring.
        """

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_description = description or _extract_docstring_description(func.__doc__) or tool_name

            input_schema, _ = _generate_input_schema(func)

            self._tools[tool_name] = ToolDefinition(
                name=tool_name,
                description=tool_description,
                input_schema=input_schema,
                handler=func,
                permission_level=permission,
                concurrent=concurrent,
            )
            return func

        return decorator

    def register_tool(
        self,
        tool: Callable | ToolDefinition,
        *,
        name: str | None = None,
        description: str | None = None,
        permission: PermissionLevel | None = None,
        concurrent: bool = True,
    ) -> ToolDefinition:
        """Register a function or ToolDefinition directly (non-decorator style)."""
        if isinstance(tool, ToolDefinition):
            if permission is not None:
                tool.permission_level = permission
            self._tools[tool.name] = tool
            return tool

        decorator = self.register(
            name=name,
            description=description,
            permission=permission if permission is not None else PermissionLevel.ASK,
            concurrent=concurrent,
        )
        decorator(tool)
        return self._tools[name or getattr(tool, "__name__", str(tool))]

    def register_with_schema(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable,
        permission: PermissionLevel = PermissionLevel.ASK,
        concurrent: bool = True,
        timeout_seconds: float | None = None,
    ) -> None:
        """Imperative registration with explicit schema."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            permission_level=permission,
            concurrent=concurrent,
            timeout_seconds=timeout_seconds,
        )

    def load_builtin(
        self,
        bundle_or_tool: str,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        permission: PermissionLevel | None = None,
        **options: Any,
    ) -> list[str]:
        """Load built-in tools into the registry by bundle name or tool name.

        Args:
            bundle_or_tool: 'filesystem', 'bash', 'web', 'memory', 'all', or a tool name like 'read_file'.
            include: Specific tool names to include.
            exclude: Tool names to skip.
            permission: Override default permission level for loaded tools.
            options: Additional bundle-specific options (e.g. sandbox, base_path).

        Returns:
            List of registered tool names.
        """
        from .builtin import (
            fetch_url,
            generate_file,
            list_directory,
            read_file,
            recall_memories,
            register_all_tools,
            register_bash_tools,
            register_filesystem_tools,
            register_memory_tools,
            register_web_tools,
            run_bash,
            save_memory,
            write_file,
        )

        name = bundle_or_tool.lower()
        if name in ("all", "builtin", "builtins"):
            return register_all_tools(self, include=include, exclude=exclude, permission=permission, **options)
        elif name in ("filesystem", "fs", "files"):
            return register_filesystem_tools(self, include=include, exclude=exclude, permission=permission, **options)
        elif name in ("bash", "shell", "terminal"):
            return register_bash_tools(self, include=include, exclude=exclude, permission=permission, **options)
        elif name in ("web", "http", "fetch"):
            return register_web_tools(self, include=include, exclude=exclude, permission=permission, **options)
        elif name in ("memory",):
            return register_memory_tools(self, include=include, exclude=exclude, permission=permission, **options)

        standalone_map = {
            "read_file": (read_file, PermissionLevel.ALLOW),
            "write_file": (write_file, PermissionLevel.ASK),
            "list_directory": (list_directory, PermissionLevel.ALLOW),
            "generate_file": (generate_file, PermissionLevel.ALLOW),
            "run_bash": (run_bash, PermissionLevel.ASK),
            "fetch_url": (fetch_url, PermissionLevel.ASK),
            "save_memory": (save_memory, PermissionLevel.ALLOW),
            "recall_memories": (recall_memories, PermissionLevel.ALLOW),
        }
        if name in standalone_map:
            fn, default_perm = standalone_map[name]
            self.register_tool(fn, permission=permission or default_perm)
            return [name]

        raise ValueError(
            f"Unknown built-in tool or bundle: {bundle_or_tool!r}. "
            f"Available bundles: 'filesystem', 'bash', 'web', 'memory', 'all'."
        )

    def get_tool_params(self) -> list[dict[str, Any]]:
        """Convert all registered tools to canonical ToolParam format."""
        params: list[dict[str, Any]] = []
        for t in self._tools.values():
            schema = t.input_schema if isinstance(t.input_schema, dict) else {}
            if not schema.get("type"):
                schema = {**schema, "type": "object"}
            if "properties" not in schema:
                schema = {**schema, "properties": {}}
            params.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": schema,
                }
            )
        return params

    def get_tools(self) -> list[ToolDefinition]:
        """Return all registered ToolDefinition objects."""
        return list(self._tools.values())

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry. Returns True if removed."""
        return self._tools.pop(name, None) is not None

    def get_tool(self, name: str) -> ToolDefinition:
        """Lookup a tool by name."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call. Handles sync and async handlers.

        Tool errors never crash the agent — they become ToolResult(is_error=True).
        """
        try:
            tool_def = self.get_tool(tool_call.name)
        except ToolNotFoundError as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=str(e),
                is_error=True,
            )

        cache_key = self._call_key(tool_call) if self._dedupe_calls else None
        if cache_key is not None and cache_key in self._call_results:
            cached = self._call_results[cache_key]
            # Same result, this call's own id.
            return ToolResult(
                tool_call_id=tool_call.id,
                content=cached.content,
                is_error=cached.is_error,
            )

        timeout = tool_def.timeout_seconds
        if timeout is None:
            timeout = self._default_timeout_seconds

        try:
            handler = tool_def.handler
            args = tool_call.input if isinstance(tool_call.input, dict) else {}
            if inspect.iscoroutinefunction(handler):
                call = handler(**args)
            else:
                loop = asyncio.get_running_loop()
                call = loop.run_in_executor(None, lambda: handler(**args))
            if timeout is not None:
                result = await asyncio.wait_for(call, timeout)
            else:
                result = await call

            outcome = ToolResult(
                tool_call_id=tool_call.id,
                content=str(result),
            )
        except asyncio.TimeoutError:
            # The model needs to know this tool is unavailable so it can move
            # on; a turn that hangs on one slow tool helps nobody.
            outcome = ToolResult(
                tool_call_id=tool_call.id,
                content=f"Tool '{tool_call.name}' timed out after {timeout}s.",
                is_error=True,
            )
        except Exception:
            outcome = ToolResult(
                tool_call_id=tool_call.id,
                content=f"Tool execution error:\n{traceback.format_exc()}",
                is_error=True,
            )

        if cache_key is not None:
            self._call_results[cache_key] = outcome
        return outcome


def normalize_tool_registry(tools: ToolRegistry | list[Any] | None) -> ToolRegistry:
    """Normalize tools argument into a ToolRegistry.

    Supports:
    - None -> empty ToolRegistry
    - ToolRegistry -> passed through
    - list of strings (bundles or tool names), callables, or ToolDefinitions
    """
    if isinstance(tools, ToolRegistry):
        return tools
    registry = ToolRegistry()
    if tools is None:
        return registry
    if isinstance(tools, (list, tuple, set)):
        for item in tools:
            if isinstance(item, str):
                registry.load_builtin(item)
            elif isinstance(item, ToolDefinition):
                registry.register_tool(item)
            elif callable(item):
                registry.register_tool(item)
            else:
                raise ValueError(f"Unsupported tool item in list: {item!r}")
        return registry
    raise TypeError(f"Expected ToolRegistry or list of tools, got {type(tools).__name__}")
