"""Tool registry and execution engine.

Handles the two-sided contract: converting Python functions into JSON schemas
the LLM can understand, and dispatching LLM tool calls back to those functions.
"""

from __future__ import annotations

import asyncio
import inspect
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

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        permission: PermissionLevel = PermissionLevel.ASK,
    ) -> Callable:
        """Decorator to register a function as a tool.

        Auto-generates JSON Schema from type hints and docstring.
        """

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_description = description or (func.__doc__ or "").split("\n")[0].strip() or tool_name

            input_schema, _ = _generate_input_schema(func)

            self._tools[tool_name] = ToolDefinition(
                name=tool_name,
                description=tool_description,
                input_schema=input_schema,
                handler=func,
                permission_level=permission,
            )
            return func

        return decorator

    def register_with_schema(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable,
        permission: PermissionLevel = PermissionLevel.ASK,
    ) -> None:
        """Imperative registration with explicit schema."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            permission_level=permission,
        )

    def get_tool_params(self) -> list[dict[str, Any]]:
        """Convert all registered tools to Anthropic ToolParam format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

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
                tool_use_id=tool_call.id,
                content=str(e),
                is_error=True,
            )

        try:
            handler = tool_def.handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**tool_call.input)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: handler(**tool_call.input))

            return ToolResult(
                tool_use_id=tool_call.id,
                content=str(result),
            )
        except Exception:
            return ToolResult(
                tool_use_id=tool_call.id,
                content=f"Tool execution error:\n{traceback.format_exc()}",
                is_error=True,
            )
