"""Tool registry and execution engine.

Handles the two-sided contract: converting Python functions into JSON schemas
the LLM can understand, and dispatching LLM tool calls back to those functions.
"""

from __future__ import annotations

from .errors import HarnessError, TransientToolError

import asyncio
import inspect
import json
import re
import logging
import traceback
import types
from copy import deepcopy
from dataclasses import replace as replace_definition
from typing import Annotated, Any, Callable, Literal, Union, get_args, get_origin, get_type_hints

from jsonschema import Draft202012Validator, validate

from .permissions import PermissionManager
from .types import DEFAULT_TIMEOUT_SECONDS, PermissionLevel, ReplayPolicy, ToolCall, ToolDefinition, ToolPolicy, ToolResult, ToolRetry
from .providers.retry import is_transient, is_transient_text


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
    if python_type is Any:
        return {}
    if python_type is type(None):
        return {"type": "null"}
    if isinstance(python_type, type) and python_type in _TYPE_MAP:
        return {"type": _TYPE_MAP[python_type]}
    origin, args = get_origin(python_type), get_args(python_type)
    if origin is Annotated:
        return _get_json_schema_type(args[0])
    if origin in (Union, types.UnionType):
        return {"anyOf": [_get_json_schema_type(arg) for arg in args]}
    if origin is Literal:
        if not args or any(type(value) not in (str, int, float, bool, type(None)) for value in args):
            raise TypeError("Tool Literal values must be JSON primitives")
        json.dumps(args, allow_nan=False)
        value_types = list(dict.fromkeys(type(value) for value in args))
        choices = [
            {**_get_json_schema_type(kind), "enum": [value for value in args if type(value) is kind]}
            for kind in value_types
        ]
        return choices[0] if len(choices) == 1 else {"anyOf": choices}
    if python_type is list or origin is list:
        return {"type": "array", "items": _get_json_schema_type(args[0]) if args else {}}
    if python_type is dict or origin is dict:
        if args and args[0] not in (str, Any):
            raise TypeError("Tool dictionary keys must be strings")
        return {"type": "object", "additionalProperties": _get_json_schema_type(args[1]) if args else {}}
    raise TypeError(f"Unsupported tool annotation {python_type!r}; use register_with_schema() for custom inputs")


def _generate_input_schema(func: Callable) -> tuple[dict[str, Any], list[str]]:
    """Generate JSON Schema input_schema from function signature and type hints.

    Returns (schema_dict, required_params_list).
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as exc:
        raise TypeError(f"Cannot resolve tool annotations for {func!r}: {exc}") from exc

    doc_descriptions = _parse_docstring_args(func.__doc__)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            raise TypeError(f"Tool parameter {name!r} must accept a named argument")
        hint = hints.get(name, str)
        prop = _get_json_schema_type(hint)

        if name in doc_descriptions:
            prop["description"] = doc_descriptions[name]

        properties[name] = prop

        # Nullable values still require an argument unless Python supplies a default.
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required

    return schema, required


logger = logging.getLogger(__name__)


class ToolNotFoundError(HarnessError, LookupError):
    pass


class ToolRegistry:
    """Registry for tools. Handles registration, schema generation, and execution."""

    def __init__(
        self,
        *,
        default_timeout_seconds: float | None = None,
        dedupe_calls: bool | None = None,
        sandbox: Any | None = None,
        retry: ToolRetry | None = None,
    ) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        # Applied to any tool that does not set its own timeout_seconds. None
        # means unset: an adopting Agent's ToolPolicy may fill it in.
        self._default_timeout_seconds = default_timeout_seconds
        # When on, an identical repeated call (same name AND same arguments)
        # within one run returns the first result instead of running again.
        # None means unset (inherit from the adopting Agent, else off).
        self._dedupe_calls = dedupe_calls
        self._call_results: dict[str, ToolResult] = {}
        # Tools registered without a timeout of their own; re-resolved when a
        # default arrives later through adopt_policy().
        self._inherited_timeouts: set[str] = set()
        # Applied to any tool registered without a retry of its own. None means
        # unset: an adopting Agent's ToolPolicy may fill it in, else the engine
        # uses ToolRetry().
        self._default_retry = retry
        self._inherited_retries: set[str] = set()
        # Default execution engine for the bash built-in.
        self.sandbox = sandbox

    def _resolve_timeout(self, timeout_seconds: float | None) -> float:
        """A tool's own timeout, else the registry default, else DEFAULT_TIMEOUT_SECONDS."""
        if timeout_seconds is not None:
            return timeout_seconds
        if self._default_timeout_seconds is not None:
            return self._default_timeout_seconds
        return DEFAULT_TIMEOUT_SECONDS

    def _resolve_retry(self, retry: ToolRetry | None) -> ToolRetry | None:
        """A tool's own retry, else the registry default, else None (the engine's default)."""
        return retry if retry is not None else self._default_retry

    @property
    def policy(self) -> ToolPolicy:
        """Effective options, including anything adopted from an AgentConfig."""
        return ToolPolicy(
            default_timeout_seconds=self._default_timeout_seconds,
            dedupe_calls=bool(self._dedupe_calls),
            retry=self._default_retry,
        )

    def adopt_policy(self, policy: ToolPolicy) -> None:
        """Fill options this registry was constructed without. Constructor values always win."""
        if self._default_timeout_seconds is None and policy.default_timeout_seconds is not None:
            self._default_timeout_seconds = policy.default_timeout_seconds
            for name in self._inherited_timeouts & self._tools.keys():
                self._tools[name] = replace_definition(
                    self._tools[name], timeout_seconds=policy.default_timeout_seconds
                )
        if self._dedupe_calls is None and policy.dedupe_calls:
            self._dedupe_calls = True
        if self._default_retry is None and policy.retry is not None:
            self._default_retry = policy.retry
            for name in self._inherited_retries & self._tools.keys():
                self._tools[name] = replace_definition(self._tools[name], retry=policy.retry)

    def reset_call_cache(self) -> None:
        """Forget deduped results. Call between turns; results are per-turn."""
        self._call_results.clear()

    @staticmethod
    def _call_key(tool_call: ToolCall) -> str | None:
        try:
            return f"{tool_call.name}:{json.dumps(tool_call.input, sort_keys=True, default=str)}"
        except Exception:
            return None  # unserialisable arguments: never dedupe

    def _store(
        self, definition: ToolDefinition, *, replace: bool = False, inherited_timeout: bool = False,
        inherited_retry: bool = False,
    ) -> ToolDefinition:
        if definition.name in self._tools and not replace:
            raise ValueError(f"Tool {definition.name!r} is already registered; pass replace=True to replace it")
        Draft202012Validator.check_schema(definition.input_schema)
        owned = replace_definition(definition, input_schema=deepcopy(definition.input_schema))
        self._tools[owned.name] = owned
        self._inherited_timeouts.discard(owned.name)
        if inherited_timeout:
            self._inherited_timeouts.add(owned.name)
        self._inherited_retries.discard(owned.name)
        if inherited_retry:
            self._inherited_retries.add(owned.name)
        return owned

    def register(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        permission: PermissionLevel | None = None,
        concurrent: bool = True,
        replay_policy: ReplayPolicy | str = "manual",
        timeout_seconds: float | None = None,
        retry: ToolRetry | None = None,
        retry_if_result: Callable[[ToolResult], bool] | None = None,
        replace: bool = False,
    ) -> Callable:
        """Decorator to register a function as a tool.

        Auto-generates JSON Schema from type hints and docstring. ``retry`` is
        the tool's ToolRetry (default: the registry's, else the engine's) and
        applies only when ``replay_policy`` is safe or idempotent.
        """

        try:
            replay_policy = ReplayPolicy(replay_policy).value
        except ValueError:
            raise ValueError("Invalid tool replay policy") from None
        inherited = timeout_seconds is None
        timeout_seconds = self._resolve_timeout(timeout_seconds)
        if timeout_seconds <= 0:
            raise ValueError("Tool timeout must be positive")

        def decorator(func: Callable) -> Callable:
            tool_name = name if name is not None else func.__name__
            tool_description = description if description is not None else (_extract_docstring_description(func.__doc__) or tool_name)

            input_schema, _ = _generate_input_schema(func)

            self._store(ToolDefinition(
                name=tool_name,
                description=tool_description,
                input_schema=input_schema,
                handler=func,
                permission_level=permission,
                concurrent=concurrent,
                replay_policy=replay_policy,
                timeout_seconds=timeout_seconds,
                retry=self._resolve_retry(retry),
                retry_if_result=retry_if_result,
            ), replace=replace, inherited_timeout=inherited, inherited_retry=retry is None)
            return func

        return decorator

    def register_tool(
        self,
        tool: Callable | ToolDefinition,
        *,
        name: str | None = None,
        description: str | None = None,
        permission: PermissionLevel | None = None,
        concurrent: bool | None = None,
        replay_policy: ReplayPolicy | str | None = None,
        timeout_seconds: float | None = None,
        retry: ToolRetry | None = None,
        retry_if_result: Callable[[ToolResult], bool] | None = None,
        replace: bool = False,
    ) -> ToolDefinition:
        """Register a function or ToolDefinition directly (non-decorator style)."""
        if isinstance(tool, ToolDefinition):
            overrides = {
                "name": name, "description": description, "permission_level": permission,
                "concurrent": concurrent, "replay_policy": replay_policy, "timeout_seconds": timeout_seconds,
                "retry": retry, "retry_if_result": retry_if_result,
            }
            definition = replace_definition(tool, **{k: v for k, v in overrides.items() if v is not None})
            inherited = definition.retry is None
            if inherited:
                definition = replace_definition(definition, retry=self._default_retry)
            return self._store(definition, replace=replace, inherited_retry=inherited)

        decorator = self.register(
            name=name,
            description=description,
            permission=permission,
            concurrent=True if concurrent is None else concurrent,
            replay_policy="manual" if replay_policy is None else replay_policy,
            timeout_seconds=timeout_seconds,
            retry=retry,
            retry_if_result=retry_if_result,
            replace=replace,
        )
        decorator(tool)
        return self._tools[name if name is not None else tool.__name__]

    def register_with_schema(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable,
        *,
        permission: PermissionLevel | None = None,
        concurrent: bool = True,
        replay_policy: ReplayPolicy | str = "manual",
        timeout_seconds: float | None = None,
        retry: ToolRetry | None = None,
        retry_if_result: Callable[[ToolResult], bool] | None = None,
        replace: bool = False,
    ) -> ToolDefinition:
        """Imperative registration with explicit schema."""
        return self._store(ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            permission_level=permission,
            concurrent=concurrent,
            replay_policy=replay_policy,
            timeout_seconds=self._resolve_timeout(timeout_seconds),
            retry=self._resolve_retry(retry),
            retry_if_result=retry_if_result,
        ), replace=replace, inherited_timeout=timeout_seconds is None, inherited_retry=retry is None)

    def load_builtin(
        self,
        bundle_or_tool: str,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        permission: PermissionLevel | None = None,
        replace: bool = False,
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
            register_all_tools,
            register_bash_tools,
            register_filesystem_tools,
            register_memory_tools,
            register_web_tools,
        )
        from .builtin._registration import select_tools

        name = bundle_or_tool.lower()
        if name in ("all", "builtin", "builtins"):
            return register_all_tools(self, include=include, exclude=exclude, permission=permission, replace=replace, **options)
        elif name in ("filesystem", "fs", "files"):
            return register_filesystem_tools(self, include=include, exclude=exclude, permission=permission, replace=replace, **options)
        elif name in ("bash", "shell", "terminal"):
            return register_bash_tools(self, include=include, exclude=exclude, permission=permission, replace=replace, **options)
        elif name in ("web", "http", "fetch"):
            return register_web_tools(self, include=include, exclude=exclude, permission=permission, replace=replace, **options)
        elif name in ("memory",):
            return register_memory_tools(self, include=include, exclude=exclude, permission=permission, replace=replace, **options)

        standalone_map = {
            "read_file": register_filesystem_tools,
            "write_file": register_filesystem_tools,
            "list_directory": register_filesystem_tools,
            "generate_file": register_filesystem_tools,
            "run_bash": register_bash_tools,
            "fetch_url": register_web_tools,
            "save_memory": register_memory_tools,
            "recall_memories": register_memory_tools,
        }
        if name in standalone_map:
            selected = select_tools(self, [name], include, exclude, replace=replace)
            return standalone_map[name](self, include=selected, permission=permission, replace=replace, **options)

        raise ValueError(
            f"Unknown built-in tool or bundle: {bundle_or_tool!r}. "
            f"Available bundles: 'filesystem', 'bash', 'web', 'memory', 'all'."
        )

    def get_tool_params(self) -> list[dict[str, Any]]:
        """Convert all registered tools to canonical ToolParam format."""
        params: list[dict[str, Any]] = []
        for t in self._tools.values():
            schema = deepcopy(t.input_schema)
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
        self._inherited_timeouts.discard(name)
        return self._tools.pop(name, None) is not None

    def get_tool(self, name: str) -> ToolDefinition:
        """Lookup a tool by name."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def resolve_tool(self, name: str) -> tuple[ToolDefinition, str]:
        """Find a tool, correcting a name the model nearly got right.

        Models misspell a tool name by case far more often than by anything
        else. Failing the call teaches nothing the model can act on, and the
        repair is unambiguous as long as exactly one tool matches. Returns the
        definition and the corrected name.
        """
        if name in self._tools:
            return self._tools[name], name
        if isinstance(name, str):
            folded = name.strip().casefold()
            matches = [known for known in self._tools if known.casefold() == folded]
            if len(matches) == 1:
                return self._tools[matches[0]], matches[0]
        raise ToolNotFoundError(
            f"No tool named {name!r}. Available tools: {', '.join(sorted(self._tools)) or 'none'}"
        )

    async def execute(
        self, tool_call: ToolCall, *, permissions: PermissionManager | None = None,
    ) -> ToolResult:
        """Validate and authorize a direct call, then execute within its timeout.

        Without a PermissionManager, unspecified tools default to ALLOW;
        explicit ASK still requires approval and DENY blocks execution.
        Agent runs handle approval/recovery in the engine before internal dispatch.
        A timeout stops waiting but cannot forcibly stop a synchronous worker thread.
        A tool registered with its own ``retry`` (and a safe or idempotent replay
        policy) is re-executed on transient failures the same way the engine does it.
        """
        call = deepcopy(tool_call)
        try:
            definition = self.get_tool(call.name)
            validate(call.input, definition.input_schema)
            policy = permissions if permissions is not None else PermissionManager()
            allowed = await policy.check_permission(call, definition)
            if not allowed:
                return ToolResult(call.id, "Permission denied", True)
            retry = definition.retry if definition.replay_policy != "manual" else None
            attempts = retry.attempts if retry is not None else 1
            attempt = 0
            while True:
                attempt += 1
                try:
                    try:
                        async with asyncio.timeout(definition.timeout_seconds):
                            result = await self._dispatch(call, definition)
                    except TimeoutError:
                        # The model needs to know this tool is unavailable so it can
                        # move on; a turn that hangs on one slow tool helps nobody.
                        raise TimeoutError(
                            f"Tool '{call.name}' timed out after {definition.timeout_seconds}s."
                        ) from None
                    if retry is not None and attempt < attempts and retry_wanted(definition, retry, result):
                        raise TransientToolError(str(result.content))
                    return result
                except Exception as exc:
                    transient = isinstance(exc, TransientToolError) or is_transient(exc)
                    if retry is None or attempt >= attempts or not transient:
                        if isinstance(exc, TransientToolError):
                            return ToolResult(call.id, f"Tool execution error: {exc}", True)
                        raise
                    logger.warning(
                        "Tool %s failed (attempt %d/%d), retrying: %s", call.name, attempt, attempts, exc,
                    )
                    wait = retry.wait_for(attempt)
                    if wait:
                        await asyncio.sleep(wait)
        except Exception as exc:
            return ToolResult(call.id, f"{type(exc).__name__}: {exc}", True)

    async def _dispatch(self, tool_call: ToolCall, definition: ToolDefinition | None = None) -> ToolResult:
        """Internal dispatch after the caller has enforced approval and timeout."""
        cache_key = self._call_key(tool_call) if self._dedupe_calls else None
        if cache_key is not None and cache_key in self._call_results:
            cached = self._call_results[cache_key]
            # Same result, this call's own id.
            return replace_definition(cached, tool_call_id=tool_call.id, tool_use_id=tool_call.id)
        outcome = await self._run_handler(tool_call, definition)
        if cache_key is not None:
            self._call_results[cache_key] = outcome
        return outcome

    async def _run_handler(self, tool_call: ToolCall, definition: ToolDefinition | None) -> ToolResult:
        try:
            tool_def = definition or self.get_tool(tool_call.name)
            if not isinstance(tool_call.input, dict):
                raise TypeError("Tool input must be an object")
            validate(tool_call.input, tool_def.input_schema)
            handler = tool_def.handler
            args = tool_call.input
            if inspect.iscoroutinefunction(handler):
                call = handler(**args)
            else:
                loop = asyncio.get_running_loop()
                import contextvars
                context = contextvars.copy_context()
                call = loop.run_in_executor(None, lambda: context.run(handler, **args))
            result = await call
            if inspect.isawaitable(result):
                result = await result

            if isinstance(result, ToolResult):
                return replace_definition(result, tool_call_id=tool_call.id, tool_use_id=tool_call.id)
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps(result, ensure_ascii=False, allow_nan=False) if isinstance(result, (dict, list)) else str(result),
            )
        except TransientToolError:
            raise  # the handler asked for a retry; the caller's policy decides
        except Exception as exc:
            # The model gets one line it can act on; the traceback is for the operator.
            logger.warning("Tool %s failed: %s", tool_call.name, traceback.format_exc())
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Tool execution error: {type(exc).__name__}: {exc}",
                is_error=True,
            )


def retry_wanted(definition: ToolDefinition, retry: ToolRetry, result: ToolResult) -> bool:
    """Whether a result that came back should be treated as a transient failure.

    The tool's ``retry_if_result`` decides first; with ``retry_error_results`` an
    error result whose text reads as throttling or overload counts too.
    """
    check = getattr(definition, "retry_if_result", None)
    if callable(check):
        try:
            if check(result):
                return True
        except Exception:
            logger.warning("retry_if_result for %s raised: %s", definition.name, traceback.format_exc())
    return bool(retry.retry_error_results and result.is_error and is_transient_text(str(result.content)))


def normalize_tool_registry(
    tools: ToolRegistry | list[Any] | None,
    *,
    policy: ToolPolicy | None = None,
    sandbox: Any | None = None,
) -> ToolRegistry:
    """Normalize tools argument into a ToolRegistry.

    Supports:
    - None -> empty ToolRegistry
    - ToolRegistry -> passed through
    - list of strings (bundles or tool names), callables, or ToolDefinitions
    """
    if isinstance(tools, ToolRegistry):
        if policy is not None:
            tools.adopt_policy(policy)
        if sandbox is not None:
            if tools.sandbox is None:
                tools.sandbox = sandbox
            elif tools.sandbox is not sandbox:
                raise ValueError("ToolRegistry is already bound to a different sandbox")
        return tools
    registry = ToolRegistry(sandbox=sandbox)
    if policy is not None:
        registry.adopt_policy(policy)  # before registering, so defaults apply at registration
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
