"""Shared validation for built-in tool selections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from harnessx.tools import ToolRegistry


def select_tools(
    registry: ToolRegistry,
    available: Iterable[str],
    include: list[str] | None,
    exclude: list[str] | None,
    *,
    replace: bool = False,
) -> list[str]:
    names = list(available)
    for label, values in (("include", include), ("exclude", exclude)):
        if isinstance(values, str):
            raise TypeError(f"{label} must be a collection of tool names")
        unknown = set(values or ()) - set(names)
        if unknown:
            raise ValueError(f"Unknown tool names in {label}: {sorted(unknown)}; available: {names}")
    selected = [name for name in names if (include is None or name in include) and name not in (exclude or ())]
    collisions = [name for name in selected if registry.has_tool(name)]
    if collisions and not replace:
        raise ValueError(f"Tools already registered: {collisions}; pass replace=True to replace them")
    return selected
