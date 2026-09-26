"""Declarative specialists, executed through the ordinary tool execution path."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .execution import RunStatus
from .skills import SkillManager
from .tools import ToolRegistry, normalize_tool_registry
from .types import AgentConfig, PermissionLevel

if TYPE_CHECKING:
    from .core import Agent


@dataclass(frozen=True)
class SubAgent:
    """Definition of a fresh specialist for each delegated task.

    ``description`` tells the parent when to delegate; ``config.system_prompt``
    instructs the specialist. Config/model, tools, and skill paths are explicit.
    ``permission`` controls delegation, independently of the child's tool checks.
    Unspecified permissions inherit the parent's permission manager. Child
    conversations are isolated, but tool handlers and that manager are shared.
    Child runs are not independently durable; delegation replay is manual.
    """

    name: str
    description: str
    config: AgentConfig
    tools: ToolRegistry | list[Any] | None = None
    skills: list[str] | None = None
    permission: PermissionLevel | None = None
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        # Keep the generated provider tool name within 64 characters.
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,54}", self.name):
            raise ValueError("SubAgent.name must be 1–55 letters, digits, underscores, or hyphens, starting with a letter or underscore")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("SubAgent.description must be a nonempty string")
        if not isinstance(self.config, AgentConfig):
            raise TypeError("SubAgent.config must be an AgentConfig")
        if self.permission is not None and not isinstance(self.permission, PermissionLevel):
            raise TypeError("SubAgent.permission must be a PermissionLevel or None")
        if (isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("SubAgent.timeout_seconds must be finite and positive")
        if self.skills is not None and (
            not isinstance(self.skills, list)
            or any(not isinstance(path, str) or not path for path in self.skills)
        ):
            raise TypeError("SubAgent.skills must be a list of nonempty paths")


def _copy_tools(registry: ToolRegistry) -> ToolRegistry:
    # Copy schemas and metadata, never deepcopy application-owned handlers.
    return normalize_tool_registry(registry.get_tools())


def prepare_subagents(subagents: list[SubAgent] | None, tools: ToolRegistry) -> tuple[SubAgent, ...]:
    """Validate every definition before registering any generated tools."""
    prepared = []
    names = set(tools.list_tools())
    for spec in subagents or []:
        if not isinstance(spec, SubAgent):
            raise TypeError("Agent.subagents accepts SubAgent definitions, not live Agent instances")
        name = f"delegate_{spec.name}"
        if name in names:
            raise ValueError(f"Duplicate subagent or tool name: {name!r}")
        names.add(name)
        registry = _copy_tools(normalize_tool_registry(spec.tools))
        if spec.skills is not None and registry.has_tool("Skill"):
            raise ValueError("A subagent with skills cannot also register a Skill tool")
        prepared.append(SubAgent(
            name=spec.name, description=spec.description, config=deepcopy(spec.config),
            tools=registry, skills=deepcopy(spec.skills), permission=spec.permission,
            timeout_seconds=spec.timeout_seconds,
        ))
    return tuple(prepared)


def install_subagent(parent: Agent, spec: SubAgent) -> None:
    # Capture bindings separately from the public definition's mutable members.
    config = deepcopy(spec.config)
    registry = _copy_tools(normalize_tool_registry(spec.tools))
    paths = deepcopy(spec.skills)

    async def delegate(task: str) -> str:
        from .core import Agent

        # Load paths before constructing an agent/provider so a bad path cannot
        # leave an allocated provider behind. Each child owns its SkillManager.
        skills = SkillManager.from_paths([*paths]) if paths is not None else None
        async with Agent(
            config=deepcopy(config), tools=_copy_tools(registry), skills=skills,
            permissions=parent.permissions,
        ) as child:
            result = await child.run(task)
            if result.status != RunStatus.COMPLETED or result.error is not None:
                detail = result.error["message"] if result.error else result.stop_reason
                raise RuntimeError(f"Subagent {spec.name!r} {result.status.value}: {detail}")
            return result.output

    parent.tools.register_tool(
        delegate, name=f"delegate_{spec.name}",
        description=f"{spec.description}\nSend a self-contained task with all necessary context. The specialist returns its result to you.",
        permission=spec.permission, concurrent=False, replay_policy="manual",
        timeout_seconds=spec.timeout_seconds,
    )
