"""Skill loading and lazy invocation.

A "skill" is a markdown file (SKILL.md) with YAML frontmatter describing
its name and one-line description, plus an instruction body the agent
loads on demand.

The agent only sees the name+description of each skill in the system prompt.
When it decides a skill applies, it calls the `Skill` tool with the skill
name and receives the full body as the tool result — keeping unused skill
instructions out of the context window.

Usage:
    skills = SkillManager.from_paths([
        "./skills/code-review",       # folder containing SKILL.md
        "./skills/triage.md",         # a SKILL.md file directly
    ])
    skills.install(agent)             # registers Skill tool + amends system prompt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .hooks import HookContext, HookEvent
from .types import PermissionLevel

if TYPE_CHECKING:
    from .core import Agent


SKILL_FILENAME = "SKILL.md"


@dataclass
class Skill:
    name: str
    description: str
    body: str
    source_path: Path


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split YAML-style frontmatter from the body. Returns (metadata, body).

    Only supports flat `key: value` pairs — sufficient for skill metadata.
    """
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines(keepends=True)
    if not lines:
        return {}, text

    # Find the closing '---'
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}, text

    metadata: dict[str, str] = {}
    for line in lines[1:end_idx]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        metadata[key.strip()] = value.strip().strip('"').strip("'")

    body = "".join(lines[end_idx + 1 :]).lstrip("\n")
    return metadata, body


def _load_skill(path: Path) -> Skill:
    """Load a single skill from a file or folder path."""
    path = path.expanduser().resolve()

    if path.is_dir():
        skill_file = path / SKILL_FILENAME
        if not skill_file.is_file():
            raise FileNotFoundError(
                f"No {SKILL_FILENAME} found in skill folder: {path}"
            )
        text = skill_file.read_text(encoding="utf-8")
        default_name = path.name
    elif path.is_file():
        text = path.read_text(encoding="utf-8")
        default_name = path.stem
    else:
        raise FileNotFoundError(f"Skill path does not exist: {path}")

    metadata, body = _parse_frontmatter(text)
    name = metadata.get("name") or default_name
    description = metadata.get("description") or f"Skill: {name}"

    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        source_path=path,
    )


class SkillManager:
    """Loads skills from disk and wires lazy invocation into an Agent.

    Lazy pattern: the model sees name+description in the system prompt and
    calls the `Skill` tool to pull a skill's body into context only when needed.
    """

    SKILLS_BLOCK_OPEN = "<available-skills>"
    SKILLS_BLOCK_CLOSE = "</available-skills>"

    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills or []:
            self.add(skill)

    @classmethod
    def from_paths(cls, paths: list[str | Path]) -> SkillManager:
        """Construct a manager by loading every path. A path may point to
        a SKILL.md file directly or to a folder containing one."""
        manager = cls()
        for raw in paths:
            manager.add(_load_skill(Path(raw)))
        return manager

    def add(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"Duplicate skill name: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(
                f"Skill '{name}' not found. Available: {sorted(self._skills)}"
            )
        return self._skills[name]

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def render_catalog(self) -> str:
        """Render the `<available-skills>` block to append to a system prompt."""
        if not self._skills:
            return ""
        lines = [self.SKILLS_BLOCK_OPEN]
        lines.append(
            "When a skill applies, call the `Skill` tool with its name to load "
            "its instructions. Only invoke skills listed here."
        )
        for skill in self._skills.values():
            lines.append(f"- {skill.name}: {skill.description}")
        lines.append(self.SKILLS_BLOCK_CLOSE)
        return "\n".join(lines)

    def install(self, agent: Agent) -> None:
        """Register the `Skill` tool on the agent and append the skills catalog
        to its system prompt. Fires HookEvent.SKILL_INVOKED on every call."""

        async def handler(skill: str) -> str:
            try:
                loaded = self.get(skill)
            except KeyError as e:
                await agent.hooks.emit(
                    HookEvent.SKILL_INVOKED,
                    HookContext(
                        event=HookEvent.SKILL_INVOKED,
                        agent=agent,
                        data={"skill": skill, "found": False},
                    ),
                )
                return str(e)

            await agent.hooks.emit(
                HookEvent.SKILL_INVOKED,
                HookContext(
                    event=HookEvent.SKILL_INVOKED,
                    agent=agent,
                    data={
                        "skill": loaded.name,
                        "found": True,
                        "source_path": str(loaded.source_path),
                        "body_chars": len(loaded.body),
                    },
                ),
            )
            return loaded.body

        agent.tools.register_with_schema(
            name="Skill",
            description=(
                "Load the full instructions for a named skill from the "
                "<available-skills> catalog. Returns the skill's body as text."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "Exact skill name from the catalog.",
                    },
                },
                "required": ["skill"],
            },
            handler=handler,
            permission=PermissionLevel.ALLOW,
        )

        catalog = self.render_catalog()
        if catalog:
            existing = agent.config.system_prompt or ""
            if catalog not in existing:
                agent.config.system_prompt = (
                    f"{existing}\n\n{catalog}" if existing else catalog
                )
