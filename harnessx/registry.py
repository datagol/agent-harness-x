"""Versioned, process-local bindings. Factories and credentials never enter storage."""

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Agent


@dataclass(frozen=True)
class AgentRef:
    name: str
    version: str = "v1"


class AgentRegistry:
    def __init__(self):
        self._factories: dict[AgentRef, Callable[[], "Agent"]] = {}

    def register(
        self, name: str, factory: Callable[[], "Agent"], *, version: str = "v1"
    ) -> AgentRef:
        ref = AgentRef(name, version)
        if ref in self._factories:
            raise ValueError(f"Agent already registered: {ref}")
        self._factories[ref] = factory
        return ref

    def create(self, ref: AgentRef) -> "Agent":
        if ref not in self._factories:
            raise ValueError(f"Missing agent binding {ref.name}@{ref.version}")
        return self._factories[ref]()


agents = AgentRegistry()
