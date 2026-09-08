"""Dataset registry for DataGOL Agent Harness benchmarks."""

from __future__ import annotations

from typing import Callable
from langsmith.schemas import Example

from .guardrails import get_guardrails_dataset
from .memory import get_memory_dataset
from .multi_agent import get_multi_agent_dataset
from .skills import get_skills_dataset
from .tool_calling import get_tool_calling_dataset

_DATASET_REGISTRY: dict[str, Callable[[], list[Example]]] = {
    "tool_calling": get_tool_calling_dataset,
    "skills": get_skills_dataset,
    "multi_agent": get_multi_agent_dataset,
    "guardrails": get_guardrails_dataset,
    "memory": get_memory_dataset,
}


def list_datasets() -> list[str]:
    """Returns the names of all registered benchmark datasets."""
    return list(_DATASET_REGISTRY.keys())


def get_dataset(name: str) -> list[Example]:
    """Retrieves a benchmark dataset by name."""
    if name == "all":
        all_examples: list[Example] = []
        for fn in _DATASET_REGISTRY.values():
            all_examples.extend(fn())
        return all_examples

    key = name.lower().replace("-", "_")
    if key not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown benchmark dataset '{name}'. Available datasets: {list_datasets() + ['all']}"
        )
    return _DATASET_REGISTRY[key]()
