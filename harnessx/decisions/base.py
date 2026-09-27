"""Async decision providers are independent of conversational LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Self

from .types import DecisionBatch, DecisionRequest


class DecisionProvider(ABC):
    @abstractmethod
    async def evaluate(
        self, request: DecisionRequest, *, timeout_seconds: float | None = None,
        max_attempts: int | None = None,
    ) -> DecisionBatch:
        """Evaluate questions; service failures return an unavailable batch."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release owned resources. Borrowed providers remain the caller's responsibility."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
