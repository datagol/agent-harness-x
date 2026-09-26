"""Scripted provider for examples that explicitly run without model services."""

from collections.abc import Iterable
from typing import Any

from harnessx import ProviderResponse
from harnessx.providers import LLMProvider


class ScriptedProvider(LLMProvider):
    """Exercise the real execution engine with fixed, provider-neutral responses."""

    def __init__(self, responses: Iterable[ProviderResponse]) -> None:
        self._responses = iter(responses)

    async def create(self, **kwargs: Any) -> ProviderResponse:
        try:
            return next(self._responses)
        except StopIteration:
            raise RuntimeError("The example exhausted its scripted responses") from None

    async def count_tokens(self, **kwargs: Any) -> int:
        return 0
