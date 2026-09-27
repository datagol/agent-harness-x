"""Public decision API type-checking fixture; never makes a live call."""

from collections.abc import Mapping
from typing import assert_type

from harnessx.decisions import (
    BooleanAnswer, BooleanCriteria, BooleanQuestion, ChoiceAnswer, ChoiceQuestion,
    DecisionBatch, DecisionProvider, DecisionRequest, JevDecisionProvider, ScoreAnswer, ScoreQuestion,
)


async def use_decisions(provider: DecisionProvider) -> None:
    request = DecisionRequest("review", "v1", {"answer": "Example"}, {
        "route": ChoiceQuestion("Route", {"sql": "Data", "general": "Other"}),
        "coverage": ScoreQuestion("Coverage", ("None", "Full")),
        "support": BooleanQuestion("Supported?", BooleanCriteria(true="Supported by evidence")),
    })
    batch = await provider.evaluate(request, timeout_seconds=1.0, max_attempts=1)
    assert_type(batch, DecisionBatch)
    assert_type(batch.choices, Mapping[str, ChoiceAnswer])
    assert_type(batch.scores["coverage"], ScoreAnswer)
    assert_type(batch.booleans["support"], BooleanAnswer)
    assert_type(DecisionBatch.from_dict(batch.to_dict()), DecisionBatch)
    assert_type(DecisionRequest.from_dict(request.to_dict()), DecisionRequest)
    async with JevDecisionProvider() as owned:
        assert_type(owned, JevDecisionProvider)
        await owned.evaluate(request)
