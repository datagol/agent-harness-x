"""Fixed decisions for offline demonstrations; these do not perform inference."""

from collections.abc import Mapping

from harnessx.decisions import Answer, DecisionBatch, DecisionProvider, DecisionRequest


class FixedDecisionProvider(DecisionProvider):
    def __init__(self, answers: Mapping[str, Answer]) -> None:
        self.answers = answers

    async def evaluate(
        self, request: DecisionRequest, *, timeout_seconds: float | None = None,
        max_attempts: int | None = None,
    ) -> DecisionBatch:
        if set(request.questions) != set(self.answers):
            raise ValueError("fixture does not match the example's questions")
        return DecisionBatch("completed", request.check_id, request.version,
                             answers=self.answers, requested_model="offline-fixture",
                             resolved_model="offline-fixture")
