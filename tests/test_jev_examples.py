import importlib
import json
import socket
from copy import deepcopy

import pytest

from harnessx import Agent, ProviderResponse, RunResult, RunStatus
from harnessx.decisions import (
    BooleanQuestion, ChoiceAnswer, ChoiceQuestion, DecisionBatch,
    DecisionError, DecisionProvider, ScoreQuestion,
)
from examples._decision_fixtures import FixedDecisionProvider
from examples._fixtures import ScriptedProvider
from examples.jev_answer_review import COVERAGE_LEVELS, review_answer
from examples.jev_classification import DOCUMENT_TYPES, classify_document
from examples.jev_routing import ROUTES, choose_route, run_routed_query


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline examples must not connect to external services")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.asyncio
@pytest.mark.parametrize("module,args", [
    ("jev_routing", ["--min-confidence", "0.8"]),
    ("jev_classification", []), ("jev_answer_review", []),
])
async def test_examples_default_offline_without_constructing_jev(module, args, monkeypatch, capsys):
    example = importlib.import_module("examples." + module)

    def forbidden():
        pytest.fail("offline examples must not construct a live decision provider")
    monkeypatch.setattr(example, "JevDecisionProvider", forbidden)
    assert await example.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "fixed offline fixtures"


@pytest.mark.asyncio
@pytest.mark.parametrize("label,confidence,expected", [
    ("sql", .9, "sql"), ("research", .9, "research"), ("general", .9, "general"),
    ("unknown", .99, "general"), ("sql", .79, "general"), ("sql", None, "general"),
    ("sql", .8, "sql"), ("arbitrary-agent-name", 1, "general"),
])
async def test_routing_uses_only_fixed_factories_and_required_confidence(label, confidence, expected):
    labels = set(ROUTES) | {label}
    probabilities = {name: 1.0 if name == label else 0.0 for name in labels}
    decisions = FixedDecisionProvider({"route": ChoiceAnswer(label, probabilities, confidence)})
    created = []

    def factory(name):
        def create():
            agent = Agent(provider=ScriptedProvider([ProviderResponse(text=name)]))
            created.append((name, agent))
            return agent
        return create
    route, batch, result = await run_routed_query(
        "query", decisions, {name: factory(name) for name in ("sql", "research", "general")}, min_confidence=.8,
    )
    assert route == result.output == expected
    assert len(created) == 1 and created[0][0] == expected and created[0][1].closed


class UnavailableProvider(DecisionProvider):
    async def evaluate(self, request, **kwargs):
        return DecisionBatch("unavailable", request.check_id, request.version,
                             error=DecisionError("timeout", "Timed out", retryable=True))


@pytest.mark.asyncio
async def test_unavailable_decision_falls_back_and_agent_failure_does_not_reroute():
    route, batch = await choose_route("query", UnavailableProvider(), min_confidence=.8)
    assert route == "general" and batch.status == "unavailable"
    created = []

    def factory():
        agent = Agent(provider=ScriptedProvider([]))
        created.append(agent)
        return agent
    route, _, result = await run_routed_query(
        "query", UnavailableProvider(), {name: factory for name in ("sql", "research", "general")}, min_confidence=.8,
    )
    assert result.status == "failed" and len(created) == 1 and created[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [True, None, -1, 1.1, float("nan"), float("inf")])
async def test_threshold_validation(threshold):
    with pytest.raises(ValueError):
        await choose_route("query", UnavailableProvider(), min_confidence=threshold)


class CaptureProvider(UnavailableProvider):
    def __init__(self):
        self.requests = []

    async def evaluate(self, request, **kwargs):
        self.requests.append(request)
        return await super().evaluate(request)


@pytest.mark.asyncio
async def test_classification_contract():
    decisions = CaptureProvider()
    result = await classify_document("Form W-2", decisions)
    submitted = decisions.requests[0]
    assert submitted.state == {"document_text": "Form W-2"}
    question = submitted.questions["document_type"]
    assert isinstance(question, ChoiceQuestion)
    assert set(question.options) == set(DOCUMENT_TYPES) == {"W2", "Deposit", "Insurance", "Payroll", "unknown"}
    assert result.status == "unavailable"


@pytest.mark.asyncio
async def test_review_uses_score_and_boolean_criteria_without_changing_run():
    result = RunResult("session", "run", output="Supplied answer")
    before = deepcopy(result)
    decisions = CaptureProvider()
    review = await review_answer(result, query="Question?", evidence="Evidence text", decisions=decisions)
    assert result == before and review.status == "unavailable"
    submitted = decisions.requests[0]
    assert submitted.state == {"query": "Question?", "answer": "Supplied answer", "evidence": "Evidence text"}
    assert isinstance(submitted.questions["coverage"], ScoreQuestion)
    assert submitted.questions["coverage"].levels == COVERAGE_LEVELS
    assert isinstance(submitted.questions["supported"], BooleanQuestion)
    assert submitted.questions["supported"].criteria.true
    assert submitted.questions["supported"].criteria.false


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "awaiting_input", "cancelled"])
async def test_review_skips_unsuccessful_runs(status):
    result = RunResult("session", "run", status=RunStatus(status))
    decisions = CaptureProvider()
    assert await review_answer(result, query="query", evidence="evidence", decisions=decisions) is None
    assert not decisions.requests


@pytest.mark.asyncio
async def test_live_routing_requires_explicit_model():
    from examples.jev_routing import main
    with pytest.raises(SystemExit) as exc:
        await main(["--live", "--min-confidence", ".8"])
    assert exc.value.code == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("module,args", [
    ("jev_routing", ["--min-confidence", ".8", "--model", "test-model"]),
    ("jev_classification", []), ("jev_answer_review", []),
])
async def test_live_flag_explicitly_selects_jev_without_live_calls(module, args, monkeypatch):
    example = importlib.import_module("examples." + module)
    selected = []

    def live():
        selected.append(True)
        return UnavailableProvider()
    monkeypatch.setattr(example, "JevDecisionProvider", live)
    monkeypatch.setattr("harnessx.core.make_provider", lambda *args: ScriptedProvider([ProviderResponse(text="fixture")]))
    assert await example.main(["--live", *args]) in (0, 1)
    assert selected == [True]
