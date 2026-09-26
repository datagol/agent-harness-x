import json
import math
import subprocess
import sys

import pytest

from harnessx.decisions import (
    BooleanAnswer, BooleanCriteria, BooleanQuestion, ChoiceAnswer, ChoiceQuestion,
    DecisionBatch, DecisionError, DecisionRequest, DecisionUsage, ScoreAnswer, ScoreQuestion,
)


def request():
    return DecisionRequest("review", "v1", {"nested": ["source"]}, {
        "route": ChoiceQuestion("Pick a route", {"sql": "Data", "general": "Writing"}),
        "coverage": ScoreQuestion("Coverage", ("None", "Complete")),
        "support": BooleanQuestion("Supported?", BooleanCriteria("All claims supported", "Unsupported claims")),
    })


def test_request_json_roundtrip_and_copies():
    original = request()
    encoded = original.to_dict()
    encoded["future_metadata"] = "ignored"
    decoded = DecisionRequest.from_dict(json.loads(json.dumps(encoded)))
    assert decoded == original
    encoded["state"]["nested"].append("new")
    assert original.state == {"nested": ["source"]}
    with pytest.raises(TypeError):
        original.questions["other"] = BooleanQuestion("New?")
    with pytest.raises(TypeError):
        original.questions["route"].options["sql"] = "changed"


def test_result_roundtrip_and_readonly_views():
    batch = DecisionBatch("completed", "review", "v1", {
        "route": ChoiceAnswer("sql", {"sql": .8, "general": .2}, .7),
        "coverage": ScoreAnswer(.75, (.25, .75), ("None", "Complete"), .6),
        "support": BooleanAnswer(.91),
    }, requested_model="fixture", attempts=1, usage=DecisionUsage(10, 5, incomplete=False))
    encoded = batch.to_dict()
    assert encoded["answers"]["coverage"]["probabilities"] == [.25, .75]
    encoded["usage"]["future"] = 1
    assert DecisionBatch.from_dict(json.loads(json.dumps(encoded))) == batch
    with pytest.raises(TypeError):
        batch.choices["route"] = batch.choices["route"]
    with pytest.raises(TypeError, match="threshold"):
        bool(batch.booleans["support"])
    unavailable = DecisionBatch("unavailable", "review", "v1", error=DecisionError("timeout", "Timed out"))
    assert DecisionBatch.from_dict(unavailable.to_dict()) == unavailable


@pytest.mark.parametrize("state", [None, 3, True, {1: "bad"}, {"bad": object()}, [math.inf], [math.nan], [set()]])
def test_invalid_state(state):
    with pytest.raises((ValueError, TypeError)):
        DecisionRequest("id", "v1", state, {"x": BooleanQuestion("Question?")})


def test_cyclic_state_rejected():
    state = []
    state.append(state)
    with pytest.raises(ValueError, match="acyclic"):
        DecisionRequest("id", "v1", state, {"x": BooleanQuestion("Question?")})


@pytest.mark.parametrize("factory", [
    lambda: ChoiceQuestion("", {"a": "A", "b": "B"}),
    lambda: ChoiceQuestion("Pick", {"a": "A"}),
    lambda: ChoiceQuestion("Pick", {"a": "", "b": "B"}),
    lambda: ScoreQuestion("Score", ("one",)),
    lambda: ScoreQuestion("Score", tuple(str(i) for i in range(11))),
    lambda: BooleanCriteria(true=""),
    lambda: BooleanQuestion("Q", criteria={"true": "Y"}),
    lambda: BooleanAnswer(True),
    lambda: BooleanAnswer(float("nan")),
    lambda: BooleanAnswer(10 ** 1000),
    lambda: ChoiceAnswer("a", {"a": .2, "b": .8}),
    lambda: ChoiceAnswer("a", {"a": .7, "b": .7}),
    lambda: ChoiceAnswer("a", {"a": .7, "b": .3}, 1.1),
    lambda: ScoreAnswer(.5, (.2, .8), ("low", "high")),
    lambda: ScoreAnswer(.8, (.2, .8), ("low", "high"), math.inf),
    lambda: DecisionUsage(-1, 0),
    lambda: DecisionUsage(True, 0),
    lambda: DecisionUsage(incomplete=False),
    lambda: DecisionBatch("completed", "id", "v1"),
    lambda: DecisionBatch("unavailable", "id", "v1", {"q": BooleanAnswer(.5)}, error=DecisionError("e", "E")),
])
def test_invalid_contracts(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(schema_version=2),
    lambda d: d.update(schema_version=True),
    lambda d: d["questions"]["route"].update(kind="future"),
    lambda d: d["questions"]["route"]["options"].update(sql=""),
    lambda d: d["questions"]["support"].update(criteria="not an object"),
    lambda d: d.pop("state"),
])
def test_request_decode_validates(mutation):
    data = request().to_dict()
    mutation(data)
    with pytest.raises(ValueError):
        DecisionRequest.from_dict(data)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(schema_version=2),
    lambda d: d["answers"]["q"].update(kind="future"),
    lambda d: d["answers"]["q"].update(probability=-1),
    lambda d: d["usage"].update(input_tokens=True),
    lambda d: d.update(status="unavailable"),
])
def test_result_decode_validates(mutation):
    data = DecisionBatch("completed", "id", "v1", {"q": BooleanAnswer(.5)}).to_dict()
    mutation(data)
    with pytest.raises(ValueError):
        DecisionBatch.from_dict(data)


def test_optional_import_isolation():
    code = '''
import importlib.abc
import sys
class NoVendor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == "typesafe_sdk" or fullname.startswith("typesafe_sdk."):
            raise ModuleNotFoundError("Optional SDK absent", name="typesafe_sdk")
sys.meta_path.insert(0, NoVendor())
import harnessx
from harnessx.decisions import JevDecisionProvider, BooleanQuestion
assert "typesafe_sdk" not in sys.modules
assert BooleanQuestion("Works without vendor")
try:
    JevDecisionProvider()
except ImportError as exc:
    assert "harnessx[jev]" in str(exc)
else:
    raise AssertionError("Missing dependency must give install guidance")
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
