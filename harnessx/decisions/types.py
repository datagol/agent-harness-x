"""Vendor-neutral decision contracts. Wire schema v1 uses JSON arrays for scores."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

JSONValue: TypeAlias = "None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]"
DecisionState: TypeAlias = str | dict[str, JSONValue] | list[JSONValue]
PROBABILITY_TOLERANCE = 1e-3


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _number(value: Any, name: str, low: float = 0, high: float = math.inf) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    try:
        valid = low <= value <= high and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} is out of range")


def _integer(value: Any, name: str, low: int = 0) -> None:
    if type(value) is not int or value < low:
        raise ValueError(f"{name} must be an integer >= {low}")


def _mapping(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key in value:
        _text(key, f"{name} key")


def _levels(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not 2 <= len(value) <= 10:
        raise ValueError("levels must contain two to ten ordered descriptions")
    for level in value:
        _text(level, "level")
    return tuple(value)


def _distribution(values: Any) -> None:
    for value in values:
        _number(value, "probability", high=1)
    if not math.isclose(sum(values), 1, rel_tol=0, abs_tol=PROBABILITY_TOLERANCE):
        raise ValueError("probabilities must sum to one within 1e-3")


def snapshot_state(state: DecisionState) -> DecisionState:
    """Copy strict JSON without converting arbitrary objects or object keys to text."""
    if not isinstance(state, (str, dict, list)):
        raise TypeError("state must be a string, JSON object, or JSON array")

    def check(value: Any) -> None:
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            _number(value, "state number", low=-math.inf)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("state object keys must be strings")
                check(item)
        else:
            raise TypeError("state contains a non-JSON value")

    try:
        check(state)
        return cast(DecisionState, json.loads(json.dumps(state, allow_nan=False)))
    except RecursionError:
        raise ValueError("state must be an acyclic JSON value") from None


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    options: Mapping[str, str]

    def __post_init__(self) -> None:
        _text(self.instructions, "instructions")
        _mapping(self.options, "options")
        if len(self.options) < 2:
            raise ValueError("choice requires at least two options")
        for description in self.options.values():
            _text(description, "option description")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True)
class ScoreQuestion:
    instructions: str
    levels: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.instructions, "instructions")
        object.__setattr__(self, "levels", _levels(self.levels))


@dataclass(frozen=True)
class BooleanCriteria:
    true: str | None = None
    false: str | None = None

    def __post_init__(self) -> None:
        for value in (self.true, self.false):
            if value is not None:
                _text(value, "boolean criterion")


@dataclass(frozen=True)
class BooleanQuestion:
    instructions: str
    criteria: BooleanCriteria | None = None

    def __post_init__(self) -> None:
        _text(self.instructions, "instructions")
        if self.criteria is not None and not isinstance(self.criteria, BooleanCriteria):
            raise TypeError("criteria must be BooleanCriteria or None")


Question: TypeAlias = ChoiceQuestion | ScoreQuestion | BooleanQuestion


@dataclass(frozen=True)
class DecisionRequest:
    check_id: str
    version: str
    state: DecisionState
    questions: Mapping[str, Question]

    def __post_init__(self) -> None:
        _text(self.check_id, "check_id")
        _text(self.version, "version")
        _mapping(self.questions, "questions")
        if not self.questions:
            raise ValueError("at least one question is required")
        if any(not isinstance(q, (ChoiceQuestion, ScoreQuestion, BooleanQuestion))
               for q in self.questions.values()):
            raise TypeError("questions must contain typed questions")
        object.__setattr__(self, "state", snapshot_state(self.state))
        object.__setattr__(self, "questions", MappingProxyType(dict(self.questions)))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "check_id": self.check_id, "version": self.version,
                "state": snapshot_state(self.state),
                "questions": {key: _tagged(q) for key, q in self.questions.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionRequest:
        _schema(data)
        try:
            questions = {}
            for key, value in data["questions"].items():
                kind = value["kind"]
                if kind == "choice":
                    question: Question = ChoiceQuestion(value["instructions"], value["options"])
                elif kind == "score":
                    question = ScoreQuestion(value["instructions"], value["levels"])
                elif kind == "boolean":
                    criteria = value.get("criteria")
                    question = BooleanQuestion(value["instructions"],
                        _construct(BooleanCriteria, criteria) if criteria is not None else None)
                else:
                    raise ValueError("unknown question kind")
                questions[key] = question
            return cls(data["check_id"], data["version"], data["state"], questions)
        except (KeyError, AttributeError, TypeError) as exc:
            raise ValueError("invalid decision request encoding") from exc


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float]
    confidence: float | None = None

    def __post_init__(self) -> None:
        _text(self.choice, "choice")
        _mapping(self.probabilities, "probabilities")
        if len(self.probabilities) < 2 or self.choice not in self.probabilities:
            raise ValueError("choice must belong to a distribution with at least two labels")
        _distribution(self.probabilities.values())
        if max(self.probabilities.values()) - self.probabilities[self.choice] > PROBABILITY_TOLERANCE:
            raise ValueError("choice must be a maximum-probability label")
        if self.confidence is not None:
            _number(self.confidence, "confidence", high=1)
        object.__setattr__(self, "probabilities", MappingProxyType(dict(self.probabilities)))


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    probabilities: tuple[float, ...]
    levels: tuple[str, ...]
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", _levels(self.levels))
        if not isinstance(self.probabilities, (list, tuple)) or len(self.probabilities) != len(self.levels):
            raise ValueError("score probabilities must match the ordered levels")
        object.__setattr__(self, "probabilities", tuple(self.probabilities))
        _distribution(self.probabilities)
        _number(self.score, "score", high=len(self.levels) - 1)
        mean = sum(i * p for i, p in enumerate(self.probabilities))
        if abs(mean - self.score) > PROBABILITY_TOLERANCE * max(1, len(self.levels) - 1):
            raise ValueError("score must agree with the distribution's weighted mean")
        if self.confidence is not None:
            _number(self.confidence, "confidence", high=1)


@dataclass(frozen=True)
class BooleanAnswer:
    probability: float

    def __post_init__(self) -> None:
        _number(self.probability, "probability", high=1)

    def __bool__(self) -> bool:
        raise TypeError("compare probability to an application-specific threshold explicitly")


Answer: TypeAlias = ChoiceAnswer | ScoreAnswer | BooleanAnswer


@dataclass(frozen=True)
class DecisionUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_dollars: float | None = None
    incomplete: bool = True

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens):
            if value is not None:
                _integer(value, "token count")
        if self.cost_dollars is not None:
            _number(self.cost_dollars, "cost_dollars")
        if type(self.incomplete) is not bool:
            raise TypeError("incomplete must be a boolean")
        if (self.input_tokens is None or self.output_tokens is None) and not self.incomplete:
            raise ValueError("unknown token usage must be marked incomplete")


@dataclass(frozen=True)
class DecisionError:
    code: str
    message: str
    status_code: int | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        _text(self.code, "code")
        _text(self.message, "message")
        if self.status_code is not None:
            _integer(self.status_code, "status_code", 100)
            if self.status_code > 599:
                raise ValueError("invalid HTTP status")
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be a boolean")


@dataclass(frozen=True)
class DecisionBatch:
    status: Literal["completed", "unavailable"]
    check_id: str
    version: str
    answers: Mapping[str, Answer] = field(default_factory=dict)
    requested_model: str | None = None
    resolved_model: str | None = None
    request_id: str | None = None
    latency_seconds: float = 0
    attempts: int = 0
    usage: DecisionUsage = field(default_factory=DecisionUsage)
    error: DecisionError | None = None

    def __post_init__(self) -> None:
        _text(self.check_id, "check_id")
        _text(self.version, "version")
        for name in ("requested_model", "resolved_model", "request_id"):
            value = getattr(self, name)
            if value is not None:
                _text(value, name)
        _number(self.latency_seconds, "latency_seconds")
        _integer(self.attempts, "attempts")
        _mapping(self.answers, "answers")
        if any(not isinstance(a, (ChoiceAnswer, ScoreAnswer, BooleanAnswer)) for a in self.answers.values()):
            raise TypeError("answers must contain typed answers")
        if not isinstance(self.usage, DecisionUsage):
            raise TypeError("usage must be DecisionUsage")
        if self.status == "completed":
            if not self.answers or self.error is not None:
                raise ValueError("completed batch requires answers and no error")
        elif self.status == "unavailable":
            if self.answers or not isinstance(self.error, DecisionError):
                raise ValueError("unavailable batch requires an error and no answers")
        else:
            raise ValueError("unknown decision status")
        object.__setattr__(self, "answers", MappingProxyType(dict(self.answers)))

    @property
    def choices(self) -> Mapping[str, ChoiceAnswer]:
        return MappingProxyType({k: a for k, a in self.answers.items() if isinstance(a, ChoiceAnswer)})

    @property
    def scores(self) -> Mapping[str, ScoreAnswer]:
        return MappingProxyType({k: a for k, a in self.answers.items() if isinstance(a, ScoreAnswer)})

    @property
    def booleans(self) -> Mapping[str, BooleanAnswer]:
        return MappingProxyType({k: a for k, a in self.answers.items() if isinstance(a, BooleanAnswer)})

    def to_dict(self) -> dict[str, Any]:
        data = _encode(self)
        data["schema_version"] = 1
        data["answers"] = {k: _tagged(a) for k, a in self.answers.items()}
        return cast(dict[str, Any], data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionBatch:
        _schema(data)
        try:
            answers = {}
            answer_types = {"choice": ChoiceAnswer, "score": ScoreAnswer, "boolean": BooleanAnswer}
            for key, value in data["answers"].items():
                if value["kind"] not in answer_types:
                    raise ValueError("unknown answer kind")
                answers[key] = _construct(answer_types[value["kind"]], value)
            values = dict(data)
            values["answers"] = answers
            values["usage"] = _construct(DecisionUsage, data["usage"])
            values["error"] = (_construct(DecisionError, data["error"])
                               if data.get("error") is not None else None)
            return cast(DecisionBatch, _construct(cls, values))
        except (KeyError, AttributeError, TypeError) as exc:
            raise ValueError("invalid decision batch encoding") from exc


def _schema(data: Mapping[str, Any]) -> None:
    if not isinstance(data, Mapping) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported decision schema version")


def _construct(cls: Any, data: Mapping[str, Any]) -> Any:
    if not isinstance(data, Mapping):
        raise TypeError("encoded object must be a mapping")
    return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


def _encode(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    return value


def _tagged(value: Question | Answer) -> dict[str, Any]:
    kind = ("choice" if isinstance(value, (ChoiceQuestion, ChoiceAnswer)) else
            "score" if isinstance(value, (ScoreQuestion, ScoreAnswer)) else "boolean")
    return {"kind": kind, **_encode(value)}
