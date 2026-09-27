"""TypeSafe adapter. Vendor imports happen only when constructing a provider."""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

from .base import DecisionProvider
from .types import (
    Answer, BooleanAnswer, BooleanQuestion, ChoiceAnswer, ChoiceQuestion, DecisionBatch,
    DecisionError, DecisionRequest, DecisionUsage, ScoreAnswer, ScoreQuestion,
    _integer, _number, _text,
)


class JevDecisionProvider(DecisionProvider):
    """Evaluate with Jev under one deadline, including retry delays.

    Injected AsyncTypeSafeClient instances are borrowed. Reported token counts cover
    the last successful response; after retries, ``usage.incomplete`` is true because
    earlier attempts may also have consumed tokens. Cost is never estimated.
    """

    def __init__(
        self, *, model: str = "jev-1.13.0", api_key: str | None = None,
        base_url: str | None = None, timeout_seconds: float = 2.0,
        max_retries: int = 1, max_state_bytes: int = 16_384, client: object | None = None,
    ) -> None:
        _text(model, "model")
        _positive_timeout(timeout_seconds)
        _integer(max_retries, "max_retries")
        _integer(max_state_bytes, "max_state_bytes", 1)
        if client is not None and (api_key is not None or base_url is not None):
            raise ValueError("client cannot be combined with api_key or base_url")
        if base_url is not None:
            _text(base_url, "base_url")
            parsed = urlsplit(base_url)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError("base_url must be an HTTP(S) API root without credentials or query")
        try:
            self._sdk = importlib.import_module("typesafe_sdk")
        except ModuleNotFoundError as exc:
            if exc.name != "typesafe_sdk":
                raise
            raise ImportError("Jev requires the optional SDK: pip install 'harnessx[jev]'") from None
        self._owns_client = client is None
        if client is None:
            key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")
            if not isinstance(key, str):
                raise TypeError("api_key must be a string")
            key = key.strip()
            if not key or not key.isascii() or not key.isprintable() or " " in key:
                raise ValueError("provide a valid api_key or set TYPESAFE_API_KEY")
            client = self._sdk.AsyncTypeSafeClient(
                api_key=key, model=model, base_url=base_url or "https://api.typesafe.ai",
                timeout=timeout_seconds, retry=self._sdk.RetryPolicy(max_retries=0),
            )
        elif not isinstance(client, self._sdk.AsyncTypeSafeClient):
            raise TypeError("client must be an AsyncTypeSafeClient")
        self._client: Any = client
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_state_bytes = max_state_bytes
        self._closed = False
        self._tasks: set[asyncio.Task[Any]] = set()
        self._close_task: asyncio.Task[None] | None = None

    async def evaluate(
        self, request: DecisionRequest, *, timeout_seconds: float | None = None,
        max_attempts: int | None = None,
    ) -> DecisionBatch:
        if self._closed:
            raise RuntimeError("decision provider is closed")
        if not isinstance(request, DecisionRequest):
            raise TypeError("request must be DecisionRequest")
        timeout = self.timeout_seconds
        if timeout_seconds is not None:
            _positive_timeout(timeout_seconds)
            timeout = min(timeout, timeout_seconds)
        attempts = self.max_retries + 1
        if max_attempts is not None:
            _integer(max_attempts, "max_attempts", 1)
            attempts = min(attempts, max_attempts)
        # Snapshot before the first await, including mutations made after construction.
        submitted = DecisionRequest(request.check_id, request.version, request.state, request.questions)
        encoded = json.dumps(submitted.state, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > self.max_state_bytes:
            raise ValueError("decision state exceeds max_state_bytes")
        questions = self._questions(submitted)
        # Track only the evaluation, never the caller's entire task: caller cleanup
        # may itself await aclose() after cancellation.
        task = asyncio.create_task(self._evaluate(submitted, questions, timeout, attempts))
        self._tasks.add(task)
        try:
            return await task
        finally:
            self._tasks.discard(task)

    def _questions(self, request: DecisionRequest) -> dict[str, Any]:
        translated: dict[str, Any] = {}
        for key, question in request.questions.items():
            if isinstance(question, ChoiceQuestion):
                translated[key] = self._sdk.Choice(instructions=question.instructions, criteria=dict(question.options))
            elif isinstance(question, ScoreQuestion):
                translated[key] = self._sdk.Score(instructions=question.instructions, criteria=list(question.levels))
            else:
                criteria = None
                if question.criteria is not None:
                    criteria = {"true": question.criteria.true, "false": question.criteria.false}
                translated[key] = self._sdk.Noul(instructions=question.instructions, criteria=criteria)
        return translated

    async def _evaluate(
        self, request: DecisionRequest, questions: dict[str, Any], timeout: float, limit: int,
    ) -> DecisionBatch:
        loop = asyncio.get_running_loop()
        start = loop.time()
        deadline = start + timeout
        attempts = 0
        request_id: str | None = None
        resolved_model: str | None = None
        usage = DecisionUsage()
        error = DecisionError("timeout", "Decision deadline exceeded.", retryable=True)
        try:
            async with asyncio.timeout_at(deadline):
                for attempt in range(limit):
                    attempts += 1
                    try:
                        response = await self._client.system_one(
                            model=self.model, state=request.state, questions=questions,
                            retry=self._sdk.RetryPolicy(max_retries=0),
                            timeout=max(deadline - loop.time(), 1e-9),
                        )
                    except self._sdk.TypeSafeAPIResponseValidationError as exc:
                        request_id = _request_id(exc.headers)
                        error = DecisionError("invalid_response", "Service returned an invalid decision response.", exc.status)
                        break
                    except self._sdk.TypeSafeAPIError as exc:
                        request_id = _request_id(exc.headers)
                        error = _http_error(exc.status)
                        delay = _retry_delay(exc.headers, attempt)
                    except self._sdk.TypeSafeAPITimeoutError:
                        error = DecisionError("timeout", "Decision request timed out.", retryable=True)
                        delay = _retry_delay({}, attempt)
                    except self._sdk.TypeSafeAPIConnectionError:
                        error = DecisionError("connection", "Could not reach the decision service.", retryable=True)
                        delay = _retry_delay({}, attempt)
                    else:
                        # request_id raises in the vendor SDK when the header is absent.
                        request_id = _request_id(response.raw_http_response.headers)
                        try:
                            _text(response.model, "resolved model")
                            resolved_model = response.model
                            usage = DecisionUsage(
                                input_tokens=response.usage.input_tokens,
                                output_tokens=response.usage.output_tokens,
                                incomplete=(attempts > 1 or response.usage.input_tokens is None
                                            or response.usage.output_tokens is None),
                            )
                            answers = self._answers(request, response.answers)
                        except (ValueError, TypeError):
                            error = DecisionError("invalid_response", "Service returned inconsistent decision answers.")
                            break
                        return DecisionBatch(
                            status="completed", check_id=request.check_id, version=request.version,
                            answers=answers, requested_model=self.model, resolved_model=resolved_model,
                            request_id=request_id, latency_seconds=loop.time() - start,
                            attempts=attempts, usage=usage,
                        )
                    if not error.retryable or attempts >= limit or delay >= deadline - loop.time():
                        break
                    await asyncio.sleep(delay)
        except TimeoutError:
            error = DecisionError("timeout", "Decision deadline exceeded.", retryable=True)
        return DecisionBatch(
            status="unavailable", check_id=request.check_id, version=request.version,
            requested_model=self.model, resolved_model=resolved_model, request_id=request_id,
            latency_seconds=loop.time() - start, attempts=attempts, usage=usage, error=error,
        )

    def _answers(self, request: DecisionRequest, received: Mapping[str, Any]) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        for key, question in request.questions.items():
            raw = received.get(key)
            if isinstance(question, ChoiceQuestion) and isinstance(raw, self._sdk.ChoiceAnswer):
                if set(raw.probabilities) != set(question.options) or raw.confidence is None:
                    raise ValueError("choice does not match the submitted options")
                answer: Answer = ChoiceAnswer(raw.choice, raw.probabilities, raw.confidence)
            elif isinstance(question, ScoreQuestion) and isinstance(raw, self._sdk.ScoreAnswer):
                indices = set(range(len(question.levels)))
                if (set(raw.probabilities) != indices or set(raw.legend) != indices
                        or tuple(raw.legend[i] for i in range(len(indices))) != question.levels
                        or raw.confidence is None):
                    raise ValueError("score does not match the submitted rubric")
                answer = ScoreAnswer(raw.score, tuple(raw.probabilities[i] for i in range(len(indices))),
                                     question.levels, raw.confidence)
            elif isinstance(question, BooleanQuestion) and isinstance(raw, self._sdk.NoulAnswer):
                answer = BooleanAnswer(raw.noul)
            else:
                raise ValueError("requested answer is missing or has an incompatible type")
            answers[key] = answer
        return answers

    async def aclose(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())
        # A cancelled closer must not interrupt cleanup or leave half-closed resources.
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        pending = tuple(self._tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self._owns_client:
            await self._client.aclose()


def _positive_timeout(value: float) -> None:
    _number(value, "timeout_seconds")
    if value <= 0:
        raise ValueError("timeout_seconds must be positive")


def _request_id(headers: Mapping[str, str]) -> str | None:
    value = headers.get("x-typesafe-request-id")
    return value if value and value.strip() else None


def _http_error(status: int) -> DecisionError:
    code = ({401: "authentication", 403: "forbidden", 429: "rate_limited"}.get(status)
            or ("invalid_request" if status in {400, 404, 405, 422} else
                "server" if status >= 500 else "unexpected_http"))
    messages = {
        "authentication": "Decision service authentication failed.",
        "forbidden": "Decision service denied access.",
        "invalid_request": "Decision service rejected the request.",
        "rate_limited": "Decision service rate limit reached.",
        "server": "Decision service failed.",
        "unexpected_http": "Decision service returned an unexpected HTTP status.",
    }
    return DecisionError(code, messages[code], status, status in {429, 500, 502, 503, 504})


def _retry_delay(headers: Mapping[str, str], attempt: int) -> float:
    for key, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
        value = headers.get(key)
        if value is None:
            continue
        try:
            delay = float(value) / divisor
        except ValueError:
            try:
                delay = parsedate_to_datetime(value).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                continue
        if math.isfinite(delay):
            return max(0, delay)
    return min(0.1 * 2 ** min(attempt, 5), 2.0)
