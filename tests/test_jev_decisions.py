"""Contract tests through the real TypeSafe SDK using its httpx2 transport."""

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json

import pytest

sdk = pytest.importorskip("typesafe_sdk")
httpx = pytest.importorskip("httpx2")

from harnessx.decisions import (
    BooleanCriteria, BooleanQuestion, ChoiceQuestion, DecisionRequest, JevDecisionProvider, ScoreQuestion,
)


def request():
    return DecisionRequest("review", "v1", {"nested": ["original"]}, {
        "route": ChoiceQuestion("Pick a route", {"sql": "Structured data", "general": "Other"}),
        "coverage": ScoreQuestion("Coverage", ("None", "Full")),
        "support": BooleanQuestion("Supported?", BooleanCriteria("Evidence supports every claim", "Unsupported claims")),
        "simple": BooleanQuestion("Relevant?"),
    })


def payload():
    return {"model": "jev-1.13.0-resolved", "usage": {"input_tokens": 12, "output_tokens": 8}, "answers": {
        "route": {"type": "choice", "choice": "sql", "confidence": .9, "probabilities": {"sql": .8, "general": .2}},
        "coverage": {"type": "score", "score": .75, "confidence": .8, "legend": {"0": "None", "1": "Full"}, "probabilities": {"0": .25, "1": .75}},
        "support": {"type": "noul", "noul": .95},
        "simple": {"type": "noul", "noul": .9},
    }}


def response(data=None, status=200, headers=None):
    return httpx.Response(status, json=payload() if data is None else data,
                          headers={"x-typesafe-request-id": "request-123", **(headers or {})})


def client(handler):
    return sdk.AsyncTypeSafeClient(api_key="test-only-key", model="borrowed-default",
                                  retry=sdk.RetryPolicy(max_retries=7), transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_real_sdk_translation_and_metadata():
    captured = []

    def handler(req):
        captured.append(req)
        return response()

    async with client(handler) as borrowed:
        async with JevDecisionProvider(client=borrowed) as provider:
            result = await provider.evaluate(request())
        assert not borrowed._http_client.is_closed
    assert result.status == "completed"
    assert result.request_id == "request-123"
    assert result.requested_model == "jev-1.13.0"
    assert result.resolved_model == "jev-1.13.0-resolved"
    assert result.choices["route"].choice == "sql"
    assert result.scores["coverage"].score == .75
    assert result.booleans["support"].probability == .95
    assert result.usage.input_tokens == 12
    assert result.usage.cost_dollars is None and not result.usage.incomplete
    assert result.attempts == 1 and result.latency_seconds >= 0
    body = json.loads(captured[0].content)
    assert body["model"] == "jev-1.13.0"
    assert body["state"] == {"nested": ["original"]}
    assert body["questions"]["route"]["type"] == "choice"
    assert body["questions"]["route"]["criteria"] == {"sql": "Structured data", "general": "Other"}
    assert body["questions"]["coverage"]["criteria"] == ["None", "Full"]
    assert body["questions"]["support"]["type"] == "noul"
    assert body["questions"]["support"]["criteria"]["true"] == "Evidence supports every claim"
    assert "criteria" not in body["questions"]["simple"]
    assert 0 < captured[0].extensions["timeout"]["read"] <= 2
    assert borrowed._config.default_model == "borrowed-default"


@pytest.mark.asyncio
@pytest.mark.parametrize("mutate", [
    lambda d: d["answers"].pop("route"),
    lambda d: d["answers"]["route"].update(type="noul", noul=.9),
    lambda d: d["answers"]["route"].update(type="future"),
    lambda d: d["answers"]["route"].update(choice="general"),
    lambda d: d["answers"]["route"].update(probabilities={"sql": .8, "other": .2}),
    lambda d: d["answers"]["route"]["probabilities"].update(general=.7),
    lambda d: d["answers"]["route"].pop("confidence"),
    lambda d: d["answers"]["route"].update(confidence=1.1),
    lambda d: d["answers"]["coverage"].update(score=.1),
    lambda d: d["answers"]["coverage"].update(legend={"0": "Full", "1": "None"}),
    lambda d: d["answers"]["coverage"]["probabilities"].update({"2": 0}),
    lambda d: d["answers"]["support"].update(noul=-.5),
    lambda d: d["usage"].update(input_tokens=-1),
    lambda d: d.update(model=""),
])
async def test_invalid_answers_unavailable_without_partial_results(mutate):
    data = payload()
    mutate(data)
    async with client(lambda req: response(data)) as borrowed:
        async with JevDecisionProvider(client=borrowed) as provider:
            result = await provider.evaluate(request())
    assert result.status == "unavailable"
    assert result.error.code == "invalid_response"
    assert result.attempts == 1
    assert not result.answers and not result.choices and not result.scores and not result.booleans


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"x-typesafe-request-id": ""}, {"x-typesafe-request-id": " "}])
async def test_absent_usage_and_request_id_stay_unknown_and_extra_answers_ignored(headers):
    data = payload()
    data["usage"] = {}
    data["answers"]["unsolicited"] = {"type": "noul", "noul": .2}
    async with client(lambda req: httpx.Response(200, json=data, headers=headers)) as borrowed:
        async with JevDecisionProvider(client=borrowed) as provider:
            result = await provider.evaluate(request())
    assert result.status == "completed"
    assert result.request_id is None
    assert result.usage.input_tokens is None and result.usage.output_tokens is None
    assert result.usage.incomplete
    assert "unsolicited" not in result.answers


@pytest.mark.asyncio
@pytest.mark.parametrize("status,code,retry", [
    (400, "invalid_request", False), (401, "authentication", False), (403, "forbidden", False),
    (422, "invalid_request", False), (429, "rate_limited", True), (500, "server", True),
    (502, "server", True), (503, "server", True), (504, "server", True), (501, "server", False),
    (418, "unexpected_http", False),
])
async def test_http_failures_retries_and_sanitized_diagnostics(status, code, retry):
    calls = []

    def handler(req):
        calls.append(req)
        return response({"message": "SECRET request state and api-key"}, status, {"retry-after": "0"})

    async with client(handler) as borrowed:
        async with JevDecisionProvider(client=borrowed) as provider:
            result = await provider.evaluate(request())
    assert len(calls) == result.attempts == (2 if retry else 1)
    assert result.error.code == code and result.error.retryable is retry
    assert "SECRET" not in json.dumps(result.to_dict())
    assert result.usage.input_tokens is None and result.usage.incomplete


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connection", "timeout", "http"])
async def test_transient_retry_success_retains_usage_uncertainty(failure):
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            if failure == "connection":
                raise httpx.ConnectError("secret", request=req)
            if failure == "timeout":
                raise httpx.ReadTimeout("secret", request=req)
            return response({}, 503, {"retry-after": "0"})
        return response()

    async with client(handler) as borrowed:
        async with JevDecisionProvider(client=borrowed) as provider:
            result = await provider.evaluate(request())
    assert result.status == "completed" and result.attempts == 2
    assert result.usage.input_tokens == 12 and result.usage.incomplete


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [
    {"retry-after": "100"}, {"retry-after-ms": "100000"},
    {"retry-after": format_datetime(datetime.now(timezone.utc) + timedelta(hours=1))},
])
async def test_retry_after_larger_than_budget_does_not_dispatch(headers):
    calls = []

    def handler(req):
        calls.append(req)
        return response({}, 429, headers)

    async with client(handler) as borrowed:
        async with JevDecisionProvider(client=borrowed) as provider:
            result = await provider.evaluate(request())
    assert result.attempts == len(calls) == 1 and result.error.code == "rate_limited"


@pytest.mark.asyncio
async def test_retry_after_delay_and_per_call_attempt_cap():
    times = []

    def handler(req):
        times.append(asyncio.get_running_loop().time())
        return response({}, 503, {"retry-after": "0.03"})

    async with client(handler) as borrowed:
        async with JevDecisionProvider(client=borrowed, max_retries=5) as provider:
            result = await provider.evaluate(request(), max_attempts=2)
            assert result.attempts == 2 and times[1] - times[0] >= .025
            result = await provider.evaluate(request(), max_attempts=1)
            assert result.attempts == 1


@pytest.mark.asyncio
async def test_deadline_cancels_transport_and_external_cancellation_propagates():
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(req):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async with client(handler) as borrowed:
        async with JevDecisionProvider(client=borrowed) as provider:
            result = await provider.evaluate(request(), timeout_seconds=.03)
            assert result.status == "unavailable" and result.error.code == "timeout"
            assert result.attempts == 1 and cancelled.is_set()
            entered.clear()
            task = asyncio.create_task(provider.evaluate(request()))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not provider._tasks


@pytest.mark.asyncio
async def test_deadline_spans_retry_delay_and_second_request():
    timeouts = []

    async def handler(req):
        timeouts.append(req.extensions["timeout"]["read"])
        if len(timeouts) == 1:
            return response({}, 503, {"retry-after": "0.01"})
        await asyncio.Event().wait()

    async with client(handler) as borrowed:
        async with JevDecisionProvider(client=borrowed, timeout_seconds=.08) as provider:
            result = await provider.evaluate(request(), timeout_seconds=5)
    assert result.error.code == "timeout" and result.attempts == 2
    assert 0 < timeouts[1] < timeouts[0] <= .08


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [b'not json', b'null', b'{"model":"x"}',
    b'{"model":"x","usage":{},"answers":{"route":{"type":"noul","noul":NaN}}}',
])
async def test_malformed_wire_responses(content):
    async with client(lambda req: httpx.Response(200, content=content)) as borrowed:
        async with JevDecisionProvider(client=borrowed) as provider:
            result = await provider.evaluate(request())
    assert result.status == "unavailable" and result.error.code == "invalid_response"


@pytest.mark.asyncio
async def test_close_cancels_pending_and_leaves_borrowed_client_open():
    entered = asyncio.Event()

    async def handler(req):
        entered.set()
        await asyncio.Event().wait()

    async with client(handler) as borrowed:
        provider = JevDecisionProvider(client=borrowed)
        task = asyncio.create_task(provider.evaluate(request()))
        await entered.wait()
        await asyncio.gather(provider.aclose(), provider.aclose())
        assert task.cancelled() and not borrowed._http_client.is_closed
        with pytest.raises(RuntimeError, match="closed"):
            await provider.evaluate(request())


@pytest.mark.asyncio
async def test_cancelled_caller_can_close_provider_in_finally_without_deadlock():
    entered = asyncio.Event()

    async def handler(req):
        entered.set()
        await asyncio.Event().wait()

    async with client(handler) as borrowed:
        provider = JevDecisionProvider(client=borrowed)

        async def caller():
            try:
                await provider.evaluate(request())
            finally:
                await provider.aclose()

        task = asyncio.create_task(caller())
        await entered.wait()
        await asyncio.wait_for(provider.aclose(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_owned_client_uses_explicit_defaults_and_closes(monkeypatch):
    created = []
    original = sdk.AsyncTypeSafeClient

    def factory(**kwargs):
        instance = original(**kwargs, transport=httpx.MockTransport(lambda req: response()))
        created.append(instance)
        return instance

    monkeypatch.setattr(sdk, "AsyncTypeSafeClient", factory)
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://unintended.example")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "unintended")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-owned-key")
    async with JevDecisionProvider() as provider:
        assert "test-owned-key" not in repr(provider)
        assert created[0]._config.base_url == "https://api.typesafe.ai"
        result = await provider.evaluate(request())
        assert result.requested_model == "jev-1.13.0"
    await provider.aclose()
    assert created[0]._http_client.is_closed


@pytest.mark.asyncio
async def test_request_snapshot_survives_mutation_during_retry():
    submitted = request()
    seen = []

    def handler(req):
        seen.append(json.loads(req.content)["state"])
        if len(seen) == 1:
            submitted.state["nested"].append("changed")
            return response({}, 503, {"retry-after": "0"})
        return response()

    async with client(handler) as borrowed:
        async with JevDecisionProvider(client=borrowed) as provider:
            assert (await provider.evaluate(submitted)).status == "completed"
    assert seen == [{"nested": ["original"]}] * 2


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_share_state():
    async def handler(req):
        state = json.loads(req.content)["state"]
        await asyncio.sleep(.001)
        return response(headers={"x-typesafe-request-id": state})

    async with client(handler) as borrowed:
        async with JevDecisionProvider(client=borrowed) as provider:
            original = request()
            results = await asyncio.gather(*(provider.evaluate(DecisionRequest(str(i), "v1", str(i), original.questions)) for i in range(10)))
    assert [r.request_id for r in results] == [str(i) for i in range(10)]


@pytest.mark.parametrize("kwargs", [
    {"timeout_seconds": True}, {"timeout_seconds": float("inf")}, {"timeout_seconds": 0},
    {"max_retries": -1}, {"max_retries": True}, {"max_state_bytes": 0}, {"model": ""},
    {"base_url": "bad"}, {"api_key": ""}, {"api_key": "bad key"}, {"client": object()},
    {"client": object(), "api_key": "key"},
])
def test_invalid_configuration_before_io(kwargs):
    with pytest.raises((ValueError, TypeError)):
        JevDecisionProvider(**kwargs)


@pytest.mark.asyncio
async def test_invalid_call_arguments_and_large_state_before_io():
    def forbidden(req):
        pytest.fail("must validate before I/O")

    async with client(forbidden) as borrowed:
        async with JevDecisionProvider(client=borrowed, max_state_bytes=2) as provider:
            for kwargs in ({}, {"timeout_seconds": True}, {"timeout_seconds": -1}, {"max_attempts": 0}):
                with pytest.raises((ValueError, TypeError)):
                    await provider.evaluate(request(), **kwargs)


@pytest.mark.asyncio
async def test_programming_errors_are_not_disguised():
    def handler(req):
        raise RuntimeError("adapter defect")

    async with client(handler) as borrowed:
        async with JevDecisionProvider(client=borrowed) as provider:
            with pytest.raises(RuntimeError, match="adapter defect"):
                await provider.evaluate(request())
