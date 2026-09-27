"""Transient provider failures are retried; everything else is not."""

import asyncio

import pytest

from harnessx.providers.retry import (
    call_with_retry,
    is_transient,
    stream_with_retry,
)
from harnessx.types import StreamChunk


class _Status(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.code = code


# --- what counts as transient -------------------------------------------------


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_rate_limits_and_server_errors_are_transient(code):
    assert is_transient(_Status(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_other_client_errors_are_not_transient(code):
    """A bad request stays bad. Retrying spends a call to get the same error."""
    assert is_transient(_Status(code)) is False


def test_timeouts_and_connection_errors_are_transient():
    assert is_transient(TimeoutError("slow")) is True
    assert is_transient(asyncio.TimeoutError()) is True
    assert is_transient(ConnectionError("reset")) is True


def test_message_text_is_a_fallback_when_there_is_no_status():
    assert is_transient(RuntimeError("RESOURCE_EXHAUSTED: quota")) is True
    assert is_transient(RuntimeError("model is overloaded")) is True
    assert is_transient(RuntimeError("invalid argument")) is False


def test_a_status_attribute_beats_the_message_text():
    """A 400 whose message happens to contain 'timeout' must not retry."""
    assert is_transient(_Status(400)) is False


# --- call_with_retry ----------------------------------------------------------


def test_a_transient_failure_is_retried_once_and_succeeds():
    async def _run():
        calls = []

        async def op():
            calls.append(1)
            if len(calls) == 1:
                raise _Status(503)
            return "ok"

        assert await call_with_retry(op, backoff_seconds=0) == "ok"
        assert len(calls) == 2
    asyncio.run(_run())


def test_a_deterministic_failure_is_not_retried():
    async def _run():
        calls = []

        async def op():
            calls.append(1)
            raise _Status(400)

        with pytest.raises(_Status):
            await call_with_retry(op, backoff_seconds=0)
        assert len(calls) == 1
    asyncio.run(_run())


def test_attempts_are_bounded():
    async def _run():
        calls = []

        async def op():
            calls.append(1)
            raise _Status(500)

        with pytest.raises(_Status):
            await call_with_retry(op, attempts=3, backoff_seconds=0)
        assert len(calls) == 3
    asyncio.run(_run())


def test_success_on_the_first_try_calls_once():
    async def _run():
        calls = []

        async def op():
            calls.append(1)
            return "fine"

        assert await call_with_retry(op, backoff_seconds=0) == "fine"
        assert len(calls) == 1
    asyncio.run(_run())


# --- stream_with_retry --------------------------------------------------------


def test_a_stream_failing_before_the_first_chunk_is_retried():
    async def _run():
        opened = []

        def open_stream():
            opened.append(1)

            async def gen():
                if len(opened) == 1:
                    raise _Status(429)
                yield StreamChunk(kind="text_delta", data="hello")

            return gen()

        chunks = [c async for c in stream_with_retry(open_stream, backoff_seconds=0)]
        assert [c.data for c in chunks] == ["hello"]
        assert len(opened) == 2
    asyncio.run(_run())


def test_a_stream_failing_AFTER_delivering_is_not_retried():
    async def _run():
        """The caller already has output; a retry would duplicate it."""
        opened = []

        def open_stream():
            opened.append(1)

            async def gen():
                yield StreamChunk(kind="text_delta", data="partial")
                raise _Status(503)

            return gen()

        received = []
        with pytest.raises(_Status):
            async for chunk in stream_with_retry(open_stream, backoff_seconds=0):
                received.append(chunk.data)

        assert received == ["partial"]
        assert len(opened) == 1, "must not re-open after delivering output"
    asyncio.run(_run())


def test_a_deterministic_stream_failure_is_not_retried():
    async def _run():
        opened = []

        def open_stream():
            opened.append(1)

            async def gen():
                raise _Status(400)
                yield  # pragma: no cover

            return gen()

        with pytest.raises(_Status):
            async for _ in stream_with_retry(open_stream, backoff_seconds=0):
                pass
        assert len(opened) == 1
    asyncio.run(_run())


def test_an_empty_stream_ends_cleanly():
    async def _run():
        def open_stream():
            async def gen():
                return
                yield  # pragma: no cover

            return gen()

        assert [c async for c in stream_with_retry(open_stream, backoff_seconds=0)] == []
    asyncio.run(_run())


def test_every_chunk_is_forwarded_in_order():
    async def _run():
        def open_stream():
            async def gen():
                for part in ("a", "b", "c"):
                    yield StreamChunk(kind="text_delta", data=part)

            return gen()

        chunks = [c async for c in stream_with_retry(open_stream, backoff_seconds=0)]
        assert [c.data for c in chunks] == ["a", "b", "c"]
    asyncio.run(_run())
