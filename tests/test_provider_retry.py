"""What counts as a transient provider failure. The retry loop itself lives in
the engine; see tests/test_retry_seams.py."""

import asyncio

import pytest

from harnessx.providers.retry import is_transient


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
