"""TokenUsage carries reasoning/thinking tokens where the provider reports them.

They are billed at the OUTPUT rate, so a consumer estimating cost from
input + output alone under-reports by the most expensive component. Before
this, the only way to reach them was to dig into ProviderResponse.raw, which
defeats the point of a vendor-agnostic usage type.
"""

from types import SimpleNamespace

import pytest

from datagol_agent_harness.types import TokenUsage


def test_thinking_tokens_defaults_to_zero():
    """A provider that does not report them must not be made to look wrong."""
    assert TokenUsage().thinking_tokens == 0


def test_thinking_tokens_is_part_of_the_usage_type():
    usage = TokenUsage(input_tokens=10, output_tokens=5, thinking_tokens=7)
    assert usage.thinking_tokens == 7


def test_gemini_maps_thoughts_token_count():
    from datagol_agent_harness.providers.gemini import _from_gemini_parts

    usage_metadata = SimpleNamespace(
        prompt_token_count=100,
        candidates_token_count=20,
        cached_content_token_count=0,
        thoughts_token_count=33,
    )
    response = _from_gemini_parts([], None, usage_metadata, raw=None)
    assert response.usage.thinking_tokens == 33
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 20


def test_gemini_without_thought_counts_reports_zero():
    """Older models and some thinking levels omit the field entirely."""
    from datagol_agent_harness.providers.gemini import _from_gemini_parts

    usage_metadata = SimpleNamespace(
        prompt_token_count=10, candidates_token_count=2, cached_content_token_count=0
    )
    response = _from_gemini_parts([], None, usage_metadata, raw=None)
    assert response.usage.thinking_tokens == 0


def test_openai_maps_reasoning_tokens_from_an_object():
    pytest.importorskip("openai")
    from datagol_agent_harness.providers.openai import _usage_from_raw

    raw = SimpleNamespace(
        prompt_tokens=50,
        completion_tokens=10,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=8),
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    assert _usage_from_raw(raw).thinking_tokens == 8


def test_openai_maps_reasoning_tokens_from_a_dict():
    """The streaming path hands back plain dicts."""
    pytest.importorskip("openai")
    from datagol_agent_harness.providers.openai import _usage_from_raw

    raw = {
        "prompt_tokens": 50,
        "completion_tokens": 10,
        "completion_tokens_details": {"reasoning_tokens": 8},
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    assert _usage_from_raw(raw).thinking_tokens == 8


def test_openai_without_reasoning_details_reports_zero():
    pytest.importorskip("openai")
    from datagol_agent_harness.providers.openai import _usage_from_raw

    raw = SimpleNamespace(
        prompt_tokens=5, completion_tokens=1,
        completion_tokens_details=None, prompt_tokens_details=None,
    )
    assert _usage_from_raw(raw).thinking_tokens == 0
