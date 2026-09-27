"""Correctness and textual matching evaluators for HarnessX."""

from __future__ import annotations

import json
import re
from typing import Any
from langsmith.schemas import Example, Run


def contains_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates whether the agent output contains required substrings.

    Looks for `contains` or `contains_all` (must all be present) and `contains_none` in example.outputs.
    """
    outputs = run.outputs or {}
    text = (outputs.get("output") or "").lower()

    example_outputs = example.outputs or {}
    must_contain = (
        example_outputs.get("contains_all")
        or example_outputs.get("contains")
        or example.metadata.get("contains")
        or []
    )
    if isinstance(must_contain, str):
        must_contain = [must_contain]

    must_not_contain = (
        example_outputs.get("contains_none")
        or example.metadata.get("contains_none")
        or []
    )
    if isinstance(must_not_contain, str):
        must_not_contain = [must_not_contain]

    if not must_contain and not must_not_contain:
        return {
            "key": "text_contains",
            "score": 1.0,
            "comment": "No text content expectations specified",
        }

    missing = [c for c in must_contain if c.lower() not in text]
    present_forbidden = [c for c in must_not_contain if c.lower() in text]

    if not missing and not present_forbidden:
        return {
            "key": "text_contains",
            "score": 1.0,
            "comment": f"All required substrings found: {must_contain}",
        }

    comments = []
    if missing:
        comments.append(f"Missing substrings: {missing}")
    if present_forbidden:
        comments.append(f"Contains forbidden substrings: {present_forbidden}")

    score = 0.0
    if must_contain:
        score = (len(must_contain) - len(missing)) / len(must_contain)
        if present_forbidden:
            score = max(0.0, score - 0.5)

    return {
        "key": "text_contains",
        "score": round(score, 2),
        "comment": "; ".join(comments),
    }


def exact_match_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates whether the agent output exactly matches expected text."""
    outputs = run.outputs or {}
    text = (outputs.get("output") or "").strip()

    example_outputs = example.outputs or {}
    expected = (
        example_outputs.get("expected_output")
        or example_outputs.get("expected")
        or example_outputs.get("answer")
    )

    if expected is None:
        return {
            "key": "exact_match",
            "score": 1.0,
            "comment": "No exact expectation specified",
        }

    expected_clean = str(expected).strip()
    match = (text.lower() == expected_clean.lower())

    return {
        "key": "exact_match",
        "score": 1.0 if match else 0.0,
        "comment": "Exact match" if match else f"Expected '{expected_clean}', got '{text}'",
    }


def regex_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates whether the agent output matches a regular expression pattern."""
    outputs = run.outputs or {}
    text = outputs.get("output") or ""

    example_outputs = example.outputs or {}
    pattern = example_outputs.get("regex_pattern") or example.metadata.get("regex_pattern")

    if not pattern:
        return {
            "key": "regex_match",
            "score": 1.0,
            "comment": "No regex pattern specified",
        }

    matched = bool(re.search(pattern, text, re.MULTILINE | re.DOTALL))
    return {
        "key": "regex_match",
        "score": 1.0 if matched else 0.0,
        "comment": f"Pattern matched: /{pattern}/" if matched else f"Failed to match pattern: /{pattern}/",
    }


def json_valid_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates whether the agent output contains valid JSON and optional required keys."""
    outputs = run.outputs or {}
    text = (outputs.get("output") or "").strip()

    example_outputs = example.outputs or {}
    required_keys = (
        example_outputs.get("json_required_keys")
        or example.metadata.get("json_required_keys")
        or []
    )

    # Attempt to parse json directly or extract from markdown code block
    json_str = text
    if "```json" in text:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
    elif "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)

    try:
        parsed = json.loads(json_str)
    except Exception as exc:
        return {
            "key": "json_valid",
            "score": 0.0,
            "comment": f"Invalid JSON output: {exc}",
        }

    if isinstance(parsed, dict) and required_keys:
        missing_keys = [k for k in required_keys if k not in parsed]
        if missing_keys:
            return {
                "key": "json_valid",
                "score": 0.5,
                "comment": f"Valid JSON but missing keys: {missing_keys}",
            }

    return {
        "key": "json_valid",
        "score": 1.0,
        "comment": "Valid JSON output",
    }
