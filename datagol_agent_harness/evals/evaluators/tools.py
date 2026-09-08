"""Tool execution evaluators for DataGOL Agent Harness."""

from __future__ import annotations

from typing import Any, Sequence
from langsmith.schemas import Example, Run


def tool_selection_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates whether the agent selected the expected tools.

    Looks for `expected_tools` and `forbidden_tools` in example.outputs or example.metadata.
    """
    outputs = run.outputs or {}
    called_tools: list[str] = outputs.get("tools_used", [])
    if not called_tools and "tool_calls" in outputs:
        called_tools = [tc["name"] for tc in outputs["tool_calls"] if "name" in tc]

    example_outputs = example.outputs or {}
    expected_tools = (
        example_outputs.get("expected_tools")
        or example.metadata.get("expected_tools")
        or []
    )
    forbidden_tools = (
        example_outputs.get("forbidden_tools")
        or example.metadata.get("forbidden_tools")
        or []
    )

    if not expected_tools and not forbidden_tools:
        # No constraints specified
        return {
            "key": "tool_selection",
            "score": 1.0,
            "comment": "No tool expectations specified",
        }

    called_set = set(called_tools)
    expected_set = set(expected_tools)
    forbidden_set = set(forbidden_tools)

    missing = expected_set - called_set
    called_forbidden = called_set & forbidden_set

    if not missing and not called_forbidden:
        return {
            "key": "tool_selection",
            "score": 1.0,
            "comment": f"Correctly called expected tools: {list(expected_set)}",
        }

    reasons: list[str] = []
    if missing:
        reasons.append(f"Missing expected tools: {list(missing)}")
    if called_forbidden:
        reasons.append(f"Called forbidden tools: {list(called_forbidden)}")

    score = 0.0
    if expected_set:
        matched = len(expected_set - missing)
        score = max(0.0, matched / len(expected_set) - (0.5 if called_forbidden else 0.0))

    return {
        "key": "tool_selection",
        "score": round(score, 2),
        "comment": "; ".join(reasons),
    }


def no_tool_errors_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates whether all executed tools succeeded without errors."""
    outputs = run.outputs or {}
    tool_calls = outputs.get("tool_calls", [])

    if not tool_calls:
        return {
            "key": "no_tool_errors",
            "score": 1.0,
            "comment": "No tools called",
        }

    errored = [tc for tc in tool_calls if tc.get("is_error")]
    if not errored:
        return {
            "key": "no_tool_errors",
            "score": 1.0,
            "comment": f"All {len(tool_calls)} tool calls succeeded",
        }

    err_names = [tc.get("name", "unknown") for tc in errored]
    return {
        "key": "no_tool_errors",
        "score": 0.0,
        "comment": f"{len(errored)} tool call(s) failed with errors: {err_names}",
    }


def tool_args_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates whether tool call arguments match expected arguments.

    Looks for `expected_args` in example.outputs, formatted as:
        {"tool_name": {"arg_key": "expected_val", ...}}
    """
    example_outputs = example.outputs or {}
    expected_args = example_outputs.get("expected_args") or example.metadata.get("expected_args")

    if not expected_args:
        return {
            "key": "tool_args_match",
            "score": 1.0,
            "comment": "No argument expectations specified",
        }

    outputs = run.outputs or {}
    tool_calls = outputs.get("tool_calls", [])

    matched_tools: list[str] = []
    mismatches: list[str] = []

    for tool_name, expected_params in expected_args.items():
        # Find matching tool calls
        matching_calls = [tc for tc in tool_calls if tc.get("name") == tool_name]
        if not matching_calls:
            mismatches.append(f"Tool '{tool_name}' was never called")
            continue

        tool_matched = False
        for call in matching_calls:
            actual_input = call.get("input", {})
            if isinstance(actual_input, dict):
                matches_all = True
                for k, expected_v in expected_params.items():
                    actual_v = actual_input.get(k)
                    if isinstance(expected_v, str) and isinstance(actual_v, str):
                        if expected_v.lower() != actual_v.lower():
                            matches_all = False
                            break
                    elif actual_v != expected_v:
                        matches_all = False
                        break
                if matches_all:
                    tool_matched = True
                    break

        if tool_matched:
            matched_tools.append(tool_name)
        else:
            mismatches.append(f"Tool '{tool_name}' arguments did not match {expected_params}")

    if not mismatches:
        return {
            "key": "tool_args_match",
            "score": 1.0,
            "comment": f"All expected arguments matched for: {matched_tools}",
        }

    score = len(matched_tools) / len(expected_args) if expected_args else 0.0
    return {
        "key": "tool_args_match",
        "score": round(score, 2),
        "comment": "; ".join(mismatches),
    }


def tool_call_count_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Evaluates whether the number of tool calls falls within expected bounds."""
    example_outputs = example.outputs or {}
    min_calls = example_outputs.get("min_tool_calls", 0)
    max_calls = example_outputs.get("max_tool_calls")

    outputs = run.outputs or {}
    tool_calls = outputs.get("tool_calls", [])
    count = len(tool_calls)

    if count < min_calls:
        return {
            "key": "tool_call_count",
            "score": 0.0,
            "comment": f"Too few tool calls: {count} < min {min_calls}",
        }

    if max_calls is not None and count > max_calls:
        return {
            "key": "tool_call_count",
            "score": 0.0,
            "comment": f"Too many tool calls: {count} > max {max_calls}",
        }

    return {
        "key": "tool_call_count",
        "score": 1.0,
        "comment": f"Tool call count ({count}) within bounds",
    }
