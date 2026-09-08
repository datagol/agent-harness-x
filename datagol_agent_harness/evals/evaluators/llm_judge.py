"""Model-graded qualitative evaluator (LLM-as-a-judge) for DataGOL Agent Harness."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from langsmith.schemas import Example, Run

from ...providers import make_provider

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_PROMPT = """You are an expert evaluator grading an LLM agent's response.

[User Prompt]
{prompt}

[Expected Criteria / Ground Truth]
{reference}

[Actual Agent Response]
{response}

Evaluate the response on:
1. Accuracy & Correctness: Does the response satisfy the user's request and match the criteria?
2. Groundedness: Are the stated facts accurate without hallucinations?
3. Clarity & Conciseness: Is the response well-structured, clear, and direct?

Respond strictly with a JSON object in this format:
```json
{{
  "score": <integer from 1 to 5, where 5 is excellent and 1 is completely wrong>,
  "reasoning": "<1-2 sentence explanation of your score>"
}}
```
"""


class LLMJudgeEvaluator:
    """Evaluates agent responses using an LLM judge."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        provider: str = "anthropic",
        prompt_template: str = DEFAULT_JUDGE_PROMPT,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.provider_name = provider
        self.prompt_template = prompt_template
        self.temperature = temperature
        self._provider = None

    def _get_provider(self):
        if self._provider is None:
            self._provider = make_provider(self.provider_name)
        return self._provider

    async def __call__(self, run: Run, example: Example) -> dict[str, Any]:
        prompt = (example.inputs or {}).get("prompt") or ""
        expected = (
            (example.outputs or {}).get("expected_output")
            or (example.outputs or {}).get("criteria")
            or "Satisfy the prompt accurately and concisely."
        )
        response = (run.outputs or {}).get("output") or ""

        formatted_prompt = self.prompt_template.format(
            prompt=prompt,
            reference=expected,
            response=response,
        )

        try:
            prov = self._get_provider()
            resp = await prov.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
                system="You are an objective AI evaluation judge.",
                tools=[],
                max_tokens=500,
                temperature=self.temperature,
            )

            # Extract response text
            judge_text = ""
            for block in getattr(resp, "content", []):
                if getattr(block, "type", "") == "text":
                    judge_text += getattr(block, "text", "")

            # Parse JSON
            match = re.search(r"\{.*?\}", judge_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                raw_score = int(parsed.get("score", 3))
                reasoning = parsed.get("reasoning", "")
                # Normalize 1-5 scale to 0.0-1.0
                normalized = max(0.0, min(1.0, (raw_score - 1) / 4.0))
                return {
                    "key": "llm_judge_score",
                    "score": round(normalized, 2),
                    "comment": f"[{raw_score}/5] {reasoning}",
                }
        except Exception as exc:
            logger.debug("LLM judge evaluation failed: %s", exc)

        return {
            "key": "llm_judge_score",
            "score": 0.5,
            "comment": "LLM judge unavailable or failed to score",
        }
