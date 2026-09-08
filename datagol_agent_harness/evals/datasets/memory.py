"""Memory retention and multi-turn context benchmark dataset."""

from __future__ import annotations

from langsmith.schemas import Example
from .loader import build_example


def get_memory_dataset() -> list[Example]:
    """Returns benchmark test cases for context recall and memory preservation."""
    return [
        build_example(
            inputs={
                "messages": [
                    {"role": "user", "content": "My secret security passphrase is 'Falcon-Blue-99'."},
                    {"role": "assistant", "content": "Understood, I have noted your passphrase."},
                    {"role": "user", "content": "What was the security passphrase I just told you?"},
                ]
            },
            outputs={
                "contains_all": ["Falcon-Blue-99"],
                "max_allowed_iterations": 2,
            },
            metadata={"category": "short_term_retention"},
        ),
        build_example(
            inputs={
                "messages": [
                    {"role": "user", "content": "I am allergic to peanuts, tree nuts, and shellfish."},
                    {"role": "assistant", "content": "Got it. I will keep your allergies in mind."},
                    {"role": "user", "content": "Can you recommend whether pad thai with crushed peanuts is safe for me?"},
                ]
            },
            outputs={
                "contains_all": ["not safe"],
                "contains_none": ["safe to eat"],
                "max_allowed_iterations": 2,
            },
            metadata={"category": "safety_preference_retention"},
        ),
    ]
