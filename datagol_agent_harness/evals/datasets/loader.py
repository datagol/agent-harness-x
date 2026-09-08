"""Dataset utilities and LangSmith sync helpers for DataGOL Agent Harness."""

from __future__ import annotations

import datetime
import logging
from typing import Any
import uuid

from langsmith.schemas import Example

logger = logging.getLogger(__name__)


def build_example(
    inputs: dict[str, Any] | str,
    outputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    example_id: uuid.UUID | None = None,
    dataset_id: uuid.UUID | None = None,
) -> Example:
    """Build a standard LangSmith Example instance."""
    now = datetime.datetime.now(datetime.timezone.utc)
    in_dict = {"prompt": inputs} if isinstance(inputs, str) else dict(inputs)
    out_dict = dict(outputs or {})
    meta_dict = dict(metadata or {})

    return Example(
        id=example_id or uuid.uuid4(),
        inputs=in_dict,
        outputs=out_dict,
        metadata=meta_dict,
        created_at=now,
        modified_at=now,
        dataset_id=dataset_id or uuid.uuid4(),
    )


def sync_dataset_to_langsmith(
    dataset_name: str,
    examples: list[Example],
    *,
    client: Any | None = None,
    description: str | None = None,
) -> str:
    """Upload or synchronize a local dataset to LangSmith.

    Returns the LangSmith dataset ID.
    """
    if client is None:
        from langsmith import Client
        client = Client()

    # Check if dataset already exists
    dataset = None
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = None

    if dataset is None:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=description or f"DataGOL Agent Harness Benchmark: {dataset_name}",
        )
        logger.info("Created LangSmith dataset '%s' (id: %s)", dataset_name, dataset.id)

    # Sync examples
    existing_examples = list(client.list_examples(dataset_id=dataset.id))
    existing_inputs = [ex.inputs for ex in existing_examples]

    created_count = 0
    for ex in examples:
        if ex.inputs not in existing_inputs:
            client.create_example(
                inputs=ex.inputs,
                outputs=ex.outputs,
                metadata=ex.metadata,
                dataset_id=dataset.id,
            )
            created_count += 1

    logger.info("Synchronized dataset '%s': %d new examples added", dataset_name, created_count)
    return str(dataset.id)
