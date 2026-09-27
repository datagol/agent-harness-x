"""Evaluation runner for HarnessX using LangSmith."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import datetime
import uuid
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence, Union

from langsmith import evaluate
from langsmith.schemas import Example, Run

from .datasets.loader import build_example, sync_dataset_to_langsmith
from .datasets.registry import get_dataset
from .evaluators import default_evaluators
from .target import AgentOrFactory, AgentTarget
from ..core import Agent

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


@dataclass
class EvaluationSummary:
    """Summary of an evaluation experiment."""

    experiment_name: str
    url: str | None
    total_examples: int
    scores: dict[str, float] = field(default_factory=dict)
    pass_rate: float = 0.0
    mean_iterations: float = 0.0
    results: list[dict[str, Any]] = field(default_factory=list)


def evaluate_agent(
    agent: AgentOrFactory,
    dataset: Union[str, Sequence[Example], Sequence[dict[str, Any]]],
    *,
    evaluators: Sequence[Callable[[Run, Example], Any]] | None = None,
    experiment_prefix: str = "harnessx-eval",
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    max_concurrency: int = 1,
    client: Any | None = None,
    offline: bool = False,
    upload_results: bool | None = None,
    print_summary: bool = True,
) -> EvaluationSummary:
    """Evaluate an agent or agent factory against a dataset using LangSmith.

    Args:
        agent: Agent instance or factory callable `(inputs) -> Agent`.
        dataset: Dataset name (e.g. 'tool_calling', 'skills', 'all'),
                 or a list of LangSmith `Example` objects, or a remote LangSmith dataset name.
        evaluators: List of evaluator functions (defaults to `default_evaluators()`).
        experiment_prefix: Prefix for the LangSmith experiment name.
        description: Description of the experiment.
        metadata: Custom metadata dictionary to attach to the experiment.
        max_concurrency: Concurrency limit for evaluations.
        client: Optional LangSmith Client.
        offline: If True, executes locally without uploading runs to LangSmith.
        upload_results: Explicit upload flag (overrides offline if provided).
        print_summary: Whether to print a formatted terminal summary.

    Returns:
        EvaluationSummary with metrics, scores, and LangSmith URL.
    """
    should_upload = not offline if upload_results is None else upload_results

    # Resolve dataset
    data_to_evaluate: Any
    if isinstance(dataset, str):
        try:
            data_to_evaluate = get_dataset(dataset)
        except ValueError:
            # Not in local registry, pass string as LangSmith dataset name
            data_to_evaluate = dataset
    else:
        data_to_evaluate = dataset

    # If uploading results, LangSmith requires the dataset to exist in the cloud
    if should_upload:
        try:
            if isinstance(dataset, str) and not isinstance(data_to_evaluate, str):
                cloud_ds_name = f"harnessx-{dataset.replace('_', '-')}"
                sync_dataset_to_langsmith(cloud_ds_name, data_to_evaluate, client=client)
                data_to_evaluate = cloud_ds_name
            elif isinstance(data_to_evaluate, (list, tuple)) and data_to_evaluate and hasattr(data_to_evaluate[0], "inputs"):
                cloud_ds_name = f"{experiment_prefix}-dataset"
                sync_dataset_to_langsmith(cloud_ds_name, list(data_to_evaluate), client=client)
                data_to_evaluate = cloud_ds_name
        except Exception as e:
            logger.warning("Failed to synchronize dataset to LangSmith: %s", e)

    eval_suite = list(evaluators) if evaluators is not None else default_evaluators()

    target = AgentTarget(
        agent,
        attach_langsmith=should_upload,
        project_name=experiment_prefix,
    )

    logger.info("Starting LangSmith evaluation experiment '%s'...", experiment_prefix)

    # Run evaluation
    if should_upload:
        results = evaluate(
            target, data=data_to_evaluate, evaluators=eval_suite,
            experiment_prefix=experiment_prefix, description=description,
            metadata=metadata, max_concurrency=max_concurrency, client=client,
            upload_results=True,
        )
    else:
        if isinstance(data_to_evaluate, str):
            raise ValueError("Offline evaluation requires local examples or a registered local dataset")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if isinstance(agent, Agent) and max_concurrency > 1:
            raise ValueError("Concurrent evaluation requires an agent factory")

        def evaluate_one(item):
            example = item if isinstance(item, Example) else build_example(item.get("inputs", item), item.get("outputs"), item.get("metadata"))
            started = datetime.datetime.now(datetime.timezone.utc)
            outputs = target(example.inputs)
            run = Run(id=uuid.uuid4(), name=experiment_prefix, run_type="chain", inputs=example.inputs,
                      outputs=outputs, start_time=started, end_time=datetime.datetime.now(datetime.timezone.utc))
            scores = []
            for evaluator in eval_suite:
                scored = evaluator(run, example)
                if isinstance(scored, dict):
                    items = scored.get("results", [scored])
                    scores.extend(SimpleNamespace(**value) if isinstance(value, dict) else value for value in items)
                else:
                    scores.append(scored)
            return {"run": run, "example": example, "evaluation_results": {"results": scores}}

        # No LangSmith client/experiment runner is constructed for offline work.
        if max_concurrency == 1:
            results = [evaluate_one(item) for item in data_to_evaluate]
        else:
            with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
                results = list(pool.map(evaluate_one, data_to_evaluate))

    # Aggregate scores & metrics
    metric_totals: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    row_results: list[dict[str, Any]] = []
    total_iterations = 0

    for item in results:
        run = item.get("run")
        ex = item.get("example")
        eval_res = item.get("evaluation_results", {}).get("results", [])

        row_info: dict[str, Any] = {
            "inputs": getattr(ex, "inputs", {}),
            "outputs": getattr(run, "outputs", {}),
            "scores": {},
        }

        iterations = (getattr(run, "outputs", {}) or {}).get("iterations", 1)
        total_iterations += iterations

        for r in eval_res:
            key = getattr(r, "key", "unknown")
            score = getattr(r, "score", None)
            if score is not None:
                numeric_score = 1.0 if score is True else (0.0 if score is False else float(score))
                metric_totals[key] = metric_totals.get(key, 0.0) + numeric_score
                metric_counts[key] = metric_counts.get(key, 0) + 1
                row_info["scores"][key] = numeric_score

        row_results.append(row_info)

    total_ex = len(row_results)
    avg_scores = {
        k: round(metric_totals[k] / metric_counts[k], 3)
        for k in metric_totals
        if metric_counts[k] > 0
    }

    # Pass rate: average of primary metrics
    pass_rate = round(sum(avg_scores.values()) / len(avg_scores), 3) if avg_scores else 1.0
    mean_iter = round(total_iterations / max(1, total_ex), 2)

    exp_url: str | None = None
    if should_upload:
        try:
            exp_url = results.url
        except Exception:
            exp_url = None

    exp_name: str = experiment_prefix
    try:
        exp_name = getattr(results, "experiment_name", None) or experiment_prefix
    except Exception:
        exp_name = experiment_prefix

    summary = EvaluationSummary(
        experiment_name=exp_name,
        url=exp_url,
        total_examples=total_ex,
        scores=avg_scores,
        pass_rate=pass_rate,
        mean_iterations=mean_iter,
        results=row_results,
    )

    if print_summary:
        _print_summary_report(summary, offline=not should_upload)

    return summary


def _print_summary_report(summary: EvaluationSummary, *, offline: bool) -> None:
    """Render a terminal evaluation report."""
    if not _RICH_AVAILABLE:
        print("\n" + "=" * 60)
        print(f"Evaluation Experiment: {summary.experiment_name}")
        print(f"Total Examples: {summary.total_examples} | Pass Rate: {summary.pass_rate * 100:.1f}%")
        print(f"Mean Iterations: {summary.mean_iterations}")
        for k, v in summary.scores.items():
            print(f"  • {k}: {v * 100:.1f}%")
        if summary.url:
            print(f"LangSmith URL: {summary.url}")
        print("=" * 60 + "\n")
        return

    console = Console()

    table = Table(title="Evaluation Metrics", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Status", justify="center")

    for metric, score in sorted(summary.scores.items()):
        pct = f"{score * 100:.1f}%"
        status = "[green]PASS[/green]" if score >= 0.8 else ("[yellow]WARN[/yellow]" if score >= 0.5 else "[red]FAIL[/red]")
        table.add_row(metric, pct, status)

    overall_color = "green" if summary.pass_rate >= 0.8 else "yellow"
    summary_text = (
        f"[bold]Experiment:[/bold] {summary.experiment_name}\n"
        f"[bold]Examples Evaluated:[/bold] {summary.total_examples}\n"
        f"[bold]Overall Pass Rate:[/bold] [{overall_color}]{summary.pass_rate * 100:.1f}%[/{overall_color}]\n"
        f"[bold]Mean Iterations:[/bold] {summary.mean_iterations}\n"
    )
    if summary.url:
        summary_text += f"\n[bold]View in LangSmith:[/bold] [link={summary.url}]{summary.url}[/link]"
    elif offline:
        summary_text += "\n[dim]Run in offline mode (traces not uploaded to LangSmith)[/dim]"

    console.print()
    console.print(Panel(summary_text, title="HarnessX Evaluation", border_style=overall_color))
    console.print(table)
    console.print()


async def evaluate_agent_async(
    agent: AgentOrFactory,
    dataset: Union[str, Sequence[Example], Sequence[dict[str, Any]]],
    *,
    evaluators: Sequence[Callable[[Run, Example], Any]] | None = None,
    experiment_prefix: str = "harnessx-eval",
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    max_concurrency: int = 1,
    client: Any | None = None,
    offline: bool = False,
    upload_results: bool | None = None,
    print_summary: bool = True,
) -> EvaluationSummary:
    """Await evaluate_agent from a running event loop.

    The synchronous runner drives each example with asyncio.run(), so it runs in
    a worker thread. Pass a factory rather than an Agent bound to the caller's
    loop; the evaluation cannot be cancelled once started; do not nest it inside
    another evaluation.
    """
    import asyncio
    import functools

    return await asyncio.to_thread(functools.partial(
        evaluate_agent, agent, dataset, evaluators=evaluators, experiment_prefix=experiment_prefix,
        description=description, metadata=metadata, max_concurrency=max_concurrency, client=client,
        offline=offline, upload_results=upload_results, print_summary=print_summary,
    ))
