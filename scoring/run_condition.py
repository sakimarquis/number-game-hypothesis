"""Prefix-dedup condition scoring: batch all unique prefixes in one scorer call."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scoring.eval_modes import SCORERS

if TYPE_CHECKING:
    from configs.settings import ExperimentCondition
    from tasks.stimuli import StimulusSet, TaskContext


def collect_unique_prefixes(stimuli: list[StimulusSet], max_examples: int) -> list[list[int]]:
    """Return stimulus prefixes up to max_examples in first-seen order, deduplicated across stimulus sets."""
    unique: dict[tuple[int, ...], list[int]] = {}
    for stimulus in stimuli:
        for k in range(min(len(stimulus.examples), max_examples)):
            prefix = tuple(stimulus.examples[: k + 1])
            unique.setdefault(prefix, list(prefix))
    return list(unique.values())


async def score_condition(
    backend,
    condition: ExperimentCondition,
    ctx: TaskContext,
    *,
    max_examples: int,
    max_retries: int,
    failure_log_path: str | None = None,
    progress_path: str | None = None,
    progress_identity: dict | None = None,
) -> tuple[dict[tuple[int, ...], Any], dict[tuple[int, ...], dict[str, Any]] | None]:
    """Score all unique stimulus prefixes for one condition in a single batched call.

    Returns the flat prefix cache plus optional per-prefix diagnostics.
    Dedup assumes scorer output is a stable function of the prefix:
    shared prefixes intentionally share the same scorer result.
    """
    prefixes = collect_unique_prefixes(ctx.task.stimuli, max_examples)
    if condition.eval_mode == "posterior":
        kwargs = {"progress_path": progress_path, "progress_identity": progress_identity}
    else:
        kwargs = {"max_retries": max_retries, "failure_log_path": failure_log_path}
    scored = await SCORERS[condition.eval_mode](backend, condition, prefixes, ctx, **kwargs)

    keys = [tuple(prefix) for prefix in prefixes]
    diagnostics = scored.prefix_diagnostics
    return (
        dict(zip(keys, scored.values)),
        dict(zip(keys, diagnostics)) if diagnostics is not None else None,
    )
