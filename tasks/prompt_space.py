"""Prompt-local hypothesis labels derived from examples and task rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tasks.prompts import TRANSFORM_NOTE, interval_label

if TYPE_CHECKING:
    from tasks.stimuli import TaskContext


@dataclass(frozen=True)
class PromptSpaceSpec:
    """Prompt-space labels shared by scoring and analysis consumers."""

    rule_labels: list[str]
    intervals: dict[str, tuple[int, int]]
    rule_families: dict[str, list[str]] | None

    @property
    def interval_labels(self) -> list[str]:
        return [interval_label(lo, hi) for lo, hi in self.intervals.values()]

    @property
    def labels(self) -> list[str]:
        return self.rule_labels + self.interval_labels + ["other"]

    @property
    def unique_labels(self) -> list[str]:
        return list(dict.fromkeys(self.labels))

    @property
    def transform_note(self) -> str:
        return TRANSFORM_NOTE if self.rule_families is not None else ""


def _floor_to(x: int, step: int, lo: int = 1) -> int:
    """Floor x to nearest multiple of step, clamped to lo."""
    return max(lo, x - x % step)


def _ceil_to(x: int, step: int, hi: int) -> int:
    """Ceil x to nearest multiple of step, clamped to hi."""
    remainder = x % step
    ceiled = x + (step - remainder) if remainder != 0 else x
    return min(hi, ceiled)


def derive_intervals(examples: list[int], domain: int) -> dict[str, tuple[int, int]]:
    """Derive 4 interval hypotheses from example set and domain.

    round10/round5 snap the example range to nearest 10/5 multiples, minmax is
    the tight example interval, and full is the entire domain.
    """
    lo, hi = min(examples), max(examples)
    return {
        "round10": (_floor_to(lo, 10), _ceil_to(hi, 10, domain)),
        "round5":  (_floor_to(lo, 5),  _ceil_to(hi, 5, domain)),
        "minmax":  (lo, hi),
        "full":    (1, domain),
    }


def build_prompt_space_spec(ctx: TaskContext, examples: list[int]) -> PromptSpaceSpec:
    """Build prompt-local labels, intervals, and grouped-family metadata for one prefix."""
    return PromptSpaceSpec(
        rule_labels=list(ctx.rule_families) if ctx.rule_families is not None else [h.description for h in ctx.rules],
        intervals=derive_intervals(examples, ctx.domain),
        rule_families=ctx.rule_families,
    )
