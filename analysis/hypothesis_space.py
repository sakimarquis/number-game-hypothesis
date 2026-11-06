"""Executable hypothesis supports and cross-mode projection into predictive space.

  - prompt_items          : prompt-local rule/interval labels with executable supports over 1..domain
  - eval_mapper / generation_mapper : hypothesis_eval label / structured item -> executable support
  - project_result        : posterior / hypothesis_eval / hypothesis_generation -> shared predictive space
  - project_trajectory    : project_result over every stored prefix of one condition
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from analysis.hypothesis_structuring import HypothesisLibrary, load_hypothesis_library
from tasks.prompt_space import build_prompt_space_spec
from tasks.prompts import interval_label
from tasks.stimuli import TaskContext

RULE_SHARE_TOP_K = (1, 3, 5)  # cutoffs written as hypothesis-compare `rule_share_top{k}` columns


# ===== Prompt-local executable supports =====

@dataclass(frozen=True)
class PromptItem:
    """One prompt label with its executable support over 1..domain.

    Grouped rule families use the mean of the supports of members consistent with the
    prompt's examples (all members when none is); `distribution` is the mean of those
    members' distributions, so each contributes equal mass.
    """

    label: str
    type: str
    support: np.ndarray
    distribution: np.ndarray


def hypothesis_support(hypothesis, domain: int) -> np.ndarray:
    support = np.zeros(domain, dtype=float)
    for value in hypothesis.support(range(1, domain + 1)):
        support[value - 1] = 1.0
    return support


def interval_support(lo: int, hi: int, domain: int) -> np.ndarray:
    support = np.zeros(domain, dtype=float)
    lo, hi = max(1, lo), min(domain, hi)
    if lo <= hi:
        support[lo - 1:hi] = 1.0
    return support


def rule_supports(ctx: TaskContext) -> dict[str, np.ndarray]:
    return {rule.description: hypothesis_support(rule, ctx.domain) for rule in ctx.rules}


def grouped_family_vectors(supports: dict[str, np.ndarray], rule_families: dict[str, list[str]],
                           examples: list[int], *, normalize_members: bool) -> dict[str, np.ndarray]:
    """Mean member vector per family; members inconsistent with `examples` drop out unless all are."""
    evidence_idx = np.asarray(examples, dtype=int) - 1
    vectors = {}
    for family, members in rule_families.items():
        consistent = [member for member in members if supports[member][evidence_idx].all()]
        vectors[family] = np.mean([
            supports[member] / supports[member].sum() if normalize_members else supports[member]
            for member in consistent or members
        ], axis=0)
    return vectors


def prompt_items(ctx: TaskContext, examples: list[int]) -> list[PromptItem]:
    """Executable supports for the unique rule and interval labels in one prompt (no 'other')."""
    spec = build_prompt_space_spec(ctx, examples)
    supports = rule_supports(ctx)
    if spec.rule_families is None:
        rule_vectors = {label: (supports[label], supports[label] / supports[label].sum()) for label in spec.rule_labels}
    else:
        grouped = grouped_family_vectors(supports, spec.rule_families, examples, normalize_members=False)
        grouped_distributions = grouped_family_vectors(supports, spec.rule_families, examples, normalize_members=True)
        rule_vectors = {label: (grouped[label], grouped_distributions[label]) for label in spec.rule_labels}

    items = [PromptItem(label, "rule", *vectors) for label, vectors in rule_vectors.items()]
    for lo, hi in dict.fromkeys(spec.intervals.values()):
        support = interval_support(lo, hi, ctx.domain)
        items.append(PromptItem(interval_label(lo, hi), "interval", support, support / support.sum()))
    return items


# ===== Cross-mode projection =====

@dataclass(frozen=True)
class ProjectedView:
    """One mode's projection into predictive space plus its rule, interval, and non-executable ("other") mass."""

    vector: np.ndarray  # [domain]; NaN when no mass is executable
    other_mass: float
    rule_mass: float = 0.0
    interval_mass: float = 0.0
    executable: tuple[tuple[str, float], ...] = ()  # (type, weight) per executable item

    @property
    def usable(self) -> bool:
        """Whether the projection contains a finite predictive target."""
        return bool(self.vector.size and np.isfinite(self.vector).all())

    def rule_share(self, k: int) -> float:
        """Rule share of weight among the k most confident executable items; NaN without any.

        Items tied at the cutoff split its remaining slots equally, so prompt order
        (Eval lists rules before intervals) cannot break ties toward rules.
        """
        if not self.executable:
            return float("nan")
        ranked = sorted(self.executable, key=lambda item: item[1], reverse=True)
        cutoff = ranked[min(k, len(ranked)) - 1][1]
        above = [item for item in ranked if item[1] > cutoff]
        tied = [item for item in ranked if item[1] == cutoff]
        fraction = (min(k, len(ranked)) - len(above)) / len(tied)
        kept = above + [(hypothesis_type, weight * fraction) for hypothesis_type, weight in tied]
        return sum(weight for hypothesis_type, weight in kept if hypothesis_type == "rule") / sum(weight for _, weight in kept)


def collapse_structured_generation(result: list[dict]) -> list[dict]:
    """Collapse duplicate executable indices (keeping the most confident) before treating confidence as mass."""
    collapsed: dict[int, dict] = {}
    out = []
    for item in result:
        if item.get("type") in {"rule", "interval"}:
            current = collapsed.get(item["index"])
            if current is None or item["confidence"] > current["confidence"]:
                collapsed[item["index"]] = dict(item)
        else:
            out.append(dict(item))
    return list(collapsed.values()) + out


def normalized_hypothesis_items(items: Iterable[tuple[object, object]]) -> list[tuple[object, float]]:
    """Drop non-positive hypothesis mass and normalize the remainder to a simplex."""
    active = [(label, float(weight)) for label, weight in items if float(weight) > 0.0]
    total = sum(weight for _, weight in active)
    return [(label, weight / total) for label, weight in active] if total > 0.0 else []


def normalize_structured_generation_confidence(result: list[dict]) -> list[dict]:
    return [
        {**item, "confidence": weight}
        for item, weight in normalized_hypothesis_items(
            (item, item["confidence"]) for item in collapse_structured_generation(result)
        )
    ]


ProjectionMapping = tuple[str, str, np.ndarray]  # (type, label, support)
Mapper = Callable[[object], ProjectionMapping | None]


def eval_mapper(ctx: TaskContext, examples: list[int]) -> Mapper:
    """Map hypothesis_eval labels of one stimulus prefix to their supports; 'other' and unknown labels map to None.

    Grouped eval prompts (Bigelow16) name base families plus a transform note;
    their executable proxy is the uniform mean over the descendant rule extensions
    consistent with `examples` (over all descendants when none is).
    """
    lookup = {item.label: (item.type, item.support) for item in prompt_items(ctx, examples)}
    return lambda label: None if label not in lookup else (lookup[label][0], label, lookup[label][1])


def generation_mapper(library: HypothesisLibrary) -> Mapper:
    """Map structured generation items to library supports; "other" items map to None."""
    return lambda item: (
        (item["type"], library.labels[item["index"]], library.vectors[item["index"]])
        if item.get("type") in {"rule", "interval"} else None
    )


def _project_weighted_items(
    items: Iterable[tuple[object, object]],
    mapper: Mapper,
    domain: int,
) -> ProjectedView:
    """Accumulate weighted executable items into predictive space; non-positive weights are fillers."""
    vector = np.zeros(domain, dtype=float)
    mass = {"other": 0.0, "rule": 0.0, "interval": 0.0}
    executable = []
    for item, raw_weight in items:
        weight = float(raw_weight)
        if weight <= 0.0:
            continue
        mapped = mapper(item)
        if mapped is None:
            mass["other"] += weight
            continue
        hypothesis_type, _label, vec = mapped
        vector += weight * vec
        mass[hypothesis_type] += weight
        executable.append((hypothesis_type, weight))

    return ProjectedView(
        # No executable mass means there is no predictive observation. Keeping
        # this missing prevents exhausted parses from becoming confident zeros.
        vector=vector if executable else np.full(domain, np.nan),
        other_mass=mass["other"],
        rule_mass=mass["rule"],
        interval_mass=mass["interval"],
        executable=tuple(executable),
    )


def project_hypothesis_eval(result: dict[str, float], examples: list[int], ctx: TaskContext) -> ProjectedView:
    """Project prompt-listed hypotheses into predictive space."""
    return _project_weighted_items(result.items(), eval_mapper(ctx, examples), ctx.domain)


def project_structured_hypothesis_generation(result: list[dict], library: HypothesisLibrary) -> ProjectedView:
    """Project collapsed, normalized structured generation through library predicates, not text matching."""
    return _project_weighted_items(((item, item["confidence"]) for item in result), generation_mapper(library), library.domain)


def project_result(eval_mode: str, result, examples: list[int], ctx: TaskContext) -> ProjectedView:
    """Project one stored prefix result of any eval mode into 1..domain predictive space.

    Stored hypothesis_eval weights are already normalized by scoring; generation
    confidences are normalized here after collapsing duplicate library indices.
    """
    if eval_mode == "posterior":
        return ProjectedView(vector=np.asarray(result, dtype=float), other_mass=0.0)
    if eval_mode == "hypothesis_eval":
        return project_hypothesis_eval(result, examples, ctx)
    return project_structured_hypothesis_generation(
        normalize_structured_generation_confidence(result), load_hypothesis_library(ctx.domain),
    )



def project_trajectory(eval_mode: str, trajectory: dict[str, dict], ctx: TaskContext) -> dict[str, dict[int, ProjectedView]]:
    """Project every stored prefix result of one condition trajectory: {stimulus: {n_examples: view}}."""
    return {
        name: {
            n: project_result(eval_mode, result, ctx.task.stimulus_map[name].examples[:n], ctx)
            for n, result in zip(ev["n_examples"], ev["results"])
        }
        for name, ev in trajectory.items()
    }
