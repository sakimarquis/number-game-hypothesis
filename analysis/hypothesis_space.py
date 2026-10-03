"""Executable hypothesis supports and cross-mode projection into predictive space.

  - prompt_items          : prompt-local rule/interval labels with executable supports over 1..domain
  - eval_lookup           : hypothesis_eval label -> (kind, support) for one prompt
  - HypothesisLibrary     : executable predicates indexed by structured generation
  - ProjectedView         : posterior / hypothesis_eval / hypothesis_generation -> shared predictive space
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from analysis.hypothesis_structuring import library_path, load_library_module
from tasks.prompt_space import build_prompt_space_spec
from tasks.prompts import interval_label
from tasks.stimuli import TaskContext


# ===== Prompt-local executable supports =====

@dataclass(frozen=True)
class PromptItem:
    """One prompt label with its executable support over 1..domain.

    Grouped rule families use the mean of member supports; `distribution` is the
    mean of member distributions, so each member contributes equal mass.
    """

    label: str
    kind: str
    support: np.ndarray
    distribution: np.ndarray


def hypothesis_support(hypothesis, domain: int) -> np.ndarray:
    support = np.zeros(domain, dtype=float)
    for value in hypothesis.extension(range(1, domain + 1)):
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
                           *, normalize_members: bool) -> dict[str, np.ndarray]:
    return {
        family: np.mean([
            supports[member] / supports[member].sum() if normalize_members else supports[member]
            for member in members
        ], axis=0)
        for family, members in rule_families.items()
    }


def prompt_items(ctx: TaskContext, examples: list[int]) -> list[PromptItem]:
    """Executable supports for the unique rule and interval labels in one prompt (no 'other')."""
    spec = build_prompt_space_spec(ctx, examples)
    supports = rule_supports(ctx)
    if spec.rule_families is None:
        rule_vectors = {label: (supports[label], supports[label] / supports[label].sum()) for label in spec.rule_labels}
    else:
        grouped = grouped_family_vectors(supports, spec.rule_families, normalize_members=False)
        grouped_distributions = grouped_family_vectors(supports, spec.rule_families, normalize_members=True)
        rule_vectors = {label: (grouped[label], grouped_distributions[label]) for label in spec.rule_labels}

    items = [PromptItem(label, "rule", *vectors) for label, vectors in rule_vectors.items()]
    for lo, hi in dict.fromkeys(spec.intervals.values()):
        support = interval_support(lo, hi, ctx.domain)
        items.append(PromptItem(interval_label(lo, hi), "interval", support, support / support.sum()))
    return items


# ===== Cross-mode projection =====

@dataclass(frozen=True)
class ProjectedView:
    """One mode's projection into predictive space plus coverage diagnostics."""

    vector: np.ndarray
    semantics: str
    total_mass: float
    matched_mass: float
    unmatched_mass: float
    total_items: int
    matched_items: int
    unmatched_items: int
    rule_mass: float = 0.0
    interval_mass: float = 0.0
    rule_items: int = 0
    interval_items: int = 0

    @property
    def coverage(self) -> float:
        return self.matched_mass / self.total_mass if self.total_mass > 0 else 0.0

    @property
    def matched_item_frac(self) -> float:
        return self.matched_items / self.total_items if self.total_items > 0 else 0.0

    @property
    def unmatched_item_frac(self) -> float:
        return self.unmatched_items / self.total_items if self.total_items > 0 else 0.0

    @property
    def usable(self) -> bool:
        """Whether the projection contains a finite predictive target."""
        return bool(self.vector.size and np.isfinite(self.vector).all())


def eval_lookup(ctx: TaskContext, examples: list[int]) -> dict[str, tuple[str, np.ndarray]]:
    """(kind, support) for every hypothesis_eval prompt label of one stimulus prefix.

    Grouped eval prompts (Bigelow16) name base families plus a transform note;
    their executable proxy is the uniform mean over descendant rule extensions.
    """
    return {item.label: (item.kind, item.support) for item in prompt_items(ctx, examples)}


@dataclass(frozen=True)
class HypothesisLibrary:
    """Executable generated-hypothesis predicates indexed by structured JSON."""

    domain: int
    vectors: dict[int, np.ndarray]
    labels: dict[int, str]


def load_hypothesis_library(domain: int) -> HypothesisLibrary:
    path = str(library_path(domain))
    return _load_hypothesis_library_cached(path, os.stat(path).st_mtime_ns, domain)


@lru_cache(maxsize=None)
def _load_hypothesis_library_cached(path: str, mtime_ns: int, domain: int) -> HypothesisLibrary:
    """Load library_dN.py as the source of executable generation semantics."""
    module = load_library_module(path, f"_number_game_library_d{domain}_{mtime_ns}")
    vectors = {
        int(index): np.asarray([float(bool(fn([x]))) for x in range(1, domain + 1)], dtype=float)
        for index, fn in module.LIBRARY.items()
    }
    labels = {
        int(index): ((fn.__doc__ or f"library:{index}").strip().splitlines()[0])
        for index, fn in module.LIBRARY.items()
    }
    return HypothesisLibrary(domain=domain, vectors=vectors, labels=labels)


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


def normalize_hypothesis_weight_dict(result: dict) -> dict:
    return dict(normalized_hypothesis_items(result.items()))


def normalize_structured_generation_confidence(result: list[dict]) -> list[dict]:
    return [
        {**item, "confidence": weight}
        for item, weight in normalized_hypothesis_items(
            (item, item["confidence"]) for item in collapse_structured_generation(result)
        )
    ]


ProjectionMapping = tuple[str, str, np.ndarray]


def map_structured_generation_item(item: dict, library: HypothesisLibrary) -> ProjectionMapping | None:
    if item.get("type") not in {"rule", "interval"}:
        return None
    return item["type"], library.labels[item["index"]], library.vectors[item["index"]]


def map_eval_label(label: str, lookup: dict[str, tuple[str, np.ndarray]]) -> ProjectionMapping | None:
    """Map one hypothesis_eval label through its prompt lookup; 'other' and unknown labels are unmatched."""
    entry = lookup.get(label)
    return None if entry is None else (entry[0], label, entry[1])


def project_posterior(values: np.ndarray) -> ProjectedView:
    vec = np.asarray(values, dtype=float)
    return ProjectedView(
        vector=vec,
        semantics="direct",
        total_mass=1.0,
        matched_mass=1.0,
        unmatched_mass=0.0,
        total_items=len(vec),
        matched_items=len(vec),
        unmatched_items=0,
    )


def _project_weighted_items(
    items: Iterable[tuple[object, object]],
    mapper: Callable[[object], ProjectionMapping | None],
    domain: int,
    semantics: str,
) -> ProjectedView:
    """Accumulate weighted executable items into predictive space."""
    vector = np.zeros(domain, dtype=float)
    total_mass = matched_mass = unmatched_mass = rule_mass = interval_mass = 0.0
    total_items = matched_items = unmatched_items = rule_items = interval_items = 0

    for item, raw_weight in items:
        weight = float(raw_weight)
        total_mass += weight
        # Item diagnostics should reflect active hypotheses, not zero-weight fillers.
        if weight <= 0.0:
            continue
        total_items += 1
        mapped = mapper(item)
        if mapped is None:
            unmatched_mass += weight
            unmatched_items += 1
            continue
        kind, _label, vec = mapped
        vector += weight * vec
        matched_mass += weight
        matched_items += 1
        if kind == "rule":
            rule_mass += weight
            rule_items += 1
        else:
            interval_mass += weight
            interval_items += 1

    return ProjectedView(
        # No executable mass means there is no predictive observation. Keeping
        # this missing prevents exhausted parses from becoming confident zeros.
        vector=vector if matched_mass > 0.0 else np.full(domain, np.nan),
        semantics=semantics,
        total_mass=total_mass,
        matched_mass=matched_mass,
        unmatched_mass=unmatched_mass,
        total_items=total_items,
        matched_items=matched_items,
        unmatched_items=unmatched_items,
        rule_mass=rule_mass,
        interval_mass=interval_mass,
        rule_items=rule_items,
        interval_items=interval_items,
    )


def project_hypothesis_eval(result: dict[str, float], examples: list[int], ctx: TaskContext) -> ProjectedView:
    """Project prompt-listed hypotheses into predictive space."""
    lookup = eval_lookup(ctx, examples)
    return _project_weighted_items(
        result.items(),
        lambda label: map_eval_label(label, lookup),
        ctx.domain,
        "direct" if ctx.rule_families is None else "family_mean",
    )


def project_structured_hypothesis_generation(result: list[dict], library: HypothesisLibrary) -> ProjectedView:
    """Project structured generation through library predicates, not text matching."""
    return _project_weighted_items(
        ((item, item["confidence"]) for item in collapse_structured_generation(result)),
        lambda item: map_structured_generation_item(item, library),
        library.domain,
        "structured_library",
    )


def summarize_projection_rows(df: pd.DataFrame, group_cols: list[str]) -> list[dict]:
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        rows.append({
            **dict(zip(group_cols, key)),
            "n_rows": int(len(group)),
            "coverage_mean": float(group["coverage"].mean()),
            "matched_mass_mean": float(group["matched_mass"].mean()),
            "unmatched_mass_mean": float(group["unmatched_mass"].mean()),
            "matched_item_frac_mean": float(group["matched_item_frac"].mean()),
            "unmatched_item_frac_mean": float(group["unmatched_item_frac"].mean()),
            "rule_mass_mean": float(group["rule_mass"].mean()),
            "interval_mass_mean": float(group["interval_mass"].mean()),
        })
    return rows
