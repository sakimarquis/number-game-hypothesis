"""Task data: stimulus sets, human ratings, and domain-valid task contexts.

Tenenbaum99 has 8 hand-written stimulus sets (Tenenbaum 1999, Ch. 5);
Bigelow16 loads 255 stimulus sets from the Bigelow & Piantadosi (2016) data.
"""

import csv
import os
from dataclasses import dataclass, replace
from functools import lru_cache, reduce
from math import gcd

import numpy as np
import pandas as pd

from tasks.hypotheses import Hypothesis, default_registry
from utils import PROJECT_ROOT


def classify_bigelow_kind(examples: list[int]) -> str:
    """Structurally classify a Bigelow16 example set into single/similarity/rule/other.

    Mirrors Tenenbaum99's operational definitions:
    - single: only one example → no structure to analyze
    - rule: arithmetic progression OR shared integer factor > 1 → points to a math rule
    - similarity: tight cluster (spread ≤ 10) without a clearer rule signature
    - other: no obvious structure (ambiguous/contrastive sets)
    """
    if len(examples) == 1:
        return "single"
    # Arithmetic progression check requires ≥3 points (2-point "AP" is trivial)
    if len(examples) >= 3:
        sorted_ex = sorted(examples)
        diffs = {sorted_ex[i + 1] - sorted_ex[i] for i in range(len(sorted_ex) - 1)}
        if len(diffs) == 1:
            return "rule"
    if reduce(gcd, examples) > 1:  # shared factor k>1 → "multiples of k" rule family
        return "rule"
    if max(examples) - min(examples) <= 10:
        return "similarity"
    return "other"


# Bigelow & Piantadosi 2016 transforms (applied to primordial bases), derived
# from the registry's registration order so the catalog cannot drift from it.
BIGELOW16_TRANSFORMS: list[str] = list(default_registry(preset="bigelow").transforms)


@dataclass
class StimulusSet:
    name: str
    examples: list[int]
    kind: str  # "single" | "rule" | "similarity" | "other"


@dataclass
class TaskData:
    name: str                     # "Tenenbaum99" | "Bigelow16"
    stimuli: list[StimulusSet]
    transforms: list[str]
    human_preds: dict | None = None  # {stimulus_name: np.ndarray} if available
    base_preset: str = "all"      # "all" (Tenenbaum99) | "bigelow" (B&P primordials)
    featured_stimuli: list[str] | None = None  # per-condition panel/decomp subset when |stimuli|>8

    @property
    def stimulus_map(self) -> dict[str, StimulusSet]:
        return {c.name: c for c in self.stimuli}


TENENBAUM99_STIMULI: list[StimulusSet] = [
    # Single-example
    StimulusSet("D={16}",              [16],             "single"),
    StimulusSet("D={60}",              [60],             "single"),
    # Rule-evoking (4 examples)
    StimulusSet("D={16,8,2,64}",       [16, 8, 2, 64],  "rule"),
    StimulusSet("D={60,80,10,30}",     [60, 80, 10, 30], "rule"),
    StimulusSet("D={81,25,4,36}",      [81, 25, 4, 36],  "rule"),
    # Similarity-evoking (4 examples)
    StimulusSet("D={16,23,19,20}",     [16, 23, 19, 20], "similarity"),
    StimulusSet("D={60,52,57,55}",     [60, 52, 57, 55], "similarity"),
    StimulusSet("D={81,98,86,93}",     [81, 98, 86, 93], "similarity"),
]


HUMAN_NATIVE_DOMAIN = 100  # Bigelow16 human data covers targets 1..100
TENENBAUM99_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "tenenbaum99_data")


@lru_cache(maxsize=1)
def _load_tenenbaum99_cached() -> tuple[tuple[frozenset, tuple[float, ...]], ...]:
    """Parse Tenenbaum99 CSVs once, keyed by example set.

    Each CSV is `target, p_yes` on ~30 sparsely sampled targets. Returns
    tuples of (example_set, preds_tuple) with NaN at unmeasured targets.
    Filenames encode the stimulus set either as a bare int (`16.csv`) or a
    bracketed list (`[16,8,2,64].csv`); matching is by set, so order does not matter.
    """
    result = []
    for fname in sorted(os.listdir(TENENBAUM99_DATA_DIR)):
        if not fname.endswith(".csv"):
            continue
        stem = fname[:-4].strip("[]")
        ex_set = frozenset(int(x) for x in stem.split(","))
        preds = np.full(HUMAN_NATIVE_DOMAIN, np.nan)
        with open(os.path.join(TENENBAUM99_DATA_DIR, fname), newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0].strip():
                    continue
                t, p = int(row[0]), float(row[1])
                if 1 <= t <= HUMAN_NATIVE_DOMAIN:
                    preds[t - 1] = p
        result.append((ex_set, tuple(preds.tolist())))
    return tuple(result)


def load_tenenbaum99() -> dict[str, np.ndarray]:
    """Load Tenenbaum (1999) sparse human ratings keyed by stimulus name.

    Values are length-100 arrays with NaN at unmeasured targets (~30 measured
    per stimulus). CSVs are matched to TENENBAUM99_STIMULI by example set (order-
    independent), since stimulus order differs between filename and stimulus.
    """
    cached = dict(_load_tenenbaum99_cached())
    human_preds = {}
    for stimulus in TENENBAUM99_STIMULI:
        preds = cached.get(frozenset(stimulus.examples))
        if preds is not None:
            human_preds[stimulus.name] = np.array(preds)
    return human_preds


def bigelow_stimulus(set_str: str) -> tuple[str, list[int]]:
    """Parse one Bigelow16 CSV `set` value such as '16_ 8_ 2' into (name, examples)."""
    examples = [int(x) for x in str(set_str).split("_ ")]
    return "D={" + ",".join(str(x) for x in examples) + "}", examples


@lru_cache(maxsize=None)
def _load_bigelow16_cached(csv_path: str) -> tuple[tuple[tuple[str, tuple[int, ...], str], ...], dict[str, np.ndarray]]:
    """Parse Bigelow16 once, returning immutable stimulus specs + baseline arrays."""
    df = pd.read_csv(csv_path)
    agg = df.groupby(["set", "target"])["rating"].mean().reset_index()
    agg.rename(columns={"rating": "p_yes"}, inplace=True)

    stimulus_specs = []
    human_preds = {}
    for set_str in sorted(df["set"].unique()):
        name, examples = bigelow_stimulus(set_str)
        stimulus_specs.append((name, tuple(examples), classify_bigelow_kind(examples)))

        preds = np.zeros(HUMAN_NATIVE_DOMAIN)
        subset = agg[agg["set"] == set_str]
        for _, row in subset.iterrows():
            t = int(row["target"])
            if 1 <= t <= HUMAN_NATIVE_DOMAIN:
                preds[t - 1] = row["p_yes"]
        human_preds[name] = preds

    return tuple(stimulus_specs), human_preds


def load_bigelow16() -> tuple[list[StimulusSet], dict[str, np.ndarray]]:
    """Load Bigelow & Piantadosi (2016) stimuli and human baselines.

    Human data covers targets 1..100 (native study resolution). Arrays are
    always length 100; callers must disable human comparison when domain != 100.

    Returns:
        stimuli: 255 StimulusSet objects with kind classified by `classify_bigelow_kind`
            (single/similarity/rule/other)
        human_preds: {stimulus_name: np.array of shape (100,)} with p(yes)
    """
    csv_path = os.path.join(PROJECT_ROOT, "dataset", "numbergame_data.csv")
    specs, cached_preds = _load_bigelow16_cached(os.path.abspath(csv_path))
    stimuli = [StimulusSet(name, list(examples), kind) for name, examples, kind in specs]
    human_preds = {name: preds.copy() for name, preds in cached_preds.items()}
    return stimuli, human_preds


def make_task(name: str, featured_stimuli: list[str] | None = None) -> TaskData:
    """Build task config. Domain-independent — returns all stimuli for the task."""
    if name == "Tenenbaum99":
        return TaskData(
            name="Tenenbaum99",
            stimuli=list(TENENBAUM99_STIMULI),
            transforms=["identity"],
            human_preds=load_tenenbaum99(),
            featured_stimuli=featured_stimuli,
        )
    if name == "Bigelow16":
        stimuli, human_preds = load_bigelow16()
        return TaskData(
            name="Bigelow16",
            stimuli=stimuli,
            transforms=BIGELOW16_TRANSFORMS,
            human_preds=human_preds,
            base_preset="bigelow",
            featured_stimuli=featured_stimuli,
        )
    raise ValueError(f"Unknown task: {name!r}")


@dataclass(frozen=True)
class TaskContext:
    """Domain-valid task data plus the rule space used by prompts and the Bayesian model.

    `task.human_preds` is None outside the human study's native domain.
    """
    task: TaskData
    domain: int
    rules: list[Hypothesis]
    intervals: list[Hypothesis]
    rule_families: dict[str, list[str]] | None


@lru_cache(maxsize=None)
def build_task_context(task: str, domain: int, natural_intervals: bool,
                       featured_stimuli: tuple[str, ...] | None = None) -> TaskContext:
    """Build domain-valid task data and rules without allocating model tensors."""
    task_data = make_task(task, featured_stimuli=list(featured_stimuli) if featured_stimuli is not None else None)
    bad = [c.name for c in task_data.stimuli if max(c.examples) > domain]
    if bad:
        raise ValueError(
            f"{task} domain={domain}: stimuli {bad} have examples outside 1..{domain}. "
            f"Remove this domain or use domain >= {max(max(c.examples) for c in task_data.stimuli)}."
        )
    if task_data.human_preds and domain != HUMAN_NATIVE_DOMAIN:
        task_data = replace(task_data, human_preds=None)
    registry = default_registry(domain, preset=task_data.base_preset)
    rules, intervals, rule_families = registry.build(task_data.transforms, natural_intervals=natural_intervals)
    return TaskContext(task_data, domain, rules, intervals, rule_families)
