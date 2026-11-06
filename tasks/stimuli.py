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


# Per-task rule space: registry base preset and the transforms composed onto its bases.
# Bigelow16 uses every registered transform (Bigelow & Piantadosi 2016 primordials).
TASK_RULE_SPACES: dict[str, tuple[str, list[str]]] = {
    "Tenenbaum99": ("all", ["identity"]),
    "Bigelow16": ("bigelow", list(default_registry(preset="bigelow").transforms)),
}


@dataclass
class StimulusSet:
    name: str
    examples: list[int]
    kind: str  # "single" | "rule" | "similarity" | "other"

    def prefix(self, n: int) -> "StimulusSet":
        """The same stimulus set truncated to its first n examples."""
        return StimulusSet(self.name, self.examples[:n], self.kind)


@dataclass
class TaskData:
    name: str                     # "Tenenbaum99" | "Bigelow16"
    stimuli: list[StimulusSet]
    human_preds: dict | None = None  # {stimulus_name: np.ndarray} if available

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


BASE_DOMAIN = 100  # the tasks' original domain 1..100: human data cover it, and extended domains should preserve its shape
TENENBAUM99_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "tenenbaum99_data")


def load_tenenbaum99() -> dict[str, np.ndarray]:
    """Load Tenenbaum (1999) sparse human ratings keyed by stimulus name.

    Each CSV is `target, p_yes` on ~30 sparsely sampled targets; values are
    length-100 arrays with NaN at unmeasured targets. Filenames encode the
    stimulus set either as a bare int (`16.csv`) or a bracketed list
    (`[16,8,2,64].csv`); matching is by set, since stimulus order differs
    between filename and stimulus.
    """
    by_example_set = {}
    for fname in sorted(os.listdir(TENENBAUM99_DATA_DIR)):
        if not fname.endswith(".csv"):
            continue
        preds = np.full(BASE_DOMAIN, np.nan)
        with open(os.path.join(TENENBAUM99_DATA_DIR, fname), newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0].strip():
                    continue
                t, p = int(row[0]), float(row[1])
                if 1 <= t <= BASE_DOMAIN:
                    preds[t - 1] = p
        by_example_set[frozenset(int(x) for x in fname[:-4].strip("[]").split(","))] = preds
    return {
        stimulus.name: by_example_set[frozenset(stimulus.examples)]
        for stimulus in TENENBAUM99_STIMULI
        if frozenset(stimulus.examples) in by_example_set
    }


def bigelow_stimulus(set_str: str) -> tuple[str, list[int]]:
    """Parse one Bigelow16 CSV `set` value such as '16_ 8_ 2' into (name, examples)."""
    examples = [int(x) for x in str(set_str).split("_ ")]
    return "D={" + ",".join(str(x) for x in examples) + "}", examples


def load_bigelow16() -> tuple[list[StimulusSet], dict[str, np.ndarray]]:
    """Load Bigelow & Piantadosi (2016) stimuli and human baselines.

    Human data covers targets 1..100 (native study resolution), so arrays are
    always length 100; build_task_context drops them outside that domain.
    Stimulus kinds come from `classify_bigelow_kind`.
    """
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "dataset", "numbergame_data.csv"))
    agg = df.groupby(["set", "target"])["rating"].mean().reset_index()

    stimuli, human_preds = [], {}
    for set_str in sorted(df["set"].unique()):
        name, examples = bigelow_stimulus(set_str)
        stimuli.append(StimulusSet(name, examples, classify_bigelow_kind(examples)))
        preds = np.zeros(BASE_DOMAIN)
        for _, row in agg[agg["set"] == set_str].iterrows():
            t = int(row["target"])
            if 1 <= t <= BASE_DOMAIN:
                preds[t - 1] = row["rating"]
        human_preds[name] = preds
    return stimuli, human_preds


@lru_cache(maxsize=None)
def make_task(name: str) -> TaskData:
    """Build task data once per process. Domain-independent — returns all stimuli for the task."""
    if name == "Tenenbaum99":
        return TaskData(name="Tenenbaum99", stimuli=list(TENENBAUM99_STIMULI), human_preds=load_tenenbaum99())
    if name == "Bigelow16":
        stimuli, human_preds = load_bigelow16()
        return TaskData(name="Bigelow16", stimuli=stimuli, human_preds=human_preds)
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
def build_task_context(task: str, domain: int) -> TaskContext:
    """Build domain-valid task data and rules without allocating model tensors."""
    task_data = make_task(task)
    bad = [c.name for c in task_data.stimuli if max(c.examples) > domain]
    if bad:
        raise ValueError(
            f"{task} domain={domain}: stimuli {bad} have examples outside 1..{domain}. "
            f"Remove this domain or use domain >= {max(max(c.examples) for c in task_data.stimuli)}."
        )
    if task_data.human_preds and domain != BASE_DOMAIN:
        task_data = replace(task_data, human_preds=None)
    preset, transforms = TASK_RULE_SPACES[task]
    rules, intervals, rule_families = default_registry(domain, preset=preset).build(transforms)
    return TaskContext(task_data, domain, rules, intervals, rule_families)
