"""Tenenbaum99 d200 extension diagnostics shared by model stages and the Bayesian baseline.

- extension mass:      probability mass on 101..200 of the normalized d200 prediction, per evidence prefix
- rule discrimination: P(yes) on rule vs non-rule targets in 101..200 for the three rule stimuli (d200)
"""

from __future__ import annotations

import numpy as np

from analysis.alpha_beta_scope import PREFIX_LENGTHS
from analysis.cache import bayesian_baseline, load_condition_trajectory
from analysis.hypothesis_space import hypothesis_support, project_trajectory
from analysis.plotter import save_csv
from configs.settings import Experiment, make_grid
from tasks.hypotheses import default_registry
from tasks.stimuli import BASE_DOMAIN, build_task_context

EXTENSION_TASK = "Tenenbaum99"
EXTENSION_DOMAIN = 200
EXTENSION_WINDOW_START = BASE_DOMAIN + 1
EXTENSION_STIMULUS_KINDS = ("rule", "similarity")
# Registry base whose extension defines rule-positive targets for each rule stimulus.
RULE_STIMULUS_BASES = {"D={16,8,2,64}": "pow_2", "D={60,80,10,30}": "mult_10", "D={81,25,4,36}": "squares"}


def _extension_stimuli(stimuli) -> list:
    return [
        stimulus for stimulus in stimuli
        if stimulus.kind in EXTENSION_STIMULUS_KINDS and len(stimulus.examples) >= max(PREFIX_LENGTHS)
    ]


def extension_mass_rows(prefix_vectors: dict[str, dict[int, np.ndarray]], stimuli) -> list[dict]:
    """One row per (stimulus, prefix) with the d200 extension mass."""
    rows = []
    for stimulus in _extension_stimuli(stimuli):
        for n, vector in prefix_vectors.get(stimulus.name, {}).items():
            if n in PREFIX_LENGTHS:
                rows.append({
                    "stimulus": stimulus.name, "stimulus_kind": stimulus.kind, "n_examples": n,
                    "extension_mass": extension_mass(vector),
                })
    return rows


def extension_mass(vector) -> float:
    """Probability mass on 101..200 after normalizing per-target d200 P(yes) into a distribution.

    Normalizing removes a model's overall yes-rate, so models compare on how much of their generalization leaves 1..100.
    """
    vector = np.asarray(vector, dtype=float)  # [EXTENSION_DOMAIN]
    total = float(vector.sum())
    return float(vector[BASE_DOMAIN:].sum() / total) if total > 0.0 else float("nan")


def rule_discrimination_rows(final_vectors: dict[str, np.ndarray]) -> list[dict]:
    """Rule-positive vs rule-negative mean P(yes) in 101..200 for each rule stimulus."""
    bases = {base.name: base for base in default_registry(EXTENSION_DOMAIN).bases}
    window = np.arange(EXTENSION_WINDOW_START - 1, EXTENSION_DOMAIN)
    rows = []
    for stimulus_name, base_name in RULE_STIMULUS_BASES.items():
        if stimulus_name not in final_vectors:
            continue
        positive = hypothesis_support(bases[base_name], EXTENSION_DOMAIN)[window] > 0
        vector = np.asarray(final_vectors[stimulus_name], dtype=float)[window]
        rows.append({
            "stimulus": stimulus_name,
            "rule_positive_mean": float(vector[positive].mean()),
            "rule_negative_mean": float(vector[~positive].mean()),
            "n_rule_positive": int(positive.sum()),
            "n_rule_negative": int((~positive).sum()),
        })
    return rows


def _full_stimulus_finals(steps_by_stimulus: dict[str, dict[int, np.ndarray]], stimuli) -> dict[str, np.ndarray]:
    return {
        stimulus.name: steps_by_stimulus[stimulus.name][len(stimulus.examples)]
        for stimulus in stimuli
        if len(stimulus.examples) in steps_by_stimulus.get(stimulus.name, {})
    }


def _prediction_steps(condition, result_key: str) -> dict[str, dict[int, np.ndarray]]:
    """Per-stimulus {n_examples: predictive vector} for one condition, projecting hypothesis modes."""
    trajectory = load_condition_trajectory(condition, result_key)
    if not trajectory:
        return {}
    views = project_trajectory(condition.eval_mode, trajectory, build_task_context(condition.task, condition.domain))
    return {name: {n: view.vector for n, view in steps.items()} for name, steps in views.items()}


def _extension_rows(base: dict, extended: dict, stimuli) -> tuple[list, list]:
    """Rows of the two extension tables for one prompt."""
    mass = [{**base, "domain": EXTENSION_DOMAIN, **row} for row in extension_mass_rows(extended, stimuli)]
    discrimination = [{**base, "domain": EXTENSION_DOMAIN, **row} for row in rule_discrimination_rows(_full_stimulus_finals(extended, stimuli))]
    return mass, discrimination


def _save_extension_tables(analysis_dir: str, stage: str, tables: tuple[list, list]) -> None:
    for rows, suffix in zip(tables, ("mass", "rule_discrimination")):
        save_csv(rows, analysis_dir, f"{stage}_{suffix}.csv")


def _write_extension_tables(experiment, analysis_dir, stage: str, eval_modes: tuple[str, ...]) -> None:
    """Write extension mass and rule discrimination rows for the selected eval modes."""
    result_key = experiment.backend.result_key
    if EXTENSION_TASK not in experiment.tasks:
        return
    stimuli = build_task_context(EXTENSION_TASK, EXTENSION_DOMAIN).task.stimuli
    tables = ([], [])
    for condition in make_grid(experiment):
        if condition.task == EXTENSION_TASK and condition.domain == EXTENSION_DOMAIN and condition.eval_mode in eval_modes:
            base = {"task": EXTENSION_TASK, "eval_mode": condition.eval_mode, "h_space": condition.h_space, "sampling": condition.sampling}
            for table, new_rows in zip(tables, _extension_rows(base, _prediction_steps(condition, result_key), stimuli)):
                table.extend(new_rows)
    _save_extension_tables(analysis_dir, stage, tables)


def run_extension(experiment: Experiment, analysis_dir: str, figure_dir: str) -> None:
    """Posterior d200 extension diagnostics."""
    _write_extension_tables(experiment, analysis_dir, "extension", ("posterior",))


def run_hypothesis_extension(experiment: Experiment, analysis_dir: str, figure_dir: str) -> None:
    """Projected hypothesis_eval/hypothesis_generation d200 extension diagnostics."""
    _write_extension_tables(
        experiment, analysis_dir, "hypothesis_extension", ("hypothesis_eval", "hypothesis_generation"),
    )


def write_bayesian_extension_tables(experiment: Experiment, analysis_dir: str) -> None:
    """Bayesian reference rows for the same two extension tables."""
    baseline = bayesian_baseline(experiment, EXTENSION_TASK, EXTENSION_DOMAIN)
    stimuli, model = baseline.ctx.task.stimuli, baseline.model
    steps = {
        c.name: {n: model.predict(c.examples[:n]).cpu().numpy() for n in range(1, len(c.examples) + 1)}
        for c in stimuli
    }
    _save_extension_tables(analysis_dir, "extension", _extension_rows({"task": EXTENSION_TASK}, steps, stimuli))
