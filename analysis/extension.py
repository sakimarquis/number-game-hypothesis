"""Tenenbaum99 d200 extension diagnostics shared by model stages and the Bayesian baseline.

- extension mass:      mean P(yes) on targets 101..200 per evidence prefix (d200)
- rule discrimination: P(yes) on rule vs non-rule targets in 101..200 for the three rule stimuli (d200)
- domain shift:        KL between the d100 prediction and the renormalized d200 mass on 1..100,
                       plus the d200 mass fraction on 101..200 (full stimuli)
"""

from __future__ import annotations

import numpy as np

from analysis.alpha_beta_scope import PREFIX_LENGTHS
from analysis.cache import bayesian_model, load_condition_trajectory, task_context, trajectory_steps
from analysis.hypothesis_space import (
    hypothesis_support,
    load_hypothesis_library,
    normalize_structured_generation_confidence,
    project_hypothesis_eval,
    project_structured_hypothesis_generation,
)
from analysis.metrics import kl_divergence
from analysis.plotter import save_csv
from configs.layout import ROOT_DOMAIN
from configs.settings import Experiment, make_grid
from tasks.hypotheses import default_registry

EXTENSION_TASK = "Tenenbaum99"
EXTENSION_DOMAIN = 200
EXTENSION_WINDOW_START = ROOT_DOMAIN + 1
EXTENSION_STIMULUS_KINDS = ("rule", "similarity")
# Registry base whose extension defines rule-positive targets for each rule stimulus.
RULE_STIMULUS_BASES = {"D={16,8,2,64}": "pow_2", "D={60,80,10,30}": "mult_10", "D={81,25,4,36}": "squares"}


def _extension_stimuli(stimuli) -> list:
    return [
        stimulus for stimulus in stimuli
        if stimulus.kind in EXTENSION_STIMULUS_KINDS and len(stimulus.examples) >= max(PREFIX_LENGTHS)
    ]


def extension_mass_rows(prefix_vectors: dict[str, dict[int, np.ndarray]], stimuli) -> list[dict]:
    """One row per (stimulus, prefix) with mean d200 mass on targets 101..200."""
    rows = []
    for stimulus in _extension_stimuli(stimuli):
        for n, vector in prefix_vectors.get(stimulus.name, {}).items():
            if n in PREFIX_LENGTHS:
                rows.append({
                    "stimulus": stimulus.name, "stimulus_kind": stimulus.kind, "n_examples": n,
                    "extension_mass": float(np.asarray(vector, dtype=float)[EXTENSION_WINDOW_START - 1:].mean()),
                })
    return rows


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


def domain_shift_rows(base_finals: dict[str, np.ndarray], extended_finals: dict[str, np.ndarray], stimuli) -> list[dict]:
    """Compare full-stimulus d100 shape against renormalized d200 mass on 1..100."""
    rows = []
    for stimulus in stimuli:
        if stimulus.name not in base_finals or stimulus.name not in extended_finals:
            continue
        base = np.asarray(base_finals[stimulus.name], dtype=float)
        extended = np.asarray(extended_finals[stimulus.name], dtype=float)
        head, tail = extended[:ROOT_DOMAIN], extended[ROOT_DOMAIN:]
        total = float(extended.sum())
        rows.append({
            "stimulus": stimulus.name,
            "stimulus_kind": stimulus.kind,
            "n_examples": len(stimulus.examples),
            "shape_kl": kl_divergence(base, head),
            "d200_tail_fraction": float(tail.sum() / total) if total > 0.0 else float("nan"),
            "d200_head_mean": float(head.mean()),
            "d200_tail_mean": float(tail.mean()),
        })
    return rows


def _full_stimulus_finals(steps_by_stimulus: dict[str, dict[int, np.ndarray]], stimuli) -> dict[str, np.ndarray]:
    return {
        stimulus.name: steps_by_stimulus[stimulus.name][len(stimulus.examples)]
        for stimulus in stimuli
        if len(stimulus.examples) in steps_by_stimulus.get(stimulus.name, {})
    }


def _prediction_steps(experiment: Experiment, condition, result_key: str) -> dict[str, dict[int, np.ndarray]]:
    """Per-stimulus {n_examples: predictive vector} for one condition, projecting hypothesis modes."""
    trajectory = load_condition_trajectory(condition, result_key)
    if not trajectory:
        return {}
    ctx = task_context(experiment, condition.task, condition.domain)
    stimulus_map = ctx.task.stimulus_map
    if condition.eval_mode == "hypothesis_generation":
        library = load_hypothesis_library(condition.domain)
    steps = {}
    for name in trajectory:
        steps[name] = {}
        for n, result in trajectory_steps(trajectory, name).items():
            if condition.eval_mode == "posterior":
                steps[name][n] = np.asarray(result, dtype=float)
            elif condition.eval_mode == "hypothesis_eval":
                steps[name][n] = project_hypothesis_eval(result, stimulus_map[name].examples[:n], ctx).vector
            else:
                steps[name][n] = project_structured_hypothesis_generation(
                    normalize_structured_generation_confidence(result), library,
                ).vector
    return steps


def _write_extension_tables(experiment, output_dir, result_key, stage: str, eval_modes: tuple[str, ...]) -> None:
    """Write extension mass, rule discrimination, and domain-shift rows for the selected eval modes."""
    if EXTENSION_TASK not in experiment.tasks:
        return
    stimuli = task_context(experiment, EXTENSION_TASK, EXTENSION_DOMAIN).task.stimuli
    by_prompt = {}
    for condition in make_grid(experiment):
        if condition.task == EXTENSION_TASK and condition.eval_mode in eval_modes:
            by_prompt.setdefault((condition.eval_mode, condition.h_space, condition.sampling), {})[condition.domain] = condition

    mass_rows, discrimination_rows, shift_rows = [], [], []
    for (eval_mode, h_space, sampling), conditions in by_prompt.items():
        base = {"task": EXTENSION_TASK, "eval_mode": eval_mode, "h_space": h_space, "sampling": sampling}
        if EXTENSION_DOMAIN not in conditions:
            continue
        extended = _prediction_steps(experiment, conditions[EXTENSION_DOMAIN], result_key)
        extended_finals = _full_stimulus_finals(extended, stimuli)
        mass_rows += [{**base, "domain": EXTENSION_DOMAIN, **row} for row in extension_mass_rows(extended, stimuli)]
        discrimination_rows += [{**base, "domain": EXTENSION_DOMAIN, **row} for row in rule_discrimination_rows(extended_finals)]
        if ROOT_DOMAIN in conditions:
            base_finals = _full_stimulus_finals(_prediction_steps(experiment, conditions[ROOT_DOMAIN], result_key), stimuli)
            shift_rows += [{**base, **row} for row in domain_shift_rows(base_finals, extended_finals, stimuli)]

    save_csv(mass_rows, output_dir, f"{stage}_mass.csv")
    save_csv(discrimination_rows, output_dir, f"{stage}_rule_discrimination.csv")
    save_csv(shift_rows, output_dir, f"{stage}_domain_shift.csv")


def run_extension(experiment: Experiment, output_dir: str, result_key: str, figure_dir: str) -> None:
    """Posterior d200 extension diagnostics."""
    _write_extension_tables(experiment, output_dir, result_key, "extension", ("posterior",))


def run_hypothesis_extension(experiment: Experiment, output_dir: str, result_key: str, figure_dir: str) -> None:
    """Projected hypothesis_eval/hypothesis_generation d200 extension diagnostics."""
    _write_extension_tables(
        experiment, output_dir, result_key, "hypothesis_extension", ("hypothesis_eval", "hypothesis_generation"),
    )


def write_bayesian_extension_tables(experiment: Experiment, output_dir: str) -> None:
    """Bayesian reference rows for the same three extension tables."""
    stimuli = task_context(experiment, EXTENSION_TASK, EXTENSION_DOMAIN).task.stimuli
    models = {domain: bayesian_model(experiment, EXTENSION_TASK, domain) for domain in (ROOT_DOMAIN, EXTENSION_DOMAIN)}
    steps = {
        domain: {
            c.name: {n: model.predict(c.examples[:n]).cpu().numpy() for n in range(1, len(c.examples) + 1)}
            for c in stimuli
        }
        for domain, model in models.items()
    }
    extended_finals = _full_stimulus_finals(steps[EXTENSION_DOMAIN], stimuli)
    base = {"task": EXTENSION_TASK}
    save_csv([{**base, "domain": EXTENSION_DOMAIN, **row} for row in extension_mass_rows(steps[EXTENSION_DOMAIN], stimuli)],
             output_dir, "extension_mass.csv")
    save_csv([{**base, "domain": EXTENSION_DOMAIN, **row} for row in rule_discrimination_rows(extended_finals)],
             output_dir, "extension_rule_discrimination.csv")
    save_csv([{**base, **row} for row in domain_shift_rows(_full_stimulus_finals(steps[ROOT_DOMAIN], stimuli), extended_finals, stimuli)],
             output_dir, "extension_domain_shift.csv")
