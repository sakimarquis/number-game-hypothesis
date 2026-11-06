"""Bayesian baseline orchestration: Bayesian + Human reference artifacts.

Run by `run_number_game.py --model bayesian`. Produces the model-level overlay
CSVs (alpha_beta_fits, divergence_metrics, prompt_effective_support), the Tenenbaum99
d200 extension references, human description top-1 metrics, stimulus-level
Bayesian decomposition plots, and human size-principle fits, so LLM-side analyses and the
cross-model summary read them as references without special-casing.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analysis.alpha_beta_scope import (
    ALPHA_BETA_COLUMNS, FULL_SCOPE, PREFIX_LENGTHS, alpha_beta_row, scope_label, stimulus_row, task_group_label,
)
from analysis.cache import bayesian_baseline
from analysis.extension import write_bayesian_extension_tables
from analysis.hypothesis_structuring import load_hypothesis_library
from analysis.metrics import kl_divergence, rated_targets, safe_correlation, shannon_entropy
from analysis.model import fit_beta, fit_beta_with_ablations
from analysis.plotter import C_ACCENT, C_HUMAN, C_INTERVAL, C_RULE, save_csv, save_fig, set_mpl
from analysis.prior_pooling import pooled_prior, pooled_priors
from analysis.condition_plots import plot_alpha_beta_all, plot_bayesian_decompositions
from analysis.trajectory import clip_to_prefix_horizon, compute_effective_support
from configs.layout import analysis_dir, bayesian_cache_dir, human_analysis_dir
from configs.settings import BAYESIAN, HUMAN, make_grid
from tasks.stimuli import BASE_DOMAIN, bigelow_stimulus
from utils import PROJECT_ROOT

HUMAN_DESCRIPTION_CSV = os.path.join(PROJECT_ROOT, "dataset", "set_descriptions_typed.csv")
HUMAN_DESCRIPTION_TYPES = ("rule", "interval", "dk", "other")
HUMAN_DESCRIPTION_TYPE_COLORS = {"rule": C_RULE, "interval": C_INTERVAL, "dk": C_ACCENT, "other": C_HUMAN}
_BASE_COLUMNS = ["task", "stimulus", "stimulus_kind", "domain", "h_space", "sampling"]
# Reference tables share the LLM stage file names so overlays read them without special-casing.
BASELINE_COLUMNS = {
    "alpha_beta_fits.csv": ALPHA_BETA_COLUMNS,
    "divergence_metrics.csv": _BASE_COLUMNS + ["eval_mode", "kl_bayes", "correlation", "entropy"],
    "prompt_effective_support.csv": _BASE_COLUMNS + ["effective_support"],
}


def _alpha_beta_scope_counts(stimuli) -> list[tuple[str, int | None, int]]:
    """Fit scopes shared by baseline and LLM aggregate size-principle rows; prefix scopes use the trajectory-horizon stimuli."""
    horizon_stimuli, _ = clip_to_prefix_horizon(stimuli, {})
    rows = [
        (scope_label(n), n, sum(len(stimulus.examples) >= n for stimulus in horizon_stimuli))
        for n in PREFIX_LENGTHS
    ]
    rows.append((scope_label(None), None, len(stimuli)))
    return [row for row in rows if row[2]]


def human_divergence_rows(task_data, task: str, domain: int, bay_preds: dict) -> list[dict]:
    """Human posterior KL and Pearson correlation against Bayesian predictions, plus entropy of dense ratings."""
    rows = []
    for stimulus in task_data.stimuli:
        if stimulus.name not in task_data.human_preds:
            continue
        human = np.asarray(task_data.human_preds[stimulus.name], dtype=float)
        bayes = np.asarray(bay_preds[stimulus.name], dtype=float)
        mask = rated_targets(human)
        if mask is None:
            continue
        h, b = human[mask], bayes[mask]
        rows.append({
            **stimulus_row(task, stimulus, domain),
            "eval_mode": "posterior",
            "kl_bayes": kl_divergence(b, h),
            "correlation": safe_correlation(b, h),
            # Entropy over a sparse subset of targets is not comparable to model entropy over the domain.
            "entropy": shannon_entropy(human, empty=0.0) if mask.all() else float("nan"),
        })
    return rows


def human_effective_support_rows(task_data, task: str, domain: int) -> list[dict]:
    """Effective support of dense human p(yes); sparse NaN targets are skipped."""
    rows = []
    for stimulus in task_data.stimuli:
        human = task_data.human_preds.get(stimulus.name)
        if human is None or np.isnan(human).any():
            continue
        rows.append({**stimulus_row(task, stimulus, domain), "effective_support": int(compute_effective_support(human))})
    return rows


def human_description_top1_rows(descriptions: pd.DataFrame) -> pd.DataFrame:
    """Executable top-1 metrics for subject-written Bigelow descriptions.

    Hypothesis size and evidence accuracy are defined only for rows mapped to a d100
    library predicate (`library_fn` = fn_NNNN); rule preference uses every typed row.
    """
    library = load_hypothesis_library(BASE_DOMAIN)
    rows = descriptions.dropna(subset=["set", "type"]).copy()
    rows["n_examples"] = rows["set"].map(lambda s: len(bigelow_stimulus(s)[1]))
    rows["rule_preference"] = rows["type"].eq("rule").astype(float)
    hypothesis_size, accuracy = [], []
    for row in rows.itertuples(index=False):
        if not isinstance(row.library_fn, str) or not row.library_fn:
            hypothesis_size.append(np.nan)
            accuracy.append(np.nan)
            continue
        support = library.vectors[int(row.library_fn.removeprefix("fn_"))] > 0
        examples = np.asarray(bigelow_stimulus(row.set)[1]) - 1
        hypothesis_size.append(float(support.mean()))
        accuracy.append(float(support[examples].all()))
    rows["hypothesis_size"] = hypothesis_size
    rows["accuracy"] = accuracy
    rows["domain"] = BASE_DOMAIN
    return rows[rows["n_examples"].isin(PREFIX_LENGTHS)][
        ["set", "type", "domain", "n_examples", "rule_preference", "hypothesis_size", "accuracy"]
    ]


def plot_human_description_types_by_n_examples(descriptions: pd.DataFrame, output_path: str) -> None:
    """Stack subject-written description type proportions by evidence prefix length."""
    set_mpl(size=8)
    work = descriptions.dropna(subset=["set", "type"]).copy()
    work["n_examples"] = work["set"].map(lambda s: len(bigelow_stimulus(s)[1]))
    counts = (
        work[work["n_examples"].isin(PREFIX_LENGTHS)]
        .groupby(["n_examples", "type"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=list(PREFIX_LENGTHS), columns=HUMAN_DESCRIPTION_TYPES, fill_value=0)
    )
    props = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    bottom = np.zeros(len(props))
    x = np.arange(len(props))
    for label in HUMAN_DESCRIPTION_TYPES:
        vals = props[label].to_numpy()
        ax.bar(x, vals, bottom=bottom, width=0.65, color=HUMAN_DESCRIPTION_TYPE_COLORS[label], label=label)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in props.index])
    ax.set_xlabel("Number of examples")
    ax.set_ylabel("Proportion")
    ax.set_title("Human Description Types by Number of Examples")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save_fig(fig, output_path)


def _write_baseline_csvs(producer: str, rows_by_name: dict[str, list[dict]]) -> dict[str, str]:
    out_dir = analysis_dir(producer)
    for name in BASELINE_COLUMNS:
        save_csv(pd.DataFrame(rows_by_name[name], columns=BASELINE_COLUMNS[name]), out_dir, name)
    return {name: os.path.join(out_dir, name) for name in BASELINE_COLUMNS}


def run_bayesian_baselines(experiment, baselines: dict) -> dict[str, str]:
    """Bayesian baseline CSVs under ``results/bayesian/analysis/``."""
    rows = {name: [] for name in BASELINE_COLUMNS}
    alpha_beta_groups = {}
    for (task, domain), baseline in baselines.items():
        task_data = baseline.ctx.task
        for fit_scope, n_examples, n_stimuli in _alpha_beta_scope_counts(task_data.stimuli):
            # Like LLM fits, every fit scope pools tasks.
            key = (domain, fit_scope, n_examples)
            bucket = alpha_beta_groups.setdefault(key, {"tasks": [], "n_stimuli": 0})
            bucket["tasks"].append(task)
            bucket["n_stimuli"] += n_stimuli
        rows["divergence_metrics.csv"].extend(
            {**stimulus_row(task, stimulus, domain), "eval_mode": "posterior", "kl_bayes": 0.0, "correlation": 1.0}
            for stimulus in task_data.stimuli
        )
        rows["prompt_effective_support.csv"].extend(
            {**stimulus_row(task, stimulus, domain),
             "effective_support": int(compute_effective_support(baseline.predictions[stimulus.name]))}
            for stimulus in task_data.stimuli
        )
    for (domain, fit_scope, n_examples), bucket in alpha_beta_groups.items():
        # The Bayesian baseline is the fixed full-Bayesian reference (α=β=1), not a fit,
        # so loss and goodness-of-fit (r²) are undefined; r²=NaN is dropped downstream.
        fit = {"alpha": 1.0, "beta": 1.0, "mse": 0.0, "r2": float("nan"), "n_stimuli": bucket["n_stimuli"]}
        rows["alpha_beta_fits.csv"].append(
            alpha_beta_row(task_group_label(bucket["tasks"]), domain, "", "", fit_scope, n_examples, fit)
        )
    paths = _write_baseline_csvs(BAYESIAN, rows)
    write_bayesian_extension_tables(experiment, analysis_dir(BAYESIAN))
    return paths


def run_human_baselines(experiment, baselines: dict) -> dict[str, str]:
    """Human baseline CSVs and description figures under ``results/human/analysis/``.

    Human size-principle fits use the default prompt's pooled prior.
    """
    prior = pooled_prior(experiment)
    rows = {name: [] for name in BASELINE_COLUMNS}
    alpha_beta_groups = {}
    for (task, domain), baseline in baselines.items():
        task_data = baseline.ctx.task
        if not task_data.human_preds:
            continue
        bucket = alpha_beta_groups.setdefault(domain, {"tasks": [], "specs": []})
        bucket["tasks"].append(task)
        priors = pooled_priors(prior, baseline.model, baseline.ctx.rule_families)
        bucket["specs"].append((baseline.model, priors, task_data.human_preds, task_data.stimuli))
        rows["divergence_metrics.csv"].extend(human_divergence_rows(task_data, task, domain, baseline.predictions))
        rows["prompt_effective_support.csv"].extend(human_effective_support_rows(task_data, task, domain))
    for domain, bucket in alpha_beta_groups.items():
        fit = {**fit_beta(bucket["specs"]), "alpha": prior["alpha"]}
        rows["alpha_beta_fits.csv"].append(
            alpha_beta_row(task_group_label(bucket["tasks"]), domain, "", "", FULL_SCOPE, None, fit)
        )
    paths = _write_baseline_csvs(HUMAN, rows)

    descriptions = pd.read_csv(HUMAN_DESCRIPTION_CSV)
    save_csv(human_description_top1_rows(descriptions), analysis_dir(HUMAN), "description_top1.csv")
    plot_human_description_types_by_n_examples(
        descriptions, os.path.join(analysis_dir(HUMAN), "description_types_by_n_examples.png"),
    )
    return paths


def build_human_analysis(experiment, baselines: dict) -> dict[tuple[str, int], dict]:
    """Fit the human size principle with ablations per (task, domain) under the default pooled prior.

    NaN targets (sparse Tenenbaum99 ratings) are masked.
    """
    prior = pooled_prior(experiment)
    fits = {}
    for (task_name, domain), baseline in baselines.items():
        task_data = baseline.ctx.task
        if not task_data.human_preds:
            continue
        priors = pooled_priors(prior, baseline.model, baseline.ctx.rule_families)
        fit = fit_beta_with_ablations(baseline.model, priors, task_data.human_preds, task_data.stimuli)
        print(f"  [{task_name} d{domain}] Human size-principle fit: β={fit['beta']:.2f} (MSE={fit['mse']:.6f})")
        fits[(task_name, domain)] = fit
    return fits


def run_baseline_analyses(experiment) -> None:
    """Generate all model-agnostic Bayesian and Human baseline artifacts."""
    conditions = make_grid(experiment)
    print(f"Baseline grid: {len(conditions)} conditions")
    print("\nBuilding Bayesian models...")
    baselines = {key: bayesian_baseline(experiment, *key) for key in dict.fromkeys(c.context_key for c in conditions)}

    # Bayesian decomposition plots: top hypotheses per stimulus for each (task, domain).
    for (task_name, domain), baseline in baselines.items():
        task_data = baseline.ctx.task
        plot_bayesian_decompositions(
            baseline.predictions,
            {c.name: baseline.model.top_hypotheses(c.examples, 15) for c in task_data.stimuli},
            [c.name for c in task_data.stimuli],
            bayesian_cache_dir(task_name, domain),
            featured=experiment.tasks[task_name].featured_stimuli,
        )

    # Human size-principle ablation panels: one set per (task, domain) where human ratings exist.
    for (task_name, domain), fit in build_human_analysis(experiment, baselines).items():
        baseline = baselines[(task_name, domain)]
        human_preds = baseline.ctx.task.human_preds
        plot_alpha_beta_all(
            fit, baseline.predictions, human_preds, [c.name for c in baseline.ctx.task.stimuli if c.name in human_preds],
            os.path.join(human_analysis_dir(task_name, domain), "alpha_beta"),
        )

    print("\nWriting Bayesian baseline CSVs...")
    run_bayesian_baselines(experiment, baselines)
    print("\nWriting Human baseline CSVs...")
    run_human_baselines(experiment, baselines)


def load_baseline_overlay(csv_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read Bayesian and Human overlay CSVs; return empty frames for missing caches."""
    def read(producer: str) -> pd.DataFrame:
        path = os.path.join(analysis_dir(producer), csv_name)
        return pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()

    return read(BAYESIAN), read(HUMAN)
