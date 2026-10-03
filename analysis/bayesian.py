"""Bayesian baseline orchestration: Bayesian + Human reference artifacts.

Run by `run_number_game.py --model bayesian`. Produces the model-level overlay
CSVs (alpha_beta_fits, divergence_metrics, effective_support), the Tenenbaum99
d200 extension references, human description top-1 metrics, stimulus-level
Bayesian decomposition plots, and human α-β fits, so LLM-side analyses and the
cross-model summary read them as references without special-casing.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analysis.alpha_beta_scope import (
    ALPHA_BETA_COLUMNS, FULL_SCOPE, PREFIX_LENGTHS, alpha_beta_row, scope_label, task_group_label,
)
from analysis.cache import build_bayesian, build_human_analysis
from analysis.extension import write_bayesian_extension_tables
from analysis.hypothesis_space import load_hypothesis_library
from analysis.metrics import safe_correlation, kl_divergence
from analysis.model import fit_alpha_beta
from analysis.plotter import C_ACCENT, C_HUMAN, C_INTERVAL, C_RULE, save_csv, save_fig, set_mpl
from analysis.posterior import plot_alpha_beta_all, plot_bayesian_decompositions
from analysis.size_principle import compute_effective_support
from configs.layout import analysis_dir, bayesian_cache_dir, human_analysis_dir
from configs.settings import make_grid
from tasks.stimuli import HUMAN_NATIVE_DOMAIN
from utils import PROJECT_ROOT

BAYESIAN = "bayesian"
HUMAN = "human"
HUMAN_DESCRIPTION_CSV = os.path.join(PROJECT_ROOT, "dataset", "set_descriptions_typed.csv")
HUMAN_DESCRIPTION_TYPES = ("rule", "interval", "dk", "other")
HUMAN_DESCRIPTION_TYPE_COLORS = {"rule": C_RULE, "interval": C_INTERVAL, "dk": C_ACCENT, "other": C_HUMAN}
_BASE_COLUMNS = ["task", "stimulus", "stimulus_kind", "domain", "h_space", "sampling"]
BASELINE_COLUMNS = {
    "alpha_beta_fits.csv": ALPHA_BETA_COLUMNS,
    "divergence_metrics.csv": _BASE_COLUMNS + ["eval_mode", "kl_bayes", "correlation"],
    "effective_support.csv": _BASE_COLUMNS + ["effective_support"],
}


def _row_base(task: str, stimulus, domain: int) -> dict:
    return {
        "task": task,
        "stimulus": stimulus.name,
        "stimulus_kind": stimulus.kind,
        "domain": domain,
        "h_space": "",
        "sampling": "",
    }


def _alpha_beta_scope_counts(stimuli) -> list[tuple[str, int | None, int]]:
    """Fit scopes shared by baseline and LLM aggregate alpha-beta rows."""
    rows = [
        (scope_label(n), n, sum(len(stimulus.examples) >= n for stimulus in stimuli))
        for n in PREFIX_LENGTHS
    ]
    rows.append((scope_label(None), None, len(stimuli)))
    return [row for row in rows if row[2]]


def human_divergence_rows(task_data, task: str, domain: int, bay_preds: dict) -> list[dict]:
    """Human posterior KL and Pearson correlation against Bayesian predictions."""
    rows = []
    for stimulus in task_data.stimuli:
        if stimulus.name not in task_data.human_preds:
            continue
        human = np.asarray(task_data.human_preds[stimulus.name], dtype=float)
        bayes = np.asarray(bay_preds[stimulus.name], dtype=float)
        mask = ~np.isnan(human)
        if mask.sum() < 2:
            continue
        h, b = human[mask], bayes[mask]
        rows.append({
            **_row_base(task, stimulus, domain),
            "eval_mode": "posterior",
            "kl_bayes": kl_divergence(b, h),
            "correlation": safe_correlation(b, h),
        })
    return rows


def human_effective_support_rows(task_data, task: str, domain: int) -> list[dict]:
    """Effective support of dense human p(yes); sparse NaN targets are skipped."""
    rows = []
    for stimulus in task_data.stimuli:
        human = task_data.human_preds.get(stimulus.name)
        if human is None or np.isnan(human).any():
            continue
        rows.append({**_row_base(task, stimulus, domain), "effective_support": int(compute_effective_support(human))})
    return rows


def human_description_top1_rows(descriptions: pd.DataFrame) -> pd.DataFrame:
    """Executable top-1 metrics for subject-written Bigelow descriptions.

    Support and evidence accuracy are defined only for rows mapped to a d100
    library predicate (`library_fn` = fn_NNNN); rule proportion uses every typed row.
    """
    library = load_hypothesis_library(HUMAN_NATIVE_DOMAIN)
    rows = descriptions.dropna(subset=["set", "type"]).copy()
    rows["n_examples"] = rows["set"].astype(str).str.split("_ ").map(len)
    rows["rule_proportion"] = rows["type"].eq("rule").astype(float)
    support_fraction, accuracy = [], []
    for row in rows.itertuples(index=False):
        if not isinstance(row.library_fn, str) or not row.library_fn:
            support_fraction.append(np.nan)
            accuracy.append(np.nan)
            continue
        support = library.vectors[int(row.library_fn.removeprefix("fn_"))] > 0
        examples = np.asarray([int(x) for x in str(row.set).split("_ ")], dtype=int) - 1
        support_fraction.append(float(support.mean()))
        accuracy.append(float(support[examples].all()))
    rows["support_fraction"] = support_fraction
    rows["accuracy"] = accuracy
    rows["domain"] = HUMAN_NATIVE_DOMAIN
    return rows[rows["n_examples"].isin(PREFIX_LENGTHS)][
        ["set", "type", "domain", "n_examples", "rule_proportion", "support_fraction", "accuracy"]
    ]


def plot_human_description_types_by_n_examples(descriptions: pd.DataFrame, output_path: str) -> None:
    """Stack subject-written description type proportions by evidence prefix length."""
    set_mpl(size=8)
    work = descriptions.dropna(subset=["set", "type"]).copy()
    work["n_examples"] = work["set"].astype(str).str.split("_ ").map(len)
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
            bucket = alpha_beta_groups.setdefault((domain, fit_scope, n_examples), {"tasks": [], "n_stimuli": 0})
            bucket["tasks"].append(task)
            bucket["n_stimuli"] += n_stimuli
        rows["divergence_metrics.csv"].extend(
            {**_row_base(task, stimulus, domain), "eval_mode": "posterior", "kl_bayes": 0.0, "correlation": 1.0}
            for stimulus in task_data.stimuli
        )
        rows["effective_support.csv"].extend(
            {**_row_base(task, stimulus, domain),
             "effective_support": int(compute_effective_support(baseline.predictions[stimulus.name]))}
            for stimulus in task_data.stimuli
        )
    for (domain, fit_scope, n_examples), bucket in alpha_beta_groups.items():
        # The Bayesian baseline is the fixed full-Bayesian reference (α=β=1), not a fit,
        # so loss and goodness-of-fit (r²) are undefined; r²=NaN is dropped downstream.
        fit = {"best_alpha": 1.0, "best_beta": 1.0, "best_loss": 0.0, "r2": float("nan"), "n_stimuli": bucket["n_stimuli"]}
        rows["alpha_beta_fits.csv"].append(
            alpha_beta_row(task_group_label(bucket["tasks"]), domain, "", "", fit_scope, n_examples, fit)
        )
    paths = _write_baseline_csvs(BAYESIAN, rows)
    write_bayesian_extension_tables(experiment, analysis_dir(BAYESIAN))
    return paths


def run_human_baselines(baselines: dict) -> dict[str, str]:
    """Human baseline CSVs and description figures under ``results/human/analysis/``."""
    rows = {name: [] for name in BASELINE_COLUMNS}
    alpha_beta_groups = {}
    for (task, domain), baseline in baselines.items():
        task_data = baseline.ctx.task
        if not task_data.human_preds:
            continue
        bucket = alpha_beta_groups.setdefault(domain, {"tasks": [], "specs": []})
        bucket["tasks"].append(task)
        bucket["specs"].append((baseline.model, task_data.human_preds, task_data.stimuli))
        rows["divergence_metrics.csv"].extend(human_divergence_rows(task_data, task, domain, baseline.predictions))
        rows["effective_support.csv"].extend(human_effective_support_rows(task_data, task, domain))
    for domain, bucket in alpha_beta_groups.items():
        fit = fit_alpha_beta(bucket["specs"])
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


def run_baseline_analyses(experiment) -> None:
    """Generate all model-agnostic Bayesian and Human baseline artifacts."""
    conditions = make_grid(experiment)
    print(f"Baseline grid: {len(conditions)} conditions")
    print("\nBuilding Bayesian models...")
    baselines = build_bayesian(experiment, conditions, recompute=True)

    # Bayesian decomposition plots: top hypotheses per stimulus for each (task, domain).
    for (task_name, domain), baseline in baselines.items():
        task_data = baseline.ctx.task
        plot_bayesian_decompositions(
            baseline.predictions,
            {c.name: baseline.model.top_hypotheses(c.examples, 15) for c in task_data.stimuli},
            [c.name for c in task_data.stimuli],
            bayesian_cache_dir(task_name, domain),
            featured=task_data.featured_stimuli,
        )

    # Human α-β ablation panels: one set per (task, domain) where human ratings exist.
    for (task_name, domain), fit in build_human_analysis(baselines).items():
        baseline = baselines[(task_name, domain)]
        human_preds = baseline.ctx.task.human_preds
        plot_alpha_beta_all(
            fit, baseline.predictions, human_preds, [c.name for c in baseline.ctx.task.stimuli if c.name in human_preds],
            os.path.join(human_analysis_dir(task_name, domain), "alpha_beta"),
            result_key="Human",
        )

    print("\nWriting Bayesian baseline CSVs...")
    run_bayesian_baselines(experiment, baselines)
    print("\nWriting Human baseline CSVs...")
    run_human_baselines(baselines)


def load_baseline_overlay(csv_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read Bayesian and Human overlay CSVs; return empty frames for missing caches."""
    def read(producer: str) -> pd.DataFrame:
        path = os.path.join(analysis_dir(producer), csv_name)
        return pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()

    return read(BAYESIAN), read(HUMAN)
