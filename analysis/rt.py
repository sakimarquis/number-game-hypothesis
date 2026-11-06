"""Human RT vs binary posterior entropy for Bigelow16."""

from __future__ import annotations

import os
from math import erfc, isfinite, sqrt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import rankdata

from analysis.cache import bayesian_baseline, load_condition_final
from analysis.metrics import safe_correlation
from analysis.plotter import C_BAYESIAN, C_LLM, save_csv, save_fig, set_mpl, with_model
from configs.layout import bayesian_cache_dir
from configs.settings import Experiment, ExperimentCondition
from tasks.stimuli import bigelow_stimulus, build_task_context
from utils import PROJECT_ROOT

set_mpl(size=8)

DEFAULT_RT_CSV = os.path.join(PROJECT_ROOT, "dataset", "numbergame_data.csv")
RT_PROMPT = ("explicit", "default")  # (h_space, sampling) of the LLM posterior compared with RTs
MIN_SUBJECTS_PER_TARGET = 5


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Bernoulli entropy in nats, vectorized over probabilities."""
    p = np.asarray(p, dtype=float)
    eps = 1e-12
    p_clipped = np.clip(p, eps, 1 - eps)
    return -(p_clipped * np.log(p_clipped) + (1 - p_clipped) * np.log(1 - p_clipped))


def load_bigelow16_rt(csv_path_or_buf) -> dict[str, dict[int, dict]]:
    """Median log10 RT per Bigelow16 stimulus and target."""
    df = pd.read_csv(csv_path_or_buf)
    df = df.dropna(subset=["rt", "target"]).copy()
    df = df[df["rt"] > 0]
    df["logrt"] = np.log10(df["rt"].astype(float))

    rt_by_stimulus: dict[str, dict[int, dict]] = {}
    for set_str, set_df in df.groupby("set"):
        stimulus_name, _examples = bigelow_stimulus(set_str)
        per_target: dict[int, dict] = {}
        for target, target_df in set_df.groupby("target"):
            per_target[int(target)] = {
                "logrt_median": float(target_df["logrt"].median()),
                "n_subjects": int(len(target_df)),
                "p_yes": float(target_df["rating"].mean()),
            }
        rt_by_stimulus[stimulus_name] = per_target
    return rt_by_stimulus


def build_rt_uncertainty_pairs(
    rt_by_stimulus: dict[str, dict[int, dict]],
    bayesian_preds: dict[str, np.ndarray],
    llm_preds: dict[str, np.ndarray] | None,
    stimulus_kind: dict[str, str],
    *,
    llm_log_preds: dict[str, np.ndarray] | None = None,
    min_subjects: int = MIN_SUBJECTS_PER_TARGET,
) -> pd.DataFrame:
    """Align RT rows with 1-based target posterior arrays and entropy columns.

    `llm_log_preds` (smoothed text-trial finals) feed H_llm; omitted, they equal `llm_preds`.
    """
    llm_log_preds = llm_preds if llm_log_preds is None else llm_log_preds
    rows = []
    for stimulus_name, targets in rt_by_stimulus.items():
        if stimulus_name not in bayesian_preds:
            continue
        bayesian = np.asarray(bayesian_preds[stimulus_name], dtype=float)
        llm = np.asarray(llm_preds[stimulus_name], dtype=float) if llm_preds and stimulus_name in llm_preds else None
        for target, info in targets.items():
            if info["n_subjects"] < min_subjects or not (1 <= target <= len(bayesian)):
                continue
            idx = target - 1
            row = {
                "stimulus": stimulus_name,
                "stimulus_kind": stimulus_kind.get(stimulus_name, ""),
                "target": int(target),
                "n_subjects": info["n_subjects"],
                "logrt_median": info["logrt_median"],
                "p_human": info["p_yes"],
                "p_bayesian": float(bayesian[idx]),
                "H_bayesian": float(binary_entropy(bayesian[idx])),
                "p_llm": np.nan,
                "H_llm": np.nan,
            }
            if llm is not None:
                row["p_llm"] = float(llm[idx])
                row["H_llm"] = float(binary_entropy(llm_log_preds[stimulus_name][idx]))
            rows.append(row)
    return pd.DataFrame(rows)


def _spearman_with_normal_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho plus a normal-approx p-value for display-only plot titles.

    rho is Pearson on average ranks (scipy.stats.spearmanr's statistic); the
    p-value is a deliberate standard-normal approximation of the t-statistic,
    not spearmanr's Student-t p-value, so it is computed here rather than taken
    from spearmanr.
    """
    rho = safe_correlation(rankdata(x), rankdata(y))
    n = len(x)
    if n < 3 or not isfinite(rho) or abs(rho) >= 1.0:
        return rho, 0.0 if abs(rho) >= 1.0 else float("nan")
    t_stat = abs(rho) * sqrt((n - 2) / (1 - rho ** 2))
    return rho, erfc(t_stat / sqrt(2.0))


def plot_rt_vs_entropy(
    df: pd.DataFrame,
    source: str,
    output_path: str,
    result_key: str = "",
) -> None:
    """Scatter H(p_source) against median log10 RT with Spearman rho in title."""
    h_col = f"H_{source}"
    sub = df.dropna(subset=[h_col, "logrt_median"])
    if sub.empty:
        return

    x = sub[h_col].to_numpy()
    y = sub["logrt_median"].to_numpy()
    rho, p_value = _spearman_with_normal_pvalue(x, y)
    label = "Bayesian" if source == "bayesian" else (result_key or "LLM")
    color = C_BAYESIAN if source == "bayesian" else C_LLM

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.scatter(x, y, s=12, color=color, alpha=0.5, edgecolors="white", linewidths=0.3)
    ax.set_xlabel(f"Binary entropy H(p_{source}) [nats]")
    ax.set_ylabel(r"log$_{10}$ median RT [ms]")
    ax.set_title(with_model(f"{label}: H(p) vs log RT (n={len(sub)}, rho={rho:.3f}, p~={p_value:.2g})", ""))
    save_fig(fig, output_path)


def run_rt_correlation(experiment: Experiment, analysis_dir: str, figure_dir: str) -> pd.DataFrame:
    """Build Bigelow16 RT/entropy pairs and save one CSV plus scatter figures."""
    result_key = experiment.backend.result_key
    if "Bigelow16" not in experiment.tasks:
        return pd.DataFrame()

    domain = experiment.tasks["Bigelow16"].domains[0]
    ctx = build_task_context("Bigelow16", domain)
    stimulus_map = ctx.task.stimulus_map
    bayesian_preds = bayesian_baseline(experiment, "Bigelow16", domain).predictions
    condition = ExperimentCondition("Bigelow16", "posterior", *RT_PROMPT, domain)
    llm_preds = load_condition_final(condition, result_key)

    rt_by_stimulus = {
        name: targets
        for name, targets in load_bigelow16_rt(DEFAULT_RT_CSV).items()
        if name in stimulus_map
    }
    stimulus_kind = {name: stimulus.kind for name, stimulus in stimulus_map.items()}
    df = build_rt_uncertainty_pairs(
        rt_by_stimulus, bayesian_preds, llm_preds, stimulus_kind,
        llm_log_preds=load_condition_final(condition, result_key, log_space=True),
    )
    if df.empty:
        print("  rt_correlation: no aligned (stimulus, target) rows, skipping")
        return df

    save_csv(df, analysis_dir, "rt_correlation_pairs.csv")
    plot_rt_vs_entropy(df, "bayesian", os.path.join(bayesian_cache_dir("Bigelow16", domain), "rt_vs_entropy_bayesian.png"))
    if llm_preds is not None:
        plot_rt_vs_entropy(df, "llm", os.path.join(figure_dir, "rt_vs_entropy_llm.png"), result_key)
    return df
