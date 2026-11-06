"""Nested-pair size-principle diagnostic on final hypothesis_eval outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.cache import load_condition_final
from analysis.hypothesis_space import prompt_items
from analysis.metrics import shannon_entropy
from analysis.plotter import save_csv, with_all_kinds
from configs.settings import Experiment, make_grid
from tasks.prompt_space import build_prompt_space_spec
from tasks.stimuli import build_task_context


# ===== Nested-pair size-principle diagnostic =====
# Compares small-vs-large posterior odds for prompt-local hypothesis pairs
# whose extensions are nested and both consistent with the observed evidence.
# Restricted to direct-semantics prompt spaces; grouped-transform families skipped.

PROB_FLOOR = 1e-6
MIN_PAIR_PROB = 1e-4
COMPARISONS = {
    "strong_vs_weak": ("strong", "weak"),
    "strong_vs_default": ("strong", "default"),
}
SLOPE_INPUT_COLS = ["log_size_ratio", "delta_log_odds", "delta_entropy", "baseline_log_odds"]


def enumerate_nested_consistent_pairs(
    prompt_labels: list[str],
    extensions: dict[str, frozenset[int]],
    examples: list[int],
) -> list[dict[str, object]]:
    """Enumerate strict small<large pairs within one prompt-local hypothesis list."""
    evidence = frozenset(examples)
    consistent = [
        label for label in prompt_labels
        if label != "other" and label in extensions and evidence.issubset(extensions[label])
    ]
    rows = []
    for small in consistent:
        small_ext = extensions[small]
        for large in consistent:
            if small == large:
                continue
            large_ext = extensions[large]
            if small_ext < large_ext:
                rows.append({
                    "small_hypothesis": small,
                    "large_hypothesis": large,
                    "small_size": len(small_ext),
                    "large_size": len(large_ext),
                    "log_size_ratio": float(np.log(len(large_ext) / len(small_ext))),
                })
    rows.sort(key=lambda row: (row["log_size_ratio"], row["small_hypothesis"], row["large_hypothesis"]))
    return rows


def compute_prompt_entropy(result: dict[str, float], prompt_labels: list[str]) -> float:
    return shannon_entropy(float(result.get(label, 0.0)) for label in prompt_labels)


def collect_pair_metrics(experiment: Experiment) -> pd.DataFrame:
    """Collect one row per nested pair and sampling condition from final hypothesis_eval outputs.

    The prompt-local pair structure depends only on (task, domain, stimulus), so
    it is precomputed once per context and reused across sampling conditions.
    """
    result_key = experiment.backend.result_key
    rows = []
    stimulus_specs_by_context = {}
    for condition in make_grid(experiment):
        if condition.eval_mode != "hypothesis_eval":
            continue
        ctx = build_task_context(condition.task, condition.domain)
        if ctx.rule_families is not None:
            print(f"    Nested pair: skipping {condition.task} d{condition.domain} (grouped transform families)")
            continue
        if condition.context_key not in stimulus_specs_by_context:
            specs = {}
            for stimulus in ctx.task.stimuli:
                labels = build_prompt_space_spec(ctx, stimulus.examples).unique_labels
                extensions = {
                    item.label: frozenset(int(i + 1) for i in np.flatnonzero(item.support > 0.0))
                    for item in prompt_items(ctx, stimulus.examples)
                }
                specs[stimulus.name] = {
                    "prompt_labels": labels,
                    "pairs": enumerate_nested_consistent_pairs(labels, extensions, stimulus.examples),
                    "n_prompt_hypotheses": len(labels),
                    "n_consistent_hypotheses": sum(
                        set(stimulus.examples) <= extension for extension in extensions.values()
                    ),
                }
            stimulus_specs_by_context[condition.context_key] = specs
        specs = stimulus_specs_by_context[condition.context_key]

        finals = load_condition_final(condition, result_key)
        if not finals:
            continue
        for stimulus in ctx.task.stimuli:
            result = finals.get(stimulus.name)
            spec = specs[stimulus.name]
            if not result or not spec["pairs"]:
                continue
            entropy = compute_prompt_entropy(result, spec["prompt_labels"])
            for pair in spec["pairs"]:
                p_small = float(result.get(pair["small_hypothesis"], 0.0))
                p_large = float(result.get(pair["large_hypothesis"], 0.0))
                if max(p_small, p_large) < MIN_PAIR_PROB:
                    continue
                rows.append({
                    "task": condition.task,
                    "stimulus": stimulus.name,
                    "stimulus_kind": stimulus.kind,
                    "domain": condition.domain,
                    "h_space": condition.h_space,
                    "sampling": condition.sampling,
                    **pair,
                    "p_small": p_small,
                    "p_large": p_large,
                    "log_odds": float(np.log(max(p_small, PROB_FLOOR)) - np.log(max(p_large, PROB_FLOOR))),
                    "entropy": entropy,
                    "n_prompt_hypotheses": spec["n_prompt_hypotheses"],
                    "n_consistent_hypotheses": spec["n_consistent_hypotheses"],
                })
    return pd.DataFrame(rows)


def compute_pair_deltas(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Pair strong/weak/default rows within the same nested hypothesis comparison."""
    if pair_df.empty:
        return pd.DataFrame()

    key_cols = [
        "task",
        "stimulus",
        "stimulus_kind",
        "domain",
        "h_space",
        "small_hypothesis",
        "large_hypothesis",
        "small_size",
        "large_size",
        "log_size_ratio",
        "n_prompt_hypotheses",
        "n_consistent_hypotheses",
    ]
    deltas = []
    for comparison, (sampling_a, sampling_b) in COMPARISONS.items():
        left = pair_df[pair_df["sampling"] == sampling_a].copy()
        right = pair_df[pair_df["sampling"] == sampling_b].copy()
        if left.empty or right.empty:
            continue
        merged = left.merge(
            right,
            on=key_cols,
            how="inner",
            suffixes=("_a", "_b"),
        )
        if merged.empty:
            continue
        merged["comparison"] = comparison
        merged["sampling_a"] = sampling_a
        merged["sampling_b"] = sampling_b
        merged["delta_log_odds"] = merged["log_odds_a"] - merged["log_odds_b"]
        merged["delta_entropy"] = merged["entropy_a"] - merged["entropy_b"]
        merged["baseline_log_odds"] = merged["log_odds_b"]
        deltas.append(merged)
    return pd.concat(deltas, ignore_index=True) if deltas else pd.DataFrame()


def _fit_ols(y: np.ndarray, columns: list[np.ndarray]) -> tuple[np.ndarray, float] | None:
    """OLS with intercept; None when the design matrix is rank-deficient."""
    X = np.column_stack([np.ones(len(y))] + columns)
    if X.shape[0] < X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    ss_res = float(np.sum((y - X @ coef) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return coef, r2


def _fit_slope_stats(group: pd.DataFrame) -> tuple[int, dict[str, float]]:
    """Shared unadjusted and entropy-adjusted nested-pair slope fits."""
    clean = group[SLOPE_INPUT_COLS].replace([np.inf, -np.inf], np.nan).dropna()
    stats = {
        "slope_unadjusted": np.nan,
        "r2_unadjusted": np.nan,
        "slope_adjusted": np.nan,
        "r2_adjusted": np.nan,
        "coef_delta_entropy": np.nan,
        "coef_baseline_log_odds": np.nan,
    }

    if len(clean) >= 2 and clean["log_size_ratio"].nunique() >= 2:
        fit = _fit_ols(clean["delta_log_odds"].to_numpy(), [clean["log_size_ratio"].to_numpy()])
        if fit is not None:
            coef, r2 = fit
            stats["slope_unadjusted"] = float(coef[1])
            stats["r2_unadjusted"] = r2

    if len(clean) >= 4 and clean["log_size_ratio"].nunique() >= 2:
        fit = _fit_ols(
            clean["delta_log_odds"].to_numpy(),
            [
                clean["log_size_ratio"].to_numpy(),
                clean["delta_entropy"].to_numpy(),
                clean["baseline_log_odds"].to_numpy(),
            ],
        )
        if fit is not None:
            coef, r2 = fit
            stats["slope_adjusted"] = float(coef[1])
            stats["coef_delta_entropy"] = float(coef[2])
            stats["coef_baseline_log_odds"] = float(coef[3])
            stats["r2_adjusted"] = r2
    return int(len(clean)), stats


def fit_stimulus_slopes(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each stimulus by its size-ratio slope.

    Within a single stimulus, ``delta_entropy`` is constant across pairs, so the
    entropy-adjusted coefficient is only identifiable when other covariates vary
    enough. The pooled fit below is the primary entropy-controlled summary.
    """
    if delta_df.empty:
        return pd.DataFrame()

    group_cols = ["task", "stimulus", "stimulus_kind", "domain", "h_space", "comparison"]
    rows = []
    for key, group in delta_df.groupby(group_cols, dropna=False):
        n_pairs_fit, slope_stats = _fit_slope_stats(group)
        row = dict(zip(group_cols, key))
        row["n_pairs"] = int(len(group))
        row["n_pairs_fit"] = n_pairs_fit
        row["mean_delta_log_odds"] = float(group["delta_log_odds"].mean())
        row["mean_delta_entropy"] = float(group["delta_entropy"].mean())
        row.update(slope_stats)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_pooled_slopes(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Fit entropy-adjusted slopes after pooling pairs across stimuli."""
    if delta_df.empty:
        return pd.DataFrame()

    rows = []
    for key, group in with_all_kinds(delta_df).groupby(
        ["task", "domain", "h_space", "comparison", "stimulus_kind"], dropna=False
    ):
        n_pairs_fit, slope_stats = _fit_slope_stats(group)
        row = {
            "task": key[0],
            "domain": key[1],
            "h_space": key[2],
            "comparison": key[3],
            "stimulus_kind": key[4],
            "n_pairs": int(len(group)),
            "n_pairs_fit": n_pairs_fit,
            "n_stimuli": int(group["stimulus"].nunique()),
        }
        row.update(slope_stats)
        rows.append(row)
    return pd.DataFrame(rows)


def run_nested_pair(experiment: Experiment, analysis_dir: str, figure_dir: str) -> pd.DataFrame:
    """Run the nested-pair size-principle diagnostic on final hypothesis_eval outputs."""
    pair_df = collect_pair_metrics(experiment)
    save_csv(pair_df, analysis_dir, "nested_pair_pair_metrics.csv")
    if pair_df.empty:
        print("  Nested pair: no direct-semantics hypothesis_eval data found")
        return pair_df

    delta_df = compute_pair_deltas(pair_df)
    save_csv(delta_df, analysis_dir, "nested_pair_paired_deltas.csv")
    if delta_df.empty:
        print("  Nested pair: no paired sampling comparisons found")
        return delta_df

    save_csv(fit_stimulus_slopes(delta_df), analysis_dir, "nested_pair_stimulus_slopes.csv")
    save_csv(fit_pooled_slopes(delta_df), analysis_dir, "nested_pair_pooled_slopes.csv")
    return delta_df
