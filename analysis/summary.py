"""Cross-model summary plots over default complete 1..4 prefix stimuli.

Summary figures read only the per-model and baseline analysis CSV caches.
Run as `python -m analysis.summary`; `run_hypothesis_analysis.py` runs the
hypothesis summary after its stages.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.alpha_beta_scope import FULL_SCOPE, PREFIX_LENGTHS
from analysis.extension import EXTENSION_DOMAIN, EXTENSION_STIMULUS_KINDS
from analysis.plotter import (
    EVAL_MODE_COLORS,
    EVAL_MODE_LABELS,
    PALETTE,
    SUMMARY_MODEL_ORDER,
    aggregate_sem,
    alpha_beta_ratio,
    model_colors,
    log_positive_ratio,
    plot_trajectory_sem,
    save_csv,
    save_fig,
    sem,
    set_mpl,
)
from configs.layout import ROOT_DOMAIN, analysis_dir, domain_file_prefix, domain_subdir
from configs.settings import RESULTS_ROOT
from tasks.stimuli import HUMAN_NATIVE_DOMAIN


@dataclass(frozen=True)
class SummaryEvalMode:
    """One evaluation mode's default summary condition and display vocabulary."""

    eval_mode: str
    h_space: str
    sampling: str

    @property
    def label(self) -> str:
        return EVAL_MODE_LABELS[self.eval_mode]

    @property
    def color(self):
        return EVAL_MODE_COLORS[self.eval_mode]


SUMMARY_EVAL_MODE_SPECS = (
    SummaryEvalMode("posterior", "implicit", "default"),
    SummaryEvalMode("hypothesis_eval", "explicit", "default"),
    SummaryEvalMode("hypothesis_generation", "implicit", "default"),
)
HYPOTHESIS_MODE_SPECS = SUMMARY_EVAL_MODE_SPECS[1:]
SUMMARY_EVAL_MODE_ORDER = tuple(spec.label for spec in SUMMARY_EVAL_MODE_SPECS)
SUMMARY_EVAL_MODE_PALETTE = {spec.label: spec.color for spec in SUMMARY_EVAL_MODE_SPECS}
SUMMARY_EVAL_MODE_LABELS = {(spec.eval_mode, spec.h_space, spec.sampling): spec.label for spec in SUMMARY_EVAL_MODE_SPECS}
HYPOTHESIS_MODES = tuple(spec.eval_mode for spec in HYPOTHESIS_MODE_SPECS)

DEFAULT_SUMMARY_DOMAINS = (ROOT_DOMAIN, EXTENSION_DOMAIN)
HUMAN_DIVERGENCE_CSV = "divergence_human_metrics.csv"
HYPOTHESIS_TOP1_CSV = "hypothesis_compare_top1_summary.csv"
HYPOTHESIS_TOP1_METRICS_CSV = "hypothesis_compare_top1_metrics.csv"
HYPOTHESIS_JS_DISTANCE_CSV = "hypothesis_compare_js_distance_summary.csv"
HYPOTHESIS_ALPHA_BETA_CSV = "alpha_beta_mode_fits.csv"
HYPOTHESIS_ENTROPY_CSV = "hypothesis_compare_entropy_summary.csv"
HYPOTHESIS_TOP1_METRICS = (
    ("top1_normalized_confidence_mean", "hypothesis_confidence.png", "Top-1 normalized confidence"),
    ("top1_support_fraction_mean", "hypothesis_support.png", "Top-1 support fraction"),
    ("top1_accuracy_mean", "hypothesis_accuracy.png", "Top-1 evidence accuracy"),
    ("top1_rule_proportion_mean", "hypothesis_rule_proportion.png", "Top-1 rule proportion"),
)
DOMAIN_HYPOTHESIS_TOP1_METRICS = (
    ("top1_rule_proportion_mean", "domain_hypothesis_rule_proportion.png", "Top-1 rule proportion"),
    ("top1_support_fraction_mean", "domain_hypothesis_support.png", "Top-1 support fraction"),
)
HYPOTHESIS_JS_DISTANCE_METRICS = (
    ("posterior_vs_eval_js_distance", HYPOTHESIS_MODE_SPECS[0]),
    ("posterior_vs_generation_js_distance", HYPOTHESIS_MODE_SPECS[1]),
)
HYPOTHESIS_SUMMARY_COLORS = tuple(spec.color for spec in HYPOTHESIS_MODE_SPECS)
HYPOTHESIS_SUMMARY_PALETTE = {spec.label: spec.color for spec in HYPOTHESIS_MODE_SPECS}
HYPOTHESIS_CONDITION_MODE_ORDER = tuple(spec.label for spec in HYPOTHESIS_MODE_SPECS)
HYPOTHESIS_CONDITION_TOP1_PANELS = (
    ("support", "top1_support_fraction_mean", "Support"),
    ("accuracy", "top1_accuracy_mean", "Accuracy"),
    ("confidence", "top1_normalized_confidence_mean", "Confidence"),
    ("rule_prop", "top1_rule_proportion_mean", "Rule prop."),
)
HUMAN_TOP1_METRIC_COLUMNS = {
    "top1_rule_proportion_mean": "rule_proportion",
    "top1_support_fraction_mean": "support_fraction",
    "top1_accuracy_mean": "accuracy",
}
HYPOTHESIS_ALPHA_BETA_METRICS = (
    ("alpha", "hypothesis_alpha.png", "α (prior weight)", 1.0),
    ("beta", "hypothesis_beta.png", "β (size principle)", 1.0),
    ("alpha_beta_ratio", "hypothesis_alpha_beta_ratio.png", "log(α / β)", 0.0),
)
ALPHA_BETA_SCATTER_PROMPTS = (
    ("default", "implicit", "default implicit", "alpha_beta_scatter_default_implicit.png"),
)
PROMPT_BAR_SPECS = (
    ("default", "implicit", "Default", "gray"),
    ("default", "explicit", "Explicit", "#5573cd"),
    ("strong", "implicit", "Strong", "#b55d60"),
    ("weak", "implicit", "Weak", "#df8920"),
)
PROMPT_BAR_ORDER = tuple(label for _sampling, _h_space, label, _color in PROMPT_BAR_SPECS)
PROMPT_BAR_PALETTE = {label: color for _sampling, _h_space, label, color in PROMPT_BAR_SPECS}
PROMPT_BAR_LABELS = {(sampling, h_space): label for sampling, h_space, label, _color in PROMPT_BAR_SPECS}
THINKING_MODE_ORDER = ("Non-thinking", "Thinking")
PROMPT_MODE_BAR_PALETTE = {
    "Default": "#404040",
    "Explicit": "midnightblue",
    "Strong": "darkred",
    "Weak": "darkorange",
}
DOMAIN_BAR_ORDER = tuple(f"d{domain}" for domain in DEFAULT_SUMMARY_DOMAINS)
DOMAIN_BAR_PALETTE = dict(zip(DOMAIN_BAR_ORDER, ("gray", "mediumseagreen")))
EXTENSION_STIMULUS_KIND_LABELS = {"rule": "Rule-derived examples", "similarity": "Interval-derived examples"}
# Posterior extension-mass figure prompt; hypothesis modes use their SummaryEvalMode prompt.
EXTENSION_POSTERIOR_PROMPT = ("explicit", "default")


@dataclass(frozen=True)
class SummaryMetric:
    name: str
    csv_name: str
    ylabel: str
    filename: str
    hline: float | None = None


SUMMARY_METRICS = (
    SummaryMetric("kl_bayes", "trajectory_metrics.csv", "KL(Bayesian || LLM)", "kl.png"),
    SummaryMetric("alpha", "trajectory_prefix_metrics.csv", "α (prior weight)", "alpha.png", 1.0),
    SummaryMetric("beta", "trajectory_prefix_metrics.csv", "β (size principle)", "beta.png", 1.0),
    SummaryMetric("alpha_beta_ratio", "trajectory_prefix_metrics.csv", "log(α / β)", "alpha_beta_ratio.png", 0.0),
)
SUMMARY_MEAN_TRAJECTORY_METRICS = {"alpha", "beta", "alpha_beta_ratio"}

THINKING_SUMMARY_MODEL_ORDER = (
    "Gemma4_A4B", "Gemma4_A4B_thinking",
    "Gemma4_E4B", "Gemma4_E4B_thinking",
    "Qwen36_A3B", "Qwen36_A3B_thinking",
    "Qwen35_4B", "Qwen35_4B_thinking",
    "GPT54_Mini", "GPT54_Mini_thinking",
    "Nemotron3_Nano", "Nemotron3_Nano_thinking",
)
HUMAN_MARKER = "x"
ALPHA_BETA_PANEL_YTICKS = (0.0, 0.5, 1.0)
ALPHA_BETA_PANEL_YTICK_LABELS = ("0", "0.5", "1")


def _analysis_csv_path(results_dir: str, producer: str, csv_name: str, domain: int) -> str:
    return os.path.join(domain_subdir(analysis_dir(producer, results_root=results_dir), domain), csv_name)


def _read_domain_csv(results_dir: str, producer: str, csv_name: str, domain: int) -> pd.DataFrame:
    """Read one per-domain analysis CSV, keeping only rows for `domain`."""
    path = _analysis_csv_path(results_dir, producer, csv_name, domain)
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["domain"] == domain].copy() if not df.empty else df


def _read_model_csv(results_dir: str, producer: str, csv_name: str) -> pd.DataFrame:
    """Read one cross-domain CSV from a producer's root analysis directory."""
    path = os.path.join(analysis_dir(producer, results_root=results_dir), csv_name)
    return pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()


def available_summary_models(
    results_dir: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    analysis_csv: str = "trajectory_metrics.csv",
    domain: int = ROOT_DOMAIN,
) -> list[str]:
    return [
        model for model in model_order
        if os.path.isfile(_analysis_csv_path(results_dir, model, analysis_csv, domain))
    ]


def available_nonthinking_models(results_dir: str, analysis_csv: str, domain: int = ROOT_DOMAIN) -> list[str]:
    """All result models with the requested cache, excluding baselines and thinking variants."""
    if not os.path.isdir(results_dir):
        return []
    skip = {"bayesian", "human", "summary"}
    candidates = [
        name for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name))
        and not name.endswith("_thinking")
        and name not in skip
        and os.path.isfile(_analysis_csv_path(results_dir, name, analysis_csv, domain))
    ]
    ordered = [model for model in SUMMARY_MODEL_ORDER if model in candidates]
    return ordered + sorted(model for model in candidates if model not in set(ordered))


def _paired_thinking_models(available) -> list[str]:
    """Base/thinking pairs (in THINKING_SUMMARY_MODEL_ORDER) where both variants are available."""
    available_set = set(available)
    return [
        variant
        for model in THINKING_SUMMARY_MODEL_ORDER
        if not model.endswith("_thinking") and {model, f"{model}_thinking"} <= available_set
        for variant in (model, f"{model}_thinking")
    ]


def available_paired_thinking_models(results_dir: str, analysis_csv: str, domain: int = ROOT_DOMAIN) -> list[str]:
    """Base/thinking pairs where both variants have the requested cache."""
    return _paired_thinking_models(available_summary_models(results_dir, THINKING_SUMMARY_MODEL_ORDER, analysis_csv, domain))


def load_metric_rows(results_dir: str, spec: SummaryMetric, models: list[str], domain: int = ROOT_DOMAIN) -> pd.DataFrame:
    """Load one metric's per-prefix LLM rows from each model's analysis cache."""
    def rows_for_model(model: str) -> pd.DataFrame:
        df = _read_domain_csv(results_dir, model, spec.csv_name, domain)
        if df.empty:
            return df
        df = df[df["source"] == "llm"].copy() if "source" in df else df
        if spec.name == "alpha_beta_ratio":
            df[spec.name] = alpha_beta_ratio(df["alpha"], df["beta"])
        return df

    return _concat_model_rows(models, rows_for_model)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _concat_model_rows(models: list[str], rows_for_model) -> pd.DataFrame:
    return _concat_frames([rows_for_model(model).assign(model=model) for model in models])


def _model_csv_rows(results_dir: str, models: list[str], csv_name: str, domain: int) -> pd.DataFrame:
    return _concat_model_rows(models, lambda model: _read_domain_csv(results_dir, model, csv_name, domain))


def _load_posterior_divergence_rows(results_dir: str, models: list[str], csv_name: str, domain: int = ROOT_DOMAIN) -> pd.DataFrame:
    """Load posterior rows from each model's divergence cache."""
    rows = _model_csv_rows(results_dir, models, csv_name, domain)
    return rows[rows["eval_mode"] == "posterior"].copy() if not rows.empty else rows


def load_alpha_beta_rows(results_dir: str, models: list[str], domain: int = ROOT_DOMAIN) -> pd.DataFrame:
    """Load task-pooled posterior alpha-beta fits."""
    return _model_csv_rows(results_dir, models, "alpha_beta_fits.csv", domain)


def load_hypothesis_top1_rows(results_dir: str, models: list[str], domain: int = ROOT_DOMAIN) -> pd.DataFrame:
    """Load per-model hypothesis top-1 summaries; model remains the CI unit."""
    return _model_csv_rows(results_dir, models, HYPOTHESIS_TOP1_CSV, domain)


def load_domain_hypothesis_top1_rows(results_dir: str, domains: tuple[int, ...] = DEFAULT_SUMMARY_DOMAINS) -> pd.DataFrame:
    """Load top-1 summaries across domains for direct d100/d200 comparison."""
    return _concat_frames([
        load_hypothesis_top1_rows(results_dir, available_nonthinking_models(results_dir, HYPOTHESIS_TOP1_CSV, domain), domain)
        for domain in domains
    ])


def load_hypothesis_top1_metric_rows(results_dir: str, models: list[str], domain: int = ROOT_DOMAIN) -> pd.DataFrame:
    """Load stimulus-level top-1 hypothesis metrics for domain-specific scatter summaries."""
    return _model_csv_rows(results_dir, models, HYPOTHESIS_TOP1_METRICS_CSV, domain)


def load_hypothesis_js_distance_rows(results_dir: str, models: list[str], domain: int = ROOT_DOMAIN) -> pd.DataFrame:
    """Load per-model posterior-vs-hypothesis JS-distance summaries."""
    return _model_csv_rows(results_dir, models, HYPOTHESIS_JS_DISTANCE_CSV, domain)


def load_hypothesis_alpha_beta_rows(results_dir: str, models: list[str], domain: int = ROOT_DOMAIN) -> pd.DataFrame:
    """Per-model eval-mode alpha-beta trajectories, averaged over prompts in log-ratio space."""
    rows = _model_csv_rows(results_dir, models, HYPOTHESIS_ALPHA_BETA_CSV, domain).dropna(subset=["n_examples"]) \
        if models else pd.DataFrame()
    if rows.empty:
        return rows
    rows["log_alpha_beta_ratio"] = np.log(alpha_beta_ratio(rows["alpha"], rows["beta"]))
    grouped = (
        rows
        .groupby(["model", "eval_mode", "n_examples"], as_index=False)
        .agg(
            alpha=("alpha", "mean"),
            beta=("beta", "mean"),
            log_alpha_beta_ratio=("log_alpha_beta_ratio", "mean"),
            n_prompts=("alpha", "size"),
        )
    )
    grouped["alpha_beta_ratio"] = np.exp(grouped["log_alpha_beta_ratio"])
    grouped["mode_label"] = grouped["eval_mode"].map(EVAL_MODE_LABELS)
    grouped["_model_order"] = grouped["model"].map({model: i for i, model in enumerate(models)})
    grouped["_mode_order"] = grouped["eval_mode"].map({spec.eval_mode: i for i, spec in enumerate(SUMMARY_EVAL_MODE_SPECS)})
    return grouped.sort_values(["_model_order", "_mode_order", "n_examples"]).drop(
        columns=["_model_order", "_mode_order"]
    ).reset_index(drop=True)


def _with_mode_labels(rows: pd.DataFrame) -> pd.DataFrame:
    """Label rows matching a SummaryEvalMode default condition; others are dropped."""
    if rows.empty:
        return rows
    keys = pd.Series(list(zip(rows["eval_mode"], rows["h_space"], rows["sampling"])), index=rows.index)
    return rows.assign(mode_label=keys.map(SUMMARY_EVAL_MODE_LABELS)).dropna(subset=["mode_label"])


def load_eval_mode_alpha_beta_rows(results_dir: str, models: list[str], domain: int = ROOT_DOMAIN) -> pd.DataFrame:
    """Default full-scope alpha-beta fits for Predict/Eval/Generate mode bars."""
    rows = _model_csv_rows(results_dir, models, HYPOTHESIS_ALPHA_BETA_CSV, domain)
    return _with_mode_labels(rows[rows["fit_scope"] == FULL_SCOPE] if not rows.empty else rows)


def load_eval_mode_kl_rows(results_dir: str, models: list[str], domain: int = ROOT_DOMAIN) -> pd.DataFrame:
    """Bayesian KL rows for Predict/Eval/Generate evaluation-mode bars."""
    return _with_mode_labels(_concat_frames([
        _model_csv_rows(results_dir, models, "divergence_metrics.csv", domain),
        _model_csv_rows(results_dir, models, "hypothesis_divergence_metrics.csv", domain),
    ]))


def load_eval_mode_predictive_entropy_rows(results_dir: str, models: list[str], domain: int = ROOT_DOMAIN) -> pd.DataFrame:
    """Endpoint predictive entropy (nats) for Predict/Eval/Generate mode bars."""
    rows = _with_mode_labels(_model_csv_rows(results_dir, models, HYPOTHESIS_ENTROPY_CSV, domain))
    if rows.empty:
        return rows
    rows = rows[rows["n_examples"] == max(PREFIX_LENGTHS)]
    return rows.assign(predictive_entropy=rows["entropy_bits_mean"] * np.log(2.0))


def complete_prefix_rows(
    df: pd.DataFrame,
    metric: str,
    sampling: str = "default",
    h_space: str = "implicit",
    domain: int | None = None,
) -> pd.DataFrame:
    """Select rows of one prompt whose units have every prefix length in PREFIX_LENGTHS."""
    work = df.copy()
    if domain is not None:
        work = work[work["domain"] == domain]
    if work.empty or metric not in work:
        return pd.DataFrame()
    work = work.dropna(subset=[metric])
    work = work[work["n_examples"].isin(PREFIX_LENGTHS)]
    work = work[(work["sampling"] == sampling) & (work["h_space"] == h_space)]

    group_cols = [
        col for col in ["model", "task", "domain", "h_space", "sampling", "stimulus", "stimulus_kind"]
        if col in work and (col not in {"stimulus", "stimulus_kind"} or work[col].notna().any())
    ]
    prefix_means = work.groupby(group_cols + ["n_examples"], as_index=False)[metric].mean()
    complete_counts = prefix_means.groupby(group_cols)["n_examples"].nunique().rename("n_prefixes").reset_index()
    complete_keys = complete_counts[complete_counts["n_prefixes"] == len(PREFIX_LENGTHS)][group_cols]
    complete = prefix_means.merge(complete_keys, on=group_cols, how="inner")
    return complete.sort_values(group_cols + ["n_examples"]).reset_index(drop=True)


def _sort_by_model_order(rows: pd.DataFrame, model_order, *extra: str, bayesian_first: bool = False) -> pd.DataFrame:
    """Sort rows by model_order (unknown models last, optional Bayesian first), then `extra` columns."""
    rank = {model: i for i, model in enumerate(model_order)}
    order = rows["model"].map(lambda model: -1 if bayesian_first and model == "Bayesian" else rank.get(model, len(rank)))
    return rows.assign(_model_order=order).sort_values(["_model_order", "model", *extra]).drop(
        columns="_model_order"
    ).reset_index(drop=True)


def prefix_summary(rows: pd.DataFrame, metric: str, group_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
                   *, group_col: str = "model") -> pd.DataFrame:
    """Aggregate prefix points over stimuli per model, or over models per evaluation mode."""
    count_col = "n_models" if group_col == "mode_label" else "n_stimuli"
    out = (
        rows.dropna(subset=[metric])
        .groupby([group_col, "n_examples"], as_index=False)
        .agg(**{metric: (metric, "mean"), "sem": (metric, sem), count_col: (metric, "size")})
    )
    rank = {group: i for i, group in enumerate(group_order)}
    return out.assign(_group_order=out[group_col].map(rank).fillna(len(rank))).sort_values(
        ["_group_order", group_col, "n_examples"]
    ).drop(columns="_group_order").reset_index(drop=True)


def mean_prefix_summary(model_points: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean trajectory over model-level prefix means; SEM is across models."""
    return (
        model_points.dropna(subset=[metric])
        .groupby("n_examples", as_index=False)
        .agg(**{metric: (metric, "mean"), "sem": (metric, sem), "n_models": (metric, "size")})
        .assign(model="Mean")
    )


def summary_colors(models: list[str]) -> dict[str, object]:
    """Use base-model colors for thinking variants."""
    bases = list(dict.fromkeys(model.removesuffix("_thinking") for model in models))
    base_colors = model_colors(bases)
    return {model: base_colors[model.removesuffix("_thinking")] for model in models}


# ===== Tenenbaum99 d200 extension diagnostics (from analysis.extension CSVs) =====

def _extension_rows(results_dir: str, models: list[str], name: str, eval_mode: str, h_space: str, sampling: str) -> pd.DataFrame:
    """LLM rows of one extension table for one eval mode and prompt, plus Bayesian reference rows."""
    csv_name = f"{'extension' if eval_mode == 'posterior' else 'hypothesis_extension'}_{name}.csv"
    llm = _concat_model_rows(models, lambda model: _read_model_csv(results_dir, model, csv_name))
    if not llm.empty:
        llm = llm[(llm["eval_mode"] == eval_mode) & (llm["h_space"] == h_space) & (llm["sampling"] == sampling)]
        llm = llm.assign(source="llm")
    bayesian = _read_model_csv(results_dir, "bayesian", f"extension_{name}.csv")
    if not bayesian.empty:
        bayesian = bayesian.assign(source="bayesian", model="Bayesian", eval_mode=eval_mode, h_space=h_space, sampling=sampling)
    return _concat_frames([llm, bayesian])


def available_extension_models(
    results_dir: str,
    name: str,
    eval_mode: str,
    h_space: str,
    sampling: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    *,
    paired: bool = False,
) -> list[str]:
    """Models whose extension table has rows for this eval mode and prompt."""
    order = THINKING_SUMMARY_MODEL_ORDER if paired else model_order
    rows = _extension_rows(results_dir, list(order), name, eval_mode, h_space, sampling)
    present = set(rows.loc[rows["source"] == "llm", "model"]) if not rows.empty else set()
    available = [model for model in order if model in present]
    return _paired_thinking_models(available) if paired else available


def load_extension_mass_rows(results_dir: str, models: list[str], eval_mode: str, h_space: str, sampling: str) -> pd.DataFrame:
    return _extension_rows(results_dir, models, "mass", eval_mode, h_space, sampling)


def load_rule_discrimination_rows(results_dir: str, models: list[str], h_space: str = "implicit",
                                  sampling: str = "default") -> pd.DataFrame:
    """Per-rule-stimulus posterior discrimination rows plus Bayesian."""
    return _extension_rows(results_dir, models, "rule_discrimination", "posterior", h_space, sampling)


def rule_discrimination_points(instance_rows: pd.DataFrame) -> pd.DataFrame:
    """Average rule-target discrimination over the three Tenenbaum99 rule stimuli per model."""
    if instance_rows.empty:
        return instance_rows
    return instance_rows.groupby("model", as_index=False, sort=False).agg(
        rule_negative_mean=("rule_negative_mean", "mean"),
        rule_positive_mean=("rule_positive_mean", "mean"),
        n_rule_stimuli=("stimulus", "size"),
    )


def plot_rule_discrimination_summary(
    rows: pd.DataFrame,
    output_path: str,
    model_order: tuple[str, ...],
    instance_rows: pd.DataFrame,
) -> pd.DataFrame:
    if rows.empty:
        return rows

    set_mpl(size=8)

    def draw_points(ax, plot_points, ordered, color_map, model_point_style):
        ax.plot([0, 1], [0, 1], color="0.45", ls=":", lw=1.0)
        instance_points = instance_rows.dropna(subset=["rule_negative_mean", "rule_positive_mean"])
        for model in ordered:
            model_instances = instance_points[instance_points["model"] == model]
            if model_instances.empty:
                continue
            color = "black" if model == "Bayesian" else color_map.get(model, PALETTE["purple"])
            ax.scatter(model_instances["rule_negative_mean"], model_instances["rule_positive_mean"],
                       color=color, marker="o", s=10, alpha=0.2, linewidths=0, zorder=2)
        for model in ordered:
            model_rows = plot_points[plot_points["model"] == model]
            if model == "Bayesian":
                ax.scatter(model_rows["rule_negative_mean"], model_rows["rule_positive_mean"],
                           color="black", marker="*", s=95, zorder=4)
            else:
                ax.scatter(model_rows["rule_negative_mean"], model_rows["rule_positive_mean"],
                           color=color_map.get(model, PALETTE["purple"]), marker="o", **model_point_style)

    _plot_scatter_with_external_legend(
        rows,
        output_path,
        "rule_negative_mean",
        "rule_positive_mean",
        "P(yes) on non-rule [101-200]",
        "P(yes) on rule [101-200]",
        ("Bayesian", *model_order),
        draw_points,
        xlim=(-0.04, 1.04),
        ylim=(-0.04, 1.04),
        xticks=[0, 0.5, 1.0],
        yticks=[0, 0.5, 1.0],
    )
    return rows


def load_domain_shift_rows(results_dir: str, models: list[str], eval_mode: str, h_space: str, sampling: str) -> pd.DataFrame:
    """Rows for d200-to-d100 shape preservation vs out-of-domain leakage."""
    return _extension_rows(results_dir, models, "domain_shift", eval_mode, h_space, sampling)


def load_eval_mode_domain_shift_rows(results_dir: str, model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER) -> pd.DataFrame:
    """Predict/Eval/Generate d200-to-d100 diagnostics for each SummaryEvalMode default condition."""
    tables = []
    for spec in SUMMARY_EVAL_MODE_SPECS:
        models = available_extension_models(results_dir, "domain_shift", spec.eval_mode, spec.h_space, spec.sampling, model_order)
        rows = load_domain_shift_rows(results_dir, models, spec.eval_mode, spec.h_space, spec.sampling)
        if not rows.empty:
            tables.append(rows.assign(mode_label=spec.label))
    return _concat_frames(tables)


def domain_shift_model_rows(rows: pd.DataFrame, model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out = rows.groupby(["source", "model"], as_index=False).agg(
        shape_kl=("shape_kl", "mean"),
        d200_tail_fraction=("d200_tail_fraction", "mean"),
        d200_tail_mean=("d200_tail_mean", "mean"),
        n_stimuli=("shape_kl", "size"),
    )
    return _sort_by_model_order(out, model_order, bayesian_first=True)


def plot_domain_shift_summary(
    rows: pd.DataFrame,
    output_path: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
) -> pd.DataFrame:
    """Scatter model-level d200 leakage against preserved d100 posterior shape."""
    summary = domain_shift_model_rows(rows, model_order)
    if summary.empty:
        return summary

    set_mpl(size=8)

    def draw_points(ax, plot_points, ordered, color_map, model_point_style):
        stimulus_rows = plot_points[plot_points["source"] == "llm"]
        for model in ordered:
            if model == "Bayesian":
                continue
            model_stimulus_rows = stimulus_rows[stimulus_rows["model"] == model]
            ax.scatter(model_stimulus_rows["d200_tail_fraction"], model_stimulus_rows["shape_kl"],
                       color=color_map.get(model, PALETTE["purple"]), s=10, alpha=0.2, linewidths=0)

        model_summary = domain_shift_model_rows(plot_points, model_order)
        for model in ordered:
            model_rows = model_summary[model_summary["model"] == model]
            if model_rows.empty:
                continue
            if model == "Bayesian":
                ax.scatter(model_rows["d200_tail_fraction"], model_rows["shape_kl"],
                           color="black", marker="*", s=95, zorder=4)
            else:
                ax.scatter(model_rows["d200_tail_fraction"], model_rows["shape_kl"],
                           color=color_map.get(model, PALETTE["purple"]), marker="o", zorder=3,
                           **model_point_style)

    _plot_scatter_with_external_legend(
        rows,
        output_path,
        "d200_tail_fraction",
        "shape_kl",
        "Mass fraction (101-200)",
        "Posterior shift (KL, 1-100)",
        ("Bayesian", *model_order),
        draw_points,
    )
    return summary


def eval_mode_domain_shift_model_rows(rows: pd.DataFrame, model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER) -> pd.DataFrame:
    """Average d200-to-d100 diagnostics per model/eval mode over models complete for every eval mode."""
    if rows.empty:
        return pd.DataFrame()
    work = rows[rows["source"] == "llm"].dropna(subset=["mode_label", "shape_kl", "d200_tail_fraction"])
    work, models = _filter_ordered_models(work, model_order)
    complete_models = [
        model for model in models
        if set(SUMMARY_EVAL_MODE_ORDER) <= set(work.loc[work["model"] == model, "mode_label"])
    ]
    work = work[work["model"].isin(complete_models)]
    if work.empty:
        return pd.DataFrame()

    grouped = work.groupby(["model", "mode_label"], as_index=False).agg(
        shape_kl=("shape_kl", "mean"),
        d200_tail_fraction=("d200_tail_fraction", "mean"),
        n_stimuli=("shape_kl", "size"),
    )
    grouped = _sort_model_category_rows(grouped, complete_models, "mode_label", SUMMARY_EVAL_MODE_ORDER)
    return grouped


def add_external_fig_legend(fig, handles, anchor, *, fontsize: float | None = None, handlelength: float = 1.0):
    """Shared outside-the-axes legend block for summary figures."""
    return fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=anchor,
        frameon=False,
        fontsize=plt.rcParams["legend.fontsize"] if fontsize is None else fontsize,
        handlelength=handlelength,
        labelspacing=0.35,
        borderaxespad=0.0,
    )


def category_patch_handles(palette: dict[str, object], labels) -> list[Patch]:
    """Legend patches colored by one categorical palette."""
    return [Patch(facecolor=palette[label], edgecolor="none", label=label) for label in labels]


def plot_eval_mode_domain_shift_summary(
    rows: pd.DataFrame,
    output_path: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
) -> pd.DataFrame:
    """Plot evaluation-mode means for the two axes used by the d200-to-d100 scatters."""
    model_rows = eval_mode_domain_shift_model_rows(rows, model_order)
    if model_rows.empty:
        return model_rows

    set_mpl(size=8)
    fig, axes = plt.subplots(1, 2, figsize=(4, 2.0), sharex=True)
    metrics = (
        ("shape_kl", "Posterior shift (KL, 1-100)"),
        ("d200_tail_fraction", "Mass fraction (101-200)"),
    )
    x = np.arange(len(SUMMARY_EVAL_MODE_ORDER), dtype=float)
    width = 1.0

    for ax, (metric, ylabel) in zip(axes, metrics):
        for idx, mode_label in enumerate(SUMMARY_EVAL_MODE_ORDER):
            values = model_rows.loc[model_rows["mode_label"] == mode_label, metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            mean = float(np.mean(values))
            ax.bar(
                x[idx], mean, width=width, color=SUMMARY_EVAL_MODE_PALETTE[mode_label],
                edgecolor="none", linewidth=0.0,
            )
            ax.errorbar(x[idx], mean, yerr=_ci95(values), color="black", capsize=2.5, elinewidth=0.8, linewidth=0.0)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(SUMMARY_EVAL_MODE_ORDER) - 0.5)
        ax.set_ylim(bottom=0.0)

    handles = category_patch_handles(SUMMARY_EVAL_MODE_PALETTE, SUMMARY_EVAL_MODE_ORDER)
    add_external_fig_legend(fig, handles, (0.80, 0.5))
    fig.subplots_adjust(left=0.12, right=0.78, bottom=0.16, wspace=0.55)
    save_fig(fig, output_path, tight_layout=False)
    return model_rows


def hypothesis_domain_shift_gap_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Eval-minus-generation gaps for hypothesis-space domain-shift metrics, paired per stimulus."""
    if rows.empty:
        return pd.DataFrame()
    return eval_generation_gap_rows(
        rows[rows["source"] == "llm"].dropna(subset=["shape_kl", "d200_tail_fraction"]),
        ("d200_tail_fraction", "shape_kl"),
        observation_cols=("task", "stimulus"),
    )


def plot_hypothesis_domain_shift_gap(rows: pd.DataFrame, output_dir: str, filename_suffix: str = "") -> pd.DataFrame:
    """Plot eval-minus-generation gaps for projected d200-to-d100 shift diagnostics."""
    return plot_eval_generation_gap_bars(
        hypothesis_domain_shift_gap_rows(rows),
        (
            ("d200_tail_fraction_gap", "Mass fraction (101-200)", "Mass fraction\n(101-200)", HYPOTHESIS_SUMMARY_COLORS[0]),
            ("shape_kl_gap", "Posterior shift (KL, 1-100)", "Posterior shift\n(KL, 1-100)", HYPOTHESIS_SUMMARY_COLORS[1]),
        ),
        os.path.join(output_dir, suffix_filename("domain_shift_eval_generation_gap.png", filename_suffix)),
    )


def ordered_scatter_models(points: pd.DataFrame, model_order: tuple[str, ...]) -> tuple[list[str], dict[str, object]]:
    """Shared model order and colors for cross-model scatter summaries."""
    present = set(points["model"])
    ordered_models = [model for model in model_order if model in present]
    ordered_models += sorted(model for model in present if model not in set(ordered_models) and model != "Human")
    if "Human" in present:
        ordered_models.append("Human")
    colors = summary_colors([model for model in ordered_models if model != "Human"])
    colors["Human"] = "black"
    return ordered_models, colors


def _filter_ordered_models(work: pd.DataFrame, model_order: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    models = [model for model in model_order if model in set(work["model"])]
    return work[work["model"].isin(models)], models


def _prepare_metric_model_rows(
    rows: pd.DataFrame,
    metric: str,
    required: set[str],
    fit_scope: str | None = None,
) -> pd.DataFrame:
    if rows.empty or not required <= set(rows.columns):
        return pd.DataFrame()

    work = rows.copy()
    if fit_scope is not None:
        if "fit_scope" not in work:
            return pd.DataFrame()
        work = work[work["fit_scope"] == fit_scope]
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    return work


def _sort_model_category_rows(
    rows: pd.DataFrame,
    models: list[str],
    category_col: str,
    category_order: tuple,
    extra_sort: tuple[str, ...] = (),
) -> pd.DataFrame:
    order_col = f"_{category_col}_order"
    rows["_model_order"] = rows["model"].map({model: i for i, model in enumerate(models)})
    rows[order_col] = rows[category_col].map({category: i for i, category in enumerate(category_order)})
    return rows.sort_values([order_col, "_model_order", *extra_sort]).drop(
        columns=[order_col, "_model_order"]
    ).reset_index(drop=True)


def _aggregate_grouped_means(
    work: pd.DataFrame,
    metric: str,
    group_cols: list[str],
    models: list[str],
    category_col: str,
    category_order: tuple,
    rename: dict[str, str] | None = None,
    extra_aggs: dict | None = None,
    count_col: str = "n_rows",
    extra_sort: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Aggregate one model-by-category metric table while preserving caller-owned filters."""
    agg_spec = {metric: (metric, "mean"), count_col: (metric, "size")}
    if extra_aggs:
        agg_spec.update(extra_aggs)
    grouped = work.groupby(group_cols, as_index=False).agg(**agg_spec)
    if rename:
        grouped = grouped.rename(columns=rename)
    return _sort_model_category_rows(grouped, models, category_col, category_order, extra_sort)


def condition_metric_model_rows(
    rows: pd.DataFrame,
    metric: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    fit_scope: str | None = None,
) -> pd.DataFrame:
    """Average each requested condition within model before cross-model error bars."""
    required = {"model", "sampling", "h_space", metric}
    work = _prepare_metric_model_rows(rows, metric, required, fit_scope=fit_scope)
    if work.empty:
        return pd.DataFrame()
    work["_prompt"] = work[["sampling", "h_space"]].apply(tuple, axis=1).map(PROMPT_BAR_LABELS)
    work = work.dropna(subset=["_prompt", metric])
    if work.empty:
        return pd.DataFrame()

    work, models = _filter_ordered_models(work, model_order)
    if work.empty:
        return pd.DataFrame()

    extra_aggs = None
    if "n_stimuli" in work:
        extra_aggs = {"n_stimuli": ("n_stimuli", "sum")}
    return _aggregate_grouped_means(
        work, metric, ["model", "_prompt"], models, "prompt", PROMPT_BAR_ORDER,
        rename={"_prompt": "prompt"}, extra_aggs=extra_aggs,
    )


def condition_mode_model_rows(
    rows: pd.DataFrame,
    metric: str,
    model_order: tuple[str, ...] = THINKING_SUMMARY_MODEL_ORDER,
    fit_scope: str | None = None,
) -> pd.DataFrame:
    """Average each paired condition/mode within model before cross-model error bars."""
    model_rows = condition_metric_model_rows(rows, metric, model_order, fit_scope=fit_scope)
    if model_rows.empty:
        return pd.DataFrame()
    model_rows["mode"] = np.where(model_rows["model"].str.endswith("_thinking"), "Thinking", "Non-thinking")
    return model_rows


def _draw_paired_mode_bars(
    ax,
    rows: pd.DataFrame,
    category_col: str,
    categories: tuple[str, ...],
    palette: dict[str, object],
    metric: str,
    ylabel: str,
    hline: float | None = None,
    hline_style: dict | None = None,
    mode_col: str = "mode",
) -> None:
    """Draw category-colored bars split by non-thinking/thinking variants."""
    x = np.arange(len(categories), dtype=float)
    width = 0.34
    offsets = {"Non-thinking": -width / 2, "Thinking": width / 2}
    for category_idx, category in enumerate(categories):
        for mode in THINKING_MODE_ORDER:
            values = rows.loc[
                (rows[category_col] == category) & (rows[mode_col] == mode), metric
            ].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            mean = float(np.mean(values))
            alpha = 0.5 if mode == "Thinking" else 1.0
            ax.bar(
                x[category_idx] + offsets[mode], mean, width=width,
                color=palette[category], alpha=alpha, edgecolor="none", linewidth=0,
            )
            ax.errorbar(
                x[category_idx] + offsets[mode], mean, yerr=_ci95(values),
                color="black", capsize=2.5, elinewidth=0.8, linewidth=0,
            )
    if hline is not None:
        style = hline_style or {"color": "black", "ls": "--", "lw": 0.8, "alpha": 0.5}
        ax.axhline(hline, **style)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    if metric in {"alpha", "beta"}:
        ax.set_yticks(ALPHA_BETA_PANEL_YTICKS)
        ax.set_yticklabels(ALPHA_BETA_PANEL_YTICK_LABELS)


def eval_mode_model_rows(
    rows: pd.DataFrame,
    metric: str,
    model_order: tuple[str, ...] = THINKING_SUMMARY_MODEL_ORDER,
) -> pd.DataFrame:
    """Average each Predict/Eval/Generate mode within model before cross-model error bars."""
    required = {"model", "mode_label", metric}
    work = _prepare_metric_model_rows(rows, metric, required)
    if work.empty:
        return pd.DataFrame()
    work = work.dropna(subset=["mode_label", metric])
    work = work[work["mode_label"].isin(SUMMARY_EVAL_MODE_ORDER)]
    work, models = _filter_ordered_models(work, model_order)
    if work.empty:
        return pd.DataFrame()

    grouped = _aggregate_grouped_means(
        work, metric, ["model", "mode_label"], models, "mode_label", SUMMARY_EVAL_MODE_ORDER,
    )
    grouped["mode"] = np.where(grouped["model"].str.endswith("_thinking"), "Thinking", "Non-thinking")
    return grouped


def _complete_eval_mode_cohort(
    panel_specs: tuple[dict, ...],
    model_order: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return base/thinking pairs complete for every evaluation mode in every panel."""
    complete_models = set(model_order)
    required_modes = set(SUMMARY_EVAL_MODE_ORDER)
    for spec in panel_specs:
        rows = spec["rows"]
        metric = spec["metric"]
        if rows.empty or not {"model", "mode_label", metric} <= set(rows.columns):
            return (), ()
        work = rows[["model", "mode_label", metric]].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna(subset=["model", "mode_label", metric])
        modes_by_model = work.groupby("model")["mode_label"].agg(set)
        complete_models &= {
            model for model, modes in modes_by_model.items()
            if required_modes <= modes
        }

    paired_models = []
    cohort_models = []
    for model in model_order:
        if model.endswith("_thinking"):
            continue
        thinking_model = f"{model}_thinking"
        if model in complete_models and thinking_model in complete_models:
            cohort_models.append(model)
            paired_models.extend((model, thinking_model))
    return tuple(paired_models), tuple(cohort_models)


def _mode_legend_handles() -> list[Patch]:
    return [
        Patch(facecolor="#404040", edgecolor="none", alpha=1.0, label="Non-thinking"),
        Patch(facecolor="#404040", edgecolor="none", alpha=0.5, label="Thinking"),
    ]


def domain_metric_model_rows(
    rows: pd.DataFrame,
    metric: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    domains: tuple[int, ...] = DEFAULT_SUMMARY_DOMAINS,
    fit_scope: str | None = None,
    sampling: str = "default",
    h_space: str = "implicit",
) -> pd.DataFrame:
    """Average each model/domain point for the default prompt condition."""
    required = {"model", "domain", "sampling", "h_space", metric}
    work = _prepare_metric_model_rows(rows, metric, required, fit_scope=fit_scope)
    if work.empty:
        return pd.DataFrame()
    work = work[(work["sampling"] == sampling) & (work["h_space"] == h_space)]
    work["domain"] = pd.to_numeric(work["domain"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["domain", metric])
    work = work[work["domain"].isin(domains)]
    if work.empty:
        return pd.DataFrame()

    present_by_domain = {
        domain: set(work.loc[work["domain"] == domain, "model"])
        for domain in domains
    }
    models = [
        model for model in model_order
        if all(model in present_by_domain[domain] for domain in domains)
    ]
    work = work[work["model"].isin(models)]
    if work.empty:
        return pd.DataFrame()

    work["domain_label"] = work["domain"].map(lambda domain: f"d{int(domain)}")
    grouped = _aggregate_grouped_means(
        work, metric, ["model", "domain", "domain_label"], models, "domain_label", DOMAIN_BAR_ORDER,
    )
    return grouped[["model", "domain", metric, "n_rows", "domain_label"]]


def _draw_domain_bars(
    ax,
    rows: pd.DataFrame,
    metric: str,
    ylabel: str,
    hline: float | None = None,
) -> None:
    """Draw one domain panel from model-level domain means."""
    sns.barplot(
        data=rows, x="domain_label", y=metric, hue="domain_label",
        order=list(DOMAIN_BAR_ORDER), hue_order=list(DOMAIN_BAR_ORDER),
        palette=DOMAIN_BAR_PALETTE, errorbar=("se", 1.96), capsize=0.12, legend=False, ax=ax,
    )
    if hline is not None:
        ax.axhline(hline, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)


def fitting_performance_model_rows(
    rows: pd.DataFrame,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    sampling: str = "default",
    h_space: str = "implicit",
    fit_scope: str = FULL_SCOPE,
) -> pd.DataFrame:
    """Average default full-scope alpha-beta fit R² within each model."""
    required = {"model", "sampling", "h_space", "r2"}
    work = _prepare_metric_model_rows(rows, "r2", required, fit_scope=fit_scope)
    if work.empty:
        return pd.DataFrame()
    work = work[(work["sampling"] == sampling) & (work["h_space"] == h_space)].dropna(subset=["r2"])
    work, models = _filter_ordered_models(work, model_order)
    if work.empty:
        return pd.DataFrame()

    extra_aggs = {"n_stimuli": ("n_stimuli", "sum")} if "n_stimuli" in work else None
    grouped = work.groupby(["model"], as_index=False).agg(
        r2=("r2", "mean"),
        n_rows=("r2", "size"),
        **(extra_aggs or {}),
    )
    grouped["_model_order"] = grouped["model"].map({model: i for i, model in enumerate(models)})
    return grouped.sort_values("_model_order").drop(columns="_model_order").reset_index(drop=True)


def plot_fitting_performance(
    model_points: pd.DataFrame,
    output_path: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
) -> pd.DataFrame:
    """Draw one per-model alpha-beta fit R² summary."""
    set_mpl(size=8)
    rows, models = _filter_ordered_models(model_points, model_order)

    colors = summary_colors(models)
    x = np.arange(len(rows), dtype=float)
    fig, ax = plt.subplots(figsize=(max(5.2, 0.62 * len(rows)), 2.5))
    ax.bar(x, rows["r2"], color=[colors[model] for model in rows["model"]], edgecolor="none")
    ax.set_xlabel("")
    ax.set_ylabel(r"$r^2$")
    ax.set_xticks(x)
    ax.set_xticklabels([model for model in rows["model"]], rotation=45, ha="right")
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylim(0.0, 1.0)
    save_fig(fig, output_path)
    return rows.assign(metric="r2")


def condition_prefix_model_rows(
    rows: pd.DataFrame,
    metric: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
) -> pd.DataFrame:
    """Average complete-prefix condition trajectories within each model."""
    required = {"model", "sampling", "h_space", "n_examples", metric}
    work = _prepare_metric_model_rows(rows, metric, required)
    if work.empty:
        return pd.DataFrame()
    work["_prompt"] = work[["sampling", "h_space"]].apply(tuple, axis=1).map(PROMPT_BAR_LABELS)
    if metric == "alpha_beta_ratio":
        work[metric] = log_positive_ratio(work[metric])
    work = work.dropna(subset=["_prompt", "n_examples", metric])
    work = work[work["n_examples"].isin(PREFIX_LENGTHS)]
    if work.empty:
        return pd.DataFrame()

    work, models = _filter_ordered_models(work, model_order)
    if work.empty:
        return pd.DataFrame()

    unit_cols = []
    for col in ("model", "_prompt", "task", "domain", "stimulus", "stimulus_kind"):
        if col in work and (col not in {"stimulus", "stimulus_kind"} or work[col].notna().any()):
            unit_cols.append(col)
    prefix_means = work.groupby(unit_cols + ["n_examples"], as_index=False)[metric].mean()
    complete_counts = (
        prefix_means.groupby(unit_cols)["n_examples"].nunique().rename("n_prefixes").reset_index()
    )
    complete_keys = complete_counts[complete_counts["n_prefixes"] == len(PREFIX_LENGTHS)][unit_cols]
    complete = prefix_means.merge(complete_keys, on=unit_cols, how="inner")
    if complete.empty:
        return pd.DataFrame()

    return _aggregate_grouped_means(
        complete, metric, ["model", "_prompt", "n_examples"], models, "prompt", PROMPT_BAR_ORDER,
        rename={"_prompt": "prompt"}, count_col="n_stimuli", extra_sort=("n_examples",),
    )


def plot_condition_trajectory(
    model_points: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str,
    hline: float | None = None,
) -> pd.DataFrame:
    """Draw condition-colored trajectories with SEM across model means."""
    set_mpl(size=8)
    series = []
    for prompt in PROMPT_BAR_ORDER:
        sub = model_points[model_points["prompt"] == prompt]
        if sub.empty:
            continue
        by_n = sub.groupby("n_examples")[metric].apply(list).to_dict()
        xs, ys, sems = aggregate_sem({int(k): v for k, v in by_n.items()})
        series.append({
            "x": xs,
            "y": ys,
            "y_err": sems,
            "label": prompt,
            "color": PROMPT_BAR_PALETTE[prompt],
        })
    if series:
        plot_trajectory_sem(
            series,
            output_path,
            xlabel="Number of examples",
            ylabel=ylabel,
            xticks=PREFIX_LENGTHS,
            hline=hline,
            hline_color="black",
        )
    return model_points.assign(metric=metric)


def plot_condition_mode_summary(
    panel_specs: tuple[dict, ...],
    output_path: str,
    model_order: tuple[str, ...] = THINKING_SUMMARY_MODEL_ORDER,
) -> dict[str, pd.DataFrame]:
    """Plot paired Non-thinking/thinking condition summaries with stacked legends."""
    return _plot_paired_mode_panels(
        panel_specs, output_path,
        rows_fn=lambda spec: condition_mode_model_rows(
            spec["rows"], spec["metric"], model_order, fit_scope=spec.get("fit_scope")
        ),
        category_col="prompt",
        categories=PROMPT_BAR_ORDER,
        palette=PROMPT_MODE_BAR_PALETTE,
        category_anchor=(0.80, 0.62),
        mode_anchor=(0.80, 0.36),
        hline_style={"color": "black", "ls": "--", "lw": 0.8, "alpha": 0.5},
        human_anchor=(0.80, 0.18),
    )


def plot_eval_mode_summary(
    panel_specs: tuple[dict, ...],
    output_path: str,
    model_order: tuple[str, ...] = THINKING_SUMMARY_MODEL_ORDER,
) -> dict[str, pd.DataFrame]:
    """Plot one complete paired cohort across all evaluation-mode panels."""
    paired_models, cohort_models = _complete_eval_mode_cohort(panel_specs, model_order)
    if not paired_models:
        return {}
    model_set = set(paired_models)
    complete_panel_specs = tuple(
        {**spec, "rows": spec["rows"][spec["rows"]["model"].isin(model_set)].copy()}
        for spec in panel_specs
    )
    panel_rows = _plot_paired_mode_panels(
        complete_panel_specs, output_path,
        rows_fn=lambda spec: eval_mode_model_rows(spec["rows"], spec["metric"], model_order),
        category_col="mode_label",
        categories=SUMMARY_EVAL_MODE_ORDER,
        palette=SUMMARY_EVAL_MODE_PALETTE,
        category_anchor=(0.80, 0.58),
        mode_anchor=(0.80, 0.30),
        hline_style={"color": "black", "ls": ":", "lw": 0.8, "alpha": 0.9},
    )
    cohort_json = json.dumps(cohort_models)
    return {
        name: rows.assign(n_pairs=len(cohort_models), cohort_models=cohort_json)
        for name, rows in panel_rows.items()
    }


def _plot_paired_mode_panels(
    panel_specs: tuple[dict, ...],
    output_path: str,
    rows_fn,
    category_col: str,
    categories: tuple[str, ...],
    palette: dict[str, object],
    category_anchor: tuple[float, float],
    mode_anchor: tuple[float, float],
    hline_style: dict,
    human_anchor: tuple[float, float] | None = None,
) -> dict[str, pd.DataFrame]:
    panel_rows = {}
    for spec in panel_specs:
        work = rows_fn(spec)
        if not work.empty:
            panel_rows[spec["name"]] = work
    if not panel_rows:
        return {}

    set_mpl(size=8)
    fig_width = 1.8 * len(panel_specs) + 1.2
    fig, axes = plt.subplots(1, len(panel_specs), figsize=(fig_width, 2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, spec in zip(axes, panel_specs):
        work = panel_rows.get(spec["name"])
        if work is None:
            ax.axis("off")
            continue
        _draw_paired_mode_bars(
            ax, work, category_col, categories, palette, spec["metric"], spec["ylabel"], spec.get("hline"),
            hline_style=hline_style,
        )

    fig.subplots_adjust(left=0.06, right=0.78, bottom=0.16, wspace=0.5)
    category_handles = category_patch_handles(palette, categories)
    fig.add_artist(add_external_fig_legend(fig, category_handles, category_anchor))
    add_external_fig_legend(fig, _mode_legend_handles(), mode_anchor)
    if human_anchor is not None:
        add_external_fig_legend(
            fig,
            [Line2D([0], [0], color="black", ls=":", lw=0.8, alpha=0.9, label="Human")],
            human_anchor,
        )
    save_fig(fig, output_path, tight_layout=False)
    return panel_rows


def plot_condition_domain_summary(
    panel_specs: tuple[dict, ...],
    output_path: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    domains: tuple[int, ...] = DEFAULT_SUMMARY_DOMAINS,
    sampling: str = "default",
    h_space: str = "implicit",
) -> dict[str, pd.DataFrame]:
    """Plot d100/d200 summaries for the default prompt condition."""
    panel_rows = {}
    for spec in panel_specs:
        work = domain_metric_model_rows(
            spec["rows"], spec["metric"], model_order,
            domains=domains, fit_scope=spec.get("fit_scope"), sampling=sampling, h_space=h_space,
        )
        if not work.empty:
            panel_rows[spec["name"]] = work
    if not panel_rows:
        return {}

    set_mpl(size=8)
    fig, axes = plt.subplots(1, len(panel_specs), figsize=(6, 2), sharex=True)
    for ax, spec in zip(axes, panel_specs):
        work = panel_rows.get(spec["name"])
        if work is None:
            ax.axis("off")
            continue
        _draw_domain_bars(ax, work, spec["metric"], spec["ylabel"], spec.get("hline"))

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, wspace=0.55)
    save_fig(fig, output_path, tight_layout=False)
    return panel_rows


def suffix_filename(filename: str, suffix: str) -> str:
    stem, ext = os.path.splitext(filename)
    return f"{stem}{suffix}{ext}"


def human_summary_point(stimulus_means: pd.DataFrame, metric: str, results_dir: str, domain: int = ROOT_DOMAIN) -> dict | None:
    """Mean human endpoint over the same complete 1..4 prefix slice (human data is d100-only)."""
    if domain != HUMAN_NATIVE_DOMAIN or metric not in {"kl_bayes", "alpha", "beta", "alpha_beta_ratio"}:
        return None
    csv_name = "divergence_metrics.csv" if metric == "kl_bayes" else "alpha_beta_fits.csv"
    human_rows = _read_domain_csv(results_dir, "human", csv_name, domain)
    if human_rows.empty:
        return None
    if metric == "kl_bayes":
        merge_keys = ["task", "domain", "stimulus", "stimulus_kind"]
        human = human_rows.merge(stimulus_means[merge_keys].drop_duplicates(), on=merge_keys, how="inner")
    else:
        full_rows = human_rows[human_rows["fit_scope"] == FULL_SCOPE]
        human = full_rows.merge(stimulus_means[["task", "domain"]].drop_duplicates(), on=["task", "domain"], how="inner")
        if human.empty:
            human = full_rows
    if human.empty:
        return None

    values = (
        pd.Series(alpha_beta_ratio(human["alpha"], human["beta"]), index=human.index)
        if metric == "alpha_beta_ratio"
        else human[metric]
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    # alpha_beta_fits rows carry n_stimuli per joint fit; divergence rows are per-stimulus.
    n_stimuli = int(human.loc[values.index, "n_stimuli"].sum()) if "n_stimuli" in human else int(len(values))
    return {"model": "Human", "n_examples": max(PREFIX_LENGTHS), metric: float(values.mean()),
            "sem": 0.0, "n_stimuli": n_stimuli}


def human_condition_hline(results_dir: str, metric: str, domain: int = ROOT_DOMAIN) -> float | None:
    """Human baseline value for condition bar reference lines (d100 only)."""
    if domain != HUMAN_NATIVE_DOMAIN or metric not in {"alpha", "beta", "kl_bayes"}:
        return None
    if metric == "kl_bayes":
        rows = _read_domain_csv(results_dir, "human", "divergence_metrics.csv", domain)
        rows = rows[rows["eval_mode"] == "posterior"] if not rows.empty else rows
    else:
        rows = _read_domain_csv(results_dir, "human", "alpha_beta_fits.csv", domain)
        rows = rows[rows["fit_scope"] == FULL_SCOPE] if not rows.empty else rows
    values = rows[metric].dropna() if not rows.empty else pd.Series(dtype=float)
    return float(values.mean()) if not values.empty else None


def plot_function(
    stimulus_means: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str,
    group_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    show_errorbar: bool = False,
    hline: float | None = None,
    human_point: dict | None = None,
    show_model_mean: bool = False,
    *, group_col: str = "model",
) -> pd.DataFrame:
    """Trajectory plot grouped by model or evaluation mode; SEM uses the input observations."""
    plot_rows = stimulus_means.copy()
    plot_human = None if human_point is None else dict(human_point)
    if metric == "alpha_beta_ratio":
        plot_rows[metric] = log_positive_ratio(plot_rows[metric]).to_numpy()
        if plot_human is not None and plot_human[metric] > 0:
            plot_human[metric] = float(np.log(plot_human[metric]))
        elif plot_human is not None:
            plot_human[metric] = np.nan

    set_mpl(size=8)
    present = set(plot_rows[group_col])
    groups = [group for group in group_order if group in present]
    colors = SUMMARY_EVAL_MODE_PALETTE if group_col == "mode_label" else summary_colors(groups)
    plot_group_df = prefix_summary(plot_rows, metric, group_order, group_col=group_col)

    series = []
    for group in groups:
        sub = plot_group_df[plot_group_df[group_col] == group].sort_values("n_examples")
        if sub.empty:
            continue
        series.append({
            "x": sub["n_examples"].tolist(),
            "y": sub[metric].tolist(),
            "y_err": sub["sem"].tolist(),
            "label": group,
            "color": colors[group],
            "fmt": "--" if group.endswith("_thinking") else "-",
            "lw": 0.8 if show_model_mean else 1.2,
            "alpha": 0.8 if show_model_mean else 0.9,
            "show_errorbar": False if show_model_mean else show_errorbar,
        })
    if show_model_mean and not plot_group_df.empty:
        mean_rows = mean_prefix_summary(plot_group_df, metric)
        if not mean_rows.empty:
            series.append({
                "x": mean_rows["n_examples"].tolist(),
                "y": mean_rows[metric].tolist(),
                "y_err": mean_rows["sem"].tolist(),
                "label": "LLM mean",
                "color": "black",
                "lw": 1.6,
                "alpha": 0.9,
                "show_errorbar": False,
            })
    if plot_human is not None:
        series.append({"x": [plot_human["n_examples"]], "y": [plot_human[metric]], "y_err": [0.0],
                       "label": "Human", "color": "black", "fmt": HUMAN_MARKER, "lw": 0.0,
                       "alpha": 0.9, "ms": 6, "mew": 1.2, "show_errorbar": False})

    if series:
        plot_trajectory_sem(
            series, output_path,
            xlabel="Number of examples", ylabel=ylabel,
            xticks=PREFIX_LENGTHS, show_errorbar=show_errorbar,
            hline=hline, hline_color="black",
        )
    group_df = prefix_summary(stimulus_means, metric, group_order, group_col=group_col)
    return pd.concat([group_df, pd.DataFrame([human_point])], ignore_index=True) if human_point is not None else group_df


def plot_thinking_mean_function(
    stimulus_means: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str,
    model_order: tuple[str, ...] = THINKING_SUMMARY_MODEL_ORDER,
    hline: float | None = None,
) -> pd.DataFrame:
    """Plot paired non-thinking/thinking model means over existing thinking runs."""
    if stimulus_means.empty:
        return pd.DataFrame()

    plot_rows = stimulus_means.copy()
    if metric == "alpha_beta_ratio":
        plot_rows[metric] = log_positive_ratio(plot_rows[metric]).to_numpy()

    model_df = prefix_summary(plot_rows, metric, model_order)
    present = set(model_df["model"])
    paired_bases = [
        model for model in model_order
        if not model.endswith("_thinking")
        and model in present
        and f"{model}_thinking" in present
    ]
    paired_models = [variant for base in paired_bases for variant in (base, f"{base}_thinking")]
    model_df = model_df[model_df["model"].isin(paired_models)]
    if model_df.empty:
        return model_df

    colors = summary_colors(paired_models)
    series = []
    for model in paired_models:
        sub = model_df[model_df["model"] == model].sort_values("n_examples")
        if sub.empty:
            continue
        series.append({
            "x": sub["n_examples"].tolist(),
            "y": sub[metric].tolist(),
            "y_err": sub["sem"].tolist(),
            "label": model,
            "color": colors[model],
            "fmt": "--" if model.endswith("_thinking") else "-",
            "lw": 0.7,
            "alpha": 0.85,
            "show_errorbar": False,
        })
    mean_specs = (
        ("Nonthinking mean", paired_bases, PALETTE["blue"], "-"),
        ("Thinking mean", [f"{model}_thinking" for model in paired_bases], PALETTE["orange"], "--"),
    )
    for label, models, color, fmt in mean_specs:
        mean_rows = mean_prefix_summary(model_df[model_df["model"].isin(models)], metric)
        if mean_rows.empty:
            continue
        series.append({
            "x": mean_rows["n_examples"].tolist(),
            "y": mean_rows[metric].tolist(),
            "y_err": mean_rows["sem"].tolist(),
            "label": label,
            "color": color,
            "fmt": fmt,
            "lw": 1.6,
            "alpha": 0.95,
            "show_errorbar": False,
        })

    if series:
        plot_trajectory_sem(
            series, output_path,
            xlabel="Number of examples", ylabel=ylabel,
            xticks=PREFIX_LENGTHS, show_errorbar=False,
            hline=hline, hline_color="black",
        )
    return model_df


def extension_mass_model_rows(
    rows: pd.DataFrame,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    out_rows = []
    work = rows.copy()
    work["extension_mass"] = pd.to_numeric(work["extension_mass"], errors="coerce")
    work = work.dropna(subset=["extension_mass"])
    for (source, model, stimulus_kind, n_examples), group in work.groupby(
        ["source", "model", "stimulus_kind", "n_examples"], sort=False
    ):
        values = group["extension_mass"].to_numpy(dtype=float)
        out_rows.append({
            "source": source,
            "model": model,
            "stimulus_kind": stimulus_kind,
            "n_examples": int(n_examples),
            "extension_mass": float(values.mean()),
            "sem": sem(values),
            "n_stimuli": int(len(values)),
        })

    out = pd.DataFrame(out_rows)
    model_rank = {model: i for i, model in enumerate(model_order)}
    kind_rank = {kind: i for i, kind in enumerate(EXTENSION_STIMULUS_KINDS)}
    out["_kind_order"] = out["stimulus_kind"].map(kind_rank)
    out["_model_order"] = out["model"].map(lambda model: -1 if model == "Bayesian" else model_rank.get(model, len(model_rank)))
    return out.sort_values(["_kind_order", "_model_order", "model", "n_examples"]).drop(
        columns=["_kind_order", "_model_order"]
    ).reset_index(drop=True)


def plot_extension_mass_summary(
    rows: pd.DataFrame,
    output_path: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
) -> pd.DataFrame:
    """Plot mean d200 extension-window mass by prefix length for rule vs similarity stimuli."""
    summary = extension_mass_model_rows(rows, model_order)
    if summary.empty:
        return summary

    set_mpl(size=8)
    fig, axes = plt.subplots(1, len(EXTENSION_STIMULUS_KINDS), figsize=(4.5, 2), sharey=True)
    present = set(summary.loc[summary["source"] == "llm", "model"])
    models = [model for model in model_order if model in present]
    colors = summary_colors(models)

    for ax, stimulus_kind in zip(axes, EXTENSION_STIMULUS_KINDS):
        kind_rows = summary[summary["stimulus_kind"] == stimulus_kind]
        for model in models:
            model_rows = kind_rows[kind_rows["model"] == model].sort_values("n_examples")
            if model_rows.empty:
                continue
            ax.plot(
                model_rows["n_examples"], model_rows["extension_mass"],
                color=colors[model], linewidth=0.8, alpha=0.8, label=model,
            )
        bayesian = kind_rows[kind_rows["source"] == "bayesian"].sort_values("n_examples")
        if not bayesian.empty:
            ax.plot(
                bayesian["n_examples"], bayesian["extension_mass"],
                color="black", linestyle="--", linewidth=2.0, alpha=0.9, label="Bayesian",
            )
        ax.set_title(EXTENSION_STIMULUS_KIND_LABELS[stimulus_kind], fontsize=plt.rcParams["axes.titlesize"] - 1)
        ax.set_xlabel("Number of examples")
        ax.set_yscale("log")
        ax.set_xticks(PREFIX_LENGTHS)
        ax.set_xlim(min(PREFIX_LENGTHS) - 0.1, max(PREFIX_LENGTHS) + 0.1)
    axes[0].set_ylabel("Domain mass (101-200)")

    handles = [
        Line2D([0], [0], color=colors[model], linewidth=0.8, alpha=0.8, label=model)
        for model in models
    ]
    handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, alpha=0.9, label="Bayesian"))
    add_external_fig_legend(
        fig, handles, (0.83, 0.5),
        fontsize=plt.rcParams["legend.fontsize"] - 1, handlelength=1.6,
    )
    fig.subplots_adjust(left=0.08, right=0.81, bottom=0.22, top=0.86, wspace=0.1)
    save_fig(fig, output_path, tight_layout=False)
    return summary


def _ci95(values: pd.Series) -> float:
    return float(1.96 * sem(values))


def human_description_top1_rows(results_dir: str, domain: int) -> pd.DataFrame:
    """Human description top-1 rows from the human baseline (d100 only; empty otherwise)."""
    if domain != HUMAN_NATIVE_DOMAIN:
        return pd.DataFrame()
    return _read_model_csv(results_dir, "human", "description_top1.csv")


def human_description_metric_by_n_examples(human_rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    value_col = HUMAN_TOP1_METRIC_COLUMNS.get(metric)
    if value_col is None or human_rows.empty:
        return pd.DataFrame()
    return (
        human_rows.dropna(subset=[value_col])
        .groupby("n_examples", as_index=False)
        .agg(value=(value_col, "mean"), n_rows=(value_col, "size"))
    )


def human_description_accuracy_support_point(human_rows: pd.DataFrame) -> pd.DataFrame:
    rows = human_rows.dropna(subset=["support_fraction", "accuracy"])
    return pd.DataFrame([{
        "model": "Human",
        "eval_mode": "hypothesis_generation",
        "support_fraction": float(rows["support_fraction"].mean()),
        "accuracy": float(rows["accuracy"].mean()),
        "n_rows": int(len(rows)),
        "domain": HUMAN_NATIVE_DOMAIN,
    }])


def hypothesis_top1_model_points(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return model-level eval/generation means for default top-1 hypothesis metrics."""
    if rows.empty or metric not in rows:
        return pd.DataFrame()
    work = rows[(rows["sampling"] == "default") & (rows["eval_mode"].isin(HYPOTHESIS_MODES))].dropna(subset=[metric])
    return work.groupby(["model", "eval_mode", "n_examples"], as_index=False)[metric].mean()


def _plot_cross_model_ci_series(
    model_points: pd.DataFrame,
    group_cols: tuple[str, ...],
    value_col: str,
    output_path: str,
    ylabel: str,
    series_specs,
    extra_series: tuple[dict, ...] = (),
    **plot_kwargs,
) -> None:
    """Plot cross-model mean±95%CI trajectories over n_examples.

    series_specs yields one (selection column/value pairs, legend label, series style
    dict) triple per trajectory; overlays arrive prebuilt via extra_series.
    """
    summary_points = (
        model_points.groupby(list(group_cols), as_index=False).agg(
            value=(value_col, "mean"),
            ci95=(value_col, _ci95),
            n_models=(value_col, "size"),
        )
    )
    series = []
    for select, label, style in series_specs:
        sub = summary_points
        for col, val in select.items():
            sub = sub[sub[col] == val]
        sub = sub.sort_values("n_examples")
        if sub.empty:
            continue
        series.append({
            "x": sub["n_examples"].astype(int).tolist(),
            "y": sub["value"].tolist(),
            "y_err": sub["ci95"].tolist(),
            "label": label,
            **style,
        })
    series.extend(extra_series)
    if series:
        plot_trajectory_sem(
            series, output_path,
            xlabel="Number of examples", ylabel=ylabel,
            xticks=PREFIX_LENGTHS, **plot_kwargs,
        )


def plot_hypothesis_top1_summary(
    rows: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str,
    human_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Plot default eval/generation top-1 trajectories with CI across models, plus d100 human lines."""
    set_mpl(size=8)
    model_points = hypothesis_top1_model_points(rows, metric)
    if model_points.empty:
        return pd.DataFrame()

    extra_series = ()
    human = human_description_metric_by_n_examples(human_rows, metric)
    if not human.empty:
        human = human.sort_values("n_examples")
        extra_series = ({
            "x": human["n_examples"].astype(int).tolist(),
            "y": human["value"].tolist(),
            "y_err": [0.0] * len(human),
            "label": "Human",
            "color": "black",
            "fmt": ":",
            "show_errorbar": False,
            "lw": 1.1,
        },)
    _plot_cross_model_ci_series(
        model_points,
        ("eval_mode", "n_examples"),
        metric,
        output_path,
        ylabel,
        tuple(({"eval_mode": spec.eval_mode}, spec.label, {"color": spec.color}) for spec in HYPOTHESIS_MODE_SPECS),
        extra_series=extra_series,
        yticks=(0, 0.5, 1),
    )
    return model_points.rename(columns={metric: "value"}).assign(metric=metric)


def plot_domain_hypothesis_top1_summary(
    rows: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str,
    domains: tuple[int, ...] = DEFAULT_SUMMARY_DOMAINS,
) -> pd.DataFrame:
    """Plot eval/generation top-1 trajectories split by d100 and d200."""
    if rows.empty or metric not in rows:
        return pd.DataFrame()
    set_mpl(size=8)
    work = rows[
        (rows["sampling"] == "default")
        & (rows["eval_mode"].isin(HYPOTHESIS_MODES))
        & (rows["domain"].isin(domains))
    ].dropna(subset=[metric])
    if work.empty:
        return pd.DataFrame()
    model_points = work.groupby(["model", "eval_mode", "domain", "n_examples"], as_index=False)[metric].mean()

    domain_styles = dict(zip(domains, ("-", "--")))
    _plot_cross_model_ci_series(
        model_points,
        ("eval_mode", "domain", "n_examples"),
        metric,
        output_path,
        ylabel,
        tuple(
            ({"eval_mode": spec.eval_mode, "domain": domain}, f"{spec.label} d{domain}",
             {"color": spec.color, "fmt": domain_styles.get(domain, "-")})
            for spec in HYPOTHESIS_MODE_SPECS
            for domain in domains
        ),
        yticks=(0, 0.5, 1),
        legend_fontsize=5.5,
    )
    return model_points.rename(columns={metric: "value"}).assign(metric=metric)


def hypothesis_js_distance_model_points(rows: pd.DataFrame) -> pd.DataFrame:
    """Model-level posterior-vs-hypothesis JS distance, each pair within its matching posterior frame."""
    if rows.empty:
        return pd.DataFrame()
    model_points = []
    for metric, spec in HYPOTHESIS_JS_DISTANCE_METRICS:
        sub = rows[
            (rows["sampling"] == "default")
            & (rows["h_space"] == spec.h_space)
            & (rows["n_examples"].isin(PREFIX_LENGTHS))
        ].dropna(subset=[metric])
        if sub.empty:
            continue
        grouped = sub.groupby(["model", "n_examples"], as_index=False)[metric].mean()
        model_points.append(grouped.rename(columns={metric: "value"}).assign(metric=metric, hypothesis_mode=spec.label))
    return _concat_frames(model_points)


def plot_hypothesis_js_distance_summary(rows: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Plot posterior-vs-eval and posterior-vs-generation JS distance across models."""
    model_df = hypothesis_js_distance_model_points(rows)
    if model_df.empty:
        return pd.DataFrame()
    _plot_cross_model_ci_series(
        model_df,
        ("hypothesis_mode", "n_examples"),
        "value",
        output_path,
        f"JS distance ({EVAL_MODE_LABELS['posterior']}, hypothesis)",
        tuple(({"hypothesis_mode": spec.label}, spec.label, {"color": spec.color}) for _metric, spec in HYPOTHESIS_JS_DISTANCE_METRICS),
    )
    return model_df


def _mode_bar_rows(points: pd.DataFrame, metric: str, panel: str) -> pd.DataFrame:
    return (
        points
        .assign(value=points[metric], hypothesis_mode=points["eval_mode"].map(EVAL_MODE_LABELS))
        .groupby(["model", "hypothesis_mode"], as_index=False)["value"].mean()
        .assign(panel=panel)
    )


def hypothesis_bar_summary_rows(top1_rows: pd.DataFrame, js_distance_rows: pd.DataFrame) -> pd.DataFrame:
    """Build one model-unit table for the compact hypothesis summary bar panels."""
    panels = []
    js_points = hypothesis_js_distance_model_points(js_distance_rows)
    if not js_points.empty:
        panels.append(js_points.groupby(["model", "hypothesis_mode"], as_index=False)["value"].mean().assign(panel="JS distance"))
    for metric, panel in (
        ("top1_normalized_confidence_mean", "Confidence"),
        ("top1_rule_proportion_mean", "Rule prop"),
    ):
        points = hypothesis_top1_model_points(top1_rows, metric)
        if not points.empty:
            panels.append(_mode_bar_rows(points, metric, panel))
    return _concat_frames(panels)


def plot_hypothesis_bar_summary(
    top1_rows: pd.DataFrame,
    js_distance_rows: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """Plot compact JS-distance/confidence/rule-proportion Eval-vs-Generate bars."""
    rows = hypothesis_bar_summary_rows(top1_rows, js_distance_rows)
    if rows.empty:
        return rows

    set_mpl(size=8)
    mode_order = HYPOTHESIS_CONDITION_MODE_ORDER
    palette = HYPOTHESIS_SUMMARY_PALETTE
    panel_specs = (
        ("JS distance", "JS distance\nto Predict"),
        ("Confidence", "Top-1\nconfidence"),
        ("Rule prop", "Top-1\nrule prop."),
    )
    fig, axes = plt.subplots(1, len(panel_specs), figsize=(2, 2), sharey=True)
    for ax, (panel, title) in zip(axes, panel_specs):
        sub = rows[rows["panel"] == panel]
        if sub.empty:
            ax.axis("off")
            continue
        sns.barplot(
            data=sub,
            x="panel",
            y="value",
            hue="hypothesis_mode",
            hue_order=mode_order,
            palette=palette,
            ax=ax,
            errorbar=("se", 1.96),
            capsize=0.0,
            err_kws={"linewidth": 0.8},
            width=0.62,
            linewidth=0,
        )
        for patch in ax.patches:
            patch.set_edgecolor("none")
            patch.set_linewidth(0)
        bar_edges = [(patch.get_x(), patch.get_x() + patch.get_width()) for patch in ax.patches]
        if bar_edges:
            ax.set_xlim(min(left for left, _right in bar_edges), max(right for _left, right in bar_edges))
        ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"] - 5, y=1.24)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([0, 0.5, 1])
        if ax is axes[0]:
            ax.tick_params(axis="y", labelsize=plt.rcParams["ytick.labelsize"] - 1)
        if ax is not axes[0]:
            ax.tick_params(axis="y", left=True, labelleft=False)
        if panel != "JS distance":
            ax.set_ylim(0, 1)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles = category_patch_handles(palette, mode_order)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.79),
        fontsize=plt.rcParams["legend.fontsize"] - 1,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.65,
    )
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.13, top=0.65, wspace=0.3)
    save_fig(fig, output_path, tight_layout=False)
    return rows


def hypothesis_condition_mode_rows(top1_rows: pd.DataFrame, js_distance_rows: pd.DataFrame) -> pd.DataFrame:
    """Build model-unit rows for Eval/Generate hypothesis bars split by thinking mode."""
    panels = []
    for panel, metric, ylabel in HYPOTHESIS_CONDITION_TOP1_PANELS:
        points = hypothesis_top1_model_points(top1_rows, metric)
        if points.empty:
            continue
        panels.append(
            points[points["n_examples"].isin(PREFIX_LENGTHS)]
            .assign(value=points[metric], hypothesis_mode=points["eval_mode"].map(EVAL_MODE_LABELS), panel=panel, ylabel=ylabel)
            .groupby(["model", "hypothesis_mode", "panel", "ylabel"], as_index=False)["value"]
            .mean()
        )
    js_points = hypothesis_js_distance_model_points(js_distance_rows)
    if not js_points.empty:
        panels.append(
            js_points.assign(panel="js_distance", ylabel="JS distance")
            .groupby(["model", "hypothesis_mode", "panel", "ylabel"], as_index=False)["value"]
            .mean()
        )
    rows = _concat_frames(panels)
    if rows.empty:
        return rows
    rows["thinking_mode"] = np.where(rows["model"].str.endswith("_thinking"), "Thinking", "Non-thinking")
    return rows


def _draw_hypothesis_condition_bars(ax, rows: pd.DataFrame, ylabel: str) -> None:
    """Draw Eval/Generate bars with thinking variants shown by transparency.

    Special case of _draw_paired_mode_bars over the hypothesis-mode category and the
    `thinking_mode` split column, with no hline and the generic `value` metric.
    """
    _draw_paired_mode_bars(
        ax, rows, "hypothesis_mode", HYPOTHESIS_CONDITION_MODE_ORDER,
        HYPOTHESIS_SUMMARY_PALETTE, "value", ylabel, mode_col="thinking_mode",
    )


def plot_hypothesis_condition_summary(
    top1_rows: pd.DataFrame,
    js_distance_rows: pd.DataFrame,
    output_path: str,
    human_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Plot paired non-thinking/thinking Eval/Generate hypothesis summaries."""
    rows = hypothesis_condition_mode_rows(top1_rows, js_distance_rows)
    if rows.empty:
        return rows

    panel_order = ("support", "accuracy", "js_distance", "confidence", "rule_prop")
    labels = rows.drop_duplicates("panel").set_index("panel")["ylabel"].to_dict()
    panel_metrics = {panel_key: metric for panel_key, metric, _ylabel in HYPOTHESIS_CONDITION_TOP1_PANELS}
    set_mpl(size=8)
    fig, axes = plt.subplots(1, len(panel_order), figsize=(8.2, 2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panel_order):
        sub = rows[rows["panel"] == panel]
        if sub.empty:
            ax.axis("off")
            continue
        _draw_hypothesis_condition_bars(ax, sub, labels[panel])
        if not human_rows.empty:
            human = human_description_metric_by_n_examples(human_rows, panel_metrics.get(panel, ""))
            if not human.empty:
                ax.axhline(float(human["value"].mean()), color="black", ls=":", lw=0.9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])

    fig.subplots_adjust(left=0.055, right=0.82, bottom=0.16, wspace=0.55)
    mode_handles = category_patch_handles(HYPOTHESIS_SUMMARY_PALETTE, HYPOTHESIS_CONDITION_MODE_ORDER)
    thinking_handles = _mode_legend_handles()
    if not human_rows.empty:
        thinking_handles.append(Line2D([0], [0], color="black", ls=":", lw=1.0, label="Human"))
    fig.add_artist(add_external_fig_legend(fig, mode_handles, (0.84, 0.58)))
    add_external_fig_legend(fig, thinking_handles, (0.84, 0.30))
    save_fig(fig, output_path, tight_layout=False)
    return rows


def hypothesis_accuracy_support_points(
    rows: pd.DataFrame,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    domain: int = ROOT_DOMAIN,
) -> pd.DataFrame:
    """Average default top-1 support and evidence accuracy across task rows."""
    if rows.empty:
        return pd.DataFrame()
    work = rows[
        (rows["domain"] == domain)
        & (rows["sampling"] == "default")
        & (rows["eval_mode"].isin(HYPOTHESIS_MODES))
        & (rows["n_examples"].isin(PREFIX_LENGTHS))
    ].dropna(subset=["top1_support_fraction", "top1_accuracy"])
    if work.empty:
        return pd.DataFrame()
    points = (
        work.groupby(["model", "eval_mode"], as_index=False)
        .agg(
            support_fraction=("top1_support_fraction", "mean"),
            accuracy=("top1_accuracy", "mean"),
            n_rows=("top1_accuracy", "size"),
        )
        .assign(domain=domain)
    )
    points["_mode_order"] = points["eval_mode"].map({mode: i for i, mode in enumerate(HYPOTHESIS_MODES)})
    return _sort_by_model_order(points.sort_values("_mode_order"), model_order).sort_values(
        "_mode_order", kind="stable",
    ).drop(columns="_mode_order").reset_index(drop=True)


def load_human_alpha_beta_scatter_rows(results_dir: str, domain: int = ROOT_DOMAIN) -> pd.DataFrame:
    """Human alpha-beta baselines in the same point schema as LLM scatter rows (d100 only)."""
    if domain != HUMAN_NATIVE_DOMAIN:
        return pd.DataFrame()
    rows = _read_domain_csv(results_dir, "human", "alpha_beta_fits.csv", domain)
    rows = rows[rows["fit_scope"] == FULL_SCOPE].dropna(subset=["alpha", "beta"]) if not rows.empty else rows
    if rows.empty:
        return pd.DataFrame()
    grouped = rows.groupby(["domain"], as_index=False).agg(
        alpha=("alpha", "mean"),
        beta=("beta", "mean"),
        n_rows=("n_stimuli", "sum"),
        n_tasks=("task", "nunique"),
    )
    return pd.concat([
        grouped.assign(model="Human", sampling=sampling, h_space=h_space, fit_scope=FULL_SCOPE)
        for sampling, h_space, *_rest in ALPHA_BETA_SCATTER_PROMPTS
    ], ignore_index=True)


def alpha_beta_scatter_points(
    rows: pd.DataFrame,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    domain: int = 100,
    human_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Mean full-scope alpha-beta points across selected task domains."""
    required = {"model", "domain", "sampling", "h_space", "fit_scope", "alpha", "beta"}
    if rows.empty or not required <= set(rows.columns):
        return pd.DataFrame()

    prompt_pairs = {(sampling, h_space) for sampling, h_space, *_ in ALPHA_BETA_SCATTER_PROMPTS}
    work = rows[
        (rows["domain"] == domain)
        & (rows["fit_scope"] == FULL_SCOPE)
    ].copy()
    work = work[work[["sampling", "h_space"]].apply(tuple, axis=1).isin(prompt_pairs)]
    if work.empty:
        return pd.DataFrame()

    work["alpha"] = pd.to_numeric(work["alpha"], errors="coerce")
    work["beta"] = pd.to_numeric(work["beta"], errors="coerce")
    work = work.dropna(subset=["alpha", "beta"])
    group_cols = ["model", "domain", "sampling", "h_space", "fit_scope"]
    points = (
        work
        .groupby(group_cols, as_index=False)
        .agg(
            alpha=("alpha", "mean"),
            beta=("beta", "mean"),
            n_rows=("alpha", "size"),
            n_tasks=("task", "nunique") if "task" in work else ("alpha", "size"),
        )
    )
    if human_rows is not None and not human_rows.empty:
        points = pd.concat([points, human_rows], ignore_index=True, sort=False)
    order = {model: i for i, model in enumerate(model_order)}
    prompt_order = {(sampling, h_space): i for i, (sampling, h_space, *_rest) in enumerate(ALPHA_BETA_SCATTER_PROMPTS)}
    points["_model_order"] = points["model"].map(lambda model: order.get(model, len(order)))
    points["_prompt_order"] = points[["sampling", "h_space"]].apply(
        lambda row: prompt_order[(row["sampling"], row["h_space"])], axis=1,
    )
    return points.sort_values(["_prompt_order", "_model_order", "model"]).drop(
        columns=["_prompt_order", "_model_order"]
    ).reset_index(drop=True)


def _scatter_model_handles(models: list[str], colors: dict[str, object]) -> list[Line2D]:
    handles = []
    for model in models:
        if model == "Bayesian":
            handles.append(Line2D([0], [0], marker="*", linestyle="", markersize=7,
                                  markerfacecolor="black", markeredgecolor="black", label="Bayesian"))
        elif model == "Human":
            handles.append(Line2D([0], [0], marker=HUMAN_MARKER, linestyle="", markersize=5,
                                  markeredgecolor="black", markeredgewidth=1.0, label="Human"))
        else:
            handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=5,
                                  markerfacecolor=colors.get(model, PALETTE["purple"]),
                                  markeredgecolor="white", markeredgewidth=0.5,
                                  alpha=0.95, label=model))
    return handles


def _plot_scatter_with_external_legend(
    points: pd.DataFrame,
    output_path: str,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    model_order: tuple[str, ...],
    draw_points,
    extra_handles: list[Line2D] | None = None,
    figsize: tuple[float, float] = (2.2, 2.0),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    reference_unit_lines: bool = False,
    dpi: int | None = 600,
) -> tuple[list[str], dict[str, object]]:
    """Shared fixed-axes scatter plot; legends live outside the exported axes."""
    plot_points = points.dropna(subset=[x_col, y_col])
    if plot_points.empty:
        return [], {}
    ordered_models, colors = ordered_scatter_models(plot_points, model_order)
    model_point_style = {"s": 42, "alpha": 0.95, "edgecolors": "white", "linewidths": 0.5}
    fig, ax = plt.subplots(figsize=figsize)
    draw_points(ax, plot_points, ordered_models, colors, model_point_style)
    if reference_unit_lines:
        ax.axvline(1.0, color="gray", ls="--", lw=0.6, alpha=0.5)
        ax.axhline(1.0, color="gray", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    handles = _scatter_model_handles(ordered_models, colors)
    if extra_handles:
        handles.extend(extra_handles)
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=plt.rcParams["legend.fontsize"] - 1,
    )
    save_fig(fig, output_path, tight_layout=False, dpi=dpi)
    return ordered_models, colors


def plot_alpha_beta_scatter_by_prompt(
    points: pd.DataFrame,
    output_dir: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
) -> pd.DataFrame:
    """Write the default-implicit full-scope alpha-beta scatter."""
    set_mpl(size=8)
    for sampling, h_space, _label, filename in ALPHA_BETA_SCATTER_PROMPTS:
        sub = points[(points["sampling"] == sampling) & (points["h_space"] == h_space)]
        if sub.empty:
            continue

        def draw_points(ax, plot_points, ordered, color_map, model_point_style):
            for model in ordered:
                model_rows = plot_points[plot_points["model"] == model]
                color = color_map.get(model, PALETTE["purple"])
                if model == "Human":
                    ax.scatter(model_rows["alpha"], model_rows["beta"], s=45, marker=HUMAN_MARKER,
                               color="black", linewidths=1.0, alpha=0.9)
                else:
                    ax.scatter(model_rows["alpha"], model_rows["beta"], color=color, **model_point_style)

        _plot_scatter_with_external_legend(
            sub,
            os.path.join(output_dir, filename),
            "alpha",
            "beta",
            "α (prior weight)",
            "β (size principle)",
            model_order,
            draw_points,
            xlim=(-0.05, 2.05),
            ylim=(-0.05, 2.05),
            xticks=[0.0, 1.0, 2.0],
            yticks=[0.0, 1.0, 2.0],
            reference_unit_lines=True,
        )
    return points


def plot_hypothesis_accuracy_support_scatter(
    points: pd.DataFrame,
    output_dir: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    filename_suffix: str = "",
    human_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write eval and generation scatter plots of accuracy vs top-1 support."""
    plot_points = points.copy()
    if human_rows is not None and not human_rows.empty:
        plot_points = pd.concat([plot_points, human_description_accuracy_support_point(human_rows)], ignore_index=True)

    mode_specs = (
        ("hypothesis_eval", f"hypothesis_eval_accuracy_support_scatter{filename_suffix}.png"),
        ("hypothesis_generation", f"hypothesis_generation_accuracy_support_scatter{filename_suffix}.png"),
    )
    set_mpl(size=8)
    for mode, filename in mode_specs:
        sub = plot_points[plot_points["eval_mode"] == mode]
        if sub.empty:
            continue

        def draw_points(ax, plot_points, _ordered, color_map, model_point_style):
            for row in plot_points.itertuples(index=False):
                if row.model == "Human":
                    ax.scatter(row.accuracy, row.support_fraction, s=38, marker=HUMAN_MARKER,
                               color="black", linewidths=1.0, alpha=0.9)
                else:
                    ax.scatter(row.accuracy, row.support_fraction,
                               color=color_map.get(row.model, PALETTE["purple"]), **model_point_style)

        _plot_scatter_with_external_legend(
            sub,
            os.path.join(output_dir, filename),
            "accuracy",
            "support_fraction",
            "Top-1 evidence accuracy",
            "Top-1 support fraction",
            model_order,
            draw_points,
            xlim=(-0.02, 1.02),
            ylim=(0.0, None),
            xticks=[0.0, 0.5, 1.0],
            yticks=[0.0, 0.3, 0.6],
        )
    return plot_points


def eval_generation_gap_rows(
    rows: pd.DataFrame,
    metrics: tuple[str, ...],
    mode_col: str = "eval_mode",
    eval_key: str = "hypothesis_eval",
    generation_key: str = "hypothesis_generation",
    observation_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Compute eval-minus-generation gaps, optionally pairing observations before model means."""
    required = {"model", mode_col, *metrics, *observation_cols}
    if rows.empty or not required <= set(rows.columns):
        return pd.DataFrame()

    index_cols = ["model", *observation_cols]
    paired = rows.pivot(index=index_cols, columns=mode_col, values=list(metrics))
    if not {eval_key, generation_key} <= set(paired.columns.get_level_values(mode_col)):
        return pd.DataFrame()

    gaps = pd.DataFrame(index=paired.index)
    for metric in metrics:
        gaps[f"{metric}_gap"] = paired[(metric, eval_key)] - paired[(metric, generation_key)]
    gaps = gaps.dropna(subset=[f"{metric}_gap" for metric in metrics])
    if not observation_cols:
        return gaps.reset_index()
    gaps = gaps.reset_index()
    gap_cols = [f"{metric}_gap" for metric in metrics]
    return gaps.groupby("model", as_index=False).agg(
        **{column: (column, "mean") for column in gap_cols},
        n_pairs=(gap_cols[0], "size"),
    )


def hypothesis_eval_generation_gap_rows(rows: pd.DataFrame, domain: int) -> pd.DataFrame:
    """Average eval-minus-generation gaps only after pairing identical observations."""
    observation_cols = ["task", "domain", "stimulus", "sampling", "n_examples"]
    metrics = ("top1_support_fraction", "top1_accuracy")
    required = {"model", "eval_mode", *observation_cols, *metrics}
    if rows.empty or not required <= set(rows.columns):
        return pd.DataFrame()

    work = rows[
        (rows["domain"] == domain)
        & (rows["sampling"] == "default")
        & (rows["eval_mode"].isin(HYPOTHESIS_MODES))
        & (rows["n_examples"].isin(PREFIX_LENGTHS))
    ].copy()
    for metric in metrics:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=list(metrics))
    if work.empty:
        return pd.DataFrame()

    gaps = eval_generation_gap_rows(work, metrics, observation_cols=tuple(observation_cols))
    return gaps.rename(columns={
        "top1_support_fraction_gap": "support_fraction_gap",
        "top1_accuracy_gap": "accuracy_gap",
    })


def plot_eval_generation_gap_bars(
    gaps: pd.DataFrame,
    metric_specs: tuple[tuple[str, str, str, str], ...],
    output_path: str,
) -> pd.DataFrame:
    """Plot mean eval-minus-generation metric gaps across models."""
    if gaps.empty:
        return pd.DataFrame()

    plot_rows = []
    for metric, label, legend_label, color in metric_specs:
        values = gaps[metric].dropna()
        plot_rows.append({
            "metric": label,
            "legend_label": legend_label,
            "value": values.mean(),
            "ci95": _ci95(values),
            "n_models": len(values),
            "color": color,
        })
    summary = pd.DataFrame(plot_rows)

    set_mpl(size=8)
    fig, ax = plt.subplots(figsize=(1.2, 2.0))
    width = 0.34
    x = np.linspace(-width / 2, width / 2, len(summary))
    bars = ax.bar(
        x,
        summary["value"],
        yerr=summary["ci95"],
        capsize=3,
        color=summary["color"],
        width=width,
        linewidth=0.0,
    )
    extent = pd.concat([summary["value"] - summary["ci95"], summary["value"] + summary["ci95"]])
    ymin = float(extent.min())
    ymax = float(extent.max())
    if ymin < 0.0:
        pad = max((ymax - ymin) * 0.12, 0.02)
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        ax.set_ylim(bottom=0.0)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.55)
    ax.set_xlim(-width, width)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Eval - Generate")
    ax.legend(
        bars,
        summary["legend_label"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        fontsize=plt.rcParams["legend.fontsize"] - 1,
        handlelength=1.0,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    save_fig(fig, output_path, tight_layout=False)
    return gaps


def plot_hypothesis_eval_generation_gap(
    rows: pd.DataFrame,
    output_dir: str,
    domain: int,
    filename_suffix: str = "",
) -> pd.DataFrame:
    """Plot mean eval-minus-generation top-1 gaps across models."""
    gaps = hypothesis_eval_generation_gap_rows(rows, domain)
    return plot_eval_generation_gap_bars(
        gaps,
        (
            ("support_fraction_gap", "Top-1 support fraction", "Top-1 support\nfraction", HYPOTHESIS_SUMMARY_COLORS[0]),
            ("accuracy_gap", "Top-1 evidence accuracy", "Top-1 evidence\naccuracy", HYPOTHESIS_SUMMARY_COLORS[1]),
        ),
        os.path.join(output_dir, f"hypothesis_eval_gen_gap{filename_suffix}.png"),
    )


def _plot_hypothesis_summary_domain(
    results_dir: str,
    output_dir: str,
    model_order: tuple[str, ...],
    domain: int,
    model_selector,
    filename_suffix: str = "",
) -> dict[str, pd.DataFrame]:
    """Draw domain-local hypothesis figures for one model cohort."""
    os.makedirs(output_dir, exist_ok=True)
    human_rows = human_description_top1_rows(results_dir, domain)
    tables = {key: [] for key in (
        "hypothesis_top1", "hypothesis_js_distance", "hypothesis_alpha_beta", "hypothesis_bar",
        "hypothesis_accuracy_support", "hypothesis_eval_generation_gap",
    )}

    top1_rows = load_hypothesis_top1_rows(results_dir, model_selector(results_dir, HYPOTHESIS_TOP1_CSV, domain), domain)
    for metric, filename, ylabel in HYPOTHESIS_TOP1_METRICS:
        tables["hypothesis_top1"].append(plot_hypothesis_top1_summary(
            top1_rows, metric, ylabel, os.path.join(output_dir, suffix_filename(filename, filename_suffix)), human_rows,
        ))

    js_rows = load_hypothesis_js_distance_rows(
        results_dir, model_selector(results_dir, HYPOTHESIS_JS_DISTANCE_CSV, domain), domain,
    )
    tables["hypothesis_js_distance"].append(plot_hypothesis_js_distance_summary(
        js_rows, os.path.join(output_dir, f"hypothesis_js_distance{filename_suffix}.png"),
    ))

    alpha_beta_rows = load_hypothesis_alpha_beta_rows(
        results_dir, model_selector(results_dir, HYPOTHESIS_ALPHA_BETA_CSV, domain), domain,
    )
    if not alpha_beta_rows.empty:
        for metric, filename, ylabel, hline in HYPOTHESIS_ALPHA_BETA_METRICS:
            tables["hypothesis_alpha_beta"].append(
                plot_function(
                    alpha_beta_rows, metric, ylabel,
                    os.path.join(output_dir, suffix_filename(filename, filename_suffix)),
                    tuple(spec.label for spec in SUMMARY_EVAL_MODE_SPECS),
                    hline=hline, show_errorbar=True, group_col="mode_label",
                ).rename(columns={metric: "value"}).assign(metric=metric)
            )

    if not top1_rows.empty and not js_rows.empty:
        tables["hypothesis_bar"].append(plot_hypothesis_bar_summary(
            top1_rows, js_rows, os.path.join(output_dir, f"hypothesis_bar_summary{filename_suffix}.png"),
        ))

    scatter_rows = load_hypothesis_top1_metric_rows(
        results_dir, model_selector(results_dir, HYPOTHESIS_TOP1_METRICS_CSV, domain), domain,
    )
    points = hypothesis_accuracy_support_points(scatter_rows, model_order=model_order, domain=domain)
    if not points.empty:
        tables["hypothesis_accuracy_support"].append(plot_hypothesis_accuracy_support_scatter(
            points, output_dir, model_order, filename_suffix=filename_suffix, human_rows=human_rows,
        ))
        tables["hypothesis_eval_generation_gap"].append(
            plot_hypothesis_eval_generation_gap(scatter_rows, output_dir, domain, filename_suffix=filename_suffix)
        )
    return {key: _concat_frames(frames) for key, frames in tables.items()}


def _run_summary_domain(
    results_dir: str,
    output_dir: str,
    thinking_output_dir: str,
    cache_dir: str,
    model_order: tuple[str, ...],
    sampling: str,
    h_space: str,
    domain: int,
) -> dict[str, pd.DataFrame]:
    """Build default-implicit four-prefix summary plots for one domain."""
    for directory in (output_dir, thinking_output_dir, cache_dir):
        os.makedirs(directory, exist_ok=True)

    # Cohorts: (model selector, model order, output dir, filename suffix) — full
    # summary view first, then paired non-thinking/thinking pairs.
    summary_cohorts = (
        (lambda root, csv_name, dom: available_summary_models(root, model_order, csv_name, dom),
         model_order, output_dir, ""),
        (available_paired_thinking_models, THINKING_SUMMARY_MODEL_ORDER, thinking_output_dir, "_thinking"),
    )
    tables = {key: [] for key in (
        "stimuli", "models", "divergence", "human_divergence", "fitting_performance", "prompt_prefix",
        "condition_task", "hypothesis_condition", "alpha_beta_scatter",
    )}

    for spec in SUMMARY_METRICS:
        for model_selector, cohort_order, cohort_output_dir, filename_suffix in summary_cohorts:
            raw = load_metric_rows(results_dir, spec, model_selector(results_dir, spec.csv_name, domain), domain)
            metric_rows = complete_prefix_rows(raw, spec.name, sampling=sampling, h_space=h_space, domain=domain) \
                if not raw.empty else pd.DataFrame()
            if metric_rows.empty:
                continue

            plot_path = os.path.join(cohort_output_dir, suffix_filename(spec.filename, filename_suffix))
            if filename_suffix:
                if spec.name in SUMMARY_MEAN_TRAJECTORY_METRICS:
                    plot_thinking_mean_function(metric_rows, spec.name, spec.ylabel, plot_path,
                                                THINKING_SUMMARY_MODEL_ORDER, hline=spec.hline)
                else:
                    plot_function(metric_rows, spec.name, spec.ylabel, plot_path, cohort_order, hline=spec.hline)
                continue

            model_df = plot_function(
                metric_rows, spec.name, spec.ylabel, plot_path, cohort_order,
                hline=spec.hline, human_point=human_summary_point(metric_rows, spec.name, results_dir, domain=domain),
                show_model_mean=spec.name in SUMMARY_MEAN_TRAJECTORY_METRICS,
            )
            tables["stimuli"].append(metric_rows.rename(columns={spec.name: "value"}).assign(metric=spec.name))
            tables["models"].append(model_df.rename(columns={spec.name: "value"}).assign(metric=spec.name))
            prompt_prefix_rows = condition_prefix_model_rows(raw, spec.name, model_order)
            if not prompt_prefix_rows.empty:
                tables["prompt_prefix"].append(plot_condition_trajectory(
                    prompt_prefix_rows, spec.name, spec.ylabel,
                    os.path.join(output_dir, f"condition_{spec.filename}"), spec.hline,
                ))

    for model_selector, cohort_order, _cohort_output_dir, filename_suffix in summary_cohorts:
        human_divergence_rows = _load_posterior_divergence_rows(
            results_dir, model_selector(results_dir, HUMAN_DIVERGENCE_CSV, domain), HUMAN_DIVERGENCE_CSV, domain,
        )
        if not human_divergence_rows.empty:
            tables["human_divergence"].append(
                condition_metric_model_rows(human_divergence_rows, "kl_human", cohort_order).assign(
                    figure=suffix_filename("human_condition_kl", filename_suffix)
                )
            )

    def paired(csv_name):
        return available_paired_thinking_models(results_dir, csv_name, domain)

    thinking_alpha_beta_rows = load_alpha_beta_rows(results_dir, paired("alpha_beta_fits.csv"), domain)
    thinking_entropy_rows = load_metric_rows(
        results_dir,
        SummaryMetric("entropy", "trajectory_prefix_metrics.csv", "Posterior entropy (nats)", "entropy.png"),
        paired("trajectory_prefix_metrics.csv"),
        domain,
    )
    if not thinking_entropy_rows.empty:
        thinking_entropy_rows = thinking_entropy_rows[thinking_entropy_rows["n_examples"] == max(PREFIX_LENGTHS)]
    human_hlines = {metric: human_condition_hline(results_dir, metric, domain) for metric in ("alpha", "beta", "kl_bayes")}
    condition_panel_rows = plot_condition_mode_summary(
        (
            {"name": "alpha", "rows": thinking_alpha_beta_rows, "metric": "alpha", "ylabel": "α (prior weight)", "fit_scope": FULL_SCOPE, "hline": human_hlines["alpha"]},
            {"name": "beta", "rows": thinking_alpha_beta_rows, "metric": "beta", "ylabel": "β (size principle)", "fit_scope": FULL_SCOPE, "hline": human_hlines["beta"]},
            {"name": "kl", "rows": _load_posterior_divergence_rows(results_dir, paired("divergence_metrics.csv"), "divergence_metrics.csv", domain),
             "metric": "kl_bayes", "ylabel": "KL with Bayesian", "hline": human_hlines["kl_bayes"]},
            {"name": "entropy", "rows": thinking_entropy_rows, "metric": "entropy", "ylabel": "Posterior entropy (nats)"},
        ),
        os.path.join(output_dir, "condition.png"),
        THINKING_SUMMARY_MODEL_ORDER,
    )
    if "kl" in condition_panel_rows:
        tables["divergence"].append(condition_panel_rows["kl"].assign(figure="condition"))

    eval_mode_models = paired(HYPOTHESIS_ALPHA_BETA_CSV)
    eval_mode_alpha_beta_rows = load_eval_mode_alpha_beta_rows(results_dir, eval_mode_models, domain)
    eval_mode_panel_rows = plot_eval_mode_summary(
        (
            {"name": "alpha", "rows": eval_mode_alpha_beta_rows, "metric": "alpha", "ylabel": "α (prior weight)"},
            {"name": "beta", "rows": eval_mode_alpha_beta_rows, "metric": "beta", "ylabel": "β (size principle)"},
            {"name": "kl", "rows": load_eval_mode_kl_rows(results_dir, eval_mode_models, domain), "metric": "kl_bayes", "ylabel": "KL with Bayesian"},
            {"name": "entropy", "rows": load_eval_mode_predictive_entropy_rows(results_dir, eval_mode_models, domain),
             "metric": "predictive_entropy", "ylabel": "Predictive entropy (nats)"},
        ),
        os.path.join(output_dir, "condition_task.png"),
        THINKING_SUMMARY_MODEL_ORDER,
    )
    for name, rows in eval_mode_panel_rows.items():
        tables["condition_task"].append(rows.assign(metric=name))

    hypothesis_condition_top1_rows = load_hypothesis_top1_rows(results_dir, paired(HYPOTHESIS_TOP1_CSV), domain)
    hypothesis_condition_js_rows = load_hypothesis_js_distance_rows(results_dir, paired(HYPOTHESIS_JS_DISTANCE_CSV), domain)
    tables["hypothesis_condition"].append(plot_hypothesis_condition_summary(
        hypothesis_condition_top1_rows, hypothesis_condition_js_rows,
        os.path.join(output_dir, "hypothesis_condition.png"),
        human_description_top1_rows(results_dir, domain),
    ))

    alpha_beta_models = available_summary_models(results_dir, model_order, "alpha_beta_fits.csv", domain)
    alpha_beta_rows = load_alpha_beta_rows(results_dir, alpha_beta_models, domain)
    fitting_points = fitting_performance_model_rows(alpha_beta_rows, model_order, sampling, h_space)
    if not fitting_points.empty:
        tables["fitting_performance"].append(
            plot_fitting_performance(fitting_points, os.path.join(output_dir, "fitting_performance.png"), model_order)
        )

    hypothesis_tables = _plot_hypothesis_summary_domain(results_dir, output_dir, model_order, domain, available_nonthinking_models)
    _plot_hypothesis_summary_domain(
        results_dir, thinking_output_dir, THINKING_SUMMARY_MODEL_ORDER, domain, available_paired_thinking_models,
        filename_suffix="_thinking",
    )

    alpha_beta_points = alpha_beta_scatter_points(
        alpha_beta_rows, model_order=model_order, domain=domain,
        human_rows=load_human_alpha_beta_scatter_rows(results_dir, domain),
    )
    if not alpha_beta_points.empty:
        tables["alpha_beta_scatter"].append(plot_alpha_beta_scatter_by_prompt(alpha_beta_points, output_dir, model_order))

    summary_tables = {**{key: _concat_frames(frames) for key, frames in tables.items()}, **hypothesis_tables}
    for table_key, filename in (
        ("stimuli", "summary_default_implicit_stimulus_metrics.csv"),
        ("models", "summary_default_implicit_model_metrics.csv"),
        ("divergence", "summary_condition_kl_by_model.csv"),
        ("human_divergence", "summary_human_condition_kl_by_model.csv"),
        ("fitting_performance", "summary_fitting_performance_by_model.csv"),
        ("prompt_prefix", "summary_condition_prefix_by_model.csv"),
        ("condition_task", "summary_condition_task_by_model.csv"),
        ("hypothesis_condition", "summary_hypothesis_condition_by_model.csv"),
        ("hypothesis_top1", "summary_hypothesis_top1_by_model.csv"),
        ("hypothesis_js_distance", "summary_hypothesis_js_distance_by_model.csv"),
        ("hypothesis_alpha_beta", "summary_hypothesis_alpha_beta_by_model.csv"),
        ("hypothesis_bar", "summary_hypothesis_bar_by_model.csv"),
        ("hypothesis_accuracy_support", "summary_hypothesis_accuracy_support_scatter.csv"),
        ("hypothesis_eval_generation_gap", "summary_hypothesis_eval_generation_gap.csv"),
        ("alpha_beta_scatter", "summary_alpha_beta_scatter.csv"),
    ):
        save_csv(summary_tables[table_key], cache_dir, domain_file_prefix(domain) + filename)
    return summary_tables


def _summary_thinking_output_dir(output_dir: str, domain: int) -> str:
    return os.path.join(output_dir, f"d{domain}_thinking")


def _condition_domain_panel_specs(results_dir: str, model_order: tuple[str, ...], domains: tuple[int, ...],
                                  paired: bool = False) -> tuple[dict, ...]:
    alpha_beta_rows, divergence_rows = [], []
    for domain in domains:
        if paired:
            alpha_beta_models = available_paired_thinking_models(results_dir, "alpha_beta_fits.csv", domain)
            divergence_models = available_paired_thinking_models(results_dir, "divergence_metrics.csv", domain)
        else:
            alpha_beta_models = available_summary_models(results_dir, model_order, "alpha_beta_fits.csv", domain)
            divergence_models = available_summary_models(results_dir, model_order, "divergence_metrics.csv", domain)
        alpha_beta_rows.append(load_alpha_beta_rows(results_dir, alpha_beta_models, domain))
        divergence_rows.append(_load_posterior_divergence_rows(results_dir, divergence_models, "divergence_metrics.csv", domain))
    alpha_beta_df = _concat_frames(alpha_beta_rows)
    return (
        {"name": "alpha", "rows": alpha_beta_df, "metric": "alpha", "ylabel": "α (prior weight)", "fit_scope": FULL_SCOPE, "hline": 1.0},
        {"name": "beta", "rows": alpha_beta_df, "metric": "beta", "ylabel": "β (size principle)", "fit_scope": FULL_SCOPE, "hline": 1.0},
        {"name": "kl", "rows": _concat_frames(divergence_rows), "metric": "kl_bayes", "ylabel": "KL(Bayesian || LLM)"},
        {"name": "loss", "rows": alpha_beta_df, "metric": "mse", "ylabel": "MSE (fit)", "fit_scope": FULL_SCOPE},
    )


def _plot_extension_summaries(
    results_dir: str,
    output_dir: str,
    cache_dir: str,
    model_order: tuple[str, ...],
    specs,
    *,
    paired: bool = False,
    filename_suffix: str = "",
) -> dict[str, pd.DataFrame]:
    """Extension-mass and domain-shift figures for each (eval_mode, h_space, sampling) spec."""
    extension_tables, shift_tables, shift_summaries = [], [], []
    for eval_mode, h_space, sampling in specs:
        mass_models = available_extension_models(results_dir, "mass", eval_mode, h_space, sampling, model_order, paired=paired)
        mass_rows = load_extension_mass_rows(results_dir, mass_models, eval_mode, h_space, sampling)
        if (mass_rows["source"] == "llm").any() if not mass_rows.empty else False:
            extension_tables.append(plot_extension_mass_summary(
                mass_rows, os.path.join(output_dir, suffix_filename(f"extension_mass_{eval_mode}.png", filename_suffix)),
                model_order,
            ).assign(eval_mode=eval_mode))

        shift_models = available_extension_models(results_dir, "domain_shift", eval_mode, h_space, sampling, model_order, paired=paired)
        if not shift_models:
            continue
        shift_rows = load_domain_shift_rows(results_dir, shift_models, eval_mode, h_space, sampling)
        shift_tables.append(shift_rows)
        shift_summaries.append(plot_domain_shift_summary(
            shift_rows, os.path.join(output_dir, suffix_filename(f"domain_shift_{eval_mode}.png", filename_suffix)),
            model_order,
        ).assign(eval_mode=eval_mode))

    extension_df = _concat_frames(extension_tables)
    shift_df = _concat_frames(shift_tables)
    save_csv(extension_df, cache_dir, suffix_filename(f"summary_extension_mass_{specs[0][0]}.csv", filename_suffix))
    save_csv(shift_df, cache_dir, suffix_filename(f"summary_domain_shift_{specs[0][0]}.csv", filename_suffix))
    return {"extension_mass": extension_df, "domain_shift": shift_df, "domain_shift_summary": _concat_frames(shift_summaries)}


def run_hypothesis_summary(
    results_dir: str = RESULTS_ROOT,
    output_dir: str | None = None,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
) -> dict[str, pd.DataFrame]:
    """Cross-model summary figures that depend on hypothesis eval/generation caches."""
    output_dir = output_dir or os.path.join(results_dir, "summary")
    cache_dir = os.path.join(output_dir, "cache")
    thinking_output_dir = _summary_thinking_output_dir(output_dir, EXTENSION_DOMAIN)
    for directory in (output_dir, cache_dir, thinking_output_dir):
        os.makedirs(directory, exist_ok=True)
    hypothesis_specs = tuple((spec.eval_mode, spec.h_space, spec.sampling) for spec in HYPOTHESIS_MODE_SPECS)

    tables = _plot_extension_summaries(results_dir, output_dir, cache_dir, model_order, hypothesis_specs)
    gap = plot_hypothesis_domain_shift_gap(tables["domain_shift"], output_dir)
    save_csv(gap, cache_dir, "summary_domain_shift_eval_generation_gap.csv")

    mode_summary = plot_eval_mode_domain_shift_summary(
        load_eval_mode_domain_shift_rows(results_dir, model_order), os.path.join(output_dir, "domain_shift_modes.png"), model_order,
    )
    save_csv(mode_summary, cache_dir, "summary_domain_shift_modes.csv")

    thinking = _plot_extension_summaries(
        results_dir, thinking_output_dir, cache_dir, THINKING_SUMMARY_MODEL_ORDER, hypothesis_specs,
        paired=True, filename_suffix="_thinking",
    )
    thinking_gap = plot_hypothesis_domain_shift_gap(thinking["domain_shift"], thinking_output_dir, "_thinking")
    save_csv(thinking_gap, cache_dir, "summary_domain_shift_eval_generation_gap_thinking.csv")
    return {**tables, "domain_shift_gap": gap, "domain_shift_modes": mode_summary}


def _plot_posterior_extension_diagnostics(
    results_dir: str,
    output_dir: str,
    cache_dir: str,
    model_order: tuple[str, ...],
    sampling: str,
    h_space: str,
    *,
    paired: bool = False,
    filename_suffix: str = "",
) -> dict[str, pd.DataFrame]:
    """Posterior-space d200 extension, rule-discrimination, and domain-shift figures."""
    extension_h_space, extension_sampling = EXTENSION_POSTERIOR_PROMPT
    mass_models = available_extension_models(results_dir, "mass", "posterior", extension_h_space, extension_sampling, model_order, paired=paired)
    mass_rows = load_extension_mass_rows(results_dir, mass_models, "posterior", extension_h_space, extension_sampling)
    mass_summary = (
        plot_extension_mass_summary(
            mass_rows, os.path.join(output_dir, suffix_filename("extension_mass_posterior.png", filename_suffix)), model_order,
        )
        if mass_models else pd.DataFrame()
    )
    save_csv(mass_summary, cache_dir, suffix_filename("summary_extension_mass_posterior.csv", filename_suffix))

    order = THINKING_SUMMARY_MODEL_ORDER if paired else model_order
    discrimination_models = available_extension_models(results_dir, "rule_discrimination", "posterior", h_space, sampling, model_order, paired=paired)
    instance_rows = load_rule_discrimination_rows(results_dir, discrimination_models, h_space, sampling)
    discrimination = rule_discrimination_points(instance_rows)
    save_csv(discrimination, cache_dir, suffix_filename("summary_rule_discrimination.csv", filename_suffix))
    plot_rule_discrimination_summary(
        discrimination, os.path.join(output_dir, suffix_filename("rule_discrimination.png", filename_suffix)), order, instance_rows,
    )

    shift_models = available_extension_models(results_dir, "domain_shift", "posterior", h_space, sampling, model_order, paired=paired)
    shift_rows = load_domain_shift_rows(results_dir, shift_models, "posterior", h_space, sampling)
    save_csv(shift_rows, cache_dir, suffix_filename("summary_domain_shift_posterior.csv", filename_suffix))
    shift_summary = plot_domain_shift_summary(
        shift_rows, os.path.join(output_dir, suffix_filename("domain_shift_posterior.png", filename_suffix)), order,
    )
    return {"extension_mass": mass_summary, "rule_discrimination": discrimination, "domain_shift": shift_summary}


def run_summary(
    results_dir: str = RESULTS_ROOT,
    output_dir: str | None = None,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    sampling: str = "default",
    h_space: str = "implicit",
    domains: tuple[int, ...] = DEFAULT_SUMMARY_DOMAINS,
) -> dict[str, pd.DataFrame]:
    """Build the root-domain summary in output_dir and sibling domain summaries under d{domain}/."""
    output_dir = output_dir or os.path.join(results_dir, "summary")
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    results = {
        domain: _run_summary_domain(
            results_dir, domain_subdir(output_dir, domain), _summary_thinking_output_dir(output_dir, domain),
            cache_dir, model_order, sampling, h_space, domain,
        )
        for domain in domains
    }
    root = results[domains[0]]

    domain_hypothesis_rows = load_domain_hypothesis_top1_rows(results_dir, domains)
    root["domain_hypothesis_top1"] = _concat_frames([
        plot_domain_hypothesis_top1_summary(domain_hypothesis_rows, metric, ylabel, os.path.join(output_dir, filename), domains)
        for metric, filename, ylabel in DOMAIN_HYPOTHESIS_TOP1_METRICS
    ])
    save_csv(root["domain_hypothesis_top1"], cache_dir, "summary_domain_hypothesis_top1_by_model.csv")

    root.update(_plot_posterior_extension_diagnostics(results_dir, output_dir, cache_dir, model_order, sampling, h_space))
    thinking_output_dir = _summary_thinking_output_dir(output_dir, EXTENSION_DOMAIN)
    os.makedirs(thinking_output_dir, exist_ok=True)
    _plot_posterior_extension_diagnostics(
        results_dir, thinking_output_dir, cache_dir, THINKING_SUMMARY_MODEL_ORDER, sampling, h_space,
        paired=True, filename_suffix="_thinking",
    )

    plot_condition_domain_summary(
        _condition_domain_panel_specs(results_dir, model_order, domains),
        os.path.join(output_dir, "condition_domain.png"),
        model_order, domains=domains, sampling=sampling, h_space=h_space,
    )
    plot_condition_domain_summary(
        _condition_domain_panel_specs(results_dir, THINKING_SUMMARY_MODEL_ORDER, domains, paired=True),
        os.path.join(output_dir, "condition_domain_thinking.png"),
        THINKING_SUMMARY_MODEL_ORDER, domains=domains, sampling=sampling, h_space=h_space,
    )
    return root


def main():
    parser = argparse.ArgumentParser(description="Cross-model default four-prefix summary plots.")
    parser.add_argument("--results-dir", default=RESULTS_ROOT)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run_summary(results_dir=args.results_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
