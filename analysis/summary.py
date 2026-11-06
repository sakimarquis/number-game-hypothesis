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
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

from analysis.alpha_beta_scope import FULL_SCOPE, PREFIX_LENGTHS
from analysis.extension import EXTENSION_DOMAIN, EXTENSION_STIMULUS_KINDS
from analysis.hypothesis_space import RULE_SHARE_TOP_K
from analysis.plotter import (
    C_INTERVAL,
    C_RULE,
    ERRBAR_ALPHA,
    EVAL_MODE_COLORS,
    EVAL_MODE_LABELS,
    LINE_ALPHA,
    PALETTE,
    SUMMARY_MODEL_ORDER,
    aggregate_sem,
    model_colors,
    plot_trajectory_sem,
    save_csv,
    save_fig,
    sem,
    set_mpl,
)
from configs.layout import analysis_dir, domain_file_prefix, domain_subdir
from configs.settings import BAYESIAN, HUMAN, RESULTS_ROOT, SUMMARY_MODE_CONDITIONS, SUMMARY_PROMPTS, THINKING_SUFFIX
from tasks.stimuli import BASE_DOMAIN, bigelow_stimulus, classify_bigelow_kind


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


SUMMARY_EVAL_MODE_SPECS = tuple(SummaryEvalMode(*condition) for condition in SUMMARY_MODE_CONDITIONS)
HYPOTHESIS_MODE_SPECS = SUMMARY_EVAL_MODE_SPECS[1:]
SUMMARY_EVAL_MODE_ORDER = tuple(spec.label for spec in SUMMARY_EVAL_MODE_SPECS)
SUMMARY_EVAL_MODE_PALETTE = {spec.label: spec.color for spec in SUMMARY_EVAL_MODE_SPECS}
SUMMARY_EVAL_MODE_LABELS = {(spec.eval_mode, spec.h_space, spec.sampling): spec.label for spec in SUMMARY_EVAL_MODE_SPECS}
HYPOTHESIS_MODES = tuple(spec.eval_mode for spec in HYPOTHESIS_MODE_SPECS)

DEFAULT_SUMMARY_DOMAINS = (BASE_DOMAIN, EXTENSION_DOMAIN)
HUMAN_DIVERGENCE_CSV = "divergence_human_metrics.csv"
HYPOTHESIS_TOP1_CSV = "hypothesis_compare_top1_summary.csv"
HYPOTHESIS_TOP1_METRICS_CSV = "hypothesis_compare_top1_metrics.csv"
HYPOTHESIS_ALPHA_BETA_CSV = "alpha_beta_mode_fits.csv"
HYPOTHESIS_ENTROPY_CSV = "hypothesis_compare_entropy_summary.csv"
HYPOTHESIS_ENTROPY_METRICS_CSV = "hypothesis_compare_entropy_metrics.csv"
CONDITION_RULE_AUC_K = 3  # top-k rule-share cutoff of the condition_eval_gen.png Rule AUC panel
HYPOTHESIS_TOP1_METRICS = (
    ("top1_normalized_confidence_mean", "hypothesis_confidence.png", "Top-1 normalized confidence"),
    ("top1_hypothesis_size_mean", "hypothesis_size.png", "Top-1 hypothesis size"),
    ("top1_accuracy_mean", "hypothesis_accuracy.png", "Top-1 evidence accuracy"),
    ("top1_rule_preference_mean", "hypothesis_rule_preference.png", "Top-1 rule preference"),
)
NON_THINKING, THINKING = THINKING_MODE_ORDER = ("Non-thinking", "Thinking")
# Paired trajectories draw thinking as dashed lines and paired bars as lighter bars; color carries the category.
THINKING_LINE_STYLES = {NON_THINKING: "-", THINKING: "--"}
THINKING_BAR_ALPHAS = {NON_THINKING: 1.0, THINKING: 0.5}
HYPOTHESIS_SUMMARY_PALETTE = {spec.label: spec.color for spec in HYPOTHESIS_MODE_SPECS}
HYPOTHESIS_CONDITION_MODE_ORDER = tuple(spec.label for spec in HYPOTHESIS_MODE_SPECS)
HYPOTHESIS_CONDITION_TOP1_PANELS = (
    ("size", "top1_hypothesis_size_mean", "Hypothesis size"),
    ("accuracy", "top1_accuracy_mean", "Accuracy"),
    ("confidence", "top1_normalized_confidence_mean", "Confidence"),
    ("rule_pref", "top1_rule_preference_mean", "Rule preference"),
)
HUMAN_TOP1_METRIC_COLUMNS = {
    "top1_rule_preference_mean": "rule_preference",
    "top1_hypothesis_size_mean": "hypothesis_size",
    "top1_accuracy_mean": "accuracy",
}
PROMPT_BAR_STYLES = {
    ("implicit", "default"): ("Default", "gray"),
    ("explicit", "default"): ("Explicit", "#5573cd"),
    ("implicit", "strong"): ("Strong", "#b55d60"),
    ("implicit", "weak"): ("Weak", "#df8920"),
}
PROMPT_BAR_ORDER = tuple(PROMPT_BAR_STYLES[prompt][0] for prompt in SUMMARY_PROMPTS)
PROMPT_BAR_PALETTE = {label: color for label, color in PROMPT_BAR_STYLES.values()}
PROMPT_BAR_LABELS = {prompt: label for prompt, (label, _color) in PROMPT_BAR_STYLES.items()}
PROMPT_MODE_BAR_PALETTE = {
    "Default": "#404040",
    "Explicit": "midnightblue",
    "Strong": "darkred",
    "Weak": "darkorange",
}
STIMULUS_KIND_LABELS = {"rule": "Rule", "similarity": "Similarity"}


@dataclass(frozen=True)
class SummaryMetric:
    name: str
    csv_name: str
    ylabel: str
    filename: str


SUMMARY_METRICS = (
    SummaryMetric("kl_bayes", "trajectory_prefix_metrics.csv", "KL(Bayesian || LLM)", "kl.png"),
    # Task-pooled prefix-scope size-principle fits under each prompt's pooled prior.
    SummaryMetric("beta", "alpha_beta_fits.csv", "Size principle (β)", "beta.png"),
)
SUMMARY_MEAN_TRAJECTORY_METRICS = {"beta"}

# Headline models drawn in summary/; every registered summary model is drawn in summary/all/.
BEST_SUMMARY_MODELS = ("Gemma4_A4B", "Qwen36_A3B", "GPT56_Luna", "Gemini36_Flash")
# (output subdir, model order, full). Only full cohorts draw the non-thinking fitting_performance,
# hypothesis accuracy/size scatter, and rule_discrimination figures.
SUMMARY_COHORTS = (("", BEST_SUMMARY_MODELS, False), ("all", SUMMARY_MODEL_ORDER, True))
# Size comparison: (size, one base model per family), families in the same order for every size.
SIZE_FAMILY_MODELS = (
    ("Small", ("Gemma4_E4B", "Qwen35_4B")),
    ("Large", ("Gemma4_A4B", "Qwen36_A3B")),
)
SIZE_COLORS = ("violet", "indigo")
SIZE_MODEL_ORDER = tuple(variant for _size, models in SIZE_FAMILY_MODELS for model in models for variant in (model, f"{model}{THINKING_SUFFIX}"))
HUMAN_MARKER = "x"


def _analysis_csv_path(results_dir: str, result_key: str, csv_name: str, domain: int) -> str:
    return os.path.join(domain_subdir(analysis_dir(result_key, results_root=results_dir), domain), csv_name)


def _read_domain_csv(results_dir: str, result_key: str, csv_name: str, domain: int) -> pd.DataFrame:
    """Read one per-domain analysis CSV, keeping only rows for `domain`."""
    path = _analysis_csv_path(results_dir, result_key, csv_name, domain)
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["domain"] == domain].copy() if not df.empty else df


def _read_model_csv(results_dir: str, result_key: str, csv_name: str) -> pd.DataFrame:
    """Read one cross-domain CSV from a result key's root analysis directory."""
    path = os.path.join(analysis_dir(result_key, results_root=results_dir), csv_name)
    return pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()


def available_summary_models(
    results_dir: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    analysis_csv: str = "trajectory_prefix_metrics.csv",
    domain: int = BASE_DOMAIN,
) -> list[str]:
    return [
        model for model in model_order
        if os.path.isfile(_analysis_csv_path(results_dir, model, analysis_csv, domain))
    ]


def thinking_model_order(model_order: tuple[str, ...]) -> tuple[str, ...]:
    """Interleave each base model with its thinking variant."""
    return tuple(variant for model in model_order for variant in (model, f"{model}{THINKING_SUFFIX}"))


def _paired_thinking_models(available) -> list[str]:
    """Base/thinking pairs, in `available` order, where both variants are available."""
    available_set = set(available)
    return [
        variant
        for model in available
        if not model.endswith(THINKING_SUFFIX) and f"{model}{THINKING_SUFFIX}" in available_set
        for variant in (model, f"{model}{THINKING_SUFFIX}")
    ]


def available_paired_thinking_models(results_dir: str, thinking_order: tuple[str, ...], analysis_csv: str,
                                     domain: int = BASE_DOMAIN) -> list[str]:
    """Base/thinking pairs where both variants have the requested cache."""
    return _paired_thinking_models(available_summary_models(results_dir, thinking_order, analysis_csv, domain))


def load_metric_rows(results_dir: str, spec: SummaryMetric, models: list[str], domain: int = BASE_DOMAIN) -> pd.DataFrame:
    """Load one metric's per-prefix LLM rows from each model's analysis cache."""
    def rows_for_model(model: str) -> pd.DataFrame:
        df = _read_domain_csv(results_dir, model, spec.csv_name, domain)
        if df.empty:
            return df
        return df[df["source"] == "llm"].copy() if "source" in df else df

    return _concat_model_rows(models, rows_for_model)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _concat_model_rows(models: list[str], rows_for_model) -> pd.DataFrame:
    return _concat_frames([rows_for_model(model).assign(model=model) for model in models])


def _model_csv_rows(results_dir: str, models: list[str], csv_name: str, domain: int) -> pd.DataFrame:
    return _concat_model_rows(models, lambda model: _read_domain_csv(results_dir, model, csv_name, domain))


def _load_posterior_divergence_rows(results_dir: str, models: list[str], csv_name: str, domain: int = BASE_DOMAIN) -> pd.DataFrame:
    """Load posterior rows from each model's divergence cache."""
    rows = _model_csv_rows(results_dir, models, csv_name, domain)
    return rows[rows["eval_mode"] == "posterior"].copy() if not rows.empty else rows


def load_alpha_beta_rows(results_dir: str, models: list[str], domain: int = BASE_DOMAIN) -> pd.DataFrame:
    """Load task-pooled posterior size-principle fits."""
    return _model_csv_rows(results_dir, models, "alpha_beta_fits.csv", domain)


def load_hypothesis_top1_rows(results_dir: str, models: list[str], domain: int = BASE_DOMAIN) -> pd.DataFrame:
    """Load per-model hypothesis top-1 summaries; model remains the CI unit."""
    return _model_csv_rows(results_dir, models, HYPOTHESIS_TOP1_CSV, domain)


def load_hypothesis_top1_metric_rows(results_dir: str, models: list[str], domain: int = BASE_DOMAIN) -> pd.DataFrame:
    """Load stimulus-level top-1 hypothesis metrics for domain-specific scatter summaries."""
    return _model_csv_rows(results_dir, models, HYPOTHESIS_TOP1_METRICS_CSV, domain)




def load_hypothesis_beta_rows(results_dir: str, models: list[str], domain: int = BASE_DOMAIN) -> pd.DataFrame:
    """Per-model default-condition eval-mode size-principle trajectories (task-pooled prefix scopes).

    Predict rows equal the default implicit prefix fits behind beta.png.
    """
    rows = _with_mode_labels(_model_csv_rows(results_dir, models, HYPOTHESIS_ALPHA_BETA_CSV, domain)) \
        if models else pd.DataFrame()
    rows = rows.dropna(subset=["n_examples"]) if not rows.empty else rows
    if rows.empty:
        return rows
    grouped = rows.groupby(["model", "eval_mode", "mode_label", "n_examples"], as_index=False)["beta"].mean()
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


def load_eval_mode_alpha_beta_rows(results_dir: str, models: list[str], domain: int = BASE_DOMAIN) -> pd.DataFrame:
    """Default full-scope size-principle fits for Predict/Eval/Generate mode bars."""
    rows = _model_csv_rows(results_dir, models, HYPOTHESIS_ALPHA_BETA_CSV, domain)
    return _with_mode_labels(rows[rows["fit_scope"] == FULL_SCOPE] if not rows.empty else rows)


def load_eval_mode_kl_rows(results_dir: str, models: list[str], domain: int = BASE_DOMAIN) -> pd.DataFrame:
    """Bayesian KL rows for Predict/Eval/Generate evaluation-mode bars."""
    return _with_mode_labels(_concat_frames([
        _model_csv_rows(results_dir, models, "divergence_metrics.csv", domain),
        _model_csv_rows(results_dir, models, "hypothesis_divergence_metrics.csv", domain),
    ]))


def load_eval_mode_predictive_entropy_rows(results_dir: str, models: list[str], domain: int = BASE_DOMAIN) -> pd.DataFrame:
    """Per-prefix predictive entropy (nats) for Predict/Eval/Generate modes."""
    rows = _with_mode_labels(_model_csv_rows(results_dir, models, HYPOTHESIS_ENTROPY_CSV, domain))
    if rows.empty:
        return rows
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


def prefix_summary(rows: pd.DataFrame, metric: str, model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER) -> pd.DataFrame:
    """Aggregate prefix points over stimuli per model."""
    out = (
        rows.dropna(subset=[metric])
        .groupby(["model", "n_examples"], as_index=False)
        .agg(**{metric: (metric, "mean"), "sem": (metric, sem), "n_stimuli": (metric, "size")})
    )
    rank = {model: i for i, model in enumerate(model_order)}
    return out.assign(_model_order=out["model"].map(rank).fillna(len(rank))).sort_values(
        ["_model_order", "model", "n_examples"]
    ).drop(columns="_model_order").reset_index(drop=True)


def mean_prefix_summary(model_points: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean trajectory over model-level prefix means; SEM is across models."""
    return (
        model_points.dropna(subset=[metric])
        .groupby("n_examples", as_index=False)
        .agg(**{metric: (metric, "mean"), "sem": (metric, sem), "n_models": (metric, "size")})
        .assign(model="Mean")
    )


def thinking_labels(models: pd.Series) -> np.ndarray:
    """Thinking/Non-thinking label of each result key, from its thinking suffix."""
    return np.where(models.str.endswith(THINKING_SUFFIX), THINKING, NON_THINKING)




def summary_colors(models: list[str]) -> dict[str, object]:
    """Use base-model colors for thinking variants."""
    bases = list(dict.fromkeys(model.removesuffix(THINKING_SUFFIX) for model in models))
    base_colors = model_colors(bases)
    return {model: base_colors[model.removesuffix(THINKING_SUFFIX)] for model in models}


# ===== Tenenbaum99 d200 extension diagnostics (from analysis.extension CSVs) =====

def _extension_rows(results_dir: str, models: list[str], name: str, eval_mode: str, h_space: str, sampling: str) -> pd.DataFrame:
    """LLM rows of one extension table for one eval mode and prompt, plus Bayesian reference rows."""
    csv_name = f"{'extension' if eval_mode == 'posterior' else 'hypothesis_extension'}_{name}.csv"
    llm = _concat_model_rows(models, lambda model: _read_model_csv(results_dir, model, csv_name))
    if not llm.empty:
        llm = llm[(llm["eval_mode"] == eval_mode) & (llm["h_space"] == h_space) & (llm["sampling"] == sampling)]
        llm = llm.assign(source="llm")
    bayesian = _read_model_csv(results_dir, BAYESIAN, f"extension_{name}.csv")
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
    """Models whose extension table has rows for this eval mode and prompt; paired callers pass a thinking order."""
    rows = _extension_rows(results_dir, list(model_order), name, eval_mode, h_space, sampling)
    present = set(rows.loc[rows["source"] == "llm", "model"]) if not rows.empty else set()
    available = [model for model in model_order if model in present]
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


def load_eval_mode_extension_mass_rows(results_dir: str, model_order: tuple[str, ...], *, paired: bool = False) -> pd.DataFrame:
    """LLM d200 extension mass for each SummaryEvalMode default condition; paired callers pass a thinking order."""
    tables = []
    for spec in SUMMARY_EVAL_MODE_SPECS:
        models = available_extension_models(results_dir, "mass", spec.eval_mode, spec.h_space, spec.sampling, model_order, paired=paired)
        rows = load_extension_mass_rows(results_dir, models, spec.eval_mode, spec.h_space, spec.sampling)
        if not rows.empty:
            tables.append(rows[rows["source"] == "llm"].assign(mode_label=spec.label))
    return _concat_frames(tables)


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
    work["_prompt"] = work[["h_space", "sampling"]].apply(tuple, axis=1).map(PROMPT_BAR_LABELS)
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
    model_order: tuple[str, ...],
    fit_scope: str | None = None,
) -> pd.DataFrame:
    """Average each paired condition/mode within model before cross-model error bars."""
    model_rows = condition_metric_model_rows(rows, metric, model_order, fit_scope=fit_scope)
    if model_rows.empty:
        return pd.DataFrame()
    model_rows["thinking"] = thinking_labels(model_rows["model"])
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
) -> None:
    """Draw category-colored bars split by non-thinking/thinking variants."""
    x = np.arange(len(categories), dtype=float)
    width = 0.34
    offsets = {NON_THINKING: -width / 2, THINKING: width / 2}
    for category_idx, category in enumerate(categories):
        for mode in THINKING_MODE_ORDER:
            values = rows.loc[
                (rows[category_col] == category) & (rows["thinking"] == mode), metric
            ].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            mean = float(np.mean(values))
            ax.bar(
                x[category_idx] + offsets[mode], mean, width=width,
                color=palette[category], alpha=THINKING_BAR_ALPHAS[mode], edgecolor="none", linewidth=0,
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


def eval_mode_model_rows(
    rows: pd.DataFrame,
    metric: str,
    model_order: tuple[str, ...],
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
    grouped["thinking"] = thinking_labels(grouped["model"])
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
        if model.endswith(THINKING_SUFFIX):
            continue
        thinking_model = f"{model}{THINKING_SUFFIX}"
        if model in complete_models and thinking_model in complete_models:
            cohort_models.append(model)
            paired_models.extend((model, thinking_model))
    return tuple(paired_models), tuple(cohort_models)


def _mode_legend_handles() -> list[Patch]:
    return [Patch(facecolor="#404040", edgecolor="none", alpha=alpha, label=mode) for mode, alpha in THINKING_BAR_ALPHAS.items()]




def model_metric_rows(
    rows: pd.DataFrame,
    metric: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    sampling: str = "default",
    h_space: str = "implicit",
    fit_scope: str | None = None,
) -> pd.DataFrame:
    """Average one metric within each model for one prompt condition."""
    required = {"model", "sampling", "h_space", metric}
    work = _prepare_metric_model_rows(rows, metric, required, fit_scope=fit_scope)
    if work.empty:
        return pd.DataFrame()
    work = work[(work["sampling"] == sampling) & (work["h_space"] == h_space)].dropna(subset=[metric])
    work, models = _filter_ordered_models(work, model_order)
    if work.empty:
        return pd.DataFrame()

    extra_aggs = {"n_stimuli": ("n_stimuli", "sum")} if "n_stimuli" in work else None
    grouped = work.groupby(["model"], as_index=False).agg(
        **{metric: (metric, "mean")},
        n_rows=(metric, "size"),
        **(extra_aggs or {}),
    )
    grouped["_model_order"] = grouped["model"].map({model: i for i, model in enumerate(models)})
    return grouped.sort_values("_model_order").drop(columns="_model_order").reset_index(drop=True)


def plot_model_metric_bars(
    panels: list[tuple[pd.DataFrame, str, str]],
    output_path: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
) -> list[pd.DataFrame]:
    """Draw one bar-per-model panel per (model_points, metric, ylabel), sharing one model legend."""
    set_mpl(size=8)
    panel_rows = [_filter_ordered_models(points, model_order)[0] for points, _metric, _ylabel in panels]
    models = [model for model in model_order if any(model in set(rows["model"]) for rows in panel_rows)]
    colors = summary_colors(models)
    fig, axes = plt.subplots(1, len(panels), figsize=(0.35 * len(models) * len(panels), 2.5), squeeze=False)
    for ax, rows, (_points, metric, ylabel) in zip(axes[0], panel_rows, panels):
        ax.bar(rows["model"].map(models.index), rows[metric], color=[colors[model] for model in rows["model"]], edgecolor="none")
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    add_external_fig_legend(fig, category_patch_handles(colors, models), (1.0, 0.5))
    save_fig(fig, output_path)
    return [rows.assign(metric=metric) for rows, (_points, metric, _ylabel) in zip(panel_rows, panels)]


def plot_human_correlation_by_kind(
    rows: pd.DataFrame,
    output_path: str,
    model_order: tuple[str, ...],
    sampling: str = "default",
    h_space: str = "implicit",
) -> pd.DataFrame:
    """Draw LLM-human correlation on rule vs similarity stimuli: mean ± SEM across per-model means."""
    work = rows[(rows["sampling"] == sampling) & (rows["h_space"] == h_space)
                & rows["stimulus_kind"].isin(STIMULUS_KIND_LABELS)].dropna(subset=["correlation"])
    work, _models = _filter_ordered_models(work, model_order)
    if work.empty:
        return pd.DataFrame()
    model_points = work.groupby(["model", "stimulus_kind"], as_index=False)["correlation"].mean()
    summary = model_points.groupby("stimulus_kind").agg(
        correlation=("correlation", "mean"), sem=("correlation", sem), n_models=("correlation", "size"),
    ).reindex(list(STIMULUS_KIND_LABELS)).reset_index()

    set_mpl(size=8)
    fig, ax = plt.subplots(figsize=(1.6, 2.5))
    x = np.arange(len(summary), dtype=float)
    ax.bar(x, summary["correlation"], yerr=summary["sem"], color=[C_RULE, C_INTERVAL],
           edgecolor="none", capsize=3, error_kw={"elinewidth": 0.8, "alpha": ERRBAR_ALPHA})
    ax.set_xticks(x, [STIMULUS_KIND_LABELS[kind] for kind in summary["stimulus_kind"]])
    ax.set_ylabel("LLM–human correlation")
    ax.set_ylim(bottom=0.0)
    save_fig(fig, output_path)
    return summary


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
    work["_prompt"] = work[["h_space", "sampling"]].apply(tuple, axis=1).map(PROMPT_BAR_LABELS)
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
    human_point: dict | None = None,
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
    if series and human_point is not None:
        series.append(human_line_series(human_point, metric))
    if series:
        plot_trajectory_sem(
            series,
            output_path,
            xlabel="Number of examples",
            ylabel=ylabel,
            xticks=PREFIX_LENGTHS,
        )
    return model_points.assign(metric=metric)


def plot_condition_mode_summary(
    panel_specs: tuple[dict, ...],
    output_path: str,
    model_order: tuple[str, ...],
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
        reference_anchor=(0.80, 0.18),
    )


def plot_eval_mode_summary(
    panel_specs: tuple[dict, ...],
    output_path: str,
    model_order: tuple[str, ...],
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
        reference_anchor=(0.80, 0.14),
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
    reference_anchor: tuple[float, float],
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
    if any(spec.get("hline") is not None for spec in panel_specs):
        add_external_fig_legend(fig, [Line2D([0], [0], **hline_style, label="Human")], reference_anchor, handlelength=1.6)
    save_fig(fig, output_path, tight_layout=False)
    return panel_rows




def suffix_filename(filename: str, suffix: str) -> str:
    stem, ext = os.path.splitext(filename)
    return f"{stem}{suffix}{ext}"


def human_summary_point(stimulus_means: pd.DataFrame, metric: str, results_dir: str, domain: int = BASE_DOMAIN) -> dict | None:
    """Mean human endpoint over the same complete 1..4 prefix slice (human data is d100-only)."""
    if domain != BASE_DOMAIN or metric not in {"kl_bayes", "beta"}:
        return None
    csv_name = "divergence_metrics.csv" if metric == "kl_bayes" else "alpha_beta_fits.csv"
    human_rows = _read_domain_csv(results_dir, HUMAN, csv_name, domain)
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

    values = human[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    # alpha_beta_fits rows carry n_stimuli per joint fit; divergence rows are per-stimulus.
    n_stimuli = int(human.loc[values.index, "n_stimuli"].sum()) if "n_stimuli" in human else int(len(values))
    return {"model": "Human", "n_examples": max(PREFIX_LENGTHS), metric: float(values.mean()),
            "sem": 0.0, "n_stimuli": n_stimuli}


def human_line_series(human_point: dict, metric: str) -> dict:
    """Human endpoint as a dotted reference line across all prefixes."""
    value = human_point[metric]
    return {"x": list(PREFIX_LENGTHS), "y": [value] * len(PREFIX_LENGTHS), "y_err": [0.0] * len(PREFIX_LENGTHS),
            "label": "Human", "color": "black", "fmt": ":", "lw": 1.2, "alpha": 0.9, "show_errorbar": False}


def human_line_handle() -> Line2D:
    return Line2D([0], [0], color="black", ls=":", lw=1.2, label="Human")


def human_condition_hline(
    results_dir: str, metric: str, domain: int = BASE_DOMAIN, stimulus_rows: pd.DataFrame | None = None,
) -> float | None:
    """Human baseline value for condition bar reference lines (d100 only).

    `stimulus_rows` restricts per-stimulus human rows to the stimuli the bars average over.
    """
    if domain != BASE_DOMAIN or metric not in {"beta", "kl_bayes", "entropy"}:
        return None
    if metric in {"kl_bayes", "entropy"}:
        rows = _read_domain_csv(results_dir, HUMAN, "divergence_metrics.csv", domain)
        rows = rows[rows["eval_mode"] == "posterior"] if not rows.empty else rows
        if stimulus_rows is not None and not rows.empty:
            keys = ["task", "domain", "stimulus"]
            rows = rows.merge(stimulus_rows[keys].drop_duplicates(), on=keys, how="inner")
    else:
        rows = _read_domain_csv(results_dir, HUMAN, "alpha_beta_fits.csv", domain)
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
    human_point: dict | None = None,
    show_model_mean: bool = False,
    *, ceiling_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-model trajectory plot; SEM uses the input observations.

    Thinking variants are dashed; when present, the legend splits into model colors and line styles.
    `ceiling_rows` (per-stimulus `metric` of a perfect Bayesian under text-trial measurement) add the
    pooled ceiling as a reference line.
    """
    plot_rows = stimulus_means.copy()

    set_mpl(size=8)
    present = set(plot_rows["model"])
    groups = [group for group in group_order if group in present]
    colors = summary_colors(groups)
    plot_group_df = prefix_summary(plot_rows, metric, group_order)

    series = []
    for group in groups:
        sub = plot_group_df[plot_group_df["model"] == group].sort_values("n_examples")
        if sub.empty:
            continue
        series.append({
            "x": sub["n_examples"].tolist(),
            "y": sub[metric].tolist(),
            "y_err": sub["sem"].tolist(),
            "label": group,
            "color": colors[group],
            "fmt": "--" if group.endswith(THINKING_SUFFIX) else "-",
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
    if ceiling_rows is not None and not ceiling_rows.empty:
        ceiling = prefix_summary(ceiling_rows.assign(model="ceiling"), metric, ("ceiling",))
        series.append({"x": ceiling["n_examples"].tolist(), "y": ceiling[metric].tolist(), "y_err": ceiling["sem"].tolist(),
                       "label": "Text-trial ceiling", "color": "gray", "fmt": ":", "lw": 1.2, "alpha": 0.9,
                       "show_errorbar": False})
    if human_point is not None:
        series.append(human_line_series(human_point, metric))

    legend_groups = None
    if any(group.endswith(THINKING_SUFFIX) for group in groups):
        bases = [group for group in groups if not group.endswith(THINKING_SUFFIX)]
        style_handles = [Line2D([0], [0], color="#404040", ls=fmt, lw=1.2, label=mode) for mode, fmt in THINKING_LINE_STYLES.items()]
        labels = {s["label"] for s in series}
        reference_handles = (
            Line2D([0], [0], color="black", lw=1.6, label="LLM mean"),
            Line2D([0], [0], color="gray", ls=":", lw=1.2, label="Text-trial ceiling"),
            human_line_handle(),
        )
        style_handles += [handle for handle in reference_handles if handle.get_label() in labels]
        legend_groups = (category_patch_handles(colors, bases), style_handles)
    if series:
        plot_trajectory_sem(
            series, output_path,
            xlabel="Number of examples", ylabel=ylabel,
            xticks=PREFIX_LENGTHS, show_errorbar=show_errorbar,
            legend_groups=legend_groups,
        )
    group_df = prefix_summary(stimulus_means, metric, group_order)
    return pd.concat([group_df, pd.DataFrame([human_point])], ignore_index=True) if human_point is not None else group_df


def plot_thinking_mean_function(
    stimulus_means: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str,
    model_order: tuple[str, ...],
    show_models: bool = True,
    human_point: dict | None = None,
) -> pd.DataFrame:
    """Plot paired non-thinking/thinking model means over existing thinking runs."""
    if stimulus_means.empty:
        return pd.DataFrame()

    model_df = prefix_summary(stimulus_means, metric, model_order)
    present = set(model_df["model"])
    paired_bases = [
        model for model in model_order
        if not model.endswith(THINKING_SUFFIX)
        and model in present
        and f"{model}{THINKING_SUFFIX}" in present
    ]
    paired_models = [variant for base in paired_bases for variant in (base, f"{base}{THINKING_SUFFIX}")]
    model_df = model_df[model_df["model"].isin(paired_models)]
    if model_df.empty:
        return model_df

    colors = summary_colors(paired_models)
    series = []
    for model in paired_models if show_models else ():
        sub = model_df[model_df["model"] == model].sort_values("n_examples")
        if sub.empty:
            continue
        series.append({
            "x": sub["n_examples"].tolist(),
            "y": sub[metric].tolist(),
            "y_err": sub["sem"].tolist(),
            "label": model,
            "color": colors[model],
            "fmt": "--" if model.endswith(THINKING_SUFFIX) else "-",
            "lw": 0.7,
            "alpha": 0.85,
            "show_errorbar": False,
        })
    mean_specs = zip(
        THINKING_MODE_ORDER,
        (paired_bases, [f"{model}{THINKING_SUFFIX}" for model in paired_bases]),
    )
    for mode, models in mean_specs:
        mean_rows = mean_prefix_summary(model_df[model_df["model"].isin(models)], metric)
        if mean_rows.empty:
            continue
        # With per-model lines drawn, the mean is one line among many; alone, it is the mode itself.
        series.append({
            "x": mean_rows["n_examples"].tolist(),
            "y": mean_rows[metric].tolist(),
            "y_err": mean_rows["sem"].tolist(),
            "label": f"{mode} mean" if show_models else mode,
            "color": "black",
            "fmt": THINKING_LINE_STYLES[mode],
            "lw": 1.6,
            "alpha": 0.9,
            "show_errorbar": not show_models,
        })
    if human_point is not None:
        series.append(human_line_series(human_point, metric))

    if series:
        plot_trajectory_sem(
            series, output_path,
            xlabel="Number of examples", ylabel=ylabel,
            xticks=PREFIX_LENGTHS, show_errorbar=False,
        )
    return model_df


def plot_size_mean_function(
    stimulus_means: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str,
    human_point: dict | None = None,
) -> pd.DataFrame:
    """Plot one mean line per size x thinking mode, averaged over SIZE_FAMILY_MODELS families.

    A (size, mode) line is drawn only when every family has that model, so lines average the same families.
    """
    model_df = prefix_summary(stimulus_means, metric, SIZE_MODEL_ORDER)
    present = set(model_df["model"])

    series, tables = [], []
    for (size, base_models), color in zip(SIZE_FAMILY_MODELS, SIZE_COLORS):
        for mode in THINKING_MODE_ORDER:
            models = [model if mode == NON_THINKING else f"{model}{THINKING_SUFFIX}" for model in base_models]
            if not set(models) <= present:
                continue
            mean_rows = mean_prefix_summary(model_df[model_df["model"].isin(models)], metric)
            tables.append(mean_rows.assign(size=size, thinking=mode))
            series.append({
                "x": mean_rows["n_examples"].tolist(),
                "y": mean_rows[metric].tolist(),
                "y_err": mean_rows["sem"].tolist(),
                "color": color,
                "fmt": THINKING_LINE_STYLES[mode],
                "lw": 1.6,
                "alpha": 0.9,
                "show_errorbar": False,
            })
    if human_point is not None:
        series.append(human_line_series(human_point, metric))
    if series:
        style_handles = [Line2D([0], [0], color="#404040", ls=fmt, lw=1.2, label=mode) for mode, fmt in THINKING_LINE_STYLES.items()]
        if human_point is not None:
            style_handles.append(human_line_handle())
        plot_trajectory_sem(
            series, output_path,
            xlabel="Number of examples", ylabel=ylabel,
            xticks=PREFIX_LENGTHS, show_errorbar=False,
            legend_groups=(category_patch_handles(dict(zip((size for size, _ in SIZE_FAMILY_MODELS), SIZE_COLORS)),
                                                  [size for size, _ in SIZE_FAMILY_MODELS]), style_handles),
        )
    return _concat_frames(tables)




def plot_extension_mass_summary(rows: pd.DataFrame, bayesian_rows: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Plot Predict/Eval/Generate d200 extension mass by prefix over base/thinking pairs complete for every mode."""
    modes_by_model = rows.groupby("model")["mode_label"].agg(set)
    complete = {model for model, modes in modes_by_model.items() if set(SUMMARY_EVAL_MODE_ORDER) <= modes}
    paired = [model for model in complete if not model.endswith(THINKING_SUFFIX) and f"{model}{THINKING_SUFFIX}" in complete]
    rows = rows[rows["model"].isin(paired + [f"{model}{THINKING_SUFFIX}" for model in paired])]
    bayesian = bayesian_rows.groupby(["stimulus_kind", "n_examples"], as_index=False)["extension_mass"].mean()

    # Rule/interval panels: color is the mode, line style is thinking, 95% CI across models, dotted Bayesian.
    model_points = rows.groupby(["model", "mode_label", "stimulus_kind", "n_examples"], as_index=False)["extension_mass"].mean()
    model_points["thinking"] = thinking_labels(model_points["model"])
    summary = model_points.groupby(["mode_label", "thinking", "stimulus_kind", "n_examples"], as_index=False).agg(
        extension_mass=("extension_mass", "mean"), ci95=("extension_mass", _ci95), n_models=("extension_mass", "size"),
    )

    set_mpl(size=8)
    fig, axes = plt.subplots(1, len(EXTENSION_STIMULUS_KINDS), figsize=(4.5, 2), sharey=True)
    for ax, stimulus_kind in zip(axes, EXTENSION_STIMULUS_KINDS):
        for mode_label in SUMMARY_EVAL_MODE_ORDER:
            color = SUMMARY_EVAL_MODE_PALETTE[mode_label]
            for thinking, fmt in THINKING_LINE_STYLES.items():
                sub = summary[
                    (summary["mode_label"] == mode_label) & (summary["thinking"] == thinking)
                    & (summary["stimulus_kind"] == stimulus_kind)
                ].sort_values("n_examples")
                if sub.empty:
                    continue
                ax.errorbar(sub["n_examples"], sub["extension_mass"], yerr=sub["ci95"], fmt="none",
                            ecolor=color, alpha=ERRBAR_ALPHA, capsize=2, elinewidth=0.8)
                ax.plot(sub["n_examples"], sub["extension_mass"], fmt, color=color, lw=1.2, alpha=LINE_ALPHA)
        kind_bayesian = bayesian[bayesian["stimulus_kind"] == stimulus_kind].sort_values("n_examples")
        ax.plot(kind_bayesian["n_examples"], kind_bayesian["extension_mass"], ":", color="black", lw=1.2, alpha=LINE_ALPHA)
        ax.set_title(STIMULUS_KIND_LABELS[stimulus_kind], fontsize=plt.rcParams["axes.titlesize"] - 1)
        ax.set_xlabel("Number of examples")
        ax.set_xticks(PREFIX_LENGTHS)
        ax.set_xlim(min(PREFIX_LENGTHS) - 0.1, max(PREFIX_LENGTHS) + 0.1)
    axes[0].set_ylim(bottom=0.0)  # after all panels draw, so the shared top covers both
    axes[0].set_ylabel("Prob. mass on 101–200")
    style_handles = [
        *(Line2D([0], [0], color="#404040", ls=fmt, lw=1.2, label=thinking) for thinking, fmt in THINKING_LINE_STYLES.items()),
        Line2D([0], [0], color="black", ls=":", lw=1.2, label="Bayesian"),
    ]
    fig.add_artist(add_external_fig_legend(fig, category_patch_handles(SUMMARY_EVAL_MODE_PALETTE, SUMMARY_EVAL_MODE_ORDER), (0.83, 0.64)))
    add_external_fig_legend(fig, style_handles, (0.83, 0.33), handlelength=1.6)
    fig.subplots_adjust(left=0.08, right=0.81, bottom=0.22, top=0.86, wspace=0.1)
    save_fig(fig, output_path, tight_layout=False)
    return summary.assign(n_cohort_pairs=len(paired))


def _ci95(values: pd.Series) -> float:
    return float(1.96 * sem(values))


def extension_decline_points(rows: pd.DataFrame) -> pd.DataFrame:
    """Per model/mode decline of d200 extension mass over prefixes, averaged over the stimuli of each kind.

    drop = m(1) - mean m(2..4); slope = least-squares slope of m over n. A stimulus with an undefined prefix
    (all-zero d200 P(yes)) is skipped, and `n_stimuli_*` counts the stimuli behind each kind.
    """
    mass = rows.set_index(["model", "mode_label", "stimulus_kind", "stimulus", "n_examples"])["extension_mass"].unstack()
    mass = mass[list(PREFIX_LENGTHS)]  # [stimulus, prefix]
    n = np.asarray(PREFIX_LENGTHS, dtype=float) - np.mean(PREFIX_LENGTHS)
    values = mass.to_numpy()
    stimuli = pd.DataFrame({
        "drop": values[:, 0] - values[:, 1:].mean(axis=1),
        "slope": values @ n / (n @ n),
    }, index=mass.index)
    points = stimuli.groupby(["model", "mode_label", "stimulus_kind"]).agg(
        drop=("drop", "mean"), slope=("slope", "mean"), n_stimuli=("drop", "count"),
    ).unstack("stimulus_kind")
    points.columns = [f"{metric}_{kind}" for metric, kind in points.columns]
    return points.reset_index()


def plot_extension_decline_bars(points: pd.DataFrame, metric: str, label: str, output_path: str,
                                model_order: tuple[str, ...]) -> None:
    """Similarity minus rule decline per model, one panel per mode; thinking bars are lighter, Bayesian dashed."""
    contrast = points[f"{metric}_similarity"] - points[f"{metric}_rule"]
    points = points.assign(contrast=contrast, base=points["model"].str.removesuffix(THINKING_SUFFIX)).dropna(subset=["contrast"])
    llm = points[points["model"] != BAYESIAN].assign(thinking=lambda d: thinking_labels(d["model"]))
    bayesian = float(points.loc[points["model"] == BAYESIAN, "contrast"].iloc[0])
    models = [model for model in model_order if model in set(llm["base"])]
    colors = summary_colors(models)
    width = 0.4
    offsets = {NON_THINKING: -width / 2, THINKING: width / 2}

    set_mpl(size=8)
    fig, axes = plt.subplots(1, len(SUMMARY_EVAL_MODE_ORDER), figsize=(6.2, 2.0), sharey=True)
    for ax, mode_label in zip(axes, SUMMARY_EVAL_MODE_ORDER):
        for row in llm[llm["mode_label"] == mode_label].itertuples(index=False):
            ax.bar(models.index(row.base) + offsets[row.thinking], row.contrast, width=width,
                   color=colors[row.base], alpha=THINKING_BAR_ALPHAS[row.thinking], edgecolor="none")
        ax.axhline(0.0, color="black", lw=0.6)
        ax.axhline(bayesian, color="black", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(mode_label, fontsize=plt.rcParams["axes.titlesize"] - 1)
        ax.set_xticks([])
    axes[0].set_ylabel(f"Similarity − rule {label}")
    handles = [*category_patch_handles(colors, models), *_mode_legend_handles(),
               Line2D([0], [0], color="black", ls="--", lw=0.8, label="Bayesian")]
    add_external_fig_legend(fig, handles, (0.84, 0.5))
    fig.subplots_adjust(left=0.09, right=0.83, bottom=0.08, top=0.88, wspace=0.08)
    save_fig(fig, output_path, tight_layout=False, dpi=600)


def load_human_description_top1_rows(results_dir: str, domain: int) -> pd.DataFrame:
    """Human description top-1 rows from the human baseline (d100 only; empty otherwise)."""
    if domain != BASE_DOMAIN:
        return pd.DataFrame()
    return _read_model_csv(results_dir, HUMAN, "description_top1.csv")


def human_description_metric_by_n_examples(human_rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    value_col = HUMAN_TOP1_METRIC_COLUMNS.get(metric)
    if value_col is None or human_rows.empty:
        return pd.DataFrame()
    return (
        human_rows.dropna(subset=[value_col])
        .groupby("n_examples", as_index=False)
        .agg(value=(value_col, "mean"), n_rows=(value_col, "size"))
    )


def human_description_accuracy_size_point(human_rows: pd.DataFrame) -> pd.DataFrame:
    rows = human_rows.dropna(subset=["hypothesis_size", "accuracy"])
    return pd.DataFrame([{
        "model": "Human",
        "eval_mode": "hypothesis_generation",
        "hypothesis_size": float(rows["hypothesis_size"].mean()),
        "accuracy": float(rows["accuracy"].mean()),
        "n_rows": int(len(rows)),
        "domain": BASE_DOMAIN,
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
    """Plot paired eval/generation top-1 trajectories split by thinking, with CI across models, plus d100 human lines."""
    set_mpl(size=8)
    model_points = hypothesis_top1_model_points(rows, metric)
    if model_points.empty:
        return pd.DataFrame()
    model_points["thinking"] = thinking_labels(model_points["model"])

    extra_series = ()
    human = human_description_metric_by_n_examples(human_rows, metric)
    if not human.empty:
        human = human.sort_values("n_examples")
        extra_series = ({
            "x": human["n_examples"].astype(int).tolist(),
            "y": human["value"].tolist(),
            "y_err": [0.0] * len(human),
            "color": "black",
            "fmt": ":",
            "show_errorbar": False,
            "lw": 1.1,
        },)
    style_handles = [Line2D([0], [0], color="#404040", ls=fmt, lw=1.2, label=thinking) for thinking, fmt in THINKING_LINE_STYLES.items()]
    if extra_series:
        style_handles.append(Line2D([0], [0], color="black", ls=":", lw=1.1, label="Human"))
    _plot_cross_model_ci_series(
        model_points,
        ("eval_mode", "thinking", "n_examples"),
        metric,
        output_path,
        ylabel,
        tuple(
            ({"eval_mode": spec.eval_mode, "thinking": thinking}, None, {"color": spec.color, "fmt": fmt})
            for spec in HYPOTHESIS_MODE_SPECS
            for thinking, fmt in THINKING_LINE_STYLES.items()
        ),
        extra_series=extra_series,
        # Accuracy sits near 1, so its range follows the data.
        yticks=None if metric == "top1_accuracy_mean" else (0, 0.5, 1),
        legend_groups=(category_patch_handles(HYPOTHESIS_SUMMARY_PALETTE, HYPOTHESIS_CONDITION_MODE_ORDER), style_handles),
    )
    return model_points.rename(columns={metric: "value"}).assign(metric=metric)










def plot_hypothesis_beta_summary(rows: pd.DataFrame, output_path: str, human_point: dict | None = None,
                                 thinking_styles: dict[str, str] = THINKING_LINE_STYLES) -> pd.DataFrame:
    """Plot paired Predict/Eval/Generate size-principle trajectories per thinking variant in thinking_styles, with CI across models."""
    if rows.empty:
        return pd.DataFrame()
    set_mpl(size=8)
    metric = "beta"
    model_points = rows[["model", "mode_label", "n_examples", metric]].copy()
    model_points["thinking"] = thinking_labels(model_points["model"])
    # A single thinking variant needs no line-style key.
    style_handles = [Line2D([0], [0], color="#404040", ls=fmt, lw=1.2, label=thinking) for thinking, fmt in thinking_styles.items()] if len(thinking_styles) > 1 else []
    extra_series = ()
    if human_point is not None:
        extra_series = (human_line_series(human_point, metric),)
        style_handles.append(human_line_handle())
    _plot_cross_model_ci_series(
        model_points,
        ("mode_label", "thinking", "n_examples"),
        metric,
        output_path,
        "Size principle (β)",
        tuple(
            ({"mode_label": spec.label, "thinking": thinking}, None, {"color": spec.color, "fmt": fmt})
            for spec in SUMMARY_EVAL_MODE_SPECS
            for thinking, fmt in thinking_styles.items()
        ),
        extra_series=extra_series,
        legend_groups=(category_patch_handles(SUMMARY_EVAL_MODE_PALETTE, SUMMARY_EVAL_MODE_ORDER), style_handles),
    )
    return model_points.rename(columns={metric: "value"}).assign(metric=metric)


def plot_hypothesis_entropy_summary(rows: pd.DataFrame, output_path: str, human_entropy: float | None = None) -> pd.DataFrame:
    """Plot paired Predict/Eval/Generate predictive-entropy trajectories split by thinking, with CI across models."""
    if rows.empty:
        return pd.DataFrame()
    set_mpl(size=8)
    metric = "predictive_entropy"
    # Tasks are averaged within each model so the model stays the CI unit.
    model_points = rows.groupby(["model", "mode_label", "n_examples"], as_index=False)[metric].mean()
    model_points["thinking"] = thinking_labels(model_points["model"])
    style_handles = [Line2D([0], [0], color="#404040", ls=fmt, lw=1.2, label=thinking) for thinking, fmt in THINKING_LINE_STYLES.items()]
    extra_series = ()
    if human_entropy is not None:
        extra_series = (human_line_series({metric: human_entropy}, metric),)
        style_handles.append(human_line_handle())
    _plot_cross_model_ci_series(
        model_points,
        ("mode_label", "thinking", "n_examples"),
        metric,
        output_path,
        "Predictive entropy (nats)",
        tuple(
            ({"mode_label": spec.label, "thinking": thinking}, None, {"color": spec.color, "fmt": fmt})
            for spec in SUMMARY_EVAL_MODE_SPECS
            for thinking, fmt in THINKING_LINE_STYLES.items()
        ),
        extra_series=extra_series,
        legend_groups=(category_patch_handles(SUMMARY_EVAL_MODE_PALETTE, SUMMARY_EVAL_MODE_ORDER), style_handles),
    )
    return model_points.rename(columns={metric: "value"}).assign(metric=metric)


def hypothesis_condition_mode_rows(top1_rows: pd.DataFrame) -> pd.DataFrame:
    """Build model-unit rows for Eval/Generate hypothesis bars split by thinking mode."""
    panels = []
    for panel, metric, ylabel in HYPOTHESIS_CONDITION_TOP1_PANELS:
        points = hypothesis_top1_model_points(top1_rows, metric)
        if points.empty:
            continue
        panels.append(
            points[points["n_examples"].isin(PREFIX_LENGTHS)]
            .assign(value=points[metric], mode_label=points["eval_mode"].map(EVAL_MODE_LABELS), panel=panel, ylabel=ylabel)
            .groupby(["model", "mode_label", "panel", "ylabel"], as_index=False)["value"]
            .mean()
        )
    rows = _concat_frames(panels)
    if rows.empty:
        return rows
    rows["thinking"] = thinking_labels(rows["model"])
    return rows


def _draw_hypothesis_condition_bars(ax, rows: pd.DataFrame, ylabel: str) -> None:
    """Draw Eval/Generate bars with thinking variants shown by transparency.

    Special case of _draw_paired_mode_bars over the hypothesis-mode category,
    with no hline and the generic `value` metric.
    """
    _draw_paired_mode_bars(
        ax, rows, "mode_label", HYPOTHESIS_CONDITION_MODE_ORDER,
        HYPOTHESIS_SUMMARY_PALETTE, "value", ylabel,
    )


def plot_hypothesis_condition_summary(
    top1_rows: pd.DataFrame,
    mass_rows: pd.DataFrame,
    output_path: str,
    human_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Plot paired non-thinking/thinking Eval/Generate hypothesis summaries."""
    rows = _concat_frames([hypothesis_condition_mode_rows(top1_rows), rule_auc_mode_rows(mass_rows, CONDITION_RULE_AUC_K)])
    if rows.empty:
        return rows

    panel_order = ("accuracy", "confidence", "rule_auc", "size", "rule_pref")
    labels = rows.drop_duplicates("panel").set_index("panel")["ylabel"].to_dict()
    panel_metrics = {panel_key: metric for panel_key, metric, _ylabel in HYPOTHESIS_CONDITION_TOP1_PANELS}
    set_mpl(size=8)
    fig, axes = plt.subplots(1, len(panel_order), figsize=(8.2, 2), sharex=True, sharey=True)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([0, 0.5, 1])
    for ax, panel in zip(axes, panel_order):
        sub = rows[rows["panel"] == panel]
        if sub.empty:
            ax.axis("off")
            continue
        _draw_hypothesis_condition_bars(ax, sub, labels[panel])
        ax.tick_params(axis="y", labelleft=True)  # shared y axis, but every panel keeps its tick labels
        if not human_rows.empty:
            if panel == "rule_auc":
                ax.axhline(human_rule_auc(human_rows), color="black", ls=":", lw=0.9)
            else:
                human = human_description_metric_by_n_examples(human_rows, panel_metrics[panel])
                if not human.empty:
                    ax.axhline(float(human["value"].mean()), color="black", ls=":", lw=0.9)

    fig.subplots_adjust(left=0.045, right=0.855, bottom=0.16, wspace=0.6)
    mode_handles = category_patch_handles(HYPOTHESIS_SUMMARY_PALETTE, HYPOTHESIS_CONDITION_MODE_ORDER)
    thinking_handles = _mode_legend_handles()
    if not human_rows.empty:
        thinking_handles.append(Line2D([0], [0], color="black", ls=":", lw=1.0, label="Human"))
    fig.add_artist(add_external_fig_legend(fig, mode_handles, (0.87, 0.62)))
    add_external_fig_legend(fig, thinking_handles, (0.87, 0.28))
    save_fig(fig, output_path, tight_layout=False)
    return rows


def rule_auc(stimuli: pd.DataFrame) -> float:
    """Criterion-free rule discrimination: P(rating of a rule stimulus > that of a similarity stimulus).

    Mann-Whitney AUC over (task, stimulus_kind, rating) rows with a finite rating; ties count
    one half, so 0.5 is chance. Only same-task pairs are compared: Eval lists a different
    candidate set per task, and the two kinds come from the tasks in different proportions.
    NaN when no task has a rated stimulus of both kinds (e.g. every top-ranked label is other).
    """
    u = pairs = 0.0
    for _task, rows in stimuli.dropna(subset=["rating"]).groupby("task"):
        on_rule = rows.loc[rows["stimulus_kind"] == "rule", "rating"]
        on_similarity = rows.loc[rows["stimulus_kind"] == "similarity", "rating"]
        if on_rule.empty or on_similarity.empty:
            continue
        u += mannwhitneyu(on_rule, on_similarity).statistic
        pairs += len(on_rule) * len(on_similarity)
    return float(u / pairs) if pairs else float("nan")


def rule_auc_mode_rows(mass_rows: pd.DataFrame, k: int) -> pd.DataFrame:
    """Model-unit Eval/Generate rule AUC rows rating stimuli by top-k rule share.

    Scored on each stimulus's full example set: stimulus kind describes the full set,
    and earlier prefixes of every Bigelow similarity stimulus are a single number.
    """
    if mass_rows.empty:
        return pd.DataFrame()
    work = mass_rows[
        (mass_rows["sampling"] == "default")
        & mass_rows["eval_mode"].isin(HYPOTHESIS_MODES)
        & mass_rows["stimulus_kind"].isin(["rule", "similarity"])
    ]
    full_set = work["n_examples"] == work.groupby(["model", "task", "stimulus", "eval_mode"])["n_examples"].transform("max")
    rows = (
        work[full_set]
        .assign(rating=work[f"rule_share_top{k}"])
        .groupby(["model", "eval_mode"])
        .apply(rule_auc, include_groups=False)
        .rename("value")
        .reset_index()
    )
    return rows.assign(
        mode_label=rows["eval_mode"].map(EVAL_MODE_LABELS), panel="rule_auc", ylabel="Rule AUC", k=k,
        thinking=thinking_labels(rows["model"]),
    ).drop(columns="eval_mode")


def human_rule_auc(human_rows: pd.DataFrame) -> float:
    """Human rule AUC over Bigelow16 sets, rating each set by rule / (rule + interval) descriptions."""
    sets = human_rows.groupby("set", as_index=False)["type"].agg(
        rating=lambda types: types.eq("rule").sum() / types.isin(["rule", "interval"]).sum(),
    )
    sets["stimulus_kind"] = sets["set"].map(lambda s: classify_bigelow_kind(bigelow_stimulus(s)[1]))
    return rule_auc(sets.assign(task="Bigelow16"))


def plot_rule_auc_top_k_summary(mass_rows: pd.DataFrame, output_path: str, human_rows: pd.DataFrame) -> pd.DataFrame:
    """Plot paired Eval/Generate rule AUC for each top-k rule-share cutoff."""
    rows = _concat_frames([rule_auc_mode_rows(mass_rows, k) for k in RULE_SHARE_TOP_K])
    if rows.empty:
        return rows
    set_mpl(size=8)
    fig, axes = plt.subplots(1, len(RULE_SHARE_TOP_K), figsize=(5.2, 2), sharex=True, sharey=True)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([0, 0.5, 1])
    for ax, k in zip(axes, RULE_SHARE_TOP_K):
        _draw_hypothesis_condition_bars(ax, rows[rows["k"] == k], "Rule AUC" if ax is axes[0] else "")
        ax.set_title(f"Top-{k}")
        if not human_rows.empty:
            ax.axhline(human_rule_auc(human_rows), color="black", ls=":", lw=0.9)

    fig.subplots_adjust(left=0.09, right=0.78, bottom=0.16, top=0.86, wspace=0.25)
    thinking_handles = _mode_legend_handles()
    if not human_rows.empty:
        thinking_handles.append(Line2D([0], [0], color="black", ls=":", lw=1.0, label="Human"))
    fig.add_artist(add_external_fig_legend(fig, category_patch_handles(HYPOTHESIS_SUMMARY_PALETTE, HYPOTHESIS_CONDITION_MODE_ORDER), (0.80, 0.58)))
    add_external_fig_legend(fig, thinking_handles, (0.80, 0.30))
    save_fig(fig, output_path, tight_layout=False)
    return rows


def hypothesis_accuracy_size_points(
    rows: pd.DataFrame,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    domain: int = BASE_DOMAIN,
) -> pd.DataFrame:
    """Average default top-1 hypothesis size and evidence accuracy across task rows."""
    if rows.empty:
        return pd.DataFrame()
    work = rows[
        (rows["domain"] == domain)
        & (rows["sampling"] == "default")
        & (rows["eval_mode"].isin(HYPOTHESIS_MODES))
        & (rows["n_examples"].isin(PREFIX_LENGTHS))
    ].dropna(subset=["top1_hypothesis_size", "top1_accuracy"])
    if work.empty:
        return pd.DataFrame()
    points = (
        work.groupby(["model", "eval_mode"], as_index=False)
        .agg(
            hypothesis_size=("top1_hypothesis_size", "mean"),
            accuracy=("top1_accuracy", "mean"),
            n_rows=("top1_accuracy", "size"),
        )
        .assign(domain=domain)
    )
    points["_mode_order"] = points["eval_mode"].map({mode: i for i, mode in enumerate(HYPOTHESIS_MODES)})
    return _sort_by_model_order(points.sort_values("_mode_order"), model_order).sort_values(
        "_mode_order", kind="stable",
    ).drop(columns="_mode_order").reset_index(drop=True)


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


def _scatter_models_with_overlap_rings(ax, xs, ys, colors, model_point_style, *, min_separation: float = 0.04) -> None:
    """Scatter model points; a point within `min_separation` (axis units) of an earlier one draws as a ring.

    Both points keep their true coordinates; the ring makes the otherwise hidden point visible.
    """
    drawn: list[tuple[float, float]] = []
    for x, y, color in zip(xs, ys, colors):
        if any(abs(x - px) < min_separation and abs(y - py) < min_separation for px, py in drawn):
            ax.scatter([x], [y], s=model_point_style["s"] * 2.4, facecolors="none", edgecolors=color,
                       linewidths=1.3, alpha=model_point_style.get("alpha", 0.95), zorder=4)
        else:
            ax.scatter([x], [y], color=color, **model_point_style)
        drawn.append((float(x), float(y)))


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
    figsize: tuple[float, float] = (2.15, 2.0),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
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


def plot_hypothesis_accuracy_size_scatter(
    points: pd.DataFrame,
    output_dir: str,
    model_order: tuple[str, ...] = SUMMARY_MODEL_ORDER,
    filename_suffix: str = "",
    human_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write eval and generation scatter plots of accuracy vs top-1 hypothesis size."""
    plot_points = points.copy()
    if human_rows is not None and not human_rows.empty:
        plot_points = pd.concat([plot_points, human_description_accuracy_size_point(human_rows)], ignore_index=True)

    mode_specs = (
        ("hypothesis_eval", f"hypothesis_eval_accuracy_size_scatter{filename_suffix}.png"),
        ("hypothesis_generation", f"hypothesis_generation_accuracy_size_scatter{filename_suffix}.png"),
    )
    set_mpl(size=8)
    for mode, filename in mode_specs:
        sub = plot_points[plot_points["eval_mode"] == mode]
        if sub.empty:
            continue

        def draw_points(ax, plot_points, _ordered, color_map, model_point_style):
            xs, ys, point_colors = [], [], []
            for row in plot_points.itertuples(index=False):
                if row.model == "Human":
                    ax.scatter(row.accuracy, row.hypothesis_size, s=38, marker=HUMAN_MARKER,
                               color="black", linewidths=1.0, alpha=0.9)
                else:
                    xs.append(row.accuracy)
                    ys.append(row.hypothesis_size)
                    point_colors.append(color_map.get(row.model, PALETTE["purple"]))
            _scatter_models_with_overlap_rings(ax, xs, ys, point_colors, model_point_style, min_separation=0.02)

        _plot_scatter_with_external_legend(
            sub,
            os.path.join(output_dir, filename),
            "accuracy",
            "hypothesis_size",
            "Top-1 evidence accuracy",
            "Top-1 hypothesis size",
            model_order,
            draw_points,
            xlim=(-0.02, 1.02),
            ylim=(0.0, None),
            xticks=[0.0, 0.5, 1.0],
            yticks=[0.0, 0.3, 0.6],
        )
    return plot_points










def _plot_hypothesis_summary_domain(
    results_dir: str,
    output_dir: str,
    model_order: tuple[str, ...],
    domain: int,
    model_selector,
    filename_suffix: str = "",
    full: bool = True,
) -> dict[str, pd.DataFrame]:
    """Draw domain-local hypothesis figures for one model cohort."""
    os.makedirs(output_dir, exist_ok=True)
    human_rows = load_human_description_top1_rows(results_dir, domain)
    tables = {"hypothesis_accuracy_size": []}

    scatter_rows = load_hypothesis_top1_metric_rows(
        results_dir, model_selector(results_dir, HYPOTHESIS_TOP1_METRICS_CSV, domain), domain,
    )
    points = hypothesis_accuracy_size_points(scatter_rows, model_order=model_order, domain=domain)
    if full and not points.empty:
        tables["hypothesis_accuracy_size"].append(plot_hypothesis_accuracy_size_scatter(
            points, output_dir, model_order, filename_suffix=filename_suffix, human_rows=human_rows,
        ))
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
    full: bool,
) -> dict[str, pd.DataFrame]:
    """Build default-implicit four-prefix summary plots for one domain."""
    for directory in (output_dir, thinking_output_dir, cache_dir):
        os.makedirs(directory, exist_ok=True)

    thinking_order = thinking_model_order(model_order)

    def base_models(root, csv_name, dom):
        return available_summary_models(root, model_order, csv_name, dom)

    def paired(root, csv_name, dom):
        return available_paired_thinking_models(root, thinking_order, csv_name, dom)

    # Cohorts: (model selector, model order, output dir, filename suffix) — full
    # summary view first, then paired non-thinking/thinking pairs.
    summary_cohorts = (
        (base_models, model_order, output_dir, ""),
        (paired, thinking_order, thinking_output_dir, THINKING_SUFFIX),
    )
    tables = {key: [] for key in (
        "stimuli", "models", "divergence", "human_divergence", "fitting_performance", "human_kl", "human_correlation_kind", "prompt_prefix",
        "condition_task", "hypothesis_condition", "rule_auc_top_k", "hypothesis_top1", "hypothesis_beta", "hypothesis_entropy",
    )}

    # The root-cohort Human size-principle endpoint, reused by the eval-mode size-principle figure.
    human_beta_point = None
    for spec in SUMMARY_METRICS:
        for model_selector, cohort_order, cohort_output_dir, filename_suffix in summary_cohorts:
            # Root kl.png also draws each available thinking variant, dashed.
            kl_root = spec.name == "kl_bayes" and not filename_suffix
            models = available_summary_models(results_dir, thinking_order, spec.csv_name, domain) if kl_root                 else model_selector(results_dir, spec.csv_name, domain)
            raw = load_metric_rows(results_dir, spec, models, domain)
            metric_rows = complete_prefix_rows(raw, spec.name, sampling=sampling, h_space=h_space, domain=domain) \
                if not raw.empty else pd.DataFrame()
            if metric_rows.empty:
                continue
            # Text-trial rows carry `<metric>_ceiling`; logits rows leave it NaN.
            ceiling_col = f"{spec.name}_ceiling"
            ceiling_rows = None
            if ceiling_col in raw and raw[ceiling_col].notna().any():
                text_rows = raw[raw[ceiling_col].notna()]
                ceiling_rows = complete_prefix_rows(
                    text_rows.assign(**{spec.name: text_rows[ceiling_col]}),
                    spec.name, sampling=sampling, h_space=h_space, domain=domain,
                )

            plot_path = os.path.join(cohort_output_dir, suffix_filename(spec.filename, filename_suffix))
            if filename_suffix:
                if spec.name in SUMMARY_MEAN_TRAJECTORY_METRICS:
                    plot_thinking_mean_function(metric_rows, spec.name, spec.ylabel, plot_path,
                                                thinking_order)
                else:
                    plot_function(metric_rows, spec.name, spec.ylabel, plot_path, cohort_order,
                                  ceiling_rows=ceiling_rows)
                if spec.name == "beta":
                    plot_thinking_mean_function(metric_rows, spec.name, spec.ylabel,
                                                os.path.join(output_dir, "beta_thinking_mean.png"),
                                                thinking_order, show_models=False,
                                                human_point=human_summary_point(metric_rows, spec.name, results_dir, domain=domain))
                continue

            if spec.name == "beta":
                size_raw = load_metric_rows(results_dir, spec, available_summary_models(
                    results_dir, SIZE_MODEL_ORDER, spec.csv_name, domain), domain)
                if not size_raw.empty:
                    size_rows = complete_prefix_rows(size_raw, spec.name, sampling=sampling, h_space=h_space, domain=domain)
                    plot_size_mean_function(
                        size_rows, spec.name, spec.ylabel, os.path.join(output_dir, suffix_filename(spec.filename, "_size")),
                        human_point=human_summary_point(size_rows, spec.name, results_dir, domain=domain),
                    )
            tables["stimuli"].append(metric_rows.rename(columns={spec.name: "value"}).assign(metric=spec.name))
            human_point = human_summary_point(metric_rows, spec.name, results_dir, domain=domain)
            if spec.name == "beta":
                human_beta_point = human_point
            model_df = plot_function(
                metric_rows, spec.name, spec.ylabel, plot_path, thinking_order if kl_root else cohort_order,
                human_point=human_point,
                show_model_mean=spec.name in SUMMARY_MEAN_TRAJECTORY_METRICS, ceiling_rows=ceiling_rows,
            )
            tables["models"].append(model_df.rename(columns={spec.name: "value"}).assign(metric=spec.name))
            prompt_prefix_rows = condition_prefix_model_rows(raw, spec.name, model_order)
            if not prompt_prefix_rows.empty:
                tables["prompt_prefix"].append(plot_condition_trajectory(
                    prompt_prefix_rows, spec.name, spec.ylabel,
                    os.path.join(output_dir, f"condition_{spec.filename}"), human_point,
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

    thinking_alpha_beta_rows = load_alpha_beta_rows(results_dir, paired(results_dir, "alpha_beta_fits.csv", domain), domain)
    thinking_entropy_rows = load_metric_rows(
        results_dir,
        SummaryMetric("entropy", "trajectory_prefix_metrics.csv", "Posterior entropy (nats)", "entropy.png"),
        paired(results_dir, "trajectory_prefix_metrics.csv", domain),
        domain,
    )
    if not thinking_entropy_rows.empty:
        thinking_entropy_rows = thinking_entropy_rows[thinking_entropy_rows["n_examples"] == max(PREFIX_LENGTHS)]
    human_hlines = {metric: human_condition_hline(results_dir, metric, domain) for metric in ("beta", "kl_bayes")}
    human_entropy = human_condition_hline(results_dir, "entropy", domain, stimulus_rows=thinking_entropy_rows)
    condition_panel_rows = plot_condition_mode_summary(
        (
            {"name": "beta", "rows": thinking_alpha_beta_rows, "metric": "beta", "ylabel": "Size principle (β)", "fit_scope": FULL_SCOPE, "hline": human_hlines["beta"]},
            {"name": "kl", "rows": _load_posterior_divergence_rows(results_dir, paired(results_dir, "divergence_metrics.csv", domain), "divergence_metrics.csv", domain),
             "metric": "kl_bayes", "ylabel": "KL with Bayesian", "hline": human_hlines["kl_bayes"]},
            {"name": "entropy", "rows": thinking_entropy_rows, "metric": "entropy", "ylabel": "Posterior entropy (nats)",
             "hline": human_entropy},
        ),
        os.path.join(output_dir, "condition.png"),
        thinking_order,
    )
    if "kl" in condition_panel_rows:
        tables["divergence"].append(condition_panel_rows["kl"].assign(figure="condition"))

    eval_mode_models = paired(results_dir, HYPOTHESIS_ALPHA_BETA_CSV, domain)
    eval_mode_alpha_beta_rows = load_eval_mode_alpha_beta_rows(results_dir, eval_mode_models, domain)
    eval_mode_entropy_rows = load_eval_mode_predictive_entropy_rows(results_dir, eval_mode_models, domain)
    eval_mode_panel_rows = plot_eval_mode_summary(
        (
            {"name": "beta", "rows": eval_mode_alpha_beta_rows, "metric": "beta", "ylabel": "Size principle (β)", "hline": human_hlines["beta"]},
            {"name": "kl", "rows": load_eval_mode_kl_rows(results_dir, eval_mode_models, domain), "metric": "kl_bayes", "ylabel": "KL with Bayesian",
             "hline": human_hlines["kl_bayes"]},
            {"name": "entropy", "rows": eval_mode_entropy_rows[eval_mode_entropy_rows["n_examples"] == max(PREFIX_LENGTHS)] if not eval_mode_entropy_rows.empty else eval_mode_entropy_rows,
             "metric": "predictive_entropy", "ylabel": "Predictive entropy (nats)", "hline": human_entropy},
            {"name": "extension_mass", "rows": load_eval_mode_extension_mass_rows(results_dir, thinking_order, paired=True),
             "metric": "extension_mass", "ylabel": "Prob. mass on 101–200"},
        ),
        os.path.join(output_dir, "condition_task.png"),
        thinking_order,
    )
    for name, rows in eval_mode_panel_rows.items():
        tables["condition_task"].append(rows.assign(metric=name))

    hypothesis_condition_top1_rows = load_hypothesis_top1_rows(results_dir, paired(results_dir, HYPOTHESIS_TOP1_CSV, domain), domain)
    human_top1_rows = load_human_description_top1_rows(results_dir, domain)
    mass_rows = _model_csv_rows(results_dir, paired(results_dir, HYPOTHESIS_ENTROPY_METRICS_CSV, domain), HYPOTHESIS_ENTROPY_METRICS_CSV, domain)
    tables["hypothesis_condition"].append(plot_hypothesis_condition_summary(
        hypothesis_condition_top1_rows, mass_rows, os.path.join(output_dir, "condition_eval_gen.png"), human_top1_rows,
    ))
    tables["rule_auc_top_k"].append(plot_rule_auc_top_k_summary(
        mass_rows, os.path.join(output_dir, "condition_rule_auc_top_k.png"), human_top1_rows,
    ))
    for metric, filename, ylabel in HYPOTHESIS_TOP1_METRICS:
        tables["hypothesis_top1"].append(plot_hypothesis_top1_summary(
            hypothesis_condition_top1_rows, metric, ylabel, os.path.join(output_dir, filename), human_top1_rows,
        ))
    hypothesis_beta_rows = load_hypothesis_beta_rows(results_dir, paired(results_dir, HYPOTHESIS_ALPHA_BETA_CSV, domain), domain)
    tables["hypothesis_beta"].append(plot_hypothesis_beta_summary(
        hypothesis_beta_rows, os.path.join(output_dir, "hypothesis_beta.png"), human_beta_point,
    ))
    plot_hypothesis_beta_summary(
        hypothesis_beta_rows, os.path.join(output_dir, "beta_prompts.png"), human_beta_point, {NON_THINKING: "-"},
    )
    tables["hypothesis_entropy"].append(plot_hypothesis_entropy_summary(
        eval_mode_entropy_rows, os.path.join(output_dir, "hypothesis_entropy_trajectory.png"), human_entropy,
    ))

    alpha_beta_models = available_summary_models(results_dir, model_order, "alpha_beta_fits.csv", domain)
    alpha_beta_rows = load_alpha_beta_rows(results_dir, alpha_beta_models, domain)
    fitting_points = model_metric_rows(alpha_beta_rows, "r2", model_order, sampling, h_space, fit_scope=FULL_SCOPE)
    human_rows = _load_posterior_divergence_rows(results_dir, base_models(results_dir, HUMAN_DIVERGENCE_CSV, domain),
                                                 HUMAN_DIVERGENCE_CSV, domain)
    if full and not fitting_points.empty and not human_rows.empty:
        human_kl_points = model_metric_rows(human_rows, "kl_human", model_order, sampling, h_space)
        fitting_table, human_kl_table = plot_model_metric_bars(
            [(fitting_points, "r2", r"$r^2$"), (human_kl_points, "kl_human", "KL(human || LLM)")],
            os.path.join(output_dir, "fitting_human_kl.png"), model_order,
        )
        tables["fitting_performance"].append(fitting_table)
        tables["human_kl"].append(human_kl_table)
    if full and not human_rows.empty:
        tables["human_correlation_kind"].append(plot_human_correlation_by_kind(
            human_rows, os.path.join(output_dir, "human_correlation_by_kind.png"), model_order, sampling, h_space,
        ))

    hypothesis_tables = _plot_hypothesis_summary_domain(results_dir, output_dir, model_order, domain, base_models, full=full)
    _plot_hypothesis_summary_domain(
        results_dir, thinking_output_dir, thinking_order, domain, paired,
        filename_suffix=THINKING_SUFFIX,
    )

    summary_tables = {**{key: _concat_frames(frames) for key, frames in tables.items()}, **hypothesis_tables}
    for table_key, filename in (
        ("stimuli", "summary_default_implicit_stimulus_metrics.csv"),
        ("models", "summary_default_implicit_model_metrics.csv"),
        ("divergence", "summary_condition_kl_by_model.csv"),
        ("human_divergence", "summary_human_condition_kl_by_model.csv"),
        ("fitting_performance", "summary_fitting_performance_by_model.csv"),
        ("human_kl", "summary_human_kl_by_model.csv"),
        ("human_correlation_kind", "summary_human_correlation_by_kind.csv"),
        ("prompt_prefix", "summary_condition_prefix_by_model.csv"),
        ("condition_task", "summary_condition_task_by_model.csv"),
        ("hypothesis_condition", "summary_hypothesis_condition_by_model.csv"),
        ("rule_auc_top_k", "summary_rule_auc_top_k_by_model.csv"),
        ("hypothesis_top1", "summary_hypothesis_top1_by_model.csv"),
        ("hypothesis_beta", "summary_hypothesis_beta_by_model.csv"),
        ("hypothesis_entropy", "summary_hypothesis_entropy_by_model.csv"),
        ("hypothesis_accuracy_size", "summary_hypothesis_accuracy_size_scatter.csv"),
    ):
        save_csv(summary_tables[table_key], cache_dir, domain_file_prefix(domain) + filename)
    return summary_tables


def _summary_thinking_output_dir(output_dir: str, domain: int) -> str:
    return os.path.join(output_dir, f"d{domain}{THINKING_SUFFIX}")


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
    full: bool = True,
) -> None:
    """Posterior-space d200 rule-discrimination figures."""
    discrimination_models = available_extension_models(results_dir, "rule_discrimination", "posterior", h_space, sampling, model_order, paired=paired)
    instance_rows = load_rule_discrimination_rows(results_dir, discrimination_models, h_space, sampling)
    discrimination = rule_discrimination_points(instance_rows)
    save_csv(discrimination, cache_dir, suffix_filename("summary_rule_discrimination.csv", filename_suffix))
    if full:
        plot_rule_discrimination_summary(
            discrimination, os.path.join(output_dir, suffix_filename("rule_discrimination.png", filename_suffix)), model_order, instance_rows,
        )


def run_summary(results_dir: str = RESULTS_ROOT, output_dir: str | None = None) -> None:
    """Build the four-prefix summary for each SUMMARY_COHORTS cohort."""
    output_dir = output_dir or os.path.join(results_dir, "summary")
    for subdir, model_order, full in SUMMARY_COHORTS:
        _run_summary_cohort(results_dir, os.path.join(output_dir, subdir), model_order, full)


def _run_summary_cohort(
    results_dir: str,
    output_dir: str,
    model_order: tuple[str, ...],
    full: bool,
    sampling: str = "default",
    h_space: str = "implicit",
    domains: tuple[int, ...] = DEFAULT_SUMMARY_DOMAINS,
) -> None:
    """Build the root-domain summary in output_dir and sibling domain summaries under d{domain}/."""
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    for domain in domains:
        _run_summary_domain(
            results_dir, domain_subdir(output_dir, domain), _summary_thinking_output_dir(output_dir, domain),
            cache_dir, model_order, sampling, h_space, domain, full,
        )

    _plot_posterior_extension_diagnostics(results_dir, output_dir, cache_dir, model_order, sampling, h_space, full=full)
    thinking_order = thinking_model_order(model_order)
    mass_rows = load_eval_mode_extension_mass_rows(results_dir, thinking_order, paired=True)
    bayesian_mass_rows = _read_model_csv(results_dir, BAYESIAN, "extension_mass.csv")
    save_csv(plot_extension_mass_summary(
        mass_rows, bayesian_mass_rows, os.path.join(output_dir, "extension_mass.png"),
    ), cache_dir, "summary_extension_mass.csv")
    decline = extension_decline_points(pd.concat([mass_rows, bayesian_mass_rows.assign(model=BAYESIAN, mode_label=BAYESIAN)]))
    save_csv(decline, cache_dir, "summary_extension_decline.csv")
    for metric, label in (("drop", "mass drop"), ("slope", "slope k")):
        plot_extension_decline_bars(decline, metric, label, os.path.join(output_dir, f"extension_decline_{metric}.png"), model_order)
    thinking_output_dir = _summary_thinking_output_dir(output_dir, EXTENSION_DOMAIN)
    os.makedirs(thinking_output_dir, exist_ok=True)
    _plot_posterior_extension_diagnostics(
        results_dir, thinking_output_dir, cache_dir, thinking_order, sampling, h_space,
        paired=True, filename_suffix=THINKING_SUFFIX,
    )


def main():
    parser = argparse.ArgumentParser(description="Cross-model default four-prefix summary plots.")
    parser.add_argument("--results-dir", default=RESULTS_ROOT)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run_summary(results_dir=args.results_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
