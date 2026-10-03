"""Prompt manipulation and hypothesis-type plots for per-model aggregate stages."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.alpha_beta_scope import PREFIX_LENGTHS
from analysis.plotter import (
    set_mpl, save_fig, with_model, filter_canonical_prompts,
    plot_trajectory_sem, plot_prompt_kind_bar, task_domain_series,
    C_BAYESIAN, C_PRIOR, C_LIKELIHOOD, EVAL_MODE_LABELS, HYP_TYPE_COLORS,
    PROMPT_PALETTE,
)

set_mpl(size=8)


def plot_prompt_sensitivity(df: pd.DataFrame, output_path: str):
    """Pooled prompt-level KL from the Bayesian posterior."""
    prompt_df = df[
        df["h_space"].isin(["explicit", "implicit"])
        & df["sampling"].isin(["default", "strong", "weak"])
    ].copy()
    if prompt_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.2), sharey=True)
    sns.barplot(
        data=prompt_df, x="h_space", y="kl_bayes", hue="h_space",
        order=["explicit", "implicit"],
        palette={"explicit": PROMPT_PALETTE["strong explicit"],
                 "implicit": PROMPT_PALETTE["strong implicit"]},
        errorbar="se", capsize=0.1, legend=False, ax=axes[0],
    )
    sns.barplot(
        data=prompt_df, x="sampling", y="kl_bayes", hue="sampling",
        order=["default", "strong", "weak"],
        palette={"default": C_BAYESIAN, "strong": C_LIKELIHOOD, "weak": C_PRIOR},
        errorbar="se", capsize=0.1, legend=False, ax=axes[1],
    )

    axes[0].set_title("h_space")
    axes[1].set_title("sampling")
    axes[0].set_ylabel(r"KL(Bayesian $\Vert$ LLM)")
    axes[1].set_ylabel("")
    for ax in axes:
        ax.set_xlabel("")
    fig.suptitle("KL with Bayesian by prompt", fontsize=11)
    save_fig(fig, output_path)


def plot_generalization_width(
    df: pd.DataFrame,
    output_path: str,
    result_key: str = "",
    baseline_overlay: dict[str, pd.DataFrame] | None = None,
):
    """Effective support grouped by canonical prompt and stimulus kind."""
    plot_prompt_kind_bar(
        df, "effective_support", "Number of values with P(yes) > 0.5",
        "Posterior Generalization Width", output_path, result_key,
        baseline_overlay=baseline_overlay)


def plot_confidence_weighted_types(df: pd.DataFrame, output_path: str, result_key: str, title: str):
    """Grouped stacked bars: confidence-weighted type proportions by sampling and prefix length.

    df must have columns: sampling, n_examples, cw_rule, cw_interval, cw_other.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    samplings = sorted(df["sampling"].unique())
    legend_handles = []
    legend_labels = []

    for panel_idx, (ax, n_examples) in enumerate(zip(axes.flat, PREFIX_LENGTHS)):
        agg = (
            df[df["n_examples"] == n_examples]
            .groupby("sampling")[["cw_rule", "cw_interval", "cw_other"]].mean()
            .reindex(samplings)
            .fillna(0.0)
        )
        x = np.arange(len(samplings))
        bottom = np.zeros(len(samplings))
        for col, label in [("cw_rule", "rule"), ("cw_interval", "interval"), ("cw_other", "other")]:
            vals = agg[col].values
            bars = ax.bar(x, vals, bottom=bottom, color=HYP_TYPE_COLORS[label], label=label, width=0.6)
            if panel_idx == 0:
                legend_handles.append(bars[0])
                legend_labels.append(label)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(samplings, fontsize=7, rotation=20, ha="right")
        ax.set_title(f"n={n_examples}")
        ax.set_ylabel("Confidence-weighted proportion" if panel_idx % 2 == 0 else "")
        ax.set_ylim(0, 1.05)

    fig.suptitle(with_model(title, result_key), fontsize=11, y=0.98)
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_fig(fig, output_path, tight_layout=False)


def plot_trajectory_rule_by_kind(df: pd.DataFrame, output_path: str):
    """All-stimulus rule-mass trajectory, lines = eval/generation x strong/weak."""
    df = filter_canonical_prompts(df).copy()
    if df.empty:
        return

    df["task_domain"] = df["task"] + "/d" + df["domain"].astype(str)
    series = []
    for eval_mode, h_space, fmt in [("hypothesis_eval", "explicit", "-s"), ("hypothesis_generation", "implicit", "--o")]:
        for sampling in ("strong", "weak"):
            sub = df[(df["eval_mode"] == eval_mode) & (df["sampling"] == sampling)].dropna(subset=["cw_rule"])
            if sub.empty:
                continue
            xs, ys, sems = task_domain_series(sub, "cw_rule")
            series.append({
                "x": xs, "y": ys, "y_err": sems, "label": f"{EVAL_MODE_LABELS[eval_mode]} {sampling}",
                "color": PROMPT_PALETTE[f"{sampling} {h_space}"], "fmt": fmt,
            })

    if series:
        plot_trajectory_sem(series, output_path, ylabel="Rule mass", figsize=(3, 2.5), xticks=PREFIX_LENGTHS)
