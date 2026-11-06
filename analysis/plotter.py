"""Matplotlib configuration, shared palette, and reusable output helpers."""

import os
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.alpha_beta_scope import FIT_SCOPE_ORDER
from configs.settings import BAYESIAN, HUMAN, MODELS, PLOT_MODEL_FIGURES, RESULTS_ROOT
from utils import resolve_path

PLOT_PARAMS = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.1}

# Core palette (seaborn "deep" — stable across all plots).
PALETTE = {"blue": "#4C72B0", "orange": "#DD8452", "green": "#55A868",
           "red": "#C44E52", "purple": "#8172B3"}
CYCLE = list(PALETTE.values())

# Semantic aliases — aliases share hex on purpose (visual vocabulary across roles:
# blue=primary/reference, orange=contrast, green=agreement, red=likelihood, purple=accent).
C_BAYESIAN = C_RULE = PALETTE["blue"]
C_LLM      = C_INTERVAL = C_BETA = PALETTE["orange"]
C_HUMAN    = C_PRIOR = PALETTE["green"]
C_LIKELIHOOD = PALETTE["red"]
C_OTHER    = C_BESTFIT = C_ACCENT = PALETTE["purple"]

HYP_TYPE_COLORS = {"rule": C_RULE, "interval": C_INTERVAL, "other": C_OTHER}
SOURCE_COLORS = {"bayesian": C_BAYESIAN, "llm": C_LLM, "human": C_HUMAN}
BASELINE_OVERLAY_STYLES = {
    "Bayesian": {"color": C_BAYESIAN, "ls": "--", "lw": 1.0},
    "Human": {"color": C_HUMAN, "ls": "-.", "lw": 1.0},
}


MODEL_COLORS = {info.display: info.summary_color for info in MODELS.values() if info.summary_color}
SUMMARY_MODEL_ORDER = tuple(MODEL_COLORS)

# Display vocabulary for the three evaluation modes; artifacts keep eval_mode keys.
EVAL_MODE_LABELS = {"posterior": "Predict", "hypothesis_eval": "Eval", "hypothesis_generation": "Generate"}
EVAL_MODE_COLORS = {"posterior": PALETTE["green"], "hypothesis_eval": PALETTE["blue"], "hypothesis_generation": PALETTE["orange"]}

KIND_ORDER = ("rule", "similarity", "single", "other")
KIND_DISPLAY = {"similarity": "sim"}
# Keyed by `{h_space}_{sampling}`, the same prompt key as ExperimentCondition.prompt.
PROMPT_ORDER = ("explicit_strong", "implicit_strong", "explicit_weak", "implicit_weak")
PROMPT_PALETTE = {
    "explicit_strong": C_LIKELIHOOD,
    "implicit_strong": "darkorange",
    "explicit_weak": "navy",
    "implicit_weak": "#4169E1",
}
def ordered_stimulus_kinds(kinds, all_label: str | None = None) -> list[str]:
    """Return stimulus-kind labels in the shared analysis order."""
    present = set(kinds)
    ordered = [k for k in KIND_ORDER if k in present]
    extras = sorted(k for k in present if k not in KIND_ORDER and k != all_label)
    return ([all_label] if all_label is not None and all_label in present else []) + ordered + extras


def with_all_kinds(df):
    """Duplicate rows into a pooled ``all`` stimulus-kind group."""
    all_rows = df.copy()
    all_rows["stimulus_kind"] = "all"
    return pd.concat([all_rows, df], ignore_index=True)


def filter_canonical_prompts(df):
    """Keep the canonical PROMPT_ORDER prompts."""
    return df[prompt_label(df).isin(PROMPT_ORDER)].copy()


def with_task_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Add the `{task}/d{domain}` group used for equal task-domain weighting."""
    return df.assign(task_domain=df["task"] + "/d" + df["domain"].astype(str))


def prompt_label(df: pd.DataFrame) -> pd.Series:
    """Per-row prompt key `{h_space}_{sampling}`, matching ExperimentCondition.prompt."""
    return df["h_space"] + "_" + df["sampling"]


def set_mpl(size=10):
    """Apply publication-quality matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": size,
        "axes.labelsize": size + 1,
        "axes.titlesize": size + 2,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size - 1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.grid": False,
    })


def save_fig(fig, path, *, tight_layout: bool = True, dpi: int | None = None):
    """Save and close a figure, applying tight layout unless layout is manual."""
    if not PLOT_MODEL_FIGURES and _is_model_figure(path):
        plt.close(fig)
        return
    if tight_layout:
        fig.tight_layout()
    save_kwargs = dict(PLOT_PARAMS)
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(path, **save_kwargs)
    plt.close(fig)
    print(f"  Saved: {path}")


def _is_model_figure(path: str) -> bool:
    """True for paths inside an LLM's results/{model}/ tree."""
    root = os.path.abspath(resolve_path(RESULTS_ROOT)) + os.sep
    path = os.path.abspath(path)
    return path.startswith(root) and path[len(root):].split(os.sep)[0] not in {"summary", BAYESIAN, HUMAN}


def save_csv(rows, output_dir: str, name: str) -> pd.DataFrame:
    """Save rows (records or a DataFrame) as CSV, creating the stage directory first."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, name), index=False)
    print(f"  {name}: {len(df)} rows")
    return df


def with_model(stem: str, result_key: str = "") -> str:
    """Append ' — {result_key}' suffix when a display name is given."""
    return f"{stem} — {result_key}" if result_key else stem


def model_colors(models: list[str]) -> dict[str, object]:
    """Shared fixed model colors for cross-model figures."""
    unknown = [model for model in models if model not in MODEL_COLORS]
    fallback = {model: CYCLE[i % len(CYCLE)] for i, model in enumerate(unknown)}
    return {model: MODEL_COLORS.get(model, fallback.get(model)) for model in models}


def all_group_rows(df: pd.DataFrame, hue_col: str | None = None) -> pd.DataFrame:
    """Prefer explicit pooled rows while keeping kind rows without a pooled peer."""
    if "stimulus_kind" not in df:
        return df
    explicit = df[df["stimulus_kind"] == "all"]
    if explicit.empty:
        return df

    unit_candidates = [
        "model", "task", "domain", "h_space", "sampling", "eval_mode",
        "comparison", "n_examples", hue_col,
    ]
    unit_cols = list(dict.fromkeys(c for c in unit_candidates if c and c in df.columns))
    if not unit_cols:
        return explicit

    kind_rows = df[df["stimulus_kind"] != "all"]
    explicit_units = explicit[unit_cols].drop_duplicates().assign(_has_explicit_all=True)
    kind_rows = kind_rows.merge(explicit_units, on=unit_cols, how="left")
    kind_rows = kind_rows[kind_rows["_has_explicit_all"].isna()].drop(columns="_has_explicit_all")
    return pd.concat([explicit, kind_rows], ignore_index=True)


def plot_kind_grouped_bar(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: str,
    hue_col: str,
    hue_order: list[str],
    palette: dict[str, str],
    result_key: str = "",
    axhline: float | None = None,
    legend_title: str = "",
    figsize: tuple[float, float] | None = None,
    baseline_overlay: dict[str, pd.DataFrame] | None = None,
):
    """Barplot pooled across and split by stimulus kind.

    X = [all, rule, sim, single, other] when present. The input must include
    stimulus_kind, ``hue_col``, and ``metric``.
    ``baseline_overlay`` maps display name to baseline rows with
    stimulus_kind plus ``metric``; means are drawn as per-group references.
    """
    if df.empty:
        return

    all_source = all_group_rows(df, hue_col)
    kind_source = df[df["stimulus_kind"] != "all"] if (df["stimulus_kind"] == "all").any() else df
    kind_order = ordered_stimulus_kinds(kind_source["stimulus_kind"]) if not kind_source.empty else []
    plot_df = pd.concat([
        all_source.assign(group="all"),
        kind_source.assign(group=kind_source["stimulus_kind"].replace(KIND_DISPLAY)),
    ], ignore_index=True)
    order = ["all"] + [KIND_DISPLAY.get(kind, kind) for kind in kind_order]

    fig, ax = plt.subplots(figsize=figsize or (max(5.2, 0.9 * len(order)), 3.4))
    sns.barplot(
        data=plot_df, x="group", y=metric, hue=hue_col,
        order=order, hue_order=hue_order, palette=palette,
        errorbar="se", capsize=0.1, ax=ax,
    )
    if axhline is not None:
        ax.axhline(axhline, color="gray", ls=":", lw=0.8)
    if baseline_overlay:
        display_to_raw = {KIND_DISPLAY.get(kind, kind): kind for kind in kind_order}
        for label, baseline_df in baseline_overlay.items():
            if baseline_df is None or baseline_df.empty:
                continue
            style = BASELINE_OVERLAY_STYLES.get(label, {"color": "gray", "ls": ":", "lw": 1.0})
            for i, group in enumerate(order):
                if group == "all":
                    sub = all_group_rows(baseline_df)
                else:
                    sub = baseline_df[baseline_df["stimulus_kind"] == display_to_raw[group]]
                if sub.empty:
                    continue
                ax.hlines(float(sub[metric].mean()), i - 0.4, i + 0.4,
                          label=label if i == 0 else None, **style)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(with_model(title, result_key))
    ax.legend(title=legend_title, fontsize=6, ncol=2, frameon=False)
    save_fig(fig, output_path)


def plot_prompt_kind_bar(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: str,
    result_key: str = "",
    axhline: float | None = None,
    baseline_overlay: dict[str, pd.DataFrame] | None = None,
):
    """Barplot for canonical prompt settings pooled across and split by kind."""
    df = filter_canonical_prompts(df)
    if df.empty:
        return
    df["prompt"] = prompt_label(df)
    plot_kind_grouped_bar(
        df, metric, ylabel, title, output_path,
        hue_col="prompt", hue_order=list(PROMPT_ORDER), palette=PROMPT_PALETTE,
        result_key=result_key, axhline=axhline,
        baseline_overlay=baseline_overlay,
    )


def plot_prompt_fit_scope_bar(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: str,
    result_key: str = "",
    axhline: float | None = None,
    baseline_overlay: dict[str, pd.DataFrame] | None = None,
):
    """Barplot for canonical prompt settings split by alpha-beta fit scope."""
    df = filter_canonical_prompts(df)
    if df.empty:
        return
    df = df.copy()
    df["prompt"] = prompt_label(df)
    order = [scope for scope in FIT_SCOPE_ORDER if scope in set(df["fit_scope"])]
    fig, ax = plt.subplots(figsize=(max(5.2, 0.9 * len(order)), 3.4))
    sns.barplot(
        data=df, x="fit_scope", y=metric, hue="prompt",
        order=order, hue_order=list(PROMPT_ORDER), palette=PROMPT_PALETTE,
        errorbar="se", capsize=0.1, ax=ax,
    )
    if axhline is not None:
        ax.axhline(axhline, color="gray", ls=":", lw=0.8)
    if baseline_overlay:
        for label, baseline_df in baseline_overlay.items():
            if baseline_df is None or baseline_df.empty:
                continue
            style = BASELINE_OVERLAY_STYLES.get(label, {"color": "gray", "ls": ":", "lw": 1.0})
            for i, scope in enumerate(order):
                sub = baseline_df[baseline_df["fit_scope"] == scope]
                if sub.empty:
                    continue
                ax.hlines(float(sub[metric].mean()), i - 0.4, i + 0.4,
                          label=label if i == 0 else None, **style)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(with_model(title, result_key))
    ax.legend(title="", fontsize=6, ncol=2, frameon=False)
    save_fig(fig, output_path)


LINE_ALPHA = 0.9
ERRBAR_ALPHA = 0.25


def _apply_errbar_alpha(container, line_alpha=LINE_ALPHA, err_alpha=ERRBAR_ALPHA):
    """Set distinct alphas for the main line vs error-bar caps/bars."""
    line, caps, bars = container
    line.set_alpha(line_alpha)
    for cap in caps:
        cap.set_alpha(err_alpha)
    for bar in bars:
        bar.set_alpha(err_alpha)


def sem(values) -> float:
    """Sample standard error of the mean (ddof=1); 0.0 for fewer than two observations."""
    vals = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def aggregate_sem(per_stimulus_values: dict[int, list[float]]):
    """{n: [val, ...]} -> (xs, means, sems). sem = std/sqrt(n); 0 if n<2."""
    xs = sorted(per_stimulus_values)
    means, sems = [], []
    for n in xs:
        vals = [v for v in per_stimulus_values[n] if v is not None and not np.isnan(v)]
        if not vals:
            means.append(np.nan)
            sems.append(np.nan)
            continue
        means.append(float(np.mean(vals)))
        sems.append(sem(vals))
    return xs, means, sems


def task_domain_series(df: pd.DataFrame, metric: str, x_col: str = "n_examples"):
    """(xs, means, sems) with equal task_domain weighting: average within each
    task_domain and x first, then across task_domain means."""
    task_domain_means = df.groupby(["task_domain", x_col], as_index=False)[metric].mean()
    by_x = task_domain_means.groupby(x_col)[metric].apply(list).to_dict()
    return aggregate_sem({int(k): v for k, v in by_x.items()})


def plot_trajectory_sem(series, output_path, xlabel="Number of examples", ylabel="",
                        title=None, hline=None, figsize=(2.15, 2),
                        xticks=None, yticks=None, show_errorbar=True, hline_color="gray",
                        legend_fontsize=6, legend_ncol=1,
                        legend_bbox_to_anchor=(1.02, 0.5), legend_groups=None):
    """Line trajectories, optionally with mean ± SEM error bars.

    series: list of {"x", "y", "y_err", "label", "color"}.
    legend_groups: optional handle lists drawn as stacked external legends in place of the series labels.
    Lines draw at LINE_ALPHA; optional error bars draw at ERRBAR_ALPHA.
    External legends are saved outside the fixed main axes by expanding the export bbox.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for s in series:
        series_show_errorbar = s.get("show_errorbar", show_errorbar)
        if series_show_errorbar:
            container = ax.errorbar(
                s["x"], np.asarray(s["y"]), yerr=np.asarray(s["y_err"]),
                fmt=s.get("fmt", "-"), color=s["color"], label="_nolegend_",
                capsize=s.get("capsize", 2), lw=s.get("lw", 1.2),
            )
            container[0].set_label(s.get("label"))
            _apply_errbar_alpha(
                container,
                line_alpha=s.get("alpha", LINE_ALPHA),
                err_alpha=s.get("err_alpha", ERRBAR_ALPHA),
            )
        else:
            ax.plot(s["x"], np.asarray(s["y"]), s.get("fmt", "-"), color=s["color"],
                    label=s.get("label"), lw=s.get("lw", 1.2), alpha=s.get("alpha", LINE_ALPHA),
                    markerfacecolor=s.get("markerfacecolor"), markersize=s.get("ms"),
                    markeredgewidth=s.get("mew"))
    if hline is not None:
        ax.axhline(hline, color=hline_color, ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if title:
        ax.set_title(title)
    has_legend = bool(legend_groups) or any(s.get("label") for s in series)
    if legend_groups:
        x, y = legend_bbox_to_anchor
        # Figure legends, unlike extra axes legends, count toward the tight export bbox.
        # Stack the groups top-down from each legend's measured height, so a long model list
        # never overlaps the line-style group below it.
        renderer = fig.canvas.get_renderer()
        gap = 0.06
        legends = [
            fig.legend(handles=handles, fontsize=legend_fontsize, loc="upper left",
                       bbox_transform=ax.transAxes, frameon=False, bbox_to_anchor=(x, 1.0))
            for handles in legend_groups
        ]
        heights = [ax.transAxes.inverted().transform_bbox(leg.get_window_extent(renderer)).height for leg in legends]
        top = y + (sum(heights) + gap * (len(legends) - 1)) / 2.0
        for leg, height in zip(legends, heights):
            leg.set_bbox_to_anchor((x, top), transform=ax.transAxes)
            top -= height + gap
    elif has_legend:
        ax.legend(
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            loc="center left",
            bbox_to_anchor=legend_bbox_to_anchor,
            frameon=False,
        )
    save_fig(fig, output_path, tight_layout=not has_legend)
