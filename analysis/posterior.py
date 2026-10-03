"""Posterior visualization for condition-level and cross-condition analysis."""

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from analysis.plotter import (
    set_mpl, save_fig, aggregate_sem, plot_trajectory_sem, with_model,
    C_BAYESIAN, C_LLM, C_HUMAN, C_ACCENT, C_PRIOR, C_LIKELIHOOD, C_BESTFIT,
    HYP_TYPE_COLORS, SOURCE_COLORS,
)
from utils import slugify_stimulus

set_mpl(size=8)

# Per-stimulus figures are drawn only for small stimulus lists; larger tasks
# (Bigelow16's 255 sets) use their curated `featured_stimuli` or skip them.
MAX_PER_STIMULUS_PLOTS = 8


def per_stimulus_plot_names(stimulus_names: list[str], featured: list[str] | None) -> list[str]:
    names = featured if featured is not None else stimulus_names
    return names if len(names) <= MAX_PER_STIMULUS_PLOTS else []


def plot_panels(
    preds: dict[str, np.ndarray],
    stimulus_names: list[str],
    title: str,
    output_path: str,
    human_preds: dict[str, np.ndarray] | None = None,
):
    """4×2 grid of bar charts, one per stimulus (up to 8).

    When `human_preds` is given, overlay human ratings as scatter points at
    measured targets (NaN entries are skipped — Tenenbaum99 data is sparse,
    Bigelow16 is dense but still plotted as points for visual consistency).
    """
    n = len(stimulus_names)
    _any = next(iter(preds.values()), None)
    n_domain = len(_any) if _any is not None else 100
    xs = np.arange(1, n_domain + 1)
    nrows, ncols = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, name in zip(axes, stimulus_names):
        p = preds.get(name, np.zeros(n_domain))
        ax.bar(xs, p, width=1.0, color=C_BAYESIAN, edgecolor="none")
        if human_preds and name in human_preds:
            hp = np.asarray(human_preds[name])
            mask = ~np.isnan(hp)
            ax.scatter(xs[mask], hp[mask], s=8, color=C_HUMAN, zorder=3,
                       edgecolors="white", linewidths=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlim(0.5, n_domain + 0.5)
        ax.set_title(name, fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    # x-label on the bottom-most visible axis of each column (handles partial grids)
    for c in range(ncols):
        rows = [r for r in range(nrows) if r * ncols + c < n]
        if rows:
            ax = axes[max(rows) * ncols + c]
            ax.tick_params(labelbottom=True)
            ax.set_xlabel(f"Number (1–{n_domain})")
    for r in range(nrows):
        if r * ncols < n:
            axes[r * ncols].set_ylabel("p(y∈C|X)", fontsize=7)
    fig.suptitle(title, fontsize=11)
    save_fig(fig, output_path)


def plot_decomposition(
    preds: np.ndarray,
    top_hyps: list[tuple[str, float, np.ndarray]],
    stimulus_name: str,
    output_path: str,
    k: int = 15,
):
    """Generalization bar chart + hypothesis extension dot plot + posterior marginal.

    Args:
        preds: shape (100,) generalization probabilities.
        top_hyps: list of (name, posterior, extension_mask_100) sorted by posterior desc.
        stimulus_name: label for the figure.
        output_path: save path.
        k: number of top hypotheses to show.
    """
    top_hyps = top_hyps[:k]
    n_domain = len(preds)
    xs = np.arange(1, n_domain + 1)

    fig = plt.figure(figsize=(14, 2 + 0.35 * k))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, k * 0.35],
                  width_ratios=[10, 1], hspace=0.05, wspace=0.05)

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.bar(xs, preds, width=1.0, color=C_BAYESIAN, edgecolor="none")
    ax_bar.set_ylabel("p(y∈C|X)")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_yticks([0, 0.5, 1.0])
    ax_bar.set_xlim(0.5, n_domain + 0.5)
    ax_bar.set_title(stimulus_name)
    ax_bar.tick_params(labelbottom=False)

    ax_dot = fig.add_subplot(gs[1, 0], sharex=ax_bar)
    for row, (name, post, ext) in enumerate(top_hyps):
        y_pos = k - 1 - row  # top hypothesis at top
        ext_np = ext.cpu().numpy() if isinstance(ext, torch.Tensor) else ext
        numbers = np.where(ext_np)[0] + 1
        ax_dot.scatter(numbers, np.full_like(numbers, y_pos, dtype=float),
                       s=4, color="#333333", marker="s", linewidths=0)
    ax_dot.set_yticks(range(k))
    ax_dot.set_yticklabels([name for name, _, _ in reversed(top_hyps)], fontsize=6)
    ax_dot.set_ylim(-0.5, k - 0.5)
    ax_dot.set_xlabel(f"Number (1–{n_domain})")

    ax_post = fig.add_subplot(gs[1, 1], sharey=ax_dot)
    posteriors = [post for _, post, _ in top_hyps]
    y_positions = [k - 1 - i for i in range(len(top_hyps))]
    ax_post.barh(y_positions, posteriors, height=0.6, color=C_LLM, edgecolor="none")
    ax_post.set_xlabel("p(h|X)", fontsize=7)
    ax_post.tick_params(labelleft=False)
    ax_post.set_xlim(0, max(posteriors) * 1.15 if posteriors else 1)

    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis("off")

    save_fig(fig, output_path, tight_layout=False)


def plot_bayesian_decompositions(
    bayesian_preds: dict[str, np.ndarray],
    decompositions: dict,
    stimulus_names: list[str],
    output_dir: str,
    featured: list[str] | None = None,
):
    """Write Bayesian-only hypothesis decompositions for one task/domain."""
    per_stimulus = per_stimulus_plot_names(stimulus_names, featured)
    if not per_stimulus:
        return

    decomp_dir = os.path.join(output_dir, "decompositions")
    os.makedirs(decomp_dir, exist_ok=True)
    for name in per_stimulus:
        if name in decompositions:
            plot_decomposition(bayesian_preds[name], decompositions[name], name,
                               os.path.join(decomp_dir, f"{slugify_stimulus(name)}.png"))


def _posterior_row(ax, xs, values, color, title):
    """One source row for plot_comparison: bars when dense, scatter when sparse."""
    values = np.asarray(values)
    mask = ~np.isnan(values)
    if mask.all():
        ax.bar(xs, values, width=1.0, color=color, edgecolor="none")
    else:
        ax.scatter(xs[mask], values[mask], s=14, color=color, zorder=3,
                   edgecolors="white", linewidths=0.3)
    ax.set_ylabel("p(y∈C|X)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])


def plot_comparison(
    bayesian: np.ndarray,
    llm: np.ndarray,
    stimulus_name: str,
    result_key: str,
    output_path: str,
    human: np.ndarray | None = None,
):
    """Stacked bar chart: Bayesian / LLM (/ Human) for one stimulus.

    When `human` is given, adds a third row. Dense human data (no NaN;
    Bigelow16) is rendered as bars; sparse human data (NaN at unmeasured
    targets; Tenenbaum99) is rendered as scatter points.
    """
    n_domain = len(bayesian)
    xs = np.arange(1, n_domain + 1)
    rows = [(bayesian, C_BAYESIAN, f"Bayesian Model — {stimulus_name}"),
            (llm,      C_LLM,      f"{result_key} — {stimulus_name}")]
    if human is not None:
        rows.append((human, C_HUMAN, f"Human — {stimulus_name}"))

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.5 * len(rows)), sharex=True)
    for ax, (values, color, title) in zip(axes, rows):
        _posterior_row(ax, xs, values, color, title)

    axes[-1].set_xlabel(f"Number (1–{n_domain})")
    axes[-1].set_xlim(0.5, n_domain + 0.5)
    save_fig(fig, output_path)


# Metric specs: (column, filename, ylabel, title_stem, sources_allowed)
# sources_allowed=None means "use every source present in curves_df".
_TRAJECTORY_METRICS = [
    ("kl_bayes",          "trajectory_kl.png",                r"KL(Bayesian $\Vert$ LLM)",   "KL trajectory",    ["llm"]),
    ("entropy",           "trajectory_entropy.png",           "Entropy",                      "Entropy",          None),
    ("effective_support", "trajectory_effective_support.png", "Effective support (p > 0.5)", "Effective support", None),
]

_SOURCE_MARKERS = {"bayesian": "o", "llm": "s", "human": "D"}
_SOURCE_ORDER = ["bayesian", "llm", "human"]


def plot_trajectory_metrics(
    curves_df,
    output_dir: str,
    result_key: str = "",
):
    """Per-metric trajectories vs prefix length, SEM across stimuli.

    Mirrors the plot_trajectory_alpha_beta style: one PNG per metric, each
    aggregating per-stimulus values at each prefix length and plotting
    mean ± SEM via plot_trajectory_sem. Assumes curves_df is already
    restricted to stimuli of a single full length, so every prefix bin
    averages the SAME concepts at growing evidence.

    curves_df: DataFrame with columns stimulus, n_examples, source,
               entropy, effective_support, kl_bayes.
    """
    os.makedirs(output_dir, exist_ok=True)
    present_sources = set(curves_df["source"].unique())

    for metric, fname, ylabel, title_stem, sources_allowed in _TRAJECTORY_METRICS:
        if metric not in curves_df.columns:
            continue
        srcs = sources_allowed if sources_allowed is not None else _SOURCE_ORDER
        series = []
        for src in srcs:
            if src not in present_sources:
                continue
            sub = curves_df[curves_df["source"] == src].dropna(subset=[metric])
            if sub.empty:
                continue
            by_n: dict[int, list[float]] = {}
            for n, v in zip(sub["n_examples"], sub[metric]):
                by_n.setdefault(int(n), []).append(float(v))
            xs, ys, sems = aggregate_sem(by_n)
            series.append({
                "x": xs, "y": ys, "y_err": sems,
                "label": None if sources_allowed == ["llm"] else src,
                "color": SOURCE_COLORS.get(src, C_ACCENT),
                "marker": _SOURCE_MARKERS.get(src, "o"),
            })
        if not series:
            continue
        plot_trajectory_sem(
            series, os.path.join(output_dir, fname),
            xlabel="Number of examples", ylabel=ylabel,
            title=with_model(title_stem, result_key),
        )


def plot_hypothesis_generation_summary(
    gen_results: dict[str, list[dict]],
    stimulus_names: list[str],
    output_dir: str,
    result_key: str = "",
):
    """Per-stimulus horizontal bar charts of generated hypotheses + confidences."""
    os.makedirs(output_dir, exist_ok=True)
    for name in stimulus_names:
        if name not in gen_results or not gen_results[name]:
            continue
        hyps = gen_results[name]
        confs = np.array([h["confidence"] for h in hyps], dtype=float)
        confs_norm = confs / confs.sum() if confs.sum() > 0 else confs
        order = np.argsort(-confs_norm)[:10]
        hyps = [hyps[i] for i in order]
        confs_norm = confs_norm[order]
        # Model-written text is data; escape `$` so matplotlib does not parse it as mathtext.
        labels = [h["hypothesis"][:50].replace("$", r"\$") for h in hyps]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(labels))))
        y = np.arange(len(labels))
        ax.barh(y, confs_norm, color=C_LLM, edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Sum-normalized confidence")
        ax.set_xlim(0, max(float(confs_norm.max()) * 1.15, 0.05) if len(confs_norm) else 1.0)
        ax.set_title(with_model(name, result_key), fontsize=9)
        ax.invert_yaxis()
        save_fig(fig, os.path.join(output_dir, f"{slugify_stimulus(name)}.png"))


def plot_hypothesis_generation_type_summary(
    metrics: dict,
    output_path: str,
    result_key: str = "",
    featured: list[str] | None = None,
):
    """Stacked bar: hypothesis type proportions per stimulus."""
    stimuli = metrics["stimuli"]
    if not stimuli:
        return
    names = [name for name in featured if name in stimuli] if featured is not None else list(stimuli.keys())
    if not names:
        return
    n = len(names)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))
    x = np.arange(n)
    bottom = np.zeros(n)
    for typ in ["rule", "interval", "other"]:
        vals = [stimuli[name]["type_proportions"].get(typ, 0) for name in names]
        ax.bar(x, vals, bottom=bottom, color=HYP_TYPE_COLORS[typ], label=typ, width=0.6)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.set_title(with_model("Hypothesis Generation Types", result_key), fontsize=9)
    save_fig(fig, output_path)


def plot_hypothesis_generation_exemplars(
    metrics: dict,
    output_path: str,
    result_key: str = "",
    per_group: int = 3,
):
    """Exemplar stimuli for four hypothesis-type patterns.

    Picks ``per_group`` stimuli per category — rule/interval/other dominated
    (argmax that type's proportion) and no-dominance (lowest max-proportion,
    i.e. most balanced). Ties broken by n_hypotheses desc so more-evidenced
    conditions win. Drawn as horizontal stacked bars, flush within a group and
    separated by a small gap between groups.
    """
    stimuli = metrics["stimuli"]
    entries = [(name, c["type_proportions"], c["n_hypotheses"])
               for name, c in stimuli.items() if c["n_hypotheses"] > 0]
    if not entries:
        return

    def pick(key_fn):
        return sorted(entries, key=key_fn)[:per_group]

    groups = [
        ("Rule-dominated",     pick(lambda e: (-e[1].get("rule", 0),     -e[2]))),
        ("Interval-dominated", pick(lambda e: (-e[1].get("interval", 0), -e[2]))),
        ("Other-dominated",    pick(lambda e: (-e[1].get("other", 0),    -e[2]))),
        ("No dominance",       pick(lambda e: ( max(e[1].values()),      -e[2]))),
    ]

    flat = [(gi, gname, name, props)
            for gi, (gname, sel) in enumerate(groups) for name, props, _ in sel]
    if not flat:
        return
    gap = 0.6
    y_pos, y = [], 0.0
    last_gi = flat[0][0]
    for gi, *_ in flat:
        if gi != last_gi:
            y += gap
            last_gi = gi
        y_pos.append(y)
        y += 1.0
    y_pos = np.array(y_pos)

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(flat) + 1.5))
    lefts = np.zeros(len(flat))
    for typ in ["rule", "interval", "other"]:
        vals = np.array([props.get(typ, 0) for _, _, _, props in flat])
        ax.barh(y_pos, vals, left=lefts, height=1.0,
                color=HYP_TYPE_COLORS[typ], label=typ,
                edgecolor="white", linewidth=0.4)
        lefts += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels([name for _, _, name, _ in flat], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion")

    # Group labels to the left of the y-tick labels
    for gi, (gname, _) in enumerate(groups):
        ys = [yp for (g, *_), yp in zip(flat, y_pos) if g == gi]
        if ys:
            ax.text(-0.32, float(np.mean(ys)), gname,
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=8, fontweight="bold")

    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.set_title(with_model("Hypothesis Type Exemplars", result_key), fontsize=9)
    fig.subplots_adjust(left=0.38)
    save_fig(fig, output_path)


def plot_all(
    bayesian_preds: dict[str, np.ndarray],
    llm_preds: dict[str, np.ndarray] | None,
    stimulus_names: list[str],
    output_dir: str,
    result_key: str = "",
    human_preds: dict[str, np.ndarray] | None = None,
    featured: list[str] | None = None,
):
    """Generate all plots for one experiment run.

    Per-stimulus plots run on `featured` when given (e.g. Bigelow16's curated
    8), otherwise on all stimuli. Sets larger than 8 without a featured
    subset skip per-stimulus plots — aggregate α-β fit + cross-condition stats suffice.
    """
    os.makedirs(output_dir, exist_ok=True)

    per_stimulus = per_stimulus_plot_names(stimulus_names, featured)
    if not per_stimulus:
        return

    plot_panels(bayesian_preds, per_stimulus, "Bayesian Model",
                os.path.join(output_dir, "bayesian_panels.png"),
                human_preds=human_preds)
    if llm_preds:
        plot_panels(llm_preds, per_stimulus, result_key,
                    os.path.join(output_dir, "llm_panels.png"),
                    human_preds=human_preds)

    if llm_preds:
        comp_dir = os.path.join(output_dir, "histogram")
        os.makedirs(comp_dir, exist_ok=True)
        for name in per_stimulus:
            if name in llm_preds:
                plot_comparison(bayesian_preds[name], llm_preds[name], name,
                                result_key, os.path.join(comp_dir, f"{slugify_stimulus(name)}.png"),
                                human=(human_preds or {}).get(name))


# ===== α-β decomposition plots =====

def plot_ablation_comparison(
    bayesian: np.ndarray,
    llm: np.ndarray,
    prior_only: np.ndarray,
    likelihood_only: np.ndarray,
    best_fit: np.ndarray,
    stimulus_name: str,
    output_path: str,
    best_alpha: float = 0.0,
    best_beta: float = 0.0,
):
    """Overlay: LLM bars + Bayesian/prior-only/likelihood-only/best-fit lines."""
    n = len(bayesian)
    xs = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(xs, llm, width=1.0, color=C_LLM, alpha=0.4, edgecolor="none", label="LLM")
    ax.plot(xs, bayesian, color=C_BAYESIAN, lw=1.5, label="Bayesian (α=1,β=1)")
    ax.plot(xs, prior_only, color=C_PRIOR, lw=1.2, ls="--", label="Prior only (β=0)")
    ax.plot(xs, likelihood_only, color=C_LIKELIHOOD, lw=1.2, ls="--", label="Likelihood only (α=0)")
    ax.plot(xs, best_fit, color=C_BESTFIT, lw=1.5, ls="-",
            label=f"Best fit (α={best_alpha:.2f},β={best_beta:.2f})")
    ax.set_ylabel("p(y∈C|X)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xlabel(f"Number (1–{n})")
    ax.set_title(stimulus_name)
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    save_fig(fig, output_path)


def plot_hypothesis_comparison(
    bayesian_hyps: dict[str, dict[str, float]],
    llm_hyps: dict[str, dict[str, float]],
    stimulus_names: list[str],
    output_path: str,
    result_key: str = "",
    top_k: int = 10,
    featured: list[str] | None = None,
):
    """Side-by-side grouped bar chart: Bayesian vs LLM posteriors, top-k by Bayesian mass per stimulus.

    Uses `featured` subset when provided (e.g. Bigelow16's curated 8), otherwise
    all stimulus_names; a 255×10 grid of bar charts would be unrenderably tall.
    """
    per_stimulus = per_stimulus_plot_names(stimulus_names, featured)
    if not per_stimulus:
        return
    n_stimuli = len([n for n in per_stimulus if n in llm_hyps])
    if n_stimuli == 0:
        return
    ncols = min(4, n_stimuli)
    nrows = (n_stimuli + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 0.35 * top_k * nrows))
    axes = np.atleast_1d(axes).flatten()

    panel_idx = 0
    for name in per_stimulus:
        if name not in llm_hyps:
            continue
        ax = axes[panel_idx]
        panel_idx += 1
        # Rank hypotheses by Bayesian mass (descending); fall back to LLM when Bayesian missing
        bay = bayesian_hyps.get(name, {})
        ranking_src = bay if bay else llm_hyps[name]
        top_hyps = [h for h, _ in sorted(ranking_src.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
        b_vals = [bay.get(h, 0.0) for h in top_hyps]
        l_vals = [llm_hyps[name].get(h, 0.0) for h in top_hyps]
        # Reverse so largest bar is at the top of the horizontal axis
        top_hyps, b_vals, l_vals = top_hyps[::-1], b_vals[::-1], l_vals[::-1]
        y = np.arange(len(top_hyps))
        ax.barh(y + 0.2, b_vals, height=0.35, color=C_BAYESIAN, label="Bayesian")
        ax.barh(y - 0.2, l_vals, height=0.35, color=C_LLM, label="LLM")
        ax.set_yticks(y)
        ax.set_yticklabels(top_hyps, fontsize=7)
        ax.set_xlabel("p(h|X)")
        ax.set_title(name, fontsize=8)
        if panel_idx == 1:
            ax.legend(fontsize=6)

    for j in range(panel_idx, len(axes)):
        axes[j].axis("off")
    fig.suptitle(with_model("Hypothesis Posteriors", result_key), fontsize=11, y=1.02)
    save_fig(fig, output_path)


def plot_alpha_beta_all(
    fit_result: dict,
    bayesian_preds: dict[str, np.ndarray],
    llm_preds: dict[str, np.ndarray],
    stimulus_names: list[str],
    output_dir: str,
):
    """Generate α-β ablation overlays; skipped for large stimulus lists."""
    stimulus_names = per_stimulus_plot_names(stimulus_names, None)
    if not stimulus_names:
        return
    os.makedirs(output_dir, exist_ok=True)
    best_alpha = fit_result["best_alpha"]
    best_beta = fit_result["best_beta"]

    for name in stimulus_names:
        if name not in fit_result["ablation_preds"]:
            continue
        ab = fit_result["ablation_preds"][name]
        slug = slugify_stimulus(name)
        plot_ablation_comparison(
            bayesian_preds[name], np.asarray(llm_preds[name]),
            ab["prior_only"], ab["likelihood_only"], ab["best_fit"],
            name, os.path.join(output_dir, f"ablation_{slug}.png"),
            best_alpha=best_alpha, best_beta=best_beta)
