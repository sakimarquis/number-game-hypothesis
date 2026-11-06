"""Hypothesis-mode comparison stages.

run_hypothesis_fit fits LLM posterior behavior to its own hypothesis_eval label
space. For every (task, domain, sampling, stimulus, prefix) where both modes ran:

  1. Build the basis matrix M[y, h] = p(y | h) over the eval prompt's label
     space (rules + intervals + 'other'). Rules and intervals reuse the
     executable supports from analysis.hypothesis_space; 'other' is
     a flat residual column (uniform over 1..domain). Per-prefix because
     interval labels shift with the example range.
  2. Fit weights w in min ||M w - p_posterior||^2 + ridge ||w||^2 s.t. w >= 0,
     normalized to a simplex. The tiny ridge breaks NNLS's default
     sparsest-solution bias, so when several hypotheses explain the
     behavior equally well the mass splits across them ("split equally").
  3. Compare w_fit against the LLM eval self-report w_eval (JS distance, top-k overlap).

Eval is always explicit, so we only fit against the explicit posterior.

run_hypothesis_compare projects posterior, hypothesis_eval, and
hypothesis_generation into predictive space per prefix and compares them.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns

from analysis.alpha_beta_scope import PREFIX_LENGTHS
from analysis.cache import load_condition_trajectory, trajectory_steps
from analysis.hypothesis_space import (
    RULE_SHARE_TOP_K,
    eval_mapper,
    generation_mapper,
    load_hypothesis_library,
    normalize_structured_generation_confidence,
    normalized_hypothesis_items,
    project_result,
    prompt_items,
)
from analysis.model import DEVICE
from analysis.metrics import HYPOTHESIS_TYPES, compute_topk_overlap_vectors, jensen_shannon_distance, shannon_entropy
from analysis.plotter import (
    EVAL_MODE_LABELS,
    PROMPT_PALETTE,
    plot_trajectory_sem,
    save_csv,
    save_fig,
    set_mpl,
    task_domain_series,
    with_model,
    with_task_domain,
)
from analysis.trajectory import plot_prompt_trajectory
from configs.settings import HYPOTHESIS_H_SPACE, Experiment, ExperimentCondition
from tasks.prompts import SAMPLING_BLOCKS
from tasks.stimuli import build_task_context

_OTHER_LABEL = "other"
_RIDGE = 1e-6
_TOP_K = 5
_EVAL = ("hypothesis_eval", HYPOTHESIS_H_SPACE["hypothesis_eval"])
_GENERATION = ("hypothesis_generation", HYPOTHESIS_H_SPACE["hypothesis_generation"])
_JS_DISTANCE_METRICS = (
    ("posterior_vs_eval_js_distance", f"JS distance ({EVAL_MODE_LABELS['posterior']}, {EVAL_MODE_LABELS['hypothesis_eval']})"),
    ("posterior_vs_generation_js_distance", f"JS distance ({EVAL_MODE_LABELS['posterior']}, {EVAL_MODE_LABELS['hypothesis_generation']})"),
)
_ENTROPY_TRAJECTORY_SPECS = (
    ("entropy_bits_mean", "hypothesis_entropy.png", "Entropy (bits)"),
    ("other_mass_mean", "hypothesis_other_mass.png", "Other mass"),
)
_ENTROPY_LINE_SPECS = tuple(
    (eval_mode, sampling, h_space, fmt)
    for eval_mode, h_spaces, fmt in (
        ("posterior", ("explicit", "implicit"), "-"),
        (_EVAL[0], (_EVAL[1],), "--"),
        (_GENERATION[0], (_GENERATION[1],), ":"),
    )
    for sampling in ("strong", "weak")
    for h_space in h_spaces
)
_TOP1_MODE_ORDER = ("hypothesis_eval", "hypothesis_generation")
_TOP1_METRICS = (
    ("top1_normalized_confidence_mean", "Norm. confidence"),
    ("top1_hypothesis_size_mean", "Hypothesis size"),
    ("top1_accuracy_mean", "Evidence accuracy"),
)
_TOP1_PAIR_METRICS = (
    ("top1_same_label_mean", "Same mapped label"),
    ("top1_support_jaccard_mean", "Support Jaccard"),
)

set_mpl(size=8)


def basis_matrix(ctx, examples):
    """Return (M, labels) where M[y_idx, h_idx] = p(y|h) on 1..domain.

    Columns: rule items, interval items, then 'other' (uniform residual).
    Per-prefix because interval labels are derived from `examples`.
    """
    items = prompt_items(ctx, examples)
    M = np.zeros((ctx.domain, len(items) + 1))
    for j, item in enumerate(items):
        M[:, j] = item.distribution
    M[:, -1] = 1.0 / ctx.domain
    return M, [item.label for item in items] + [_OTHER_LABEL]


def _spectral_radius(A, n_iters=20):
    """Largest eigenvalue per batch of PSD matrices via power iteration.

    Used to set the FISTA step size; iterations are pure matmul so they
    vectorize across the batch dim without LAPACK calls.
    """
    # A = MᵀM + ridge·I is entrywise nonnegative, so a fixed all-ones start overlaps the top
    # eigenvector and keeps the step size, and hence the fitted weights, reproducible.
    B, H, _ = A.shape
    v = torch.ones(B, H, dtype=A.dtype, device=A.device) / H ** 0.5
    for _ in range(n_iters):
        v = (A @ v.unsqueeze(-1)).squeeze(-1)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
    Av = (A @ v.unsqueeze(-1)).squeeze(-1)
    return (v * Av).sum(dim=-1)


def fit_behavior_weights_batched(M_batch, p_batch, ridge=_RIDGE, n_steps=500):
    """Batched bounded ridge regression: min ||M w - p||^2 + ridge ||w||^2 s.t. w >= 0.

    Vectorized FISTA across the batch dim: pure matmul/relu, runs on GPU when
    CUDA is available, and avoids per-problem dense linear solves.

    FISTA keeps every iterate nonnegative. Closed-form ridge plus clipping can
    leak mass to wrong hypotheses when overlapping columns require small
    negative coefficients in the unconstrained optimum.

    With identical columns, the L2 ridge term selects the symmetric bounded
    optimum, splitting mass evenly across tied hypotheses.

    Args:
        M_batch: (B, D, H) basis matrices.
        p_batch: (B, D)    posterior predictions.
    Returns:
        (B, H) weights on the simplex (rows sum to 1, or 0 if degenerate).
    """
    M_t = torch.as_tensor(M_batch, dtype=torch.float64, device=DEVICE)
    p_t = torch.as_tensor(p_batch, dtype=torch.float64, device=DEVICE)
    B, _D, H = M_t.shape
    eye = torch.eye(H, dtype=torch.float64, device=DEVICE)
    A = M_t.transpose(-1, -2) @ M_t + ridge * eye
    b_vec = (M_t.transpose(-1, -2) @ p_t.unsqueeze(-1)).squeeze(-1)

    lipschitz = 2.0 * _spectral_radius(A).clamp(min=1e-12)        # (B,)
    inv_L = (1.0 / lipschitz).unsqueeze(-1)                       # (B, 1)

    w = torch.zeros(B, H, dtype=torch.float64, device=DEVICE)
    y = w.clone()
    t = torch.tensor(1.0, dtype=torch.float64, device=DEVICE)
    for _ in range(n_steps):
        grad = 2.0 * ((A @ y.unsqueeze(-1)).squeeze(-1) - b_vec)
        w_new = (y - inv_L * grad).clamp(min=0.0)
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt())
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new

    s = w.sum(dim=-1, keepdim=True)
    w = torch.where(s > 0, w / s, w)
    return w.cpu().numpy()


class Top1Stats(NamedTuple):
    row: dict
    support_mask: np.ndarray | None


def top1_stats(items, examples, mapper, skip_other_top1: bool = False) -> Top1Stats:
    """Top normalized hypothesis plus executable support diagnostics.

    `top1_type` is rule/interval for an executable top-1, "other" for an answer with no
    executable top-1, and "" for an empty (exhausted) answer. Confidence, size, and accuracy
    are defined only for an executable top-1: an all-"other" Eval answer would otherwise
    score confidence 1.0.
    """
    normalized = sorted(normalized_hypothesis_items(items), key=lambda item: item[1], reverse=True)
    base = {
        "top1_label": "",
        "top1_type": "",
        "top1_normalized_confidence": float("nan"),
        "top1_hypothesis_size": float("nan"),
        "top1_accuracy": float("nan"),
    }
    if not normalized:
        return Top1Stats(base, None)

    mapped = None
    for raw_label, confidence in normalized:
        mapped = mapper(raw_label)
        if mapped is not None or not skip_other_top1:
            break
    else:
        raw_label, confidence = normalized[0]
    display_label = str(raw_label["hypothesis"]) if isinstance(raw_label, dict) else raw_label

    base.update({"top1_label": display_label, "top1_type": "other"})
    if mapped is None:
        return Top1Stats(base, None)

    hypothesis_type, label, vec = mapped
    # vec is a 0/1 support, or a grouped family's mean over member supports (see PromptItem):
    # its mean is the members' mean |h| / domain, and it is 1 on an example only if every member accepts it.
    vec = np.asarray(vec, dtype=float)
    evidence_idx = np.asarray(examples, dtype=int) - 1
    base.update({
        "top1_label": label,
        "top1_type": hypothesis_type,
        "top1_normalized_confidence": confidence,
        "top1_hypothesis_size": float(vec.mean()),
        "top1_accuracy": float(bool((vec[evidence_idx] == 1.0).all())),
    })
    return Top1Stats(base, vec > 0.0)


def compare_top1_stats(eval_top1: Top1Stats, gen_top1: Top1Stats) -> dict:
    """Label and support agreement for eval-vs-generation top-1 hypotheses."""
    # Only executable top-1s (those with a support) can share a mapped label.
    same = (
        eval_top1.support_mask is not None and gen_top1.support_mask is not None
        and eval_top1.row["top1_label"] == gen_top1.row["top1_label"]
    )
    if eval_top1.support_mask is None or gen_top1.support_mask is None:
        jaccard = float("nan")
    else:
        union = np.logical_or(eval_top1.support_mask, gen_top1.support_mask).sum()
        jaccard = float(np.logical_and(eval_top1.support_mask, gen_top1.support_mask).sum() / union) if union else float("nan")
    return {"top1_same_label": float(same), "top1_support_jaccard": jaccard}


def compare_weights(w_fit, w_eval, top_k=_TOP_K):
    """JS distance and top-k label overlap between fit and eval weight vectors."""
    return {
        "js_distance": jensen_shannon_distance(w_fit, w_eval),
        "topk_overlap": compute_topk_overlap_vectors(w_fit, w_eval, k=top_k),
    }


def _view_js_distance(views: dict, mode_a: tuple[str, str], mode_b: tuple[str, str]) -> float:
    if mode_a not in views or mode_b not in views:
        return float("nan")
    return jensen_shannon_distance(views[mode_a].vector, views[mode_b].vector)


def _collect_problems(experiment: Experiment):
    """Return one fit problem per (posterior condition, stimulus, prefix) with a matching eval step in the eval h_space."""
    result_key = experiment.backend.result_key
    problems = []
    for task, factors in experiment.tasks.items():
        if not {"posterior", "hypothesis_eval"}.issubset(factors.eval_modes):
            continue
        for domain in factors.domains:
            ctx = build_task_context(task, domain)
            for sampling in factors.samplings:
                post_traj = load_condition_trajectory(ExperimentCondition(task, "posterior", _EVAL[1], sampling, domain), result_key)
                eval_traj = load_condition_trajectory(ExperimentCondition(task, *_EVAL, sampling, domain), result_key)
                if not post_traj or not eval_traj:
                    continue

                for stimulus_name, stimulus in ctx.task.stimulus_map.items():
                    eval_by_n = trajectory_steps(eval_traj, stimulus_name)
                    for n, p_step in trajectory_steps(post_traj, stimulus_name).items():
                        eval_step = eval_by_n.get(n)
                        if eval_step is None:
                            continue
                        M, labels = basis_matrix(ctx, stimulus.examples[:n])
                        w_eval = np.array([float(eval_step.get(label, 0.0)) for label in labels])
                        w_eval_sum = w_eval.sum()
                        problems.append({
                            "M": M,
                            "p": np.asarray(p_step, dtype=np.float64),
                            "w_eval": w_eval / w_eval_sum if w_eval_sum > 0 else w_eval,
                            "row": {
                                "task": task, "domain": domain,
                                "h_space": _EVAL[1], "sampling": sampling,
                                "stimulus": stimulus_name, "stimulus_kind": stimulus.kind,
                                "n_examples": n, "n_labels": len(labels),
                            },
                        })
    return problems


def _solve_problem_group(probs):
    """Batched FISTA over problems sharing the same domain (D); pad H to max."""
    D = probs[0]["M"].shape[0]
    H_max = max(prob["M"].shape[1] for prob in probs)
    M_pad = np.zeros((len(probs), D, H_max), dtype=np.float64)
    p_stack = np.zeros((len(probs), D), dtype=np.float64)
    for slot, prob in enumerate(probs):
        M_pad[slot, :, :prob["M"].shape[1]] = prob["M"]
        p_stack[slot] = prob["p"]
    W_pad = fit_behavior_weights_batched(M_pad, p_stack)
    return [W_pad[slot, :prob["M"].shape[1]] for slot, prob in enumerate(probs)]


def run_hypothesis_fit(experiment: Experiment, analysis_dir: str, figure_dir: str):
    """Fit explicit-posterior to eval label space; compare against eval self-report.

    All (condition, stimulus, prefix) sub-problems are collected, then dispatched
    to the GPU in one batched FISTA pass per domain.
    """
    problems = _collect_problems(experiment)
    if not problems:
        print("  Hypothesis fit: no overlapping posterior+eval data")
        return pd.DataFrame()

    by_domain = {}
    for prob in problems:
        by_domain.setdefault(prob["M"].shape[0], []).append(prob)

    rows = []
    for group in by_domain.values():
        for prob, w_fit in zip(group, _solve_problem_group(group)):
            rows.append({**prob["row"], **compare_weights(w_fit, prob["w_eval"])})

    df = save_csv(rows, analysis_dir, "hypothesis_fit_metrics.csv")
    n_valid = int(df["js_distance"].notna().sum())
    print(f"  hypothesis_fit: {n_valid}/{len(rows)} valid on {DEVICE}; "
          f"mean JS distance(fit, eval) = {df['js_distance'].mean():.3f}, "
          f"mean top-{_TOP_K} overlap = {df['topk_overlap'].mean():.3f}")
    return df


def _ordered_samplings(df: pd.DataFrame) -> list[str]:
    present = set(df["sampling"].unique())
    return [s for s in SAMPLING_BLOCKS if s in present]


def summarize_hypothesis_compare(metric_rows, entropy_rows, top1_rows, pair_rows) -> tuple[pd.DataFrame, ...]:
    metric_df, entropy_df, top1_df, pair_df = (pd.DataFrame(rows) for rows in (metric_rows, entropy_rows, top1_rows, pair_rows))
    metric_summary = entropy_summary = top1_summary = pair_summary = pd.DataFrame()
    if not metric_df.empty:
        metric_summary = (
            metric_df
            .groupby(["domain", "n_examples", "h_space", "sampling"], dropna=False)[[name for name, _ in _JS_DISTANCE_METRICS]]
            .mean()
            .reset_index()
        )
    if not entropy_df.empty:
        entropy_summary = (
            entropy_df
            .groupby(["task", "domain", "n_examples", "h_space", "sampling", "eval_mode"], dropna=False)
            .agg(
                entropy_bits_mean=("entropy_bits", "mean"),
                other_mass_mean=("other_mass", "mean"),
                rule_mass_mean=("rule_mass", "mean"),
                interval_mass_mean=("interval_mass", "mean"),
                n_rows=("entropy_bits", "size"),
            )
            .reset_index()
        )
    if not top1_df.empty:
        # Type shares count every answered prefix ("other" included); exhausted answers are no response.
        answered = top1_df["top1_type"].ne("")
        top1_df = top1_df.assign(**{
            f"top1_{typ}_preference": top1_df["top1_type"].eq(typ).astype(float).where(answered)
            for typ in HYPOTHESIS_TYPES
        })
        top1_summary = (
            top1_df
            .groupby(["domain", "n_examples", "sampling", "eval_mode"], dropna=False)
            .agg(
                top1_normalized_confidence_mean=("top1_normalized_confidence", "mean"),
                top1_hypothesis_size_mean=("top1_hypothesis_size", "mean"),
                top1_accuracy_mean=("top1_accuracy", "mean"),
                top1_rule_preference_mean=("top1_rule_preference", "mean"),
                top1_interval_preference_mean=("top1_interval_preference", "mean"),
                top1_other_preference_mean=("top1_other_preference", "mean"),
                n_rows=("top1_accuracy", "size"),
            )
            .reset_index()
        )
    if not pair_df.empty:
        pair_summary = (
            pair_df
            .groupby(["domain", "n_examples", "sampling"], dropna=False)
            .agg(
                top1_same_label_mean=("top1_same_label", "mean"),
                top1_support_jaccard_mean=("top1_support_jaccard", "mean"),
                n_rows=("top1_same_label", "size"),
            )
            .reset_index()
        )
    return metric_summary, entropy_summary, top1_summary, pair_summary


def _plot_js_distance_trajectories(metric_df: pd.DataFrame, output_dir: str) -> None:
    if metric_df.empty:
        return
    # Rows are pooled across task/domains (one "all" group), not task_domain-weighted.
    plot_df = metric_df.assign(source="llm", task_domain="all")
    for metric, ylabel in _JS_DISTANCE_METRICS:
        plot_prompt_trajectory(plot_df, metric, ylabel, os.path.join(output_dir, f"hypothesis_{metric}.png"), include_bayesian=False)


def _plot_eval_generation_trajectory(metric_df: pd.DataFrame, output_path: str) -> None:
    if metric_df.empty:
        return
    plot_df = metric_df[metric_df["sampling"].isin(["strong", "weak"])].dropna(subset=["eval_vs_generation_js_distance"])
    if plot_df.empty:
        return
    plot_df = with_task_domain(plot_df)
    series = []
    for sampling in ("strong", "weak"):
        sub = plot_df[plot_df["sampling"] == sampling]
        if sub.empty:
            continue
        xs, ys, sems = task_domain_series(sub, "eval_vs_generation_js_distance")
        series.append({"x": xs, "y": ys, "y_err": sems, "label": sampling, "color": PROMPT_PALETTE[f"explicit_{sampling}"]})
    if series:
        plot_trajectory_sem(
            series, output_path,
            ylabel=f"JS distance ({EVAL_MODE_LABELS['hypothesis_eval']}, {EVAL_MODE_LABELS['hypothesis_generation']})",
            figsize=(3, 2.5), xticks=PREFIX_LENGTHS,
        )


def _plot_entropy_trajectories(summary_df: pd.DataFrame, output_dir: str) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.dropna(subset=["eval_mode", "sampling", "h_space"])
    if plot_df.empty:
        return
    plot_df = with_task_domain(plot_df)
    xticks = tuple(sorted(int(n) for n in plot_df["n_examples"].dropna().unique()))
    for metric, filename, ylabel in _ENTROPY_TRAJECTORY_SPECS:
        series = []
        for eval_mode, sampling, h_space, fmt in _ENTROPY_LINE_SPECS:
            # Posterior projection is direct, so its other mass is always 0.
            if metric == "other_mass_mean" and eval_mode == "posterior":
                continue
            sub = plot_df[
                (plot_df["eval_mode"] == eval_mode) & (plot_df["sampling"] == sampling) & (plot_df["h_space"] == h_space)
            ].dropna(subset=[metric])
            if sub.empty:
                continue
            xs, ys, sems = task_domain_series(sub, metric)
            h_space_label = f" {h_space}" if eval_mode == "posterior" else ""
            series.append({
                "x": xs, "y": ys, "y_err": sems,
                "label": f"{EVAL_MODE_LABELS[eval_mode]} {sampling}{h_space_label}",
                "color": PROMPT_PALETTE[f"{h_space}_{sampling}"],
                "fmt": fmt,
            })
        if series:
            plot_trajectory_sem(
                series, os.path.join(output_dir, filename),
                ylabel=ylabel, figsize=(3.4, 2.6), xticks=xticks,
            )


def _plot_top1_metric_grid(summary_df: pd.DataFrame, output_path: str, result_key: str) -> None:
    """n×metric bar grid of eval/generation top-1 summaries."""
    if summary_df.empty:
        return
    samplings = _ordered_samplings(summary_df)
    plot_df = summary_df.assign(mode_label=summary_df["eval_mode"].map(EVAL_MODE_LABELS))
    hue_order = [EVAL_MODE_LABELS[m] for m in _TOP1_MODE_ORDER if m in set(plot_df["eval_mode"])]
    n_values = sorted(plot_df["n_examples"].unique())[:len(PREFIX_LENGTHS)]
    fig, axes = plt.subplots(len(n_values), len(_TOP1_METRICS), figsize=(14.5, 3.0 * len(n_values)), sharex=True)
    axes = np.asarray(axes).reshape(len(n_values), len(_TOP1_METRICS))
    for row_idx, n_examples in enumerate(n_values):
        sub_n = plot_df[plot_df["n_examples"] == n_examples]
        for col_idx, (metric, label) in enumerate(_TOP1_METRICS):
            ax = axes[row_idx, col_idx]
            sns.barplot(data=sub_n, x="sampling", y=metric, order=samplings, errorbar=None, ax=ax,
                        hue="mode_label", hue_order=hue_order)
            ax.set_title(label if row_idx == 0 else "")
            ax.set_xlabel("")
            ax.set_ylabel(f"n={n_examples}" if col_idx == 0 else "")
            ax.tick_params(axis="x", rotation=30)
            ax.set_ylim(0.0, 1.05)
            if row_idx == 0 and col_idx == len(_TOP1_METRICS) - 1:
                ax.legend(frameon=False, fontsize=7)
            elif ax.get_legend():
                ax.get_legend().remove()
    fig.suptitle(with_model("Eval vs generation top-1 hypothesis summaries", result_key), fontsize=12)
    save_fig(fig, output_path)


def _plot_top1_pair_metric_panels(summary_df: pd.DataFrame, output_path: str, result_key: str) -> None:
    """Plot agreement metrics as two compact panels with n=1..4 grouped per sampling."""
    plot_df = summary_df[summary_df["n_examples"].isin(PREFIX_LENGTHS)] if not summary_df.empty else summary_df
    if plot_df.empty:
        return
    samplings = _ordered_samplings(plot_df)
    plot_df = plot_df.assign(n_label="n=" + plot_df["n_examples"].astype(int).astype(str))
    hue_order = [f"n={n}" for n in PREFIX_LENGTHS if n in set(plot_df["n_examples"])]

    fig, axes = plt.subplots(1, len(_TOP1_PAIR_METRICS), figsize=(7.2, 2.7), sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, (metric, label) in zip(axes, _TOP1_PAIR_METRICS):
        sns.barplot(
            data=plot_df, x="sampling", y=metric, order=samplings,
            hue="n_label", hue_order=hue_order,
            palette=sns.color_palette("deep", len(hue_order)),
            errorbar=None, ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("Agreement" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0.0, 1.05)
        if ax is axes[-1]:
            ax.legend(title="", frameon=False, fontsize=7)
        elif ax.get_legend():
            ax.get_legend().remove()
    fig.suptitle(with_model("Eval-generation top-1 agreement", result_key), fontsize=12)
    save_fig(fig, output_path)


def run_hypothesis_compare(experiment: Experiment, analysis_dir: str, figure_dir: str):
    """Compare projected posterior, eval, and generation predictions per prefix.

    JS-distance pairs are formed within matching h_spaces where possible:
      - posterior_vs_eval:        explicit posterior vs explicit eval
      - posterior_vs_generation:  implicit posterior vs implicit generation
      - eval_vs_generation:       explicit eval vs implicit generation (cross-h_space, unavoidable)
    """
    result_key = experiment.backend.result_key
    metric_rows, entropy_rows, top1_rows, pair_rows = [], [], [], []

    for task, factors in experiment.tasks.items():
        for domain in factors.domains:
            ctx = build_task_context(task, domain)
            generation_library = (
                load_hypothesis_library(domain) if "hypothesis_generation" in factors.eval_modes else None
            )

            for sampling in factors.samplings:
                def trajectory(eval_mode, h_space):
                    if eval_mode not in factors.eval_modes:
                        return None
                    # Posterior views here feed only entropy and JS distance, so text-trial results are smoothed.
                    return load_condition_trajectory(
                        ExperimentCondition(task, eval_mode, h_space, sampling, domain), result_key,
                        log_space=eval_mode == "posterior",
                    )

                eval_traj = trajectory(*_EVAL)
                gen_traj = trajectory(*_GENERATION)
                post_trajs = {h: trajectory("posterior", h) for h in factors.h_spaces}
                if not eval_traj and not gen_traj and not any(post_trajs.values()):
                    continue

                for stimulus_name, stimulus in ctx.task.stimulus_map.items():
                    eval_by_n = trajectory_steps(eval_traj, stimulus_name)
                    gen_by_n = trajectory_steps(gen_traj, stimulus_name)
                    post_by_n = {h: trajectory_steps(t, stimulus_name) for h, t in post_trajs.items()}
                    all_ns = set(eval_by_n) | set(gen_by_n) | {n for steps in post_by_n.values() for n in steps}

                    for n_examples in sorted(all_ns):
                        examples = stimulus.examples[:n_examples]
                        row_base = {
                            "task": task, "domain": domain,
                            "stimulus": stimulus_name, "stimulus_kind": stimulus.kind,
                            "sampling": sampling, "n_examples": n_examples,
                        }

                        views = {}
                        for h, steps in post_by_n.items():
                            if n_examples in steps:
                                views[("posterior", h)] = project_result("posterior", steps[n_examples], examples, ctx)

                        eval_top1 = gen_top1 = None
                        if n_examples in eval_by_n:
                            evaluated = eval_by_n[n_examples]
                            views[_EVAL] = project_result(_EVAL[0], evaluated, examples, ctx)
                            eval_top1 = top1_stats(evaluated.items(), examples, eval_mapper(ctx, examples), skip_other_top1=True)
                            top1_rows.append({**row_base, "eval_mode": "hypothesis_eval", **eval_top1.row})

                        if n_examples in gen_by_n:
                            gen_norm = normalize_structured_generation_confidence(gen_by_n[n_examples])
                            views[_GENERATION] = project_result(_GENERATION[0], gen_by_n[n_examples], examples, ctx)
                            gen_top1 = top1_stats(
                                ((item, item["confidence"]) for item in gen_norm), examples, generation_mapper(generation_library),
                            )
                            top1_rows.append({**row_base, "eval_mode": "hypothesis_generation", **gen_top1.row})

                        if eval_top1 is not None and gen_top1 is not None:
                            pair_rows.append({**row_base, **compare_top1_stats(eval_top1, gen_top1)})

                        for (eval_mode, h_space), view in views.items():
                            entropy_rows.append({
                                **row_base, "eval_mode": eval_mode, "h_space": h_space,
                                "entropy_bits": shannon_entropy(view.vector, base=2),
                                "other_mass": view.other_mass,
                                "rule_mass": view.rule_mass,
                                "interval_mass": view.interval_mass,
                                **{f"rule_share_top{k}": view.rule_share(k) for k in RULE_SHARE_TOP_K},
                            })

                        for h_space in factors.h_spaces:
                            if ("posterior", h_space) not in views:
                                continue
                            metric_rows.append({
                                **row_base,
                                "h_space": h_space,
                                "posterior_vs_eval_js_distance": _view_js_distance(views, ("posterior", h_space), _EVAL),
                                "posterior_vs_generation_js_distance": _view_js_distance(views, ("posterior", h_space), _GENERATION),
                                "eval_vs_generation_js_distance": _view_js_distance(views, _EVAL, _GENERATION),
                            })

    metric_df = save_csv(metric_rows, analysis_dir, "hypothesis_compare_js_distance_metrics.csv")
    entropy_df = save_csv(entropy_rows, analysis_dir, "hypothesis_compare_entropy_metrics.csv")
    top1_df = save_csv(top1_rows, analysis_dir, "hypothesis_compare_top1_metrics.csv")
    save_csv(pair_rows, analysis_dir, "hypothesis_compare_top1_pair_metrics.csv")
    metric_summary, entropy_summary, top1_summary, pair_summary = summarize_hypothesis_compare(
        metric_rows, entropy_rows, top1_rows, pair_rows,
    )
    save_csv(metric_summary, analysis_dir, "hypothesis_compare_js_distance_summary.csv")
    save_csv(entropy_summary, analysis_dir, "hypothesis_compare_entropy_summary.csv")
    save_csv(top1_summary, analysis_dir, "hypothesis_compare_top1_summary.csv")
    save_csv(pair_summary, analysis_dir, "hypothesis_compare_top1_pair_summary.csv")

    _plot_js_distance_trajectories(metric_df, figure_dir)
    _plot_eval_generation_trajectory(metric_df, os.path.join(figure_dir, "hypothesis_eval_vs_generation_js_distance.png"))
    _plot_entropy_trajectories(entropy_summary, figure_dir)
    _plot_top1_metric_grid(top1_summary, os.path.join(figure_dir, "hypothesis_top1_summary.png"), result_key)
    _plot_top1_pair_metric_panels(pair_summary, os.path.join(figure_dir, "hypothesis_top1_agreement.png"), result_key)

    print(f"  hypothesis_compare: {len(metric_df)} JS-distance rows, "
          f"{len(entropy_df)} entropy rows, {len(top1_df)} top-1 rows")
    return metric_df
