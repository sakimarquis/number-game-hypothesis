"""Trajectory analysis: posterior metrics over evidence-prefix length.

Computes and plots KL/entropy/effective-support/α-β trajectories from the
posterior prefix caches reconstructed by analysis.cache.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from analysis.alpha_beta_scope import PREFIX_LENGTHS
from analysis.metrics import kl_divergence
from analysis.model import NumberGameModel, fit_alpha_beta
from analysis.plotter import (
    set_mpl, aggregate_sem, plot_trajectory_sem, with_model,
    alpha_beta_ratio, log_positive_ratio, task_domain_series,
    ordered_stimulus_kinds, plot_prompt_kind_bar,
    CYCLE, C_HUMAN, C_ALPHA, C_BETA, C_ACCENT,
    PROMPT_ORDER, PROMPT_PALETTE,
)
from tasks.stimuli import StimulusSet

set_mpl(size=8)


def clip_to_prefix_horizon(
    stimuli: list[StimulusSet],
    trajectories: dict[str, dict],
    prefix_horizon: int,
) -> tuple[list[StimulusSet], dict[str, dict]]:
    """Restrict to stimuli with at least N examples, truncated to their first N.

    N is `prefix_horizon` capped by the longest stimulus, so every prefix bin
    averages over the same stimuli. Trajectories keep only those stimuli and
    prefixes of at most N examples.
    """
    n = min(prefix_horizon, max(len(c.examples) for c in stimuli))
    clipped = [StimulusSet(c.name, c.examples[:n], c.kind) for c in stimuli if len(c.examples) >= n]
    names = {c.name for c in clipped}
    subset = {}
    for name, ev in trajectories.items():
        if name not in names:
            continue
        keep = [(ne, r) for ne, r in zip(ev["n_examples"], ev["results"]) if ne <= n]
        if keep:
            ns, rs = map(list, zip(*keep))
            subset[name] = {"n_examples": ns, "results": rs}
    return clipped, subset


def bayesian_prefix_trajectories(model: NumberGameModel, stimuli: list[StimulusSet]) -> dict[str, dict]:
    """Bayesian posterior trajectory in the same {n_examples, results} shape as LLM rows."""
    return {
        c.name: {
            "n_examples": list(range(1, len(c.examples) + 1)),
            "results": [model.predict(c.examples[: i + 1]).cpu().numpy() for i in range(len(c.examples))],
        }
        for c in stimuli
    }


def compute_trajectory(
    model: NumberGameModel,
    stimuli: list[StimulusSet],
    trajectories: dict[str, dict],
) -> dict[str, dict]:
    """Compute KL divergence + Bayesian weight at each evidence prefix.

    Returns {stimulus_name: {"n_examples": [...], "kl": [...],
                             "bayesian_weight": [...], "target_hyp": str}}
    """
    stimulus_map = {c.name: c for c in stimuli}
    out = {}
    for stimulus_name, ev in trajectories.items():
        stimulus = stimulus_map[stimulus_name]
        # Identify target hypothesis from full-evidence posterior
        full_post = model.math_rule_posteriors(stimulus.examples)
        target = max((k for k in full_post if k != "other"), key=full_post.get)

        kls, bay_ws = [], []
        for n, llm_pred in zip(ev["n_examples"], ev["results"]):
            prefix = stimulus.examples[:n]
            bay_ws.append(model.math_rule_posteriors(prefix).get(target, 0.0))
            kls.append(kl_divergence(model.predict(prefix).cpu().numpy(), np.asarray(llm_pred)))

        out[stimulus_name] = {
            "n_examples": ev["n_examples"],
            "kl": kls,
            "bayesian_weight": bay_ws,
            "target_hyp": target,
        }
    return out


def compute_trajectory_alpha_beta(
    model: NumberGameModel,
    stimuli: list[StimulusSet],
    trajectories: dict[str, dict],
) -> dict:
    """Fit α-β at each evidence prefix length across all stimuli.

    For each prefix n, builds truncated stimuli (examples[:n]) and collects the
    corresponding LLM predictions, then runs fit_alpha_beta. Pass clipped
    inputs (clip_to_prefix_horizon) so every prefix averages the same stimuli.

    Returns {"n_examples": [...], "alpha": [...], "beta": [...], "loss": [...], "n_stimuli": [...]}
    """
    stimulus_map = {c.name: c for c in stimuli}
    out = {"n_examples": [], "alpha": [], "beta": [], "loss": [], "n_stimuli": []}
    for n in sorted({n for ev in trajectories.values() for n in ev["n_examples"]}):
        llm_preds, trunc_stimuli = {}, []
        for stimulus_name, ev in trajectories.items():
            if n not in ev["n_examples"]:
                continue
            stimulus = stimulus_map[stimulus_name]
            llm_preds[stimulus_name] = np.asarray(ev["results"][ev["n_examples"].index(n)])
            trunc_stimuli.append(StimulusSet(stimulus.name, stimulus.examples[:n], stimulus.kind))

        fit = fit_alpha_beta([(model, llm_preds, trunc_stimuli)])
        if fit is None:
            continue
        out["n_examples"].append(n)
        out["alpha"].append(fit["best_alpha"])
        out["beta"].append(fit["best_beta"])
        out["loss"].append(fit["best_loss"])
        out["n_stimuli"].append(fit["n_stimuli"])
    return out


def plot_trajectory_alpha_beta(trajectory: dict, output_path: str, result_key: str = ""):
    """Joint α and β vs evidence prefix length."""
    if not trajectory["n_examples"]:
        return

    plot_trajectory_sem(
        [{"x": trajectory["n_examples"], "y": trajectory["alpha"], "y_err": [0.0] * len(trajectory["alpha"]),
          "label": "α (prior weight)", "color": C_ALPHA, "marker": "o"},
         {"x": trajectory["n_examples"], "y": trajectory["beta"], "y_err": [0.0] * len(trajectory["beta"]),
          "label": "β (size principle)", "color": C_BETA, "marker": "s"}],
        output_path, ylabel="Parameter value",
        title=with_model("α-β Trajectory", result_key), hline=1.0,
    )


def plot_trajectory_alpha_beta_ratio(trajectory: dict, output_path: str, result_key: str = ""):
    """Joint log(α/β) vs prefix length; hline=0 marks balance."""
    xs, ys = [], []
    for n, a, b in zip(trajectory["n_examples"], trajectory["alpha"], trajectory["beta"]):
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0 or b <= 0.0:
            continue
        xs.append(n)
        ys.append(float(np.log(alpha_beta_ratio(a, b))))
    if not xs:
        return

    plot_trajectory_sem(
        [{"x": xs, "y": ys, "y_err": [0.0] * len(xs),
          "label": "log(α / β)", "color": C_ACCENT, "marker": "D"}],
        output_path, ylabel="log(α / β)",
        title=with_model("log(α/β) Trajectory", result_key), hline=0.0,
    )


def compute_trajectory_rate(kl_trajectory: list[float]) -> float:
    """Linear rate of KL decrease: (KL_first - KL_last) / (n_steps - 1).

    Positive = converging toward Bayesian; negative = diverging.
    """
    clean = [v for v in kl_trajectory if not np.isnan(v)]
    if len(clean) < 2:
        return 0.0
    return (clean[0] - clean[-1]) / (len(clean) - 1)


def plot_prompt_trajectory(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str,
    include_bayesian: bool = True,
):
    """Prompt trajectories with equal task_domain weighting.

    LLM rows use the four canonical prompt colors from hspace_alpha.png;
    Bayesian rows use green.
    """
    plot_df = df.dropna(subset=[metric]).copy()
    if metric == "alpha_beta_ratio":
        plot_df[metric] = log_positive_ratio(plot_df[metric]).to_numpy()
        plot_df = plot_df.dropna(subset=[metric])
    if plot_df.empty:
        return
    has_prompt = plot_df["sampling"].notna() & plot_df["h_space"].notna()
    plot_df["prompt"] = (plot_df["sampling"].astype("string") + " " + plot_df["h_space"].astype("string")).where(has_prompt)

    series = []
    bayes = plot_df[plot_df["source"] == "bayesian"]
    if include_bayesian and not bayes.empty:
        xs, ys, sems = task_domain_series(bayes, metric)
        series.append({"x": xs, "y": ys, "y_err": sems, "label": "Bayesian", "color": C_HUMAN, "marker": "o"})

    llm = plot_df[plot_df["source"] == "llm"]
    for prompt in PROMPT_ORDER:
        sub = llm[llm["prompt"] == prompt]
        if sub.empty:
            continue
        xs, ys, sems = task_domain_series(sub, metric)
        series.append({"x": xs, "y": ys, "y_err": sems, "label": prompt,
                       "color": PROMPT_PALETTE[prompt], "marker": "s"})

    if series:
        plot_trajectory_sem(
            series, output_path, ylabel=ylabel, figsize=(3, 2.5), xticks=PREFIX_LENGTHS,
            hline=0.0 if metric == "alpha_beta_ratio" else None,
        )


def plot_sampling_alpha_beta_trajectory(df: pd.DataFrame, sampling: str, output_path: str):
    """α/β trajectories for one sampling frame with equal task_domain weighting."""
    sub = df[df["sampling"] == sampling].dropna(subset=["alpha", "beta"])
    if sub.empty:
        return
    series = []
    for metric, label, color, marker in [
        ("alpha", "α (prior weight)", C_ALPHA, "o"),
        ("beta", "β (size principle)", C_BETA, "s"),
    ]:
        xs, ys, sems = task_domain_series(sub, metric)
        series.append({"x": xs, "y": ys, "y_err": sems, "label": label, "color": color, "marker": marker})

    plot_trajectory_sem(
        series, output_path, ylabel="Parameter value", hline=1.0, figsize=(3, 2.5), xticks=PREFIX_LENGTHS,
    )


def plot_trajectory_rate(df: pd.DataFrame, output_path: str, result_key: str = ""):
    """Trajectory rate grouped by canonical prompt and stimulus kind."""
    # trajectory_rate is constant across n_examples within a condition; dedupe before aggregating
    # so each (task, stimulus, domain, h_space, sampling) contributes one value.
    conditions = df.drop_duplicates(subset=["task", "stimulus", "domain", "h_space", "sampling"])
    plot_prompt_kind_bar(
        conditions, "trajectory_rate", "Trajectory rate  (ΔKL per step)",
        "Convergence rate by stimulus kind", output_path, result_key,
        axhline=0.0)


KIND_COLORS = {
    "single":     CYCLE[0],
    "rule":       CYCLE[1],
    "similarity": CYCLE[2],
    "other":      CYCLE[3],
    "all":        "black",
}
KIND_MARKERS = {"single": "o", "rule": "s", "similarity": "D", "other": "^", "all": "*"}


def plot_narrowing_aggregate(rows, output_path, result_key: str = ""):
    """Unique generated hypothesis count vs evidence prefix length."""
    buckets: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for kind in (row["stimulus_kind"], "all"):
            buckets[kind][int(row["n_examples"])].append(float(row["n_unique"]))
    if not buckets:
        return

    series = []
    for kind in [*ordered_stimulus_kinds(k for k in buckets if k != "all"), "all"]:
        xs, means, sems = aggregate_sem(buckets[kind])
        series.append({
            "x": xs, "y": means, "y_err": sems, "label": kind,
            "color": KIND_COLORS[kind], "marker": KIND_MARKERS[kind],
        })
    plot_trajectory_sem(
        series, output_path, ylabel="Unique hypotheses",
        title=with_model("Hypothesis set narrowing", result_key), figsize=(3, 2.5),
    )
