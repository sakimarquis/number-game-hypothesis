"""Trajectory analysis: posterior metrics over evidence-prefix length.

Computes and plots KL/entropy/effective-support/size-principle trajectories from the
posterior prefix caches reconstructed by analysis.cache.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from analysis.alpha_beta_scope import PREFIX_LENGTHS
from analysis.cache import smooth_text_trials, step_trials
from analysis.metrics import kl_divergence, rated_targets, shannon_entropy
from analysis.model import NumberGameModel, fit_beta
from analysis.plotter import (
    set_mpl, aggregate_sem, plot_trajectory_sem, with_model,
    task_domain_series,
    ordered_stimulus_kinds, plot_prompt_kind_bar,
    CYCLE, C_HUMAN, C_BETA,
    PROMPT_ORDER, PROMPT_PALETTE, prompt_label,
)
from tasks.stimuli import StimulusSet

set_mpl(size=8)


def compute_effective_support(preds: np.ndarray, threshold: float = 0.5) -> int:
    """Number of targets with p(yes) > threshold."""
    return int((preds > threshold).sum())


def size_principle_curves(
    trajectories: dict[str, dict],
    stimuli: list,
    label: str,
    bayesian_trajectories: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Compute size-principle metrics per prefix step across a trajectory.

    Each prefix step contributes one row, binned by its own prefix length as
    n_examples — so a single size-4 stimulus yields 4 points (lengths 1..4).
    Tenenbaum99 and Bigelow16 go through the same path; asymmetry between
    tasks comes only from how many prefix steps each stimulus carries.

    Args:
        trajectories: {name: {"n_examples": [...], "results": [step_preds, ...]}}
        stimuli: list of StimulusSet objects (for full example lookup)
        label: source identifier ("bayesian", "llm", "human")
        bayesian_trajectories: same shape — if given, KL(bayesian || preds) per step

    Returns:
        DataFrame with columns: stimulus, stimulus_kind, n_examples, entropy, effective_support,
        source, and with `bayesian_trajectories` kl_bayes plus kl_bayes_ceiling: the KL a perfect
        Bayesian would score under the same number of text trials (NaN for logits results).
    """
    stimulus_map = {c.name: c for c in stimuli}
    rows = []
    for name, traj in trajectories.items():
        stimulus = stimulus_map.get(name)
        if stimulus is None:
            continue
        bay_steps = (bayesian_trajectories or {}).get(name)
        for i, (n, step, trials) in enumerate(zip(traj["n_examples"], traj["results"], step_trials(traj))):
            preds = np.asarray(step, dtype=float)
            log_preds = smooth_text_trials(preds, trials)  # entropy and KL only; support stays raw
            row = {
                "stimulus": name,
                "stimulus_kind": stimulus.kind,
                "n_examples": n,
                "entropy": shannon_entropy(log_preds, empty=0.0),
                "effective_support": compute_effective_support(preds),
                "source": label,
            }
            if bay_steps is not None and i < len(bay_steps["results"]):
                bay_preds = np.asarray(bay_steps["results"][i], dtype=float)
                row["kl_bayes"] = kl_divergence(bay_preds, log_preds)
                row["kl_bayes_ceiling"] = text_trial_kl_ceiling(bay_preds, trials) if trials else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def human_kl_curves(
    trajectories: dict[str, dict],
    stimuli: list,
    human_preds: dict[str, np.ndarray],
) -> pd.DataFrame:
    """KL from full human ratings to each LLM prefix prediction."""
    stimulus_map = {c.name: c for c in stimuli}
    rows = []
    for name, traj in trajectories.items():
        stimulus = stimulus_map.get(name)
        if stimulus is None or name not in human_preds:
            continue
        human = np.asarray(human_preds[name], dtype=float)
        mask = rated_targets(human)
        if mask is None:
            continue
        for n, step, trials in zip(traj["n_examples"], traj["results"], step_trials(traj)):
            preds = smooth_text_trials(step, trials)
            rows.append({
                "stimulus": name,
                "stimulus_kind": stimulus.kind,
                "n_examples": n,
                "kl_human": kl_divergence(human[mask], preds[mask]),
                "source": "llm",
            })
    return pd.DataFrame(rows)


def clip_to_prefix_horizon(
    stimuli: list[StimulusSet],
    trajectories: dict[str, dict],
) -> tuple[list[StimulusSet], dict[str, dict]]:
    """Restrict to stimuli with at least N examples, truncated to their first N.

    N is the longest analyzed prefix length capped by the longest stimulus, so
    every prefix bin averages over the same stimuli. Trajectories keep only
    those stimuli and prefixes of at most N examples.
    """
    n = min(max(PREFIX_LENGTHS), max(len(c.examples) for c in stimuli))
    clipped = [c.prefix(n) for c in stimuli if len(c.examples) >= n]
    names = {c.name for c in clipped}
    subset = {}
    for name, ev in trajectories.items():
        if name not in names:
            continue
        keep = [i for i, ne in enumerate(ev["n_examples"]) if ne <= n]
        if keep:
            subset[name] = {key: [ev[key][i] for i in keep] for key in ("n_examples", "results", "trials") if key in ev}
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


def text_trial_kl_ceiling(p: np.ndarray, trials: int, draws: int = 200) -> float:
    """Expected smoothed KL(p || p remeasured with `trials` text trials): a perfect Bayesian's score.

    `p` is the Bayesian per-target P(yes); the fixed seed keeps analysis CSVs reproducible.
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    counts = np.random.default_rng(0).binomial(trials, p, size=(draws, p.size))  # [draws, domain]
    return float(np.mean([kl_divergence(p, smooth_text_trials(row / trials, trials)) for row in counts]))


def compute_trajectory_beta(
    model: NumberGameModel,
    priors,
    stimuli: list[StimulusSet],
    trajectories: dict[str, dict],
) -> dict:
    """Fit the size principle β under fixed `priors` at each evidence prefix length across all stimuli.

    For each prefix n, builds truncated stimuli (examples[:n]) and collects the
    corresponding LLM predictions, then runs fit_beta. Pass clipped
    inputs (clip_to_prefix_horizon) so every prefix averages the same stimuli.

    Returns {"n_examples": [...], "beta": [...], "mse": [...], "n_stimuli": [...]}
    """
    stimulus_map = {c.name: c for c in stimuli}
    out = {"n_examples": [], "beta": [], "mse": [], "n_stimuli": []}
    for n in sorted({n for ev in trajectories.values() for n in ev["n_examples"]}):
        llm_preds, trunc_stimuli = {}, []
        for stimulus_name, ev in trajectories.items():
            if n not in ev["n_examples"]:
                continue
            stimulus = stimulus_map[stimulus_name]
            llm_preds[stimulus_name] = np.asarray(ev["results"][ev["n_examples"].index(n)])
            trunc_stimuli.append(stimulus.prefix(n))

        fit = fit_beta([(model, priors, llm_preds, trunc_stimuli)])
        if fit is None:
            continue
        out["n_examples"].append(n)
        out["beta"].append(fit["beta"])
        out["mse"].append(fit["mse"])
        out["n_stimuli"].append(fit["n_stimuli"])
    return out


def plot_trajectory_beta(trajectory: dict, output_path: str, result_key: str = ""):
    """Size principle β vs evidence prefix length."""
    if not trajectory["n_examples"]:
        return

    plot_trajectory_sem(
        [{"x": trajectory["n_examples"], "y": trajectory["beta"], "y_err": [0.0] * len(trajectory["beta"]),
          "label": "Size principle (β)", "color": C_BETA, "marker": "s"}],
        output_path, ylabel="Size principle (β)",
        title=with_model("Size-principle Trajectory", result_key), hline=1.0,
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

    LLM rows use the four canonical prompt colors of the fit-scope bars;
    Bayesian rows use green.
    """
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return
    plot_df["prompt"] = prompt_label(plot_df)  # NaN for Bayesian rows without a prompt

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
        plot_trajectory_sem(series, output_path, ylabel=ylabel, figsize=(3, 2.5), xticks=PREFIX_LENGTHS)


def plot_trajectory_rate(df: pd.DataFrame, output_path: str, result_key: str = ""):
    """Trajectory rate of LLM prefix KL, one value per condition stimulus, grouped by canonical prompt and stimulus kind."""
    keys = ["task", "stimulus", "stimulus_kind", "domain", "h_space", "sampling"]
    llm = df[df["source"] == "llm"].dropna(subset=["kl_bayes"]).sort_values("n_examples")
    conditions = llm.groupby(keys, as_index=False)["kl_bayes"].agg(compute_trajectory_rate).rename(
        columns={"kl_bayes": "trajectory_rate"})
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
