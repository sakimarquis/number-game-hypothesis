"""Analysis cache reads: condition trajectories and memoized Bayesian baselines.

Pure I/O + glue. No plotting; baseline plotting lives in analysis/bayesian.py.
"""

import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from configs.layout import bayesian_cache_dir, bayesian_cache_paths, condition_data_path
from configs.settings import BayesianConfig, Experiment, ExperimentCondition
from analysis.model import NumberGameModel
from scoring.cache import load_prefix_cache
from tasks.stimuli import TASK_RULE_SPACES, TaskContext, TaskData, build_task_context, make_task
from utils import load_json, save_json

BAYESIAN_CACHE_SCHEMA = 1


@dataclass(frozen=True)
class BayesianBaseline:
    """One task/domain context with its Bayesian model and cached predictions."""
    ctx: TaskContext
    model: NumberGameModel
    predictions: dict[str, np.ndarray]
    rule_posteriors: dict[str, dict[str, float]]  # {stimulus: {rule description | "other": mass}}


def bayesian_baseline(experiment: Experiment, task: str, domain: int) -> BayesianBaseline:
    """Bayesian baseline for one task/domain, memoized per run.

    Loads cached predictions whose identity matches the live experiment; missing or stale baselines are rebuilt and saved.
    """
    return _bayesian_baseline(task, domain, experiment.bayesian, bayesian_cache_dir(task, domain))


def bayesian_cache_identity(config: BayesianConfig, task: str, domain: int) -> dict:
    """Describe every configured input that changes one Bayesian baseline."""
    preset, transforms = TASK_RULE_SPACES[task]
    return {
        "schema": BAYESIAN_CACHE_SCHEMA,
        "task": task,
        "domain": domain,
        "bayesian": asdict(config),
        "task_config": {
            "base_preset": preset,
            "transforms": list(transforms),
            "stimuli": [asdict(stimulus) for stimulus in build_task_context(task, domain).task.stimuli],
        },
    }


@lru_cache(maxsize=None)
def _bayesian_baseline(task: str, domain: int, config: BayesianConfig, cache_dir: str) -> BayesianBaseline:
    # cache_dir keys the memo so a different results root never reuses another root's baseline.
    ctx = build_task_context(task, domain)
    model = NumberGameModel(ctx.rules, ctx.intervals, lam=config.lam, sigma=config.sigma, domain=domain)
    paths = bayesian_cache_paths(task, domain)
    identity = bayesian_cache_identity(config, task, domain)
    # The identity file is written last, so a matching identity implies complete prediction files.
    if os.path.exists(paths["identity"]) and load_json(paths["identity"]) == identity:
        preds = {k: np.array(v) for k, v in load_json(paths["predictions"]).items()}
        rule_posteriors = load_json(paths["rule_posteriors"])
        print(f"  [{task} d{domain}] Loaded cached Bayesian predictions")
    else:
        preds = {c.name: model.predict(c.examples).cpu().numpy() for c in ctx.task.stimuli}
        rule_posteriors = {c.name: model.rule_posteriors(c.examples) for c in ctx.task.stimuli}
        save_json(paths["predictions"], preds)
        save_json(paths["rule_posteriors"], rule_posteriors)
        save_json(paths["identity"], identity)
        print(f"  [{task} d{domain}] Computed and saved Bayesian predictions")
    return BayesianBaseline(ctx, model, preds, rule_posteriors)


def reconstruct_trajectory(cache: dict[tuple[int, ...], object], task_data: TaskData,
                           stimulus_set_names: list[str] | None = None) -> dict:
    """Rebuild per-stimulus trajectory from a flat prefix cache.

    Returns {stimulus_name: {"n_examples": [...], "results": [step1, ...]}}.
    Stimuli with no cached prefixes are omitted. Trajectories stop at the
    first missing prefix (handles max_examples cutoff). ``stimulus_set_names``
    restricts reconstruction to the exact scored full input sets when a prefix
    cache is partial, preventing shared-prefix ghost trajectories.
    """
    out = {}
    stimuli = (
        task_data.stimuli if stimulus_set_names is None
        else [task_data.stimulus_map[n] for n in stimulus_set_names]
    )
    for stimulus in stimuli:
        steps, ns = [], []
        for k in range(len(stimulus.examples)):
            p = tuple(stimulus.examples[: k + 1])
            if p not in cache:
                break
            steps.append(cache[p])
            ns.append(len(p))
        if steps:
            out[stimulus.name] = {"n_examples": ns, "results": steps}
    return out


def trajectory_steps(trajectory: dict | None, stimulus: str) -> dict[int, object]:
    """Map one stimulus's trajectory to {n_examples: result}; {} when absent."""
    if not trajectory or stimulus not in trajectory:
        return {}
    return dict(zip(trajectory[stimulus]["n_examples"], trajectory[stimulus]["results"]))


def condition_analysis_path(condition: ExperimentCondition, result_key: str,
                            *, results_root: str | None = None) -> str:
    """Analysis consumes structured generations; scoring retains its raw file."""
    name = "hypothesis_generation_structured" if condition.eval_mode == "hypothesis_generation" else condition.eval_mode
    return condition_data_path(condition, result_key, name, results_root=results_root)


@lru_cache(maxsize=None)
def _load_condition_trajectory_cached(data_path: str, mtime_ns: int, task: str) -> dict:
    """Cache trajectory reconstruction for one immutable condition payload version."""
    raw = load_json(data_path)
    task_data = make_task(task)
    trajectory = reconstruct_trajectory(
        load_prefix_cache(raw), task_data,
        stimulus_set_names=raw["experiment_condition"].get("stimulus_set_names"),
    )
    if raw.get("cache_identity", {}).get("mode_config", {}).get("method") == "text_trials":
        # Each text-trial vector is a frequency over that prefix's valid vectors, which can be fewer than n_trials.
        trials = {tuple(json.loads(k)): v["valid_vectors"] for k, v in raw["posterior_diagnostics"]["per_prefix"].items()}
        for name, ev in trajectory.items():
            examples = task_data.stimulus_map[name].examples
            ev["trials"] = [trials[tuple(examples[:n])] for n in ev["n_examples"]]
    return trajectory


def load_condition_trajectory(condition: ExperimentCondition, result_key: str,
                              *, results_root: str | None = None, log_space: bool = False) -> dict | None:
    """Load one condition's trajectory, reconstructed from the cached condition payload.

    Returns {stimulus_name: {"n_examples": [...], "results": [step1, ...]}} or None; text-trial
    posteriors also carry "trials" (valid vectors per step). `log_space` smooths those results.
    """
    data_path = condition_analysis_path(condition, result_key, results_root=results_root)
    if not os.path.exists(data_path):
        return None
    trajectory = _load_condition_trajectory_cached(data_path, os.stat(data_path).st_mtime_ns, condition.task) or None
    if log_space and trajectory:
        # Copy: the cached trajectory keeps raw frequencies for linear metrics.
        trajectory = {
            name: {**ev, "results": [smooth_text_trials(r, n) for r, n in zip(ev["results"], ev["trials"])]}
            if "trials" in ev else ev
            for name, ev in trajectory.items()
        }
    return trajectory


def smooth_text_trials(vector, trials: int | None) -> np.ndarray:
    """Jeffreys-smoothed marginals (k + 0.5) / (n + 1) for a vector of k/n text-trial frequencies.

    Log-space metrics (KL, entropy, JS) use this instead of the raw frequencies, whose zeros only mean
    "not chosen in n trials" and would otherwise be read as impossible. `trials=None` (logits) is unchanged.
    """
    vector = np.asarray(vector, dtype=float)
    return vector if trials is None else (vector * trials + 0.5) / (trials + 1)


def step_trials(ev: dict) -> list[int | None]:
    """Valid text trials behind each step's result; None for every step of a logits or non-posterior result."""
    return ev.get("trials") or [None] * len(ev["results"])


def trajectory_finals(trajectory: dict, eval_mode: str) -> dict:
    """Extract the final (last-prefix) result per stimulus; posterior finals become float arrays."""
    finals = {stimulus_name: ev["results"][-1] for stimulus_name, ev in trajectory.items()}
    if eval_mode == "posterior":
        finals = {stimulus_name: np.asarray(v, dtype=float) for stimulus_name, v in finals.items()}
    return finals


def load_condition_final(condition: ExperimentCondition, result_key: str,
                         *, results_root: str | None = None, log_space: bool = False) -> dict | None:
    """Load one condition's final (last-prefix) result per stimulus, or None."""
    traj = load_condition_trajectory(condition, result_key, results_root=results_root, log_space=log_space)
    return trajectory_finals(traj, condition.eval_mode) if traj else None
