"""Cache I/O: prefix caches, condition trajectories, Bayesian baselines, and task contexts.

Pure I/O + glue. No plotting; baseline plotting lives in analysis/bayesian.py.
"""

import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from configs.layout import bayesian_cache_paths, condition_data_path, human_analysis_cache_path
from configs.settings import Experiment, ExperimentCondition
from analysis.model import NumberGameModel, compute_alpha_beta_ablations, fit_alpha_beta
from tasks.stimuli import TaskContext, TaskData, build_task_context, make_task
from utils import load_json, save_json, group_by


BAYESIAN_CACHE_SCHEMA = 1


def task_context(experiment: Experiment, task: str, domain: int) -> TaskContext:
    """Domain-valid task data and rule space for one configured task/domain."""
    featured = experiment.tasks[task].featured_stimuli
    return build_task_context(
        task, domain, experiment.bayesian.natural_intervals,
        tuple(featured) if featured is not None else None,
    )


_model_cache: dict[tuple, NumberGameModel] = {}


def bayesian_model(experiment: Experiment, task: str, domain: int) -> NumberGameModel:
    """NumberGameModel for one task/domain, memoized across calls in a run."""
    featured = experiment.tasks[task].featured_stimuli
    key = (task, domain, experiment.bayesian, None if featured is None else tuple(featured))
    if key not in _model_cache:
        ctx = task_context(experiment, task, domain)
        _model_cache[key] = NumberGameModel(
            ctx.rules, ctx.intervals, lam=experiment.bayesian.lam, sigma=experiment.bayesian.sigma,
            max_integer=domain,
        )
    return _model_cache[key]


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


def serialize_prefix_cache(cache: dict[tuple[int, ...], object]) -> dict[str, object]:
    """Encode tuple prefix keys for the on-disk JSON payload."""
    return {
        json.dumps(list(prefix), separators=(",", ":")): result
        for prefix, result in cache.items()
    }


def load_prefix_cache(raw: dict) -> dict[tuple[int, ...], object]:
    """Decode the on-disk flat prefix-cache payload."""
    return {tuple(json.loads(k)): v for k, v in raw["prefix_cache"].items()}


def condition_analysis_path(condition: ExperimentCondition, result_key: str,
                            *, results_root: str | None = None) -> str:
    """Analysis consumes structured generations; scoring retains its raw file."""
    name = "hypothesis_generation_structured" if condition.eval_mode == "hypothesis_generation" else condition.eval_mode
    return condition_data_path(condition, result_key, name, results_root=results_root)


@lru_cache(maxsize=None)
def _load_condition_trajectory_cached(data_path: str, mtime_ns: int, task: str) -> dict:
    """Cache trajectory reconstruction for one immutable condition payload version."""
    raw = load_json(data_path)
    return reconstruct_trajectory(
        load_prefix_cache(raw), make_task(task),
        stimulus_set_names=raw["experiment_condition"].get("stimulus_set_names"),
    )


def load_condition_trajectory(condition: ExperimentCondition, result_key: str,
                              *, results_root: str | None = None) -> dict | None:
    """Load one condition's trajectory, reconstructed from the cached condition payload.

    Returns {stimulus_name: {"n_examples": [...], "results": [step1, ...]}} or None.
    """
    data_path = condition_analysis_path(condition, result_key, results_root=results_root)
    if not os.path.exists(data_path):
        return None
    return _load_condition_trajectory_cached(data_path, os.stat(data_path).st_mtime_ns, condition.task) or None


def trajectory_finals(trajectory: dict, eval_mode: str) -> dict:
    """Extract the final (last-prefix) result per stimulus; posterior finals become float arrays."""
    finals = {stimulus_name: ev["results"][-1] for stimulus_name, ev in trajectory.items()}
    if eval_mode == "posterior":
        finals = {stimulus_name: np.asarray(v, dtype=float) for stimulus_name, v in finals.items()}
    return finals


def load_condition_final(condition: ExperimentCondition, result_key: str,
                         *, results_root: str | None = None) -> dict | None:
    """Load one condition's final (last-prefix) result per stimulus, or None."""
    traj = load_condition_trajectory(condition, result_key, results_root=results_root)
    return trajectory_finals(traj, condition.eval_mode) if traj else None


@lru_cache(maxsize=None)
def _load_bayesian_cached(preds_path: str, preds_mtime_ns: int, hyps_path: str, hyps_mtime_ns: int) -> tuple[dict, dict]:
    """Cache Bayesian baseline loads for one immutable file pair version."""
    preds = {k: np.array(v) for k, v in load_json(preds_path).items()}
    return preds, load_json(hyps_path)


def bayesian_cache_identity(experiment: Experiment, task: str, domain: int) -> dict:
    """Describe every configured input that changes one Bayesian baseline."""
    task_data = task_context(experiment, task, domain).task
    return {
        "schema": BAYESIAN_CACHE_SCHEMA,
        "task": task,
        "domain": domain,
        "bayesian": asdict(experiment.bayesian),
        "task_config": {
            "base_preset": task_data.base_preset,
            "transforms": list(task_data.transforms),
            "stimuli": [asdict(stimulus) for stimulus in task_data.stimuli],
        },
    }


def load_bayesian(task: str, domain: int, expected_identity: dict,
                  *, results_root: str | None = None) -> tuple[dict, dict]:
    """Load Bayesian predictions and hypothesis posteriors for (task, domain) when their identity matches.

    Returns (preds: {stimulus_name: np.ndarray}, hyp_preds: {stimulus_name: {hyp: float}}), or empty dicts.
    """
    paths = bayesian_cache_paths(task, domain, results_root=results_root)
    if not os.path.exists(paths["predictions"]) or not os.path.exists(paths["hypotheses"]):
        return {}, {}
    if load_json(paths["identity"]) != expected_identity:
        return {}, {}
    return _load_bayesian_cached(
        paths["predictions"], os.stat(paths["predictions"]).st_mtime_ns,
        paths["hypotheses"], os.stat(paths["hypotheses"]).st_mtime_ns,
    )


@dataclass(frozen=True)
class BayesianBaseline:
    """One task/domain context with its Bayesian model and cached predictions."""
    ctx: TaskContext
    model: NumberGameModel
    predictions: dict[str, np.ndarray]
    hypotheses: dict[str, dict[str, float]]


def build_bayesian(experiment: Experiment, conditions, *, recompute: bool = False) -> dict[tuple[str, int], BayesianBaseline]:
    """Build Bayesian baselines for all unique (task, domain) pairs.

    Loads predictions only when their identity matches the live experiment.
    With ``recompute=True``, missing or stale baselines are rebuilt and saved;
    otherwise they fail closed so LLM runs cannot silently change baselines.
    """
    baselines = {}
    for (task_name, domain), _ in group_by(conditions, lambda c: c.context_key):
        ctx = task_context(experiment, task_name, domain)
        model = bayesian_model(experiment, task_name, domain)
        identity = bayesian_cache_identity(experiment, task_name, domain)
        preds, hyp_preds = load_bayesian(task_name, domain, identity)
        if preds and hyp_preds:
            print(f"  [{task_name} d{domain}] Loaded cached Bayesian predictions")
        elif not recompute:
            raise FileNotFoundError(
                f"Missing or stale Bayesian baseline for {task_name} d{domain}. "
                "Run `python run_number_game.py --model bayesian` first."
            )
        else:
            preds = {c.name: model.predict(c.examples).cpu().numpy() for c in ctx.task.stimuli}
            hyp_preds = {c.name: model.math_rule_posteriors(c.examples) for c in ctx.task.stimuli}
            paths = bayesian_cache_paths(task_name, domain)
            save_json(paths["predictions"], preds)
            save_json(paths["hypotheses"], hyp_preds)
            save_json(paths["identity"], identity)
            print(f"  [{task_name} d{domain}] Computed and saved Bayesian predictions")
        baselines[(task_name, domain)] = BayesianBaseline(ctx, model, preds, hyp_preds)
    return baselines


def build_human_analysis(baselines: dict[tuple[str, int], BayesianBaseline]) -> dict[tuple[str, int], dict]:
    """Fit human α-β per (task, domain) and cache the fits under results/human/analysis/.

    NaN targets (Tenenbaum99 sparse sampling) are masked inside fit_alpha_beta.
    """
    fits = {}
    for (task_name, domain), baseline in baselines.items():
        task_data = baseline.ctx.task
        if not task_data.human_preds:
            continue
        fit = fit_alpha_beta([(baseline.model, task_data.human_preds, task_data.stimuli)])
        fit = {
            **fit,
            **compute_alpha_beta_ablations(
                baseline.model, task_data.human_preds, task_data.stimuli, fit["best_alpha"], fit["best_beta"],
            ),
        }
        save_json(human_analysis_cache_path(task_name, domain), {
            key: fit[key] for key in ("best_alpha", "best_beta", "best_loss", "n_stimuli", "ablation_mse")
        })
        print(f"  [{task_name} d{domain}] Human α-β fit: "
              f"α={fit['best_alpha']:.2f} β={fit['best_beta']:.2f} "
              f"(MSE={fit['best_loss']:.6f})")
        fits[(task_name, domain)] = fit
    return fits


def load_human_alpha_beta_fit(task_name: str, domain: int) -> dict | None:
    """Load cached model-agnostic human alpha-beta fit for one task/domain."""
    return load_json(human_analysis_cache_path(task_name, domain))
