"""Typed experiment config loader."""

import os

from configs.settings import (
    BayesianConfig,
    Experiment,
    HypothesisEvalConfig,
    HypothesisGenerationConfig,
    PosteriorConfig,
    RUN_PROFILE_NAMES,
    RunProfile,
    ScoringPolicy,
    TaskGrid,
    is_baseline_run,
    load_generation_sampling,
    make_grid,
    resolve_backend_config,
    resolve_model,
)
from utils import PROJECT_ROOT, load_yaml


def _load_run_profile(raw: dict, override: str | None) -> RunProfile:
    name = override or raw.get("RUN_PROFILE", "full")
    if name not in RUN_PROFILE_NAMES:
        raise ValueError(f"Unknown run profile {name!r}; expected one of {RUN_PROFILE_NAMES}")
    if name != "mini":
        return RunProfile()

    configured = raw.get("MINI_CONDITIONS")
    if not isinstance(configured, dict) or not configured:
        raise ValueError("RUN_PROFILE 'mini' requires a non-empty MINI_CONDITIONS mapping")

    keys = set()
    for eval_mode, prompts in configured.items():
        if not isinstance(prompts, list) or not prompts or not all(isinstance(prompt, str) for prompt in prompts):
            raise ValueError(f"MINI_CONDITIONS.{eval_mode} must be a non-empty list of prompt names")
        for prompt in prompts:
            h_space, separator, sampling = prompt.partition("_")
            if not separator or not h_space or not sampling:
                raise ValueError(
                    f"Invalid mini condition prompt {prompt!r}; expected '<h_space>_<sampling>'"
                )
            keys.add((eval_mode, h_space, sampling))
    return RunProfile(condition_keys=frozenset(keys))


def _validate_run_profile(experiment: Experiment) -> None:
    required = experiment.run_profile.condition_keys
    if required is None:
        return

    available_by_task = {task: set() for task in experiment.tasks}
    for condition in make_grid(experiment):
        available_by_task[condition.task].add(
            (condition.eval_mode, condition.h_space, condition.sampling)
        )

    missing = {
        task: sorted(required - available)
        for task, available in available_by_task.items()
        if not required.issubset(available)
    }
    if missing:
        details = "; ".join(
            f"{task}: {condition_keys}" for task, condition_keys in missing.items()
        )
        raise ValueError(f"Run profile 'mini' contains conditions unavailable in the task grid: {details}")


def load_experiment(
    *,
    backend: str | None = None,
    thinking: bool = False,
    model_override: str | None = None,
    run_profile: str | None = None,
) -> Experiment:
    """Parse configs/number_game.yml into the typed Experiment bundle; `thinking` is the CLI flag."""
    raw = load_yaml(os.path.join(PROJECT_ROOT, "configs", "number_game.yml"))
    model = model_override or raw["MODEL"]
    baseline_run = is_baseline_run(model)
    model_info = resolve_model(raw["MODEL"] if baseline_run else model)
    if baseline_run:
        backend = None
        thinking = False

    tasks = {}
    for name in raw["tasks"]:
        block = raw[name]
        tasks[name] = TaskGrid(
            eval_modes=list(block["eval_modes"]),
            h_spaces=list(block["h_spaces"]),
            samplings=list(block["samplings"]),
            domains=list(block["domains"]),
            featured_stimuli=list(block["featured_stimuli"]) if block.get("featured_stimuli") else None,
        )

    scoring = ScoringPolicy(
        max_examples=raw["SCORING"]["max_examples"],
        trajectory_prefix_horizon=raw["SCORING"]["trajectory_prefix_horizon"],
        max_retries=raw["SCORING"]["max_retries"],
    )
    bayesian = BayesianConfig(
        lam=raw["BAYESIAN"]["lam"],
        sigma=raw["BAYESIAN"]["sigma"],
        natural_intervals=raw["BAYESIAN"]["natural_intervals"],
    )
    posterior = PosteriorConfig(
        method=raw["POSTERIOR"]["method"],
        n_trials=raw["POSTERIOR"]["n_trials"],
        temperature=raw["POSTERIOR"]["temperature"],
        thinking_mode=raw["POSTERIOR"]["thinking_mode"],
        thinking_budget=raw["POSTERIOR"]["thinking_budget"],
        top_logprobs_local=raw["POSTERIOR"]["top_logprobs_local"],
        top_logprobs_openai=raw["POSTERIOR"]["top_logprobs_openai"],
    )
    hypothesis_eval = HypothesisEvalConfig(
        max_tokens=raw["HYPOTHESIS_EVAL"]["max_tokens"],
        thinking_budget=raw["HYPOTHESIS_EVAL"]["thinking_budget"],
    )
    backend_config = resolve_backend_config(
        model_info,
        backend=backend,
        thinking_settings=raw.get("THINKING", {}),
        posterior=posterior,
        thinking=thinking,
    )
    hypothesis_generation = HypothesisGenerationConfig(
        max_tokens=raw["HYPOTHESIS_GENERATION"]["max_tokens"],
        n_hypotheses=raw["HYPOTHESIS_GENERATION"]["n_hypotheses"],
        thinking_budget=raw["HYPOTHESIS_GENERATION"]["thinking_budget"],
        sampling=load_generation_sampling(model_info.id, backend_config.thinking),
    )

    experiment = Experiment(
        tasks=tasks,
        scoring=scoring,
        bayesian=bayesian,
        posterior=posterior,
        hypothesis_eval=hypothesis_eval,
        hypothesis_generation=hypothesis_generation,
        backend=backend_config,
        model=model,
        seed=raw["SEED"],
        overwrite=raw["OVERWRITE"],
        concurrency=raw["CONCURRENCY"],
        run_profile=_load_run_profile(raw, run_profile),
    )
    _validate_run_profile(experiment)
    return experiment
