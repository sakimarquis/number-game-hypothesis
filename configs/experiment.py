"""Typed experiment config loader."""

import os

from configs.settings import (
    BayesianConfig,
    Experiment,
    HypothesisEvalConfig,
    HypothesisGenerationConfig,
    MINI_CONDITIONS,
    PosteriorConfig,
    RUN_PROFILES,
    ScoringPolicy,
    TaskGrid,
    is_baseline_run,
    load_generation_sampling,
    make_grid,
    resolve_backend_config,
    resolve_model,
)
from utils import PROJECT_ROOT, load_yaml


def _validate_run_profile(experiment: Experiment) -> None:
    if experiment.run_profile not in RUN_PROFILES:
        raise ValueError(f"Unknown run profile {experiment.run_profile!r}; expected one of {RUN_PROFILES}")
    if experiment.run_profile != "mini":
        return

    available_by_task = {task: set() for task in experiment.tasks}
    for condition in make_grid(experiment):
        available_by_task[condition.task].add(
            (condition.eval_mode, condition.h_space, condition.sampling)
        )

    missing = {
        task: sorted(MINI_CONDITIONS - available)
        for task, available in available_by_task.items()
        if not MINI_CONDITIONS <= available
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

    scoring = ScoringPolicy(**raw["SCORING"])
    bayesian = BayesianConfig(**raw["BAYESIAN"])
    posterior = PosteriorConfig(**raw["POSTERIOR"])
    backend_config = resolve_backend_config(
        model_info,
        backend=backend,
        thinking_settings=raw["THINKING"],
        thinking=thinking,
    )
    generation_tokens = dict(raw["HYPOTHESIS_GENERATION"])
    if backend_config.thinking and model_info.thinking_generation_tokens:
        generation_tokens.update(model_info.thinking_generation_tokens)

    experiment = Experiment(
        tasks=tasks,
        scoring=scoring,
        bayesian=bayesian,
        posterior=posterior,
        hypothesis_eval=HypothesisEvalConfig(**raw["HYPOTHESIS_EVAL"]),
        hypothesis_generation=HypothesisGenerationConfig(
            **generation_tokens,
            sampling=load_generation_sampling(model_info.id, backend_config.thinking),
        ),
        backend=backend_config,
        model=model,
        seed=raw["SEED"],
        overwrite=raw["OVERWRITE"],
        concurrency=raw["CONCURRENCY"],
        run_profile=run_profile or raw["RUN_PROFILE"],
    )
    _validate_run_profile(experiment)
    return experiment
