"""Number game factorial experiment runner.

Iterates experiment factors (task, eval_mode, h_space, sampling, domain), with
hypothesis_eval fixed to explicit h_space. ``--model bayesian`` builds the
Bayesian/Human baselines; LLM runs load those cached baselines for analysis and
otherwise score only their own conditions within each task/domain/prompt group.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, replace

from backends import create_backend
from configs.cli import add_common_cli_args, experiment_from_args
from configs.layout import condition_data_path, parse_failure_path, text_posterior_progress_dir
from configs.settings import EVAL_MODES, RUN_PROFILE_NAMES, is_baseline_run, make_grid
from analysis.cache import (
    bayesian_cache_identity,
    load_bayesian,
    load_prefix_cache,
    serialize_prefix_cache,
    task_context,
)
from scoring.eval_modes import (
    TEXT_TRIAL_FORMAT,
    TextTrialTransportFailure,
    posterior_vector_max_tokens,
    text_trial_messages,
    valid_cached_result,
)
from scoring.run_condition import collect_unique_prefixes, score_condition
from scoring.text_trial_progress import remove_text_trial_progress
from tasks.prompt_space import build_prompt_space_spec
from utils import load_json, seed_everything, save_json, group_by


ALL_SELECTOR = "all"
PROMPT_VERSION = "number_game_prompt_v1"
# Online checkpointed text-trial runs keep this many contexts in flight so one
# context's slow tail does not idle the backend's global request semaphore.
CHECKPOINTED_CONTEXT_WORKERS = 2


@dataclass(frozen=True)
class RunSelectors:
    task: str | None = None
    domain: int | None = None
    prompt: str | None = None
    eval_mode: str | None = None


def _selector(value: str | None, cast=None):
    if value is None or value == ALL_SELECTOR:
        return None
    return cast(value) if cast is not None else value


def _apply_posterior_method(experiment, requested: str | None):
    """Resolve the posterior method: CLI override, else text_trials without logprobs, else YAML."""
    if is_baseline_run(experiment.model):
        return experiment
    supports_logprobs = experiment.backend.supports_logprobs
    method = requested or (experiment.posterior.method if supports_logprobs else "text_trials")
    if method == "logits" and not supports_logprobs:
        raise RuntimeError(
            "POSTERIOR.method='logits' requires backend logprob support; "
            "use --posterior-method text_trials with --allow-text-posterior."
        )
    return replace(experiment, posterior=replace(experiment.posterior, method=method))


def _select_conditions(conditions, selectors: RunSelectors):
    selected = [
        condition for condition in conditions
        if (selectors.task is None or condition.task == selectors.task)
        and (selectors.domain is None or condition.domain == selectors.domain)
        and (selectors.prompt is None or condition.prompt == selectors.prompt)
        and (selectors.eval_mode is None or condition.eval_mode == selectors.eval_mode)
    ]
    if not selected:
        raise ValueError(f"No experiment conditions match selectors: {selectors}")
    return selected


def _conditions_for_run(experiment, selectors: RunSelectors):
    """Build the default grid, or an ad hoc eval-mode grid for explicit CLI runs."""
    if selectors.eval_mode is not None:
        tasks = {
            task_name: replace(factors, eval_modes=[selectors.eval_mode])
            for task_name, factors in experiment.tasks.items()
            if selectors.task is None or task_name == selectors.task
        }
        experiment = replace(experiment, tasks=tasks)
    conditions = [
        condition for condition in make_grid(experiment)
        if experiment.run_profile.includes(condition)
    ]
    return _select_conditions(conditions, selectors)


def _load_task_contexts(experiment, conditions):
    """Build task contexts and require their precomputed Bayesian baselines."""
    contexts = {}
    for (task_name, domain), _ in group_by(conditions, lambda c: c.context_key):
        preds, hyp_preds = load_bayesian(task_name, domain, bayesian_cache_identity(experiment, task_name, domain))
        if not preds or not hyp_preds:
            raise RuntimeError(
                f"Missing Bayesian baseline for {task_name} d{domain}. "
                "Run `python run_number_game.py --model bayesian` first."
            )
        print(f"  [{task_name} d{domain}] Loaded cached Bayesian predictions")
        contexts[(task_name, domain)] = task_context(experiment, task_name, domain)
    return contexts


def _prompt_semantics_fingerprint(ctx, condition, max_examples: int) -> dict:
    """Hash the canonical task and prompt-space inputs for one raw cache."""
    prefixes = collect_unique_prefixes(ctx.task.stimuli, max_examples)
    prompt_spaces = []
    for prefix in prefixes:
        spec = build_prompt_space_spec(ctx, prefix)
        prompt_spaces.append({
            "prefix": prefix,
            "labels": spec.labels,
            "intervals": spec.intervals,
            "grouped_rule_families": spec.rule_families,
        })
    semantics = {
        "version": PROMPT_VERSION,
        "condition": condition.metadata(),
        "stimuli": [
            {"name": stimulus.name, "examples": list(stimulus.examples), "kind": stimulus.kind}
            for stimulus in ctx.task.stimuli
        ],
        "rules": [rule.description for rule in ctx.rules],
        "prompt_spaces": prompt_spaces,
    }
    encoded = json.dumps(semantics, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "version": PROMPT_VERSION,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "stimulus_count": len(semantics["stimuli"]),
        "prefix_count": len(prompt_spaces),
    }


def _raw_cache_identity(experiment, condition, ctx) -> dict:
    """Return the semantic runtime inputs that produced one raw eval-mode cache.

    Execution controls (retries, concurrency, run profile, selectors, overwrite
    policy, Batch vs online) are deliberately excluded.
    """
    config = experiment.backend
    binding = config.binding
    mode_config = condition.mode_config
    if condition.eval_mode != "posterior":
        mode_identity = asdict(mode_config)
    elif mode_config.method == "logits":
        mode_identity = {
            "method": "logits",
            "thinking_mode": mode_config.thinking_mode,
            "thinking_budget": mode_config.thinking_budget,
            "top_logprobs": config.top_logprobs,
        }
    else:
        mode_identity = {
            "method": "text_trials",
            "text_format": TEXT_TRIAL_FORMAT,
            "n_trials": mode_config.n_trials,
            "temperature": mode_config.temperature,
            "thinking_budget": mode_config.thinking_budget,
        }
    return {
        "model_id": config.model_info.id,
        "backend": config.backend,
        "thinking": config.thinking,
        "reasoning_effort": config.reasoning_effort,
        "renderer": binding.thinking_renderer if config.enable_thinking else binding.renderer,
        "thinking_protocol": binding.thinking_protocol if config.thinking else None,
        "seed": experiment.seed,
        "scoring": {"max_examples": experiment.scoring.max_examples},
        "prompt_semantics": _prompt_semantics_fingerprint(ctx, condition, experiment.scoring.max_examples),
        "mode_config": mode_identity,
    }


def _raw_cache_problem(raw: dict, experiment, condition, ctx) -> str | None:
    """Why an existing raw cache cannot be reused as-is, or None when it matches."""
    identity = raw.get("cache_identity")
    expected = _raw_cache_identity(experiment, condition, ctx)
    # Extra recorded keys (historical provenance) do not invalidate a cache.
    if not isinstance(identity, dict) or any(identity.get(key) != value for key, value in expected.items()):
        return "incompatible with the requested semantics"
    metadata = raw["experiment_condition"]
    if (
        {key: metadata.get(key) for key in condition.metadata()} != condition.metadata()
        or metadata.get("stimulus_set_names") != [stimulus.name for stimulus in ctx.task.stimuli]
    ):
        return "incompatible with the requested condition"
    cached = load_prefix_cache(raw)
    prefixes = collect_unique_prefixes(ctx.task.stimuli, experiment.scoring.max_examples)
    if set(cached) != {tuple(prefix) for prefix in prefixes} or not all(
        valid_cached_result(condition, cached[tuple(prefix)], build_prompt_space_spec(ctx, prefix))
        for prefix in prefixes
    ):
        return "incomplete or malformed"
    return None


def _load_matching_cache(experiment, condition, ctx):
    """Return the reusable raw cache, None when absent or overwriting, or fail closed."""
    if experiment.overwrite:
        return None
    data_path = condition_data_path(condition, experiment.backend.result_key, condition.eval_mode)
    raw = load_json(data_path)
    if raw is None:
        return None
    problem = _raw_cache_problem(raw, experiment, condition, ctx)
    if problem is not None:
        raise RuntimeError(
            f"Existing raw cache is {problem} and was left unchanged: {data_path}. "
            "Use --refresh-raw-cache to explicitly replace it."
        )
    return raw


def _estimate_message_tokens(messages: list[dict]) -> int:
    """Approximate provider input tokens from message characters."""
    characters = sum(len(str(message["role"])) + len(str(message["content"])) for message in messages)
    return (characters + 3) // 4


def _preflight_text_posterior(experiment, backend, conditions, contexts, *, allow_text_posterior: bool) -> None:
    """Print the pending text-trial request/token estimate and require explicit opt-in."""
    text_conditions = [
        condition for condition in conditions
        if condition.eval_mode == "posterior"
        and condition.mode_config.method == "text_trials"
        and _load_matching_cache(experiment, condition, contexts[condition.context_key]) is None
    ]
    if not text_conditions:
        return

    totals = dict.fromkeys(("prefixes", "requests", "judgments", "batch_jobs", "input_tokens", "output_tokens"), 0)
    print("\nText posterior preflight (explicit opt-in required):")
    for (task, domain), context_conditions in group_by(text_conditions, lambda condition: condition.context_key):
        ctx = contexts[(task, domain)]
        prefixes = collect_unique_prefixes(ctx.task.stimuli, experiment.scoring.max_examples)
        n_trials = context_conditions[0].mode_config.n_trials
        vector_requests = len(context_conditions) * len(prefixes) * n_trials
        thinking_headroom = (
            max(context_conditions[0].mode_config.thinking_budget, backend.minimum_thinking_headroom_tokens)
            if experiment.backend.thinking
            else 0
        )
        row = {
            "prefixes": len(context_conditions) * len(prefixes),
            "requests": vector_requests,
            "judgments": vector_requests * domain,
            "batch_jobs": len(context_conditions) if backend.batch_api_enabled else 0,
            "input_tokens": n_trials * sum(
                _estimate_message_tokens(text_trial_messages(condition, prefix, ctx, backend.text_trial_encoding))
                for condition in context_conditions
                for prefix in prefixes
            ),
            "output_tokens": vector_requests * backend.completion_token_limit(
                posterior_vector_max_tokens(domain), thinking_headroom,
            ),
        }
        print(
            f"  {task} d{domain}: {len(context_conditions)} conditions, "
            f"{len(prefixes)} unique prefixes/condition, "
            f"{row['requests']:,} logical samples (vector requests), "
            f"{row['judgments']:,} target judgments, "
            f"{row['batch_jobs']:,} initial Batch jobs, input ~= {row['input_tokens']:,} tokens, "
            f"generated <= {row['output_tokens']:,} tokens"
        )
        for key, value in row.items():
            totals[key] += value

    print(
        f"  Total: {totals['prefixes']:,} condition-prefixes, "
        f"{totals['requests']:,} logical samples (vector requests), "
        f"{totals['judgments']:,} target judgments, "
        f"{totals['batch_jobs']:,} initial Batch jobs, input ~= {totals['input_tokens']:,} tokens, "
        f"generated <= {totals['output_tokens']:,} tokens "
        "(provider retries excluded)"
    )
    if not allow_text_posterior:
        raise RuntimeError(
            "Text posterior is disabled by default. Review the preflight estimate, "
            "then re-run with --allow-text-posterior."
        )


async def _run_eval_conditions(eval_conditions, ctx, backend, experiment):
    """Load or atomically publish each eval-mode cache for one prompt group (or single Batch condition)."""
    print(f"\n  Prompt: {eval_conditions[0].prompt}")
    result_key = experiment.backend.result_key

    for condition in eval_conditions:
        data_path = condition_data_path(condition, result_key, condition.eval_mode)
        identity = _raw_cache_identity(experiment, condition, ctx)
        progress_path = (
            text_posterior_progress_dir(condition, result_key)
            if condition.eval_mode == "posterior"
            and condition.mode_config.method == "text_trials"
            and backend.checkpoints_text_trials
            else None
        )
        if _load_matching_cache(experiment, condition, ctx) is not None:
            print(f"    [{condition.eval_mode}] Found existing prefix_cache, skipping")
            if progress_path is not None:
                remove_text_trial_progress(progress_path, expected_identity=identity)
            await backend.release_batch_sidecars(condition)
            continue

        if progress_path is not None and experiment.overwrite:
            remove_text_trial_progress(progress_path)
        print(f"  [{condition.eval_mode}] Scoring {len(ctx.task.stimuli)} stimulus sets, "
              f"max_examples={experiment.scoring.max_examples}...")
        cache, prefix_diagnostics = await score_condition(
            backend, condition, ctx,
            max_examples=experiment.scoring.max_examples,
            max_retries=experiment.scoring.max_retries,
            failure_log_path=parse_failure_path(condition, result_key),
            progress_path=progress_path,
            progress_identity=identity if progress_path is not None else None,
        )
        payload = {
            "experiment_condition": {
                **condition.metadata(),
                "stimulus_set_names": [stimulus.name for stimulus in ctx.task.stimuli],
            },
            "cache_identity": identity,
            "prefix_cache": serialize_prefix_cache(cache),
        }
        if prefix_diagnostics is not None:
            payload["posterior_diagnostics"] = {
                "text_format": TEXT_TRIAL_FORMAT,
                "per_prefix": serialize_prefix_cache(prefix_diagnostics),
            }
        save_json(data_path, payload)
        if progress_path is not None:
            remove_text_trial_progress(progress_path, expected_identity=identity)
        await backend.release_batch_sidecars(condition)


async def _run_bounded(items, n_workers: int, run_item) -> None:
    """Run items FIFO on at most n_workers tasks; the first failure cancels the rest."""
    queue = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)

    async def worker():
        while not queue.empty():
            await run_item(queue.get_nowait())

    workers = [asyncio.create_task(worker()) for _ in range(min(n_workers, len(items)))]
    try:
        await asyncio.gather(*workers)
    except BaseException:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise


def _print_context_header(task_name: str, domain: int) -> None:
    print(f"\n{'=' * 60}\nContext: {task_name}, domain 1..{domain}\n{'=' * 60}")


async def _run_batch_api_context(conditions, ctx, backend, experiment):
    """Run condition-wide Batch jobs through a bounded FIFO worker pool."""
    ordered = [
        condition
        for _, eval_conditions in group_by(conditions, lambda c: c.prompt_key)
        for condition in eval_conditions
    ]
    await _run_bounded(
        ordered, backend.batch_job_concurrency,
        lambda condition: _run_eval_conditions([condition], ctx, backend, experiment),
    )


async def _run_online_contexts(conditions, contexts, backend, experiment) -> None:
    """Score contexts online; checkpointed text-trial failures are deferred until later contexts finish."""
    deferred = []

    async def run_context(item):
        (task_name, domain), ctx_conditions = item
        _print_context_header(task_name, domain)
        for _, eval_conditions in group_by(ctx_conditions, lambda c: c.prompt_key):
            try:
                await _run_eval_conditions(eval_conditions, contexts[(task_name, domain)], backend, experiment)
            except TextTrialTransportFailure as exc:
                deferred.append(exc)
                print(
                    "    [posterior] transport retries exhausted; "
                    "continuing later conditions with durable progress retained"
                )

    await _run_bounded(
        list(group_by(conditions, lambda c: c.context_key)),
        CHECKPOINTED_CONTEXT_WORKERS if backend.checkpoints_text_trials else 1,
        run_context,
    )
    if len(deferred) == 1:
        raise deferred[0]
    if deferred:
        raise TextTrialTransportFailure(
            f"{len(deferred)} text-trial conditions exhausted transport retries; "
            "later conditions completed and durable progress was retained"
        ) from deferred[0]


async def _run_experiment(experiment, backend, selectors=RunSelectors(), *, allow_text_posterior: bool = False):
    """Score selected raw conditions without importing the analysis stack."""
    conditions = _conditions_for_run(experiment, selectors)
    print(
        f"Experiment grid: {len(conditions)} conditions "
        f"(profile={experiment.run_profile.name}, overwrite={experiment.overwrite})"
    )

    print("\nLoading Bayesian baselines...")
    contexts = _load_task_contexts(experiment, conditions)
    _preflight_text_posterior(experiment, backend, conditions, contexts, allow_text_posterior=allow_text_posterior)

    if backend.batch_api_enabled:
        for (task_name, domain), ctx_conditions in group_by(conditions, lambda c: c.context_key):
            _print_context_header(task_name, domain)
            await _run_batch_api_context(ctx_conditions, contexts[(task_name, domain)], backend, experiment)
    else:
        await _run_online_contexts(conditions, contexts, backend, experiment)


def _analysis_command(experiment) -> list[str]:
    command = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_analysis.py"),
        "--model", experiment.model,
        "--backend", experiment.backend.backend,
        "--include-per-condition",
    ]
    if experiment.backend.thinking:
        command.append("--thinking")
    return command


def main():
    parser = argparse.ArgumentParser(description="Number game factorial experiment")
    add_common_cli_args(parser, model_required=False)
    parser.add_argument("--port", "-p", type=int, default=None,
                        help="Local OpenAI-compatible server port")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Print top_logprobs when binary scoring falls back to 0.5")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Override CONCURRENCY from configs/number_game.yml")
    parser.add_argument("--run-profile", choices=RUN_PROFILE_NAMES, default=None,
                        help="LLM scoring schedule: full canonical grid or request-minimal mini subset.")
    parser.add_argument("--online", action="store_true",
                        help="Use online generateContent instead of the configured Gemini Batch API.")
    parser.add_argument("--allow-text-posterior", action="store_true",
                        help="Confirm the preflighted cost of explicit text-trial posterior scoring.")
    parser.add_argument("--posterior-method", choices=["logits", "text_trials"], default=None,
                        help="Posterior protocol override. Backends without logprobs default to text_trials.")
    parser.add_argument("--refresh-raw-cache", action="store_true",
                        help="Explicitly rescore and replace selected raw caches.")
    parser.add_argument("--task", default=None, metavar="TASK",
                        help="Task selector: Tenenbaum99, Bigelow16, or all.")
    parser.add_argument("--domain", default=None,
                        help="Domain selector, e.g. 100, 200, or all.")
    parser.add_argument("--prompt", default=None,
                        help="Prompt selector as h_space_sampling, e.g. explicit_strong, or all.")
    parser.add_argument("--eval-mode", dest="eval_mode", default=None,
                        choices=[ALL_SELECTOR, *EVAL_MODES],
                        help="Eval-mode selector: posterior, hypothesis_eval, hypothesis_generation, or all.")
    args = parser.parse_args()

    if args.port is not None:
        os.environ["LOCAL_BASE_URL"] = f"http://localhost:{args.port}/v1"
    if args.debug:
        os.environ["DEBUG_BINARY_LOGPROBS"] = "1"

    experiment = experiment_from_args(args, run_profile=args.run_profile)
    if args.task not in (None, ALL_SELECTOR) and args.task not in experiment.tasks:
        parser.error(f"--task must be one of {[ALL_SELECTOR, *experiment.tasks]}")
    experiment = _apply_posterior_method(experiment, args.posterior_method)
    selectors = RunSelectors(
        task=_selector(args.task),
        domain=_selector(args.domain, cast=int),
        prompt=_selector(args.prompt),
        eval_mode=_selector(args.eval_mode),
    )
    if args.refresh_raw_cache:
        experiment = replace(experiment, overwrite=True)
    if args.concurrency is not None:
        experiment = replace(experiment, concurrency=args.concurrency)
    if args.online:
        if experiment.backend.backend != "google_cloud":
            parser.error("--online is only valid with the google_cloud backend")
        experiment = replace(experiment, backend=replace(experiment.backend, execution="online"))
    seed_everything(experiment.seed)
    if is_baseline_run(experiment.model):
        if args.refresh_raw_cache:
            parser.error("--refresh-raw-cache requires an LLM model")
        # Deferred: keeps pandas/matplotlib out of remote-scoring runs.
        from analysis import bayesian

        bayesian.run_baseline_analyses(experiment)
        print("\nDone.")
        return

    backend = create_backend(experiment.backend, concurrency=experiment.concurrency)
    asyncio.run(_run_experiment(experiment, backend, selectors, allow_text_posterior=args.allow_text_posterior))
    command = _analysis_command(experiment)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
