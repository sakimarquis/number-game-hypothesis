"""Per-condition and per-model analysis stages.

Each aggregate stage writes CSV caches whose file names start with the stage's
registry key, e.g. stage `alpha_beta` writes `alpha_beta_fits.csv`.
"""
from __future__ import annotations

import os
from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd

from analysis.alpha_beta_scope import FULL_SCOPE, PREFIX_LENGTHS, alpha_beta_row, scope_label, stimulus_row, task_group_label
from analysis.bayesian import load_baseline_overlay
from analysis.cache import bayesian_baseline, load_condition_final, load_condition_trajectory, trajectory_finals
from analysis.extension import run_extension, run_hypothesis_extension
from analysis.hypothesis_fit import run_hypothesis_compare, run_hypothesis_fit
from analysis.hypothesis_space import collapse_structured_generation, project_hypothesis_eval, project_result, project_trajectory
from analysis.metrics import (
    HYPOTHESIS_TYPES, compute_hypothesis_generation_metrics, hypothesis_type,
    kl_divergence, rated_targets, safe_correlation, type_proportions,
)
from analysis.model import fit_beta, fit_beta_with_ablations
from analysis.plotter import plot_prompt_fit_scope_bar, plot_prompt_kind_bar, save_csv, with_task_domain
from analysis.prior_pooling import DEFAULT_PROMPT, pooled_prior, pooled_priors
from analysis.condition_plots import (
    plot_all, plot_alpha_beta_all,
    plot_hypothesis_comparison,
    plot_hypothesis_generation_exemplars,
    plot_hypothesis_generation_summary,
    plot_hypothesis_generation_type_summary,
    plot_trajectory_metrics,
)
from analysis.prompt_effect import (
    plot_confidence_weighted_types,
    plot_generalization_width,
    plot_prompt_sensitivity,
    plot_trajectory_rule_mass,
)
from analysis.rt import run_rt_correlation
from analysis.size_principle import run_nested_pair
from analysis.trajectory import (
    bayesian_prefix_trajectories, clip_to_prefix_horizon, compute_effective_support, human_kl_curves, size_principle_curves,
    compute_trajectory_beta,
    plot_narrowing_aggregate, plot_prompt_trajectory, plot_trajectory_beta, plot_trajectory_rate,
)
from configs import layout
from configs.layout import condition_data_path, condition_result_dir, domain_subdir
from configs.settings import Experiment, make_grid
from tasks.hypotheses import aggregate_by_rule_family
from tasks.stimuli import StimulusSet, build_task_context
from utils import group_by


def _conditions(experiment: Experiment, *eval_modes: str):
    return [condition for condition in make_grid(experiment) if condition.eval_mode in eval_modes]


def _row_base(condition, stimulus) -> dict:
    return stimulus_row(condition.task, stimulus, condition.domain, condition.h_space, condition.sampling)


# ==================================================================
# Per-condition stages
# ==================================================================

def run_trajectory_metrics(model, ctx, llm_trajectories, output_dir, result_key):
    """Per-prefix KL, entropy, and support over a fixed stimulus subset."""
    human_preds = ctx.task.human_preds
    clipped_stimuli, clipped_trajectories = clip_to_prefix_horizon(ctx.task.stimuli, llm_trajectories)
    bay_traj = bayesian_prefix_trajectories(model, [c for c in clipped_stimuli if c.name in llm_trajectories])

    curves = [
        size_principle_curves(bay_traj, clipped_stimuli, "bayesian"),
        size_principle_curves(clipped_trajectories, clipped_stimuli, "llm", bayesian_trajectories=bay_traj),
    ]
    if human_preds and not any(np.isnan(v).any() for v in human_preds.values()):  # dense ratings only
        # Shape: {name: {n_examples, results}} with one full-evidence human rating.
        human_traj = {
            c.name: {"n_examples": [len(c.examples)], "results": [human_preds[c.name]]}
            for c in clipped_stimuli if c.name in human_preds
        }
        curves.append(size_principle_curves(human_traj, clipped_stimuli, "human", bayesian_trajectories=bay_traj))
    if human_preds:
        curves.append(human_kl_curves(clipped_trajectories, clipped_stimuli, human_preds))
    plot_trajectory_metrics(pd.concat(curves, ignore_index=True), output_dir, result_key)


def analyze_posterior_condition_results(finals, trajectories, ctx, model, priors, bayesian_preds, out_dir, result_key, featured):
    """Analyze one posterior condition's final predictions and prefix trajectories under its pooled prior."""
    task_data = ctx.task
    plot_all(bayesian_preds, finals, [c.name for c in task_data.stimuli], out_dir, result_key,
             human_preds=task_data.human_preds, featured=featured)
    # Full-stimulus-set size-principle fit + per-stimulus ablation plots.
    fit = fit_beta_with_ablations(model, priors, finals, task_data.stimuli)
    plot_alpha_beta_all(fit, bayesian_preds, finals, [c.name for c in task_data.stimuli], os.path.join(out_dir, "alpha_beta"))
    print(f"    Best fit: size principle={fit['beta']:.2f} (MSE={fit['mse']:.6f})")
    run_trajectory_metrics(model, ctx, trajectories, out_dir, result_key)

    beta_traj = compute_trajectory_beta(model, priors, *clip_to_prefix_horizon(task_data.stimuli, trajectories))
    plot_trajectory_beta(beta_traj, os.path.join(out_dir, "trajectory_beta.png"), result_key)
    print(f"    Size-principle trajectory: n={beta_traj['n_examples']}, "
          f"β={[f'{b:.2f}' for b in beta_traj['beta']]}")


def analyze_hypothesis_condition_results(all_finals, ctx, rule_posteriors, out_dir, result_key, featured):
    """Run hypothesis-mode plots and caches after generation structuring."""
    task_data = ctx.task
    stimulus_names = [c.name for c in task_data.stimuli]

    if "hypothesis_eval" in all_finals:
        if ctx.rule_families:
            rule_posteriors = {
                stimulus_name: aggregate_by_rule_family(posteriors, ctx.rule_families)
                for stimulus_name, posteriors in rule_posteriors.items()
            }
        plot_hypothesis_comparison(
            rule_posteriors, all_finals["hypothesis_eval"], stimulus_names,
            os.path.join(out_dir, "hypothesis_comparison.png"), result_key,
            featured=featured)

    if "hypothesis_generation" in all_finals:
        plot_hypothesis_generation_summary(
            all_finals["hypothesis_generation"], stimulus_names,
            os.path.join(out_dir, "hypothesis_generation"), result_key)
        gen_metrics = compute_hypothesis_generation_metrics(all_finals["hypothesis_generation"], stimulus_names)
        plot_hypothesis_generation_type_summary(
            gen_metrics, os.path.join(out_dir, "hypothesis_generation_types.png"),
            result_key, featured=featured)
        if task_data.name == "Bigelow16":
            plot_hypothesis_generation_exemplars(
                gen_metrics, os.path.join(out_dir, "hypothesis_generation_exemplars.png"), result_key)
        for stimulus_name, metrics in gen_metrics["stimuli"].items():
            type_counts = metrics["type_counts"]
            print(f"    {stimulus_name}: rule={type_counts['rule']} "
                  f"int={type_counts['interval']} other={type_counts['other']}")


def run_per_condition(experiment: Experiment):
    """Run posterior condition-local plots."""
    result_key = experiment.backend.result_key
    for condition in _conditions(experiment, "posterior"):
        trajectories = load_condition_trajectory(condition, result_key)
        if not trajectories:
            continue
        baseline = bayesian_baseline(experiment, condition.task, condition.domain)
        print(f"  {condition.task} d{condition.domain} prompt: {condition.prompt}")
        analyze_posterior_condition_results(
            trajectory_finals(trajectories, "posterior"), trajectories, baseline.ctx, baseline.model,
            _fit_prior(experiment, condition)[1], baseline.predictions, condition_result_dir(condition, result_key), result_key,
            experiment.tasks[condition.task].featured_stimuli,
        )


def run_hypothesis_per_condition(experiment: Experiment):
    """Run condition-local hypothesis figures from structured generation caches."""
    result_key = experiment.backend.result_key
    hypothesis_conditions = _conditions(experiment, "hypothesis_eval", "hypothesis_generation")
    for _, eval_conditions in group_by(hypothesis_conditions, lambda c: c.prompt_group):
        all_finals = {}
        for condition in eval_conditions:
            final = load_condition_final(condition, result_key)
            if final:
                all_finals[condition.eval_mode] = final
        if not all_finals:
            continue
        condition = eval_conditions[0]
        baseline = bayesian_baseline(experiment, condition.task, condition.domain)
        print(f"  {condition.task} d{condition.domain} prompt: {condition.prompt}")
        analyze_hypothesis_condition_results(
            all_finals, baseline.ctx, baseline.rule_posteriors,
            condition_result_dir(condition, result_key), result_key,
            experiment.tasks[condition.task].featured_stimuli,
        )


# ==================================================================
# Per-model aggregate stages
# ==================================================================

def _alpha_beta_fit_groups(
    stimulus_map: dict[str, StimulusSet],
    trajectory: dict[str, dict],
) -> list[tuple[str, int | None, list[StimulusSet], dict[str, np.ndarray], dict[str, int]]]:
    """Build task-local α-β fit scopes: prefix lengths and full stimulus sets.

    Prefix scopes fit the trajectory-horizon stimuli (clip_to_prefix_horizon), so every prefix
    length of a task fits the same stimuli as the trajectory stage.
    """
    groups = []

    def collect_scope(n: int | None, stimuli):
        scope_stimuli = []
        scope_preds = {}
        counts = {"n_valid": 0, "n_exhausted": 0, "n_usable": 0}
        for stimulus in stimuli:
            ev = trajectory.get(stimulus.name)
            if ev is None:
                continue
            target_n = len(stimulus.examples) if n is None else n
            if len(stimulus.examples) < target_n or target_n not in ev["n_examples"]:
                continue
            index = ev["n_examples"].index(target_n)
            result = np.asarray(ev["results"][index], dtype=float)
            statuses = ev.get("projection_status")
            if statuses is not None:
                status = statuses[index]
            elif result.size == 0:
                status = "exhausted"
            else:
                status = "usable" if result.ndim == 1 and np.isfinite(result).all() else "unusable"

            counts["n_exhausted"] += status == "exhausted"
            counts["n_valid"] += status != "exhausted"
            if status != "usable":
                continue
            counts["n_usable"] += 1
            scope_stimuli.append(stimulus.prefix(target_n))
            scope_preds[stimulus.name] = result
        if sum(counts.values()):
            groups.append((scope_label(n), n, scope_stimuli, scope_preds, counts))

    horizon_stimuli, _ = clip_to_prefix_horizon(list(stimulus_map.values()), {})
    for n in PREFIX_LENGTHS:
        collect_scope(n, horizon_stimuli)
    collect_scope(None, stimulus_map.values())
    return groups


def _projected_trajectory(condition, result_key) -> dict[str, dict] | None:
    """Project one hypothesis-mode condition's trajectory into predictive vectors plus projection status."""
    trajectory = load_condition_trajectory(condition, result_key)
    if not trajectory:
        return None
    views = project_trajectory(condition.eval_mode, trajectory, build_task_context(condition.task, condition.domain))
    return {
        name: {
            "n_examples": list(ev["n_examples"]),
            "results": [views[name][n].vector for n in ev["n_examples"]],
            "projection_status": [
                "exhausted" if not result else "usable" if views[name][n].usable else "unusable"
                for n, result in zip(ev["n_examples"], ev["results"])
            ],
        }
        for name, ev in trajectory.items()
    }


def _fit_prior(experiment, condition) -> tuple:
    """Pooled prior α and hypothesis prior weights for one condition's fits.

    Predict uses its own prompt's pooled prior; Eval and Generate use the default prompt's.
    """
    prompt = (condition.h_space, condition.sampling) if condition.eval_mode == "posterior" else DEFAULT_PROMPT
    prior = pooled_prior(experiment, *prompt)
    baseline = bayesian_baseline(experiment, condition.task, condition.domain)
    return prior["alpha"], pooled_priors(prior, baseline.model, baseline.ctx.rule_families)


def _add_alpha_beta_groups(bucket_map, key_prefix: tuple, task: str, model, prior: tuple, fit_groups):
    """Every fit scope pools tasks; `prior` is the condition's (α, prior weights)."""
    alpha, priors = prior
    for fit_scope, n_examples, stimuli, preds, counts in fit_groups:
        bucket = bucket_map.setdefault(
            (*key_prefix, fit_scope, n_examples),
            {"tasks": [], "specs": [], "alpha": alpha, "counts": {"n_valid": 0, "n_exhausted": 0, "n_usable": 0}},
        )
        bucket["tasks"].append(task)
        if stimuli:
            bucket["specs"].append((model, priors, preds, stimuli))
        for count_name, count in counts.items():
            bucket["counts"][count_name] += count


def _posterior_fit_groups(condition, result_key):
    trajectory = load_condition_trajectory(condition, result_key)
    if not trajectory:
        return []
    return _alpha_beta_fit_groups(build_task_context(condition.task, condition.domain).task.stimulus_map, trajectory)


def run_divergence(experiment: Experiment, analysis_dir: str, figure_dir: str, *, eval_modes: tuple[str, ...]):
    """Divergence of each final result from the Bayesian prediction, plus posterior human KL."""
    result_key = experiment.backend.result_key
    stage = "divergence" if "posterior" in eval_modes else "hypothesis_divergence"
    rows, human_rows = [], []
    for condition in _conditions(experiment, *eval_modes):
        final = load_condition_final(condition, result_key)
        if not final:
            continue
        log_final = load_condition_final(condition, result_key, log_space=True)
        baseline = bayesian_baseline(experiment, condition.task, condition.domain)
        ctx, bay_preds = baseline.ctx, baseline.predictions
        for stimulus in ctx.task.stimuli:
            if stimulus.name not in final:
                continue
            result = final[stimulus.name]
            view = project_result(condition.eval_mode, result, stimulus.examples, ctx)
            log_vector = log_final[stimulus.name] if condition.eval_mode == "posterior" else view.vector
            metrics = {
                "kl_bayes": kl_divergence(bay_preds[stimulus.name], log_vector),
                "correlation": safe_correlation(bay_preds[stimulus.name], view.vector),
            }
            if condition.eval_mode != "posterior":
                metrics["other_mass"] = view.other_mass
            rows.append({**_row_base(condition, stimulus), "eval_mode": condition.eval_mode, **metrics})

            human = (ctx.task.human_preds or {}).get(stimulus.name) if condition.eval_mode == "posterior" else None
            mask = None if human is None else rated_targets(human)
            if mask is not None:
                human_rows.append({
                    **_row_base(condition, stimulus),
                    "eval_mode": "posterior",
                    "kl_human": kl_divergence(human[mask], log_vector[mask]),
                    "correlation": safe_correlation(human[mask], result[mask]),
                })

    df = save_csv(rows, analysis_dir, f"{stage}_metrics.csv")
    if human_rows:  # domains without human data (d200) get no file; summaries treat a missing file as no rows
        save_csv(human_rows, analysis_dir, f"{stage}_human_metrics.csv")
    if "posterior" in eval_modes:
        post_df = df[df["eval_mode"] == "posterior"] if not df.empty else df
        if not post_df.empty:
            bay_div, human_div = load_baseline_overlay("divergence_metrics.csv")
            plot_prompt_kind_bar(
                post_df, "kl_bayes", "KL(Bayesian || LLM)", "KL by stimulus kind",
                os.path.join(figure_dir, "divergence_by_type.png"),
                result_key=result_key,
                baseline_overlay={"Bayesian": bay_div, "Human": human_div},
            )
    return df


def _fit_alpha_beta_buckets(buckets: dict, *, with_eval_mode: bool) -> list[dict]:
    rows = []
    for key, bucket in buckets.items():
        fit = fit_beta(bucket["specs"])
        if fit is None:
            continue
        fit = {**fit, "alpha": bucket["alpha"]}
        if with_eval_mode:
            domain, eval_mode, h_space, sampling, fit_scope, n_examples = key
            rows.append(alpha_beta_row(
                task_group_label(bucket["tasks"]), domain, h_space, sampling, fit_scope, n_examples,
                {**fit, **bucket["counts"]}, eval_mode=eval_mode,
            ))
        else:
            domain, h_space, sampling, fit_scope, n_examples = key
            rows.append(alpha_beta_row(task_group_label(bucket["tasks"]), domain, h_space, sampling, fit_scope, n_examples, fit))
            if fit_scope == FULL_SCOPE:
                print(f"    Size-principle fit: {rows[-1]['task']} d{domain} {h_space}_{sampling} β={fit['beta']:.2f}")
    return rows


def run_alpha_beta(experiment: Experiment, analysis_dir: str, figure_dir: str):
    """Fit the posterior size principle under each prompt's pooled prior after pooling task stimuli.

    Each row fixes domain/h_space/sampling and one fit scope: n=1..4 prefixes
    or the full stimulus sets. The task column names the pooled task group.
    """
    result_key = experiment.backend.result_key
    buckets = {}
    for condition in _conditions(experiment, "posterior"):
        _add_alpha_beta_groups(
            buckets, (condition.domain, condition.h_space, condition.sampling), condition.task,
            bayesian_baseline(experiment, condition.task, condition.domain).model, _fit_prior(experiment, condition),
            _posterior_fit_groups(condition, result_key),
        )
    df = save_csv(_fit_alpha_beta_buckets(buckets, with_eval_mode=False), analysis_dir, "alpha_beta_fits.csv")

    if not df.empty:
        bay_ab, human_ab = load_baseline_overlay("alpha_beta_fits.csv")
        overlay = {"Bayesian": bay_ab, "Human": human_ab}
        for metric, ylabel, title, filename, axhline in (
            ("mse", "MSE (fit)", "MSE (fit)", "loss.png", None),
            ("beta", "Size principle (β)", "Size principle (β) by fit scope", "domain_beta.png", 1.0),
        ):
            plot_prompt_fit_scope_bar(
                df, metric, ylabel, title, os.path.join(figure_dir, filename),
                result_key=result_key, axhline=axhline, baseline_overlay=overlay,
            )
        prefix_rows = with_task_domain(df.dropna(subset=["n_examples"])).assign(source="llm")
        plot_prompt_trajectory(prefix_rows, "beta", "Size principle (β)", os.path.join(figure_dir, "trajectory_beta.png"),
                               include_bayesian=False)
    return df


def run_alpha_beta_modes(experiment: Experiment, analysis_dir: str, figure_dir: str):
    """Fit posterior, eval, and structured-generation projections in shared predictive space."""
    result_key = experiment.backend.result_key
    buckets = {}
    for condition in make_grid(experiment):
        if condition.eval_mode == "posterior":
            groups = _posterior_fit_groups(condition, result_key)
        else:
            projected = _projected_trajectory(condition, result_key)
            if not projected:
                continue
            groups = _alpha_beta_fit_groups(build_task_context(condition.task, condition.domain).task.stimulus_map, projected)
        _add_alpha_beta_groups(
            buckets, (condition.domain, condition.eval_mode, condition.h_space, condition.sampling), condition.task,
            bayesian_baseline(experiment, condition.task, condition.domain).model, _fit_prior(experiment, condition), groups,
        )
    return save_csv(_fit_alpha_beta_buckets(buckets, with_eval_mode=True), analysis_dir, "alpha_beta_mode_fits.csv")


def run_prompt_effects(experiment: Experiment, analysis_dir: str, figure_dir: str):
    """Compute posterior prompt sensitivity and effective support."""
    result_key = experiment.backend.result_key
    sensitivity_rows, support_rows = [], []
    for condition in _conditions(experiment, "posterior"):
        final = load_condition_final(condition, result_key)
        if not final:
            continue
        log_final = load_condition_final(condition, result_key, log_space=True)
        bayesian_preds = bayesian_baseline(experiment, condition.task, condition.domain).predictions
        for stimulus in build_task_context(condition.task, condition.domain).task.stimuli:
            llm_post = final.get(stimulus.name)
            if llm_post is None:
                continue
            support_rows.append({**_row_base(condition, stimulus), "effective_support": compute_effective_support(llm_post)})
            sensitivity_rows.append({**_row_base(condition, stimulus), "kl_bayes": kl_divergence(bayesian_preds[stimulus.name], log_final[stimulus.name])})

    if sensitivity_rows:
        sens_df = save_csv(sensitivity_rows, analysis_dir, "prompt_sensitivity.csv")
        plot_prompt_sensitivity(sens_df, os.path.join(figure_dir, "prompt_sensitivity.png"))
    if support_rows:
        supp_df = save_csv(support_rows, analysis_dir, "prompt_effective_support.csv")
        bay_support, human_support = load_baseline_overlay("prompt_effective_support.csv")
        plot_generalization_width(
            supp_df, os.path.join(figure_dir, "generalization_width.png"), result_key,
            baseline_overlay={"Bayesian": bay_support, "Human": human_support})
    return sensitivity_rows, support_rows


TRAJECTORY_PLOT_SPECS = [
    ("kl_bayes", "trajectory_kl.png", r"KL(Bayesian $\Vert$ LLM)", False),
    ("entropy", "trajectory_entropy.png", "Entropy", True),
    ("effective_support", "trajectory_effective_support.png", "Effective support (p > 0.5)", True),
]


def _annotate_trajectory_rows(df, task, domain, source, h_space=None, sampling=None):
    """Add cross-condition identifiers expected by the shared trajectory plotter."""
    if df.empty:
        return []
    df = with_task_domain(df.assign(task=task, domain=domain)).assign(source=source)
    if h_space is not None:
        df = df.assign(h_space=h_space, sampling=sampling)
    return df.to_dict("records")


def run_posterior_trajectory(experiment: Experiment, analysis_dir: str, figure_dir: str):
    """Compute posterior KL, entropy, and support prefix trajectories."""
    result_key = experiment.backend.result_key
    metric_rows, llm_loaded = [], False

    for (task, domain), conditions in group_by(_conditions(experiment, "posterior"), lambda c: c.context_key):
        baseline = bayesian_baseline(experiment, task, domain)
        ctx, model = baseline.ctx, baseline.model
        bayes_stimuli, _ = clip_to_prefix_horizon(ctx.task.stimuli, {})
        bayes_traj = bayesian_prefix_trajectories(model, bayes_stimuli)
        metric_rows.extend(_annotate_trajectory_rows(
            size_principle_curves(bayes_traj, bayes_stimuli, "bayesian", bayesian_trajectories=bayes_traj),
            task, domain, "bayesian"))

        for condition in conditions:
            ev = load_condition_trajectory(condition, result_key)
            if not ev:
                continue
            traj_stimuli, traj_ev = clip_to_prefix_horizon(ctx.task.stimuli, ev)
            traj_stimuli = [c for c in traj_stimuli if c.name in traj_ev]
            llm_loaded = True
            prompt = (condition.h_space, condition.sampling)
            metric_rows.extend(_annotate_trajectory_rows(
                size_principle_curves(traj_ev, traj_stimuli, "llm", bayesian_trajectories=bayes_traj),
                task, domain, "llm", *prompt))
            if ctx.task.human_preds:
                metric_rows.extend(_annotate_trajectory_rows(
                    human_kl_curves(traj_ev, traj_stimuli, ctx.task.human_preds), task, domain, "llm", *prompt))

    # Bayesian rows alone lack the prompt columns the plots group by.
    if llm_loaded:
        metrics_df = save_csv(metric_rows, analysis_dir, "trajectory_prefix_metrics.csv")
        plot_trajectory_rate(metrics_df, os.path.join(figure_dir, "trajectory_rate.png"), result_key)
        for metric, filename, ylabel, include_bayesian in TRAJECTORY_PLOT_SPECS:
            plot_prompt_trajectory(metrics_df, metric, ylabel, os.path.join(figure_dir, filename),
                                   include_bayesian=include_bayesian)


def run_generation_narrowing(experiment: Experiment, analysis_dir: str, figure_dir: str):
    """Count unique structured generations over evidence."""
    result_key = experiment.backend.result_key
    rows = []
    for condition in _conditions(experiment, "hypothesis_generation"):
        trajectory = load_condition_trajectory(condition, result_key)
        if not trajectory:
            continue
        stimulus_map = build_task_context(condition.task, condition.domain).task.stimulus_map
        for stimulus_name, ed in trajectory.items():
            base = _row_base(condition, stimulus_map[stimulus_name])
            for n, generated in zip(ed["n_examples"], ed["results"]):
                rows.append({**base, "n_examples": n, "n_unique": len({item["hypothesis"] for item in generated})})

    if rows:
        save_csv(rows, analysis_dir, "generation_narrowing_counts.csv")
        plot_narrowing_aggregate(rows, os.path.join(figure_dir, "narrowing.png"), result_key)


def _type_columns(props: dict[str, float]) -> dict[str, float]:
    return {f"cw_{typ}": value for typ, value in props.items()}


def run_hypothesis_types(experiment: Experiment, analysis_dir: str, figure_dir: str):
    """Hypothesis-type composition for eval and structured generation, per final and per prefix."""
    result_key = experiment.backend.result_key
    rows, eval_rows, trajectory_rows = [], [], []

    for condition in _conditions(experiment, "hypothesis_eval", "hypothesis_generation"):
        trajectory = load_condition_trajectory(condition, result_key)
        if not trajectory:
            continue
        ctx = build_task_context(condition.task, condition.domain)
        stimulus_map = ctx.task.stimulus_map

        if condition.eval_mode == "hypothesis_eval":
            for stimulus_name, ed in trajectory.items():
                stimulus = stimulus_map[stimulus_name]
                for n, result in zip(ed["n_examples"], ed["results"]):
                    if not result:
                        continue
                    view = project_hypothesis_eval(result, stimulus.examples[:n], ctx)
                    eval_rows.append({
                        **_row_base(condition, stimulus), "eval_mode": condition.eval_mode, "n_examples": n,
                        **_type_columns(type_proportions([
                            ("rule", view.rule_mass), ("interval", view.interval_mass), ("other", view.other_mass),
                        ])),
                    })
            continue

        finals = trajectory_finals(trajectory, condition.eval_mode)
        metrics = compute_hypothesis_generation_metrics(finals, list(stimulus_map))
        for stimulus_name, cm in metrics["stimuli"].items():
            rows.append({
                **_row_base(condition, stimulus_map[stimulus_name]),
                "n_hypotheses": cm["n_hypotheses"],
                **cm["type_counts"],
                **{f"other_{reason}": count for reason, count in cm["other_reasons"].items()},
                **{f"{typ}_prop": value for typ, value in cm["type_proportions"].items()},
                **_type_columns(cm["confidence_weighted_types"]),
            })
        for stimulus_name, ed in trajectory.items():
            for n, result in zip(ed["n_examples"], ed["results"]):
                if not result:
                    continue
                types = [hypothesis_type(item) for item in result]
                counts = {typ: types.count(typ) for typ in HYPOTHESIS_TYPES}
                trajectory_rows.append({
                    **_row_base(condition, stimulus_map[stimulus_name]), "eval_mode": condition.eval_mode,
                    "n_examples": n, **counts,
                    **{f"{typ}_prop": count / len(result) for typ, count in counts.items()},
                    **_type_columns(type_proportions(
                        (hypothesis_type(item), item["confidence"]) for item in collapse_structured_generation(result)
                    )),
                    "n_hypotheses": len(result),
                })

    if not rows:
        print("  Hypothesis generation: no data found")
        return

    df = save_csv(rows, analysis_dir, "hypothesis_types_metrics.csv")
    rule_parts = []
    if eval_rows:
        eval_df = save_csv(eval_rows, analysis_dir, "hypothesis_types_eval_trajectory.csv")
        plot_confidence_weighted_types(
            eval_df, os.path.join(figure_dir, "confidence_weighted_types_eval.png"),
            result_key, title="Confidence-Weighted Types: Eval")
        rule_parts.append(eval_df)
    if trajectory_rows:
        traj_df = save_csv(trajectory_rows, analysis_dir, "hypothesis_types_generation_trajectory.csv")
        plot_confidence_weighted_types(
            traj_df, os.path.join(figure_dir, "confidence_weighted_types_gen.png"),
            result_key, title="Confidence-Weighted Types: Generate")
        rule_parts.append(traj_df)
    if rule_parts:
        plot_trajectory_rule_mass(pd.concat(rule_parts, ignore_index=True),
                                  os.path.join(figure_dir, "trajectory_rule_mass.png"))
    return df


DEFAULT_STAGES = {
    "divergence": partial(run_divergence, eval_modes=("posterior",)),
    "alpha_beta": run_alpha_beta,
    "prompt": run_prompt_effects,
    "trajectory": run_posterior_trajectory,
    "rt_correlation": run_rt_correlation,
}

HYPOTHESIS_STAGES = {
    "nested_pair": run_nested_pair,
    "alpha_beta_mode": run_alpha_beta_modes,
    "generation_narrowing": run_generation_narrowing,
    "hypothesis_divergence": partial(run_divergence, eval_modes=("hypothesis_eval", "hypothesis_generation")),
    "hypothesis_types": run_hypothesis_types,
    "hypothesis_compare": run_hypothesis_compare,
    "hypothesis_fit": run_hypothesis_fit,
}

# Cross-domain stages run once per model over every configured domain and write
# to the model's root analysis/figure directories.
DEFAULT_MODEL_STAGES = {"extension": run_extension}
HYPOTHESIS_MODEL_STAGES = {"hypothesis_extension": run_hypothesis_extension}


def _assert_structured_generation_available(experiment) -> None:
    """Hypothesis-generation analysis requires structured predicate mapping."""
    missing = []
    for condition in _conditions(experiment, "hypothesis_generation"):
        raw_path = condition_data_path(condition, experiment.backend.result_key, "hypothesis_generation")
        structured_path = condition_data_path(condition, experiment.backend.result_key, "hypothesis_generation_structured")
        if os.path.exists(raw_path) and not os.path.exists(structured_path):
            missing.append(structured_path)
    if missing:
        preview = "\n  ".join(missing[:10])
        suffix = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            "Hypothesis analysis requires hypothesis_generation_structured.json; "
            "raw hypothesis_generation.json is not an analysis input. Missing:\n  "
            f"{preview}{suffix}"
        )


def _print_stage(name: str) -> None:
    print(f"\n{'=' * 40}\nStage: {name}\n{'=' * 40}")


def _run_stage_registry(stages: dict, model_stages: dict, experiment, *, per_condition_fn=None) -> str:
    result_key = experiment.backend.result_key
    model_dir = layout.model_root(result_key)
    output_dir = layout.analysis_dir(result_key)
    print(f"Analyzing: {result_key}\nResults dir: {model_dir}\nAnalysis cache dir: {output_dir}")

    domains = list(dict.fromkeys(domain for factors in experiment.tasks.values() for domain in factors.domains))
    for domain in domains:
        domain_experiment = replace(experiment, tasks={
            name: replace(factors, domains=[domain])
            for name, factors in experiment.tasks.items()
            if domain in factors.domains
        })
        domain_output_dir = domain_subdir(output_dir, domain)
        domain_figure_dir = domain_subdir(model_dir, domain)
        os.makedirs(domain_output_dir, exist_ok=True)
        os.makedirs(domain_figure_dir, exist_ok=True)
        print(f"\nDomain: d{domain} (cache {domain_output_dir}, figures {domain_figure_dir})")

        if per_condition_fn is not None:
            _print_stage("per_condition")
            per_condition_fn(domain_experiment)
        for stage_name, stage_fn in stages.items():
            _print_stage(stage_name)
            stage_fn(domain_experiment, domain_output_dir, domain_figure_dir)

    for stage_name, stage_fn in model_stages.items():
        _print_stage(stage_name)
        stage_fn(experiment, output_dir, model_dir)
    return output_dir


def run_all_stages(experiment, include_per_condition: bool = False) -> str:
    """Run the default non-hypothesis pipeline stages for one model."""
    return _run_stage_registry(
        DEFAULT_STAGES, DEFAULT_MODEL_STAGES, experiment,
        per_condition_fn=run_per_condition if include_per_condition else None,
    )


def run_hypothesis_stages(experiment) -> str:
    """Run hypothesis-only analysis stages for one model."""
    _assert_structured_generation_available(experiment)
    return _run_stage_registry(
        HYPOTHESIS_STAGES, HYPOTHESIS_MODEL_STAGES, experiment, per_condition_fn=run_hypothesis_per_condition,
    )
