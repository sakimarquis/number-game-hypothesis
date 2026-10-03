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

from analysis.alpha_beta_scope import FULL_SCOPE, PREFIX_LENGTHS, alpha_beta_row, scope_label, task_group_label
from analysis.bayesian import load_baseline_overlay
from analysis.cache import (
    bayesian_cache_identity, bayesian_model, build_bayesian, load_bayesian,
    load_condition_final, load_condition_trajectory, load_human_alpha_beta_fit,
    task_context, trajectory_finals, trajectory_steps,
)
from analysis.extension import run_extension, run_hypothesis_extension
from analysis.hypothesis_fit import run_hypothesis_compare, run_hypothesis_fit
from analysis.hypothesis_space import (
    collapse_structured_generation,
    load_hypothesis_library,
    normalize_hypothesis_weight_dict,
    normalize_structured_generation_confidence,
    project_hypothesis_eval,
    project_posterior,
    project_structured_hypothesis_generation,
    summarize_projection_rows,
)
from analysis.metrics import (
    compute_hypothesis_generation_metrics, compute_topk_overlap, hypothesis_type,
    kl_divergence, safe_correlation, type_proportions,
)
from analysis.model import compute_alpha_beta_ablations, fit_alpha_beta
from analysis.plotter import (
    alpha_beta_ratio, plot_prompt_fit_scope_bar, plot_prompt_kind_bar, save_csv, with_all_kinds,
)
from analysis.posterior import (
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
    plot_trajectory_rule_by_kind,
)
from analysis.rt import run_rt_correlation
from analysis.size_principle import compute_effective_support, human_kl_curves, run_nested_pair, size_principle_curves
from analysis.trajectory import (
    bayesian_prefix_trajectories, clip_to_prefix_horizon,
    compute_trajectory, compute_trajectory_alpha_beta, compute_trajectory_rate,
    plot_narrowing_aggregate, plot_prompt_trajectory,
    plot_sampling_alpha_beta_trajectory, plot_trajectory_alpha_beta,
    plot_trajectory_alpha_beta_ratio, plot_trajectory_rate,
)
from configs.layout import (
    analysis_dir, condition_cache_path, condition_data_path, condition_result_dir, domain_subdir, model_root,
)
from configs.settings import Experiment, make_grid
from tasks.hypotheses import aggregate_by_rule_family
from tasks.stimuli import StimulusSet
from utils import group_by, save_json


def _conditions(experiment: Experiment, *eval_modes: str):
    return [condition for condition in make_grid(experiment) if condition.eval_mode in eval_modes]


def _bayesian_predictions(experiment: Experiment, task: str, domain: int) -> tuple[dict, dict]:
    return load_bayesian(task, domain, bayesian_cache_identity(experiment, task, domain))


def _row_base(condition, stimulus) -> dict:
    """Identifying key for a cross-condition row: task, stimulus, kind, domain, prompt."""
    return {
        "task": condition.task, "stimulus": stimulus.name, "stimulus_kind": stimulus.kind,
        "domain": condition.domain, "h_space": condition.h_space, "sampling": condition.sampling,
    }


# ==================================================================
# Per-condition stages
# ==================================================================

def run_posterior_analysis(model, bayesian_preds, llm_finals, stimuli, output_dir, result_key, condition):
    """Full-stimulus-set α-β fit + per-stimulus ablation plots.

    The fit (aggregate + ablation MSEs) is always dumped to JSON so
    large-stimulus tasks like Bigelow16 — whose per-stimulus plots are
    skipped — remain fully analyzable downstream.
    """
    fit = fit_alpha_beta([(model, llm_finals, stimuli)])
    fit = {
        **fit,
        **compute_alpha_beta_ablations(model, llm_finals, stimuli, fit["best_alpha"], fit["best_beta"]),
    }
    # Keep bulky ablation arrays out of the JSON cache after plotting consumes them.
    save_json(condition_cache_path(condition, result_key, "alpha_beta_fit.json"),
              {k: v for k, v in fit.items() if k != "ablation_preds"})

    human_fit = load_human_alpha_beta_fit(condition.task, condition.domain)
    if human_fit:
        print(f"    Human fit (cached): α={human_fit['best_alpha']:.2f}, "
              f"β={human_fit['best_beta']:.2f} (MSE={human_fit['best_loss']:.6f})")

    plot_alpha_beta_all(fit, bayesian_preds, llm_finals, [c.name for c in stimuli],
                        os.path.join(output_dir, "alpha_beta"))
    print(f"    Best fit: α={fit['best_alpha']:.2f}, "
          f"β={fit['best_beta']:.2f} (MSE={fit['best_loss']:.6f})")
    return fit, human_fit


def _dense_human_preds(human_preds):
    return (
        human_preds
        if human_preds and not any(np.any(np.isnan(v)) for v in human_preds.values())
        else None
    )


def run_trajectory_metrics(model, ctx, llm_trajectories, output_dir, result_key, condition, prefix_horizon: int):
    """Per-prefix KL, entropy, and support over a fixed stimulus subset."""
    human_preds = ctx.task.human_preds
    clipped_stimuli, clipped_trajectories = clip_to_prefix_horizon(ctx.task.stimuli, llm_trajectories, prefix_horizon)
    bay_traj = bayesian_prefix_trajectories(model, [c for c in clipped_stimuli if c.name in llm_trajectories])

    curves = [
        size_principle_curves(bay_traj, clipped_stimuli, "bayesian"),
        size_principle_curves(clipped_trajectories, clipped_stimuli, "llm", bayesian_trajectories=bay_traj),
    ]
    dense_human = _dense_human_preds(human_preds)
    if dense_human:
        # Shape: {name: {n_examples, results}} with one full-evidence human rating.
        human_traj = {
            c.name: {"n_examples": [len(c.examples)], "results": [dense_human[c.name]]}
            for c in clipped_stimuli if c.name in dense_human
        }
        curves.append(size_principle_curves(human_traj, clipped_stimuli, "human", bayesian_trajectories=bay_traj))
    if human_preds:
        curves.append(human_kl_curves(clipped_trajectories, clipped_stimuli, human_preds))
    plot_trajectory_metrics(pd.concat(curves, ignore_index=True), output_dir, result_key)

    save_json(condition_cache_path(condition, result_key, "trajectory_kl.json"), compute_trajectory(model, clipped_stimuli, clipped_trajectories))


def analyze_posterior_condition_results(finals, trajectories, ctx, model, bayesian_preds, out_dir, result_key,
                                        condition, prefix_horizon: int):
    """Analyze one posterior condition's final predictions and prefix trajectories."""
    task_data = ctx.task
    plot_all(bayesian_preds, finals, [c.name for c in task_data.stimuli], out_dir, result_key,
             human_preds=task_data.human_preds, featured=task_data.featured_stimuli)
    run_posterior_analysis(model, bayesian_preds, finals, task_data.stimuli, out_dir, result_key, condition)
    run_trajectory_metrics(model, ctx, trajectories, out_dir, result_key, condition, prefix_horizon)

    ab_traj = compute_trajectory_alpha_beta(model, *clip_to_prefix_horizon(task_data.stimuli, trajectories, prefix_horizon))
    save_json(condition_cache_path(condition, result_key, "trajectory_alpha_beta.json"), ab_traj)
    plot_trajectory_alpha_beta(ab_traj, os.path.join(out_dir, "trajectory_alpha_beta.png"), result_key)
    plot_trajectory_alpha_beta_ratio(ab_traj, os.path.join(out_dir, "trajectory_alpha_beta_ratio.png"), result_key)
    print(f"    α-β trajectory: n={ab_traj['n_examples']}, "
          f"α={[f'{a:.2f}' for a in ab_traj['alpha']]}, "
          f"β={[f'{b:.2f}' for b in ab_traj['beta']]}")


def analyze_hypothesis_condition_results(all_finals, ctx, bayesian_hyp_preds, out_dir, result_key, condition):
    """Run hypothesis-interface plots and caches after generation structuring."""
    task_data = ctx.task
    stimulus_names = [c.name for c in task_data.stimuli]

    if "hypothesis_eval" in all_finals:
        bay_hyps = bayesian_hyp_preds
        if ctx.rule_families:
            bay_hyps = {
                stimulus_name: aggregate_by_rule_family(hypotheses, ctx.rule_families)
                for stimulus_name, hypotheses in bay_hyps.items()
            }
        plot_hypothesis_comparison(
            bay_hyps, all_finals["hypothesis_eval"], stimulus_names,
            os.path.join(out_dir, "hypothesis_comparison.png"), result_key,
            featured=task_data.featured_stimuli)

    if "hypothesis_generation" in all_finals:
        plot_hypothesis_generation_summary(
            all_finals["hypothesis_generation"], stimulus_names,
            os.path.join(out_dir, "hypothesis_generation"), result_key)
        gen_metrics = compute_hypothesis_generation_metrics(all_finals["hypothesis_generation"], stimulus_names)
        save_json(condition_cache_path(condition, result_key, "hypothesis_generation_metrics.json"), gen_metrics)
        plot_hypothesis_generation_type_summary(
            gen_metrics, os.path.join(out_dir, "hypothesis_generation_types.png"),
            result_key, featured=task_data.featured_stimuli)
        if task_data.name == "Bigelow16":
            plot_hypothesis_generation_exemplars(
                gen_metrics, os.path.join(out_dir, "hypothesis_generation_exemplars.png"), result_key)
        for stimulus_name, metrics in gen_metrics["stimuli"].items():
            type_counts = metrics["type_counts"]
            print(f"    {stimulus_name}: rule={type_counts['rule']} "
                  f"int={type_counts['interval']} other={type_counts['other']}")


def run_per_condition(experiment: Experiment, result_key: str):
    """Run posterior condition-local plots and caches."""
    baselines = build_bayesian(experiment, make_grid(experiment))
    for condition in _conditions(experiment, "posterior"):
        trajectories = load_condition_trajectory(condition, result_key)
        if not trajectories:
            continue
        baseline = baselines[condition.context_key]
        print(f"  {condition.task} d{condition.domain} prompt: {condition.prompt}")
        analyze_posterior_condition_results(
            trajectory_finals(trajectories, "posterior"), trajectories, baseline.ctx, baseline.model,
            baseline.predictions, condition_result_dir(condition, result_key), result_key, condition,
            prefix_horizon=experiment.scoring.trajectory_prefix_horizon,
        )


def run_hypothesis_per_condition(experiment: Experiment, result_key: str):
    """Run condition-local hypothesis figures from structured generation caches."""
    baselines = build_bayesian(experiment, make_grid(experiment))
    hypothesis_conditions = _conditions(experiment, "hypothesis_eval", "hypothesis_generation")
    for _, eval_conditions in group_by(hypothesis_conditions, lambda c: c.prompt_key):
        all_finals = {}
        for condition in eval_conditions:
            final = load_condition_final(condition, result_key)
            if final:
                all_finals[condition.eval_mode] = final
        if not all_finals:
            continue
        condition = eval_conditions[0]
        baseline = baselines[condition.context_key]
        print(f"  {condition.task} d{condition.domain} prompt: {condition.prompt}")
        analyze_hypothesis_condition_results(
            all_finals, baseline.ctx, baseline.hypotheses,
            condition_result_dir(condition, result_key), result_key, condition,
        )


# ==================================================================
# Per-model aggregate stages
# ==================================================================

def _alpha_beta_fit_groups(
    stimuli: list[StimulusSet],
    trajectory: dict[str, dict] | None,
    full_preds: dict[str, np.ndarray] | None = None,
) -> list[tuple[str, int | None, list[StimulusSet], dict[str, np.ndarray], dict[str, int]]]:
    """Build task-local α-β fit scopes (prefix lengths plus full stimulus sets) before cross-task pooling."""
    groups = []
    if trajectory is None:
        return groups
    stimulus_map = {stimulus.name: stimulus for stimulus in stimuli}

    def collect_scope(n: int | None):
        scope_stimuli = []
        scope_preds = {}
        counts = {"n_valid": 0, "n_exhausted": 0, "n_usable": 0}
        for stimulus_name, ev in trajectory.items():
            stimulus = stimulus_map[stimulus_name]
            target_n = len(stimulus.examples) if n is None else n
            if len(stimulus.examples) < target_n or target_n not in ev["n_examples"]:
                continue
            index = ev["n_examples"].index(target_n)
            result = np.asarray(
                full_preds[stimulus_name]
                if n is None and full_preds is not None and stimulus_name in full_preds
                else ev["results"][index],
                dtype=float,
            )
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
            scope_stimuli.append(StimulusSet(stimulus.name, stimulus.examples[:target_n], stimulus.kind))
            scope_preds[stimulus.name] = result
        if sum(counts.values()):
            groups.append((scope_label(n), n, scope_stimuli, scope_preds, counts))

    for n in PREFIX_LENGTHS:
        collect_scope(n)
    collect_scope(None)
    return groups


def _project_hypothesis_trajectory(trajectory: dict[str, dict] | None, vector_for_step) -> dict[str, dict] | None:
    """Apply a mode-specific projection while preserving the trajectory cache shape."""
    if not trajectory:
        return None
    projected = {}
    for stimulus_name in trajectory:
        ns, results, statuses = [], [], []
        for n, result in trajectory_steps(trajectory, stimulus_name).items():
            view = vector_for_step(stimulus_name, n, result)
            ns.append(n)
            results.append(np.asarray(view.vector, dtype=float))
            statuses.append("exhausted" if not result else "usable" if view.usable else "unusable")
        projected[stimulus_name] = {"n_examples": ns, "results": results, "projection_status": statuses}
    return projected or None


def _projected_trajectory(experiment, condition, result_key) -> dict[str, dict] | None:
    """Project one hypothesis-mode condition's normalized trajectory into predictive vectors."""
    trajectory = load_condition_trajectory(condition, result_key)
    if not trajectory:
        return None
    ctx = task_context(experiment, condition.task, condition.domain)
    if condition.eval_mode == "hypothesis_eval":
        stimulus_map = ctx.task.stimulus_map
        return _project_hypothesis_trajectory(
            trajectory,
            lambda name, n, result: project_hypothesis_eval(
                normalize_hypothesis_weight_dict(result), stimulus_map[name].examples[:n], ctx,
            ),
        )
    library = load_hypothesis_library(condition.domain)
    return _project_hypothesis_trajectory(
        trajectory,
        lambda _name, _n, result: project_structured_hypothesis_generation(
            normalize_structured_generation_confidence(result), library,
        ),
    )


def _add_alpha_beta_groups(bucket_map, key_prefix: tuple, task: str, model, fit_groups):
    for fit_scope, n_examples, stimuli, preds, counts in fit_groups:
        bucket = bucket_map.setdefault(
            (*key_prefix, fit_scope, n_examples),
            {"tasks": [], "specs": [], "counts": {"n_valid": 0, "n_exhausted": 0, "n_usable": 0}},
        )
        bucket["tasks"].append(task)
        if stimuli:
            bucket["specs"].append((model, preds, stimuli))
        for count_name, count in counts.items():
            bucket["counts"][count_name] += count


def _posterior_fit_groups(experiment, condition, result_key):
    finals = load_condition_final(condition, result_key)
    if not finals:
        return []
    stimuli = task_context(experiment, condition.task, condition.domain).task.stimuli
    return _alpha_beta_fit_groups(stimuli, load_condition_trajectory(condition, result_key), finals)


def _projection_divergence_metrics(view, bayesian_probs):
    """Compare executable predictions and retain projection coverage diagnostics."""
    return {
        "kl_bayes": kl_divergence(bayesian_probs, view.vector),
        "correlation": safe_correlation(bayesian_probs, view.vector),
        "comparison_space": "predictive_1_to_domain",
        "projection_semantics": view.semantics,
        "coverage": view.coverage,
        "matched_mass": view.matched_mass,
        "unmatched_mass": view.unmatched_mass,
    }


def run_divergence(experiment: Experiment, output_dir: str, result_key: str, figure_dir: str,
                   *, stage: str, eval_modes: tuple[str, ...]):
    """Divergence of each final result from the Bayesian prediction, plus posterior human KL."""
    rows, human_rows = [], []
    for condition in _conditions(experiment, *eval_modes):
        final = load_condition_final(condition, result_key)
        if not final:
            continue
        ctx = task_context(experiment, condition.task, condition.domain)
        bay_preds, bay_hyps = _bayesian_predictions(experiment, condition.task, condition.domain)
        library = load_hypothesis_library(condition.domain) if condition.eval_mode == "hypothesis_generation" else None
        for stimulus in ctx.task.stimuli:
            if stimulus.name not in final:
                continue
            result = final[stimulus.name]
            if condition.eval_mode == "posterior":
                metrics = {
                    "kl_bayes": kl_divergence(bay_preds[stimulus.name], result),
                    "correlation": safe_correlation(bay_preds[stimulus.name], result),
                }
            elif condition.eval_mode == "hypothesis_eval":
                metrics = _projection_divergence_metrics(
                    project_hypothesis_eval(normalize_hypothesis_weight_dict(result), stimulus.examples, ctx),
                    bay_preds[stimulus.name],
                )
            else:
                view = project_structured_hypothesis_generation(normalize_structured_generation_confidence(result), library)
                metrics = _projection_divergence_metrics(view, bay_preds[stimulus.name])
                bayesian_hypotheses = bay_hyps[stimulus.name]
                bay_topk = sorted(bayesian_hypotheses, key=bayesian_hypotheses.get, reverse=True)[:5]
                metrics["topk_overlap"] = compute_topk_overlap([h["hypothesis"] for h in result[:5]], bay_topk)
            rows.append({**_row_base(condition, stimulus), "eval_mode": condition.eval_mode, **metrics})

            human = (ctx.task.human_preds or {}).get(stimulus.name) if condition.eval_mode == "posterior" else None
            if human is not None:
                # KL(Human || LLM) on the shared mask of rated targets.
                mask = ~np.isnan(human)
                if mask.sum() >= 2:
                    human_rows.append({
                        **_row_base(condition, stimulus),
                        "eval_mode": "posterior",
                        "kl_human": kl_divergence(human[mask], result[mask]),
                        "correlation": safe_correlation(human[mask], result[mask]),
                    })

    df = save_csv(rows, output_dir, f"{stage}_metrics.csv")
    if "posterior" in eval_modes:
        save_csv(human_rows, output_dir, f"{stage}_human_metrics.csv")
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


def run_cross_mode_projection(experiment: Experiment, output_dir: str, result_key: str, figure_dir: str):
    """Project all three interfaces into 1..N predictive space."""
    projection_rows = []
    for condition in make_grid(experiment):
        final = load_condition_final(condition, result_key)
        if not final:
            continue
        ctx = task_context(experiment, condition.task, condition.domain)
        library = load_hypothesis_library(condition.domain) if condition.eval_mode == "hypothesis_generation" else None
        for stimulus in ctx.task.stimuli:
            if stimulus.name not in final:
                continue
            result = final[stimulus.name]
            if condition.eval_mode == "posterior":
                view = project_posterior(result)
            elif condition.eval_mode == "hypothesis_eval":
                view = project_hypothesis_eval(result, stimulus.examples, ctx)
            else:
                view = project_structured_hypothesis_generation(normalize_structured_generation_confidence(result), library)
            projection_rows.append({
                **_row_base(condition, stimulus),
                "eval_mode": condition.eval_mode,
                "projection_semantics": view.semantics,
                **{field: getattr(view, field) for field in (
                    "coverage", "total_mass", "matched_mass", "unmatched_mass", "total_items",
                    "matched_items", "unmatched_items", "matched_item_frac", "unmatched_item_frac",
                    "rule_mass", "interval_mass", "rule_items", "interval_items",
                )},
            })

    proj_df = save_csv(projection_rows, output_dir, "cross_mode_projection_metrics.csv")
    if proj_df.empty:
        return proj_df
    pooled = with_all_kinds(proj_df)
    save_csv(
        summarize_projection_rows(pooled, ["task", "domain", "h_space", "sampling", "eval_mode", "stimulus_kind"]),
        output_dir, "cross_mode_projection_summary.csv",
    )
    save_csv(
        summarize_projection_rows(pooled, ["task", "eval_mode", "stimulus_kind"]),
        output_dir, "cross_mode_projection_summary_pooled.csv",
    )
    return proj_df


def _fit_alpha_beta_buckets(buckets: dict, *, with_eval_mode: bool) -> list[dict]:
    rows = []
    for key, bucket in buckets.items():
        fit = fit_alpha_beta(bucket["specs"])
        if fit is None:
            continue
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
                print(f"    α-β fit: {rows[-1]['task']} d{domain} {h_space}_{sampling} "
                      f"α={fit['best_alpha']:.2f} β={fit['best_beta']:.2f}")
    return rows


def run_alpha_beta(experiment: Experiment, output_dir: str, result_key: str, figure_dir: str):
    """Fit posterior α-β after pooling task stimuli.

    Each row fixes domain/h_space/sampling and one fit scope: n=1..4 prefixes
    or the full stimulus sets. The task column names the pooled task group.
    """
    buckets = {}
    for condition in _conditions(experiment, "posterior"):
        _add_alpha_beta_groups(
            buckets, (condition.domain, condition.h_space, condition.sampling), condition.task,
            bayesian_model(experiment, condition.task, condition.domain),
            _posterior_fit_groups(experiment, condition, result_key),
        )
    df = save_csv(_fit_alpha_beta_buckets(buckets, with_eval_mode=False), output_dir, "alpha_beta_fits.csv")

    if not df.empty:
        bay_ab, human_ab = load_baseline_overlay("alpha_beta_fits.csv")
        overlay = {"Bayesian": bay_ab, "Human": human_ab}
        for metric, ylabel, title, filename, axhline in (
            ("alpha", "α* (prior weight)", "h_space -> α by fit scope", "hspace_alpha.png", 1.0),
            ("mse", "loss", "loss", "loss.png", None),
            ("beta", "β* (size principle)", "β by fit scope", "domain_beta.png", 1.0),
        ):
            plot_prompt_fit_scope_bar(
                df, metric, ylabel, title, os.path.join(figure_dir, filename),
                result_key=result_key, axhline=axhline, baseline_overlay=overlay,
            )
    return df


def run_alpha_beta_modes(experiment: Experiment, output_dir: str, result_key: str, figure_dir: str):
    """Fit posterior, eval, and structured-generation projections in shared predictive space."""
    buckets = {}
    for condition in make_grid(experiment):
        if condition.eval_mode == "posterior":
            groups = _posterior_fit_groups(experiment, condition, result_key)
        else:
            projected = _projected_trajectory(experiment, condition, result_key)
            if not projected:
                continue
            stimuli = task_context(experiment, condition.task, condition.domain).task.stimuli
            groups = _alpha_beta_fit_groups(stimuli, projected)
        _add_alpha_beta_groups(
            buckets, (condition.domain, condition.eval_mode, condition.h_space, condition.sampling), condition.task,
            bayesian_model(experiment, condition.task, condition.domain), groups,
        )
    return save_csv(_fit_alpha_beta_buckets(buckets, with_eval_mode=True), output_dir, "alpha_beta_mode_fits.csv")


def run_prompt_effects(experiment: Experiment, output_dir: str, result_key: str, figure_dir: str):
    """Compute posterior prompt sensitivity and effective support."""
    sensitivity_rows, support_rows = [], []
    for condition in _conditions(experiment, "posterior"):
        final = load_condition_final(condition, result_key)
        if not final:
            continue
        bayesian_preds, _ = _bayesian_predictions(experiment, condition.task, condition.domain)
        for stimulus in task_context(experiment, condition.task, condition.domain).task.stimuli:
            llm_post = final.get(stimulus.name)
            if llm_post is None:
                continue
            support_rows.append({**_row_base(condition, stimulus), "effective_support": compute_effective_support(llm_post)})
            sensitivity_rows.append({**_row_base(condition, stimulus), "kl_bayes": kl_divergence(bayesian_preds[stimulus.name], llm_post)})

    if sensitivity_rows:
        sens_df = save_csv(sensitivity_rows, output_dir, "prompt_sensitivity.csv")
        plot_prompt_sensitivity(sens_df, os.path.join(figure_dir, "prompt_sensitivity.png"))
    if support_rows:
        supp_df = save_csv(support_rows, output_dir, "prompt_effective_support.csv")
        bay_support, human_support = load_baseline_overlay("effective_support.csv")
        plot_generalization_width(
            supp_df, os.path.join(figure_dir, "generalization_width.png"), result_key,
            baseline_overlay={"Bayesian": bay_support, "Human": human_support})
    return sensitivity_rows, support_rows


TRAJECTORY_PLOT_SPECS = [
    ("kl_bayes", "trajectory_kl.png", r"KL(Bayesian $\Vert$ LLM)", False),
    ("entropy", "trajectory_entropy.png", "Entropy", True),
    ("effective_support", "trajectory_effective_support.png", "Effective support (p > 0.5)", True),
    ("alpha_beta_ratio", "trajectory_alpha_beta_ratio.png", "log(α / β)", True),
]
ALPHA_BETA_SAMPLINGS = ("default", "strong", "weak")


def _annotate_trajectory_rows(df, task, domain, source, h_space=None, sampling=None):
    """Add cross-condition identifiers expected by the shared trajectory plotter."""
    if df.empty:
        return []
    df = df.assign(task=task, domain=domain, task_domain=f"{task}/d{domain}", source=source)
    if h_space is not None:
        df = df.assign(h_space=h_space, sampling=sampling)
    return df.to_dict("records")


def run_posterior_trajectory(experiment: Experiment, output_dir: str, result_key: str, figure_dir: str):
    """Compute posterior KL, support, and α-β prefix trajectories."""
    kl_rows, metric_rows, alpha_beta_rows = [], [], []
    prefix_horizon = experiment.scoring.trajectory_prefix_horizon

    for (task, domain), conditions in group_by(_conditions(experiment, "posterior"), lambda c: c.context_key):
        ctx = task_context(experiment, task, domain)
        model = bayesian_model(experiment, task, domain)
        bayes_stimuli, _ = clip_to_prefix_horizon(ctx.task.stimuli, {}, prefix_horizon)
        bayes_traj = bayesian_prefix_trajectories(model, bayes_stimuli)
        metric_rows.extend(_annotate_trajectory_rows(
            size_principle_curves(bayes_traj, bayes_stimuli, "bayesian", bayesian_trajectories=bayes_traj),
            task, domain, "bayesian"))

        for condition in conditions:
            ev = load_condition_trajectory(condition, result_key)
            if not ev:
                continue
            traj_stimuli, traj_ev = clip_to_prefix_horizon(ctx.task.stimuli, ev, prefix_horizon)
            traj_stimuli = [c for c in traj_stimuli if c.name in traj_ev]
            prompt = (condition.h_space, condition.sampling)
            metric_rows.extend(_annotate_trajectory_rows(
                size_principle_curves(traj_ev, traj_stimuli, "llm", bayesian_trajectories=bayes_traj),
                task, domain, "llm", *prompt))
            if ctx.task.human_preds:
                metric_rows.extend(_annotate_trajectory_rows(
                    human_kl_curves(traj_ev, traj_stimuli, ctx.task.human_preds), task, domain, "llm", *prompt))

            ab_traj = compute_trajectory_alpha_beta(model, *clip_to_prefix_horizon(ctx.task.stimuli, ev, prefix_horizon))
            for n, alpha, beta, loss, n_stimuli in zip(
                ab_traj["n_examples"], ab_traj["alpha"], ab_traj["beta"], ab_traj["loss"], ab_traj["n_stimuli"],
            ):
                alpha_beta_rows.append({
                    "task": task, "domain": domain, "h_space": condition.h_space, "sampling": condition.sampling,
                    "task_domain": f"{task}/d{domain}", "source": "llm", "n_examples": n,
                    "alpha": alpha, "beta": beta, "mse": loss, "n_stimuli": n_stimuli,
                    "alpha_beta_ratio": alpha_beta_ratio(alpha, beta),
                })

            stimulus_map = ctx.task.stimulus_map
            for stimulus_name, cd in compute_trajectory(model, traj_stimuli, traj_ev).items():
                rate = compute_trajectory_rate(cd["kl"])
                for n, kl, weight in zip(cd["n_examples"], cd["kl"], cd["bayesian_weight"]):
                    kl_rows.append({
                        **_row_base(condition, stimulus_map[stimulus_name]),
                        "n_examples": n, "kl_bayes": kl, "bayesian_weight": weight, "trajectory_rate": rate,
                    })

    if kl_rows:
        kl_df = save_csv(kl_rows, output_dir, "trajectory_metrics.csv")
        plot_trajectory_rate(kl_df, os.path.join(figure_dir, "trajectory_rate.png"), result_key)

    if metric_rows or alpha_beta_rows:
        metrics_df = save_csv(metric_rows + alpha_beta_rows, output_dir, "trajectory_prefix_metrics.csv")
        for metric, filename, ylabel, include_bayesian in TRAJECTORY_PLOT_SPECS:
            plot_prompt_trajectory(metrics_df, metric, ylabel, os.path.join(figure_dir, filename),
                                   include_bayesian=include_bayesian)
        for sampling in ALPHA_BETA_SAMPLINGS:
            plot_sampling_alpha_beta_trajectory(
                metrics_df, sampling, os.path.join(figure_dir, f"trajectory_alpha_beta_{sampling}.png"),
            )


def run_generation_narrowing(experiment: Experiment, output_dir: str, result_key: str, figure_dir: str):
    """Count unique structured generations over evidence."""
    rows = []
    for condition in _conditions(experiment, "hypothesis_generation"):
        trajectory = load_condition_trajectory(condition, result_key)
        if not trajectory:
            continue
        stimulus_map = task_context(experiment, condition.task, condition.domain).task.stimulus_map
        for stimulus_name, ed in trajectory.items():
            base = _row_base(condition, stimulus_map[stimulus_name])
            for n, generated in zip(ed["n_examples"], ed["results"]):
                rows.append({**base, "n_examples": n, "n_unique": len({item["hypothesis"] for item in generated})})

    if rows:
        save_csv(rows, output_dir, "generation_narrowing_counts.csv")
        plot_narrowing_aggregate(rows, os.path.join(figure_dir, "narrowing.png"), result_key)


def _type_columns(props: dict[str, float]) -> dict[str, float]:
    return {f"cw_{typ}": value for typ, value in props.items()}


def run_hypothesis_generation(experiment: Experiment, output_dir: str, result_key: str, figure_dir: str):
    """Hypothesis-type composition for eval and structured generation, per final and per prefix."""
    rows, eval_rows, trajectory_rows = [], [], []

    for condition in _conditions(experiment, "hypothesis_eval", "hypothesis_generation"):
        trajectory = load_condition_trajectory(condition, result_key)
        if not trajectory:
            continue
        ctx = task_context(experiment, condition.task, condition.domain)
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
                            ("rule", view.rule_mass), ("interval", view.interval_mass), ("other", view.unmatched_mass),
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
                **{f"{typ}_prop": value for typ, value in cm["type_proportions"].items()},
                **_type_columns(cm["confidence_weighted_types"]),
                "mean_confidence": cm["mean_confidence"],
                "top1_confidence_norm": cm["top1_confidence_norm"],
                "confidence_entropy": cm["confidence_entropy"],
            })
        for stimulus_name, ed in trajectory.items():
            for n, result in zip(ed["n_examples"], ed["results"]):
                if not result:
                    continue
                types = [hypothesis_type(item) for item in result]
                counts = {typ: types.count(typ) for typ in ("rule", "interval", "other")}
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

    df = save_csv(rows, output_dir, "hypothesis_generation_metrics.csv")
    rule_parts = []
    if eval_rows:
        eval_df = save_csv(eval_rows, output_dir, "hypothesis_generation_confidence_weighted_types_eval.csv")
        plot_confidence_weighted_types(
            eval_df, os.path.join(figure_dir, "confidence_weighted_types_eval.png"),
            result_key, title="Confidence-Weighted Types: Eval")
        rule_parts.append(eval_df)
    if trajectory_rows:
        traj_df = save_csv(trajectory_rows, output_dir, "hypothesis_generation_trajectory.csv")
        plot_confidence_weighted_types(
            traj_df, os.path.join(figure_dir, "confidence_weighted_types_gen.png"),
            result_key, title="Confidence-Weighted Types: Generate")
        rule_parts.append(traj_df)
    if rule_parts:
        plot_trajectory_rule_by_kind(pd.concat(rule_parts, ignore_index=True),
                                     os.path.join(figure_dir, "trajectory_rule_by_kind.png"))
    return df


DEFAULT_STAGES = {
    "divergence": partial(run_divergence, stage="divergence", eval_modes=("posterior",)),
    "alpha_beta": run_alpha_beta,
    "prompt": run_prompt_effects,
    "trajectory": run_posterior_trajectory,
    "rt_correlation": run_rt_correlation,
}

HYPOTHESIS_STAGES = {
    "cross_mode": run_cross_mode_projection,
    "nested_pair": run_nested_pair,
    "alpha_beta_mode": run_alpha_beta_modes,
    "generation_narrowing": run_generation_narrowing,
    "hypothesis_divergence": partial(
        run_divergence, stage="hypothesis_divergence", eval_modes=("hypothesis_eval", "hypothesis_generation"),
    ),
    "hypothesis_generation": run_hypothesis_generation,
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
    model_dir = model_root(result_key)
    output_dir = analysis_dir(result_key)
    # Validate every baseline once before stages begin, without recomputing
    # model-agnostic baselines during an LLM analysis run.
    build_bayesian(experiment, make_grid(experiment))
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
            per_condition_fn(domain_experiment, result_key)
        for stage_name, stage_fn in stages.items():
            _print_stage(stage_name)
            stage_fn(domain_experiment, domain_output_dir, result_key, domain_figure_dir)

    for stage_name, stage_fn in model_stages.items():
        _print_stage(stage_name)
        stage_fn(experiment, output_dir, result_key, model_dir)
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
