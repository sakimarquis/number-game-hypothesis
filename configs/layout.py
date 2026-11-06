"""Single source of truth for filesystem paths under RESULTS_ROOT."""

import os

from configs.settings import HUMAN, RESULTS_ROOT, ExperimentCondition
from tasks.stimuli import BASE_DOMAIN
from utils import resolve_path


# Aggregate artifacts for the base domain live directly under their root; other
# domains use a `d{domain}/` subdirectory (or a `d{domain}_` file prefix).
def domain_subdir(base_dir: str, domain: int) -> str:
    return base_dir if domain == BASE_DOMAIN else os.path.join(base_dir, f"d{domain}")


def domain_file_prefix(domain: int) -> str:
    return "" if domain == BASE_DOMAIN else f"d{domain}_"


def _results_root(results_root: str | None) -> str:
    return resolve_path(RESULTS_ROOT) if results_root is None else results_root


def model_root(result_key: str, *, results_root: str | None = None) -> str:
    """Per-model root: RESULTS_ROOT/{result_key}/"""
    return os.path.join(_results_root(results_root), result_key)


def condition_result_dir(condition: ExperimentCondition, result_key: str, *, results_root: str | None = None) -> str:
    """Result path: results/{model}/{task}/d{domain}/{h_space}_{sampling}/"""
    root = model_root(result_key, results_root=results_root)
    return os.path.join(root, condition.task, f"d{condition.domain}", condition.prompt)


def analysis_dir(result_key: str, *, results_root: str | None = None) -> str:
    """Flat analysis-cache dir for one model: {model_root}/analysis/"""
    return os.path.join(model_root(result_key, results_root=results_root), "analysis")


def condition_data_path(condition: ExperimentCondition, result_key: str, name: str,
                        *, results_root: str | None = None) -> str:
    """Condition data file: {model}/{task}/d{domain}/{prompt}/data/{name}.json"""
    return os.path.join(condition_result_dir(condition, result_key, results_root=results_root), "data", f"{name}.json")


def condition_batch_key(condition: ExperimentCondition, suffix: str) -> str:
    return f"{condition.task}-d{condition.domain}-{condition.h_space}-{condition.sampling}-{suffix}"


def parse_failure_path(condition: ExperimentCondition, result_key: str) -> str:
    """Sibling parse-failure log: {condition_dir}/data/{eval_mode}_parse_failures.json"""
    return condition_data_path(condition, result_key, f"{condition.eval_mode}_parse_failures")


def text_posterior_progress_dir(condition: ExperimentCondition, result_key: str) -> str:
    """Transient request-level progress for resumable online text-trial scoring."""
    return os.path.join(
        condition_result_dir(condition, result_key), "data", ".posterior_text_trials_progress",
    )


def bayesian_cache_dir(task_name: str, domain: int, *, results_root: str | None = None) -> str:
    """Cached Bayesian predictions: results/bayesian/{task}/d{domain}/"""
    return os.path.join(_results_root(results_root), "bayesian", task_name, f"d{domain}")


def bayesian_cache_paths(task_name: str, domain: int, *, results_root: str | None = None) -> dict[str, str]:
    """Bayesian prediction, rule-posterior, and identity files for one task/domain."""
    cache_dir = bayesian_cache_dir(task_name, domain, results_root=results_root)
    return {
        "predictions": os.path.join(cache_dir, "bayesian.json"),
        "rule_posteriors": os.path.join(cache_dir, "bayesian_hypotheses.json"),
        "identity": os.path.join(cache_dir, "cache_identity.json"),
    }


def human_analysis_dir(task_name: str, domain: int) -> str:
    """Model-agnostic human alpha-beta analysis: results/human/{task}/d{domain}/"""
    return os.path.join(model_root(HUMAN), task_name, f"d{domain}")
