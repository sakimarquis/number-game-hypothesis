"""Shared alpha-beta fit-scope vocabulary and analysis row schemas."""

# Evidence prefix lengths analyzed as separate fit scopes and trajectory points.
PREFIX_LENGTHS = (1, 2, 3, 4)
FULL_SCOPE = "full"
FIT_SCOPE_ORDER = tuple(f"n={n}" for n in PREFIX_LENGTHS) + (FULL_SCOPE,)
ALPHA_BETA_COLUMNS = ["task", "domain", "h_space", "sampling", "fit_scope", "n_examples", "alpha", "beta", "mse", "r2", "n_stimuli"]


def scope_label(n_examples: int | None) -> str:
    return FULL_SCOPE if n_examples is None else f"n={n_examples}"


def task_group_label(tasks: list[str]) -> str:
    """Pooled task group recorded on task-pooled fit rows, e.g. 'Tenenbaum99+Bigelow16'."""
    return "+".join(dict.fromkeys(tasks))


def stimulus_row(task: str, stimulus, domain: int, h_space: str = "", sampling: str = "") -> dict:
    """Identifying columns of a per-stimulus analysis row; baselines leave the prompt empty."""
    return {
        "task": task, "stimulus": stimulus.name, "stimulus_kind": stimulus.kind,
        "domain": domain, "h_space": h_space, "sampling": sampling,
    }


def alpha_beta_row(
    task: str,
    domain: int,
    h_space: str,
    sampling: str,
    fit_scope: str,
    n_examples: int | None,
    fit: dict,
    *,
    eval_mode: str | None = None,
) -> dict:
    """Shared row schema for all alpha-beta cache writers; `eval_mode` selects the per-mode schema."""
    row = dict(zip(ALPHA_BETA_COLUMNS, (
        task, domain, h_space, sampling, fit_scope, n_examples,
        fit["alpha"], fit["beta"], fit["mse"], fit["r2"], fit["n_stimuli"],
    )))
    if eval_mode is None:
        return row
    # `task` keeps its leading position; eval_mode follows it.
    return {"task": task, "eval_mode": eval_mode, **row, **{key: fit[key] for key in ("n_valid", "n_exhausted", "n_usable")}}
