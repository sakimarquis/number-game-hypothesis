"""Shared alpha-beta fit-scope vocabulary and row schema."""

# Evidence prefix lengths analyzed as separate fit scopes and trajectory points.
PREFIX_LENGTHS = (1, 2, 3, 4)
FULL_SCOPE = "full"
FIT_SCOPE_ORDER = tuple(f"n={n}" for n in PREFIX_LENGTHS) + (FULL_SCOPE,)
ALPHA_BETA_COLUMNS = ["task", "domain", "h_space", "sampling", "fit_scope", "n_examples", "alpha", "beta", "mse", "r2", "n_stimuli"]
ALPHA_BETA_MODE_COLUMNS = [
    "task", "eval_mode", "domain", "h_space", "sampling", "fit_scope", "n_examples",
    "alpha", "beta", "mse", "r2", "n_stimuli", "n_valid", "n_exhausted", "n_usable",
]


def scope_label(n_examples: int | None) -> str:
    return FULL_SCOPE if n_examples is None else f"n={n_examples}"


def task_group_label(tasks: list[str]) -> str:
    """Pooled task group recorded on task-pooled fit rows, e.g. 'Tenenbaum99+Bigelow16'."""
    return "+".join(dict.fromkeys(tasks))


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
    row = {
        "task": task,
        "domain": domain,
        "h_space": h_space,
        "sampling": sampling,
        "fit_scope": fit_scope,
        "n_examples": n_examples,
        "alpha": fit["best_alpha"],
        "beta": fit["best_beta"],
        "mse": fit["best_loss"],
        "r2": fit["r2"],
        "n_stimuli": fit["n_stimuli"],
    }
    if eval_mode is None:
        return row
    row = {**row, "eval_mode": eval_mode, **{key: fit[key] for key in ("n_valid", "n_exhausted", "n_usable")}}
    return {key: row[key] for key in ALPHA_BETA_MODE_COLUMNS}
