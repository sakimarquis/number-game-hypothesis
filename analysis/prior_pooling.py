"""Pooled prior of the size-principle fit, and the 4-fold comparison that chose it.

The α-β fits (`alpha_beta` stages, trajectories, human baselines) fix the prior to a pooled prior learned
once per prompt from the Predict data of every summary model and its thinking variant at d100, plus humans
for the default prompt: log p(h) = α log p0(h) + a[base(h)] + c[transform(h)], with one α and one offset per
rule base and transform shared by every source and evidence count n, while β is one per (source, n). Rules take
their base and transform from the task's rule families, and all intervals share one base. `pooled_prior`
caches each prompt's prior under `results/prior_pooling/`; `pooled_priors` applies it to a Bayesian model.

`python -m analysis.prior_pooling` reruns the comparison of how the prior parameter is pooled
(`prior_pooling_cv.csv`). Rows are the unique (task, prefix) inputs of each source, tasks pooled; humans
contribute their full sets at n = |set|. β is anchored at 1 with the repo's alpha-beta lambda.
"""

import os
from dataclasses import asdict

import numpy as np
import torch

from analysis.cache import bayesian_baseline, condition_analysis_path, load_condition_trajectory
from analysis.metrics import safe_correlation
from analysis.model import ALPHA_BETA_REG_LAMBDA as REG
from analysis.plotter import SUMMARY_MODEL_ORDER, save_csv
from configs.experiment import load_experiment
from configs.settings import BAYESIAN, HUMAN, RESULTS_ROOT, THINKING_SUFFIX, ExperimentCondition
from tasks.stimuli import BASE_DOMAIN, build_task_context
from utils import load_json, resolve_path, save_json

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
TASKS = ("Tenenbaum99", "Bigelow16")
SHAPE_REG = 1e-3
FOLDS = 4
PRIOR_SCHEMA = 1
DEFAULT_PROMPT = ("implicit", "default")
# Each config: which groups share alpha (None = alpha fixed at 1), the anchor weight on log alpha, and the prior form.
CONFIGS = {
    "1_alpha_per_source_n": {"alpha": "source_n", "alpha_reg": REG, "prior": None},
    "2_alpha_per_source_n_reg0.1": {"alpha": "source_n", "alpha_reg": 0.1, "prior": None},
    "3_alpha_per_n": {"alpha": "n", "alpha_reg": REG, "prior": None},
    "4_alpha_per_source": {"alpha": "source", "alpha_reg": REG, "prior": None},
    "5_alpha_plus_pooled_shape": {"alpha": "source_n", "alpha_reg": REG, "prior": "shape"},  # alpha log p0 + v
    "6_rank1": {"alpha": None, "alpha_reg": 0.0, "prior": "rank1"},  # log p0 + v0 + w_(source, n) v
    "7_alpha_per_n_plus_pooled_shape": {"alpha": "n", "alpha_reg": REG, "prior": "shape"},
    "8_alpha_per_source_plus_pooled_shape": {"alpha": "source", "alpha_reg": REG, "prior": "shape"},
    "9_alpha_global_plus_pooled_shape": {"alpha": "all", "alpha_reg": REG, "prior": "shape"},
}
PRIOR_CONFIG = CONFIGS["9_alpha_global_plus_pooled_shape"]
_PRIORS: dict[tuple[str, str], dict] = {}


def main():
    experiment = load_experiment(model_override=BAYESIAN)
    rows = load_rows(*DEFAULT_PROMPT)
    hyps = {task: task_hypotheses(experiment, task) for task in TASKS}
    out = []
    for name, config in CONFIGS.items():
        print(f"{name} ...", flush=True)
        out += held_out_r2(rows, hyps, config, name)
    df = save_csv(out, prior_dir(), "prior_pooling_cv.csv")
    table = df[df.n_examples == "all"].pivot_table(index="source", columns="config", values="r2")
    print(table.round(3).to_string())
    print("LLM mean:\n" + table.drop(index=HUMAN).mean().round(3).to_string())


def prior_dir() -> str:
    return os.path.join(resolve_path(RESULTS_ROOT), "prior_pooling")


def pooled_prior(experiment, h_space: str = "implicit", sampling: str = "default") -> dict:
    """One prompt's pooled prior {alpha, base, transform}; refit when its sources or settings change."""
    key = (h_space, sampling)
    if key in _PRIORS:
        return _PRIORS[key]
    identity = {
        "schema": PRIOR_SCHEMA, "h_space": h_space, "sampling": sampling, "sources": prior_sources(h_space, sampling),
        "bayesian": asdict(experiment.bayesian), "beta_reg": REG, "shape_reg": SHAPE_REG,
    }
    path = os.path.join(prior_dir(), f"pooled_prior_{h_space}_{sampling}.json")
    prior = load_json(path) if os.path.exists(path) else None
    if prior is None or prior["identity"] != identity:
        print(f"  Fitting pooled prior for {h_space}_{sampling} over {len(identity['sources'])} sources")
        batch = Batch(load_rows(h_space, sampling), {task: task_hypotheses(experiment, task) for task in TASKS})
        params = fit(batch, PRIOR_CONFIG, train=torch.ones_like(batch.n, dtype=torch.bool))
        prior = {
            "identity": identity,
            "alpha": float(torch.exp(params["log_alpha"][0])),
            "base": dict(zip(batch.bases, params["a"].tolist())),
            "transform": dict(zip(batch.transforms, params["c"].tolist())),
        }
        save_json(path, prior)
    _PRIORS[key] = prior
    return prior


def pooled_priors(prior: dict, model, rule_families) -> torch.Tensor:
    """Hypothesis prior weights exp(α log p0 + a[base] + c[transform]) for one Bayesian model."""
    offsets = torch.tensor([prior["base"].get(base, 0.0) + prior["transform"].get(transform, 0.0)
                            for base, transform in hypothesis_labels(model, rule_families)], dtype=DT, device=model.priors.device)
    return torch.exp(prior["alpha"] * torch.log(model.priors) + offsets)


def hypothesis_labels(model, rule_families) -> list[tuple[str, str]]:
    """(base, transform) of each hypothesis in the Bayesian model's order; intervals share one base."""
    base_of = {rule: base for base, members in (rule_families or {}).items() for rule in members}
    labels = []
    for description in model.descriptions[: model.n_math]:
        base = base_of.get(description, description)
        labels.append((base, description[len(base) + 2: -1] if description != base else "identity"))
    return labels + [("interval", "identity")] * (len(model.descriptions) - model.n_math)


def prior_sources(h_space: str, sampling: str) -> list[str]:
    """Summary models (and thinking variants) with Predict caches for this prompt; humans join the default prompt."""
    llms = [
        source for model in SUMMARY_MODEL_ORDER for source in (model, f"{model}{THINKING_SUFFIX}")
        if any(os.path.exists(condition_analysis_path(_condition(task, h_space, sampling), source)) for task in TASKS)
    ]
    return ([HUMAN] if (h_space, sampling) == DEFAULT_PROMPT else []) + llms


def _condition(task: str, h_space: str, sampling: str) -> ExperimentCondition:
    return ExperimentCondition(task, "posterior", h_space, sampling, BASE_DOMAIN)


def load_rows(h_space: str, sampling: str) -> list[tuple[str, str, tuple[int, ...], np.ndarray]]:
    """(source, task, examples, target) for each unique input of each prior source."""
    rows = []
    for source in prior_sources(h_space, sampling):
        if source == HUMAN:
            for task in TASKS:
                task_data = build_task_context(task, BASE_DOMAIN).task
                rows += [(HUMAN, task, tuple(s.examples), task_data.human_preds[s.name]) for s in task_data.stimuli]
            continue
        seen = set()
        for task in TASKS:
            examples = build_task_context(task, BASE_DOMAIN).task.stimulus_map
            for name, ev in (load_condition_trajectory(_condition(task, h_space, sampling), source) or {}).items():
                for n, result in zip(ev["n_examples"], ev["results"]):
                    key = (task, tuple(examples[name].examples[:n]))
                    if key not in seen:
                        seen.add(key)
                        rows.append((source, *key, np.asarray(result, dtype=float)))
    return rows


def task_hypotheses(experiment, task: str) -> dict:
    """Extensions, baseline log prior, and (base, transform) labels of one task's d100 Bayesian model."""
    model = bayesian_baseline(experiment, task, BASE_DOMAIN).model
    labels = hypothesis_labels(model, build_task_context(task, BASE_DOMAIN).rule_families)
    return {"ext": model.extensions.to(DT), "logp0": torch.log(model.priors), "labels": labels}


def held_out_r2(rows, hyps, config: dict, name: str) -> list[dict]:
    """Pooled r2 of 4-fold held-out predictions per source and per (source, n); folds are keyed by input."""
    batch = Batch(rows, hyps)
    keys = sorted({(task, ex) for _s, task, ex, _t in rows})
    perm = torch.randperm(len(keys), generator=torch.Generator().manual_seed(0)).tolist()
    fold_of = {key: perm[i] % FOLDS for i, key in enumerate(keys)}
    folds = torch.tensor([fold_of[(task, ex)] for _s, task, ex, _t in rows], device=DEV)
    pred = torch.zeros_like(batch.target)
    for fold in range(FOLDS):
        params = fit(batch, config, train=folds != fold)
        with torch.no_grad():
            pred[folds == fold] = batch.predict(config, params)[folds == fold]
    n_params = sum(p.numel() for p in init_params(batch, config).values())
    out = []
    for source in dict.fromkeys(r[0] for r in rows):
        for n in ("all", 1, 2, 3, 4):
            sel = (batch.source == batch.sources.index(source)) & ((batch.n == n) if n != "all" else True)
            if sel.any():
                m = batch.mask[sel]
                r = safe_correlation(pred[sel][m].cpu().numpy(), batch.target[sel][m].cpu().numpy())
                out.append({"config": name, "source": source, "n_examples": n, "n_inputs": int(sel.sum()),
                            "r2": r ** 2, "n_params": n_params})
    return out


class Batch:
    """Padded per-row tensors; rows of one task share that task's extensions."""

    def __init__(self, rows, hyps):
        self.sources = list(dict.fromkeys(r[0] for r in rows))
        self.source = torch.tensor([self.sources.index(r[0]) for r in rows], device=DEV)
        self.n = torch.tensor([len(r[2]) for r in rows], device=DEV)
        target = np.stack([r[3] for r in rows])
        self.mask = torch.tensor(np.isfinite(target), device=DEV)
        self.target = torch.tensor(np.nan_to_num(target), dtype=DT, device=DEV)
        self.bases = list(dict.fromkeys(b for h in hyps.values() for b, _ in h["labels"]))
        self.transforms = list(dict.fromkeys(c for h in hyps.values() for _, c in h["labels"]))
        S, H = len(rows), max(h["ext"].shape[0] for h in hyps.values())
        self.logw0 = torch.full((S, H), -torch.inf, dtype=DT, device=DEV)  # log p0 on consistent hypotheses
        self.loglik = torch.zeros(S, H, dtype=DT, device=DEV)  # -n log|h|
        self.ia = torch.zeros(S, H, dtype=torch.long, device=DEV)
        self.ic = torch.zeros(S, H, dtype=torch.long, device=DEV)
        self.tasks = {}
        for task, h in hyps.items():
            idx = torch.tensor([i for i, r in enumerate(rows) if r[1] == task], device=DEV)
            ext, k = h["ext"], h["ext"].shape[0]
            shown = torch.zeros(len(idx), ext.shape[1], dtype=DT, device=DEV)
            for j, i in enumerate(idx.tolist()):
                shown[j, [x - 1 for x in rows[i][2]]] = 1
            consistent = (shown @ (1 - ext).T) == 0  # [rows, hypotheses]
            self.logw0[idx, :k] = torch.where(consistent, h["logp0"], -torch.inf)
            self.loglik[idx, :k] = -self.n[idx, None] * torch.log(ext.sum(1))
            self.ia[idx, :k] = torch.tensor([self.bases.index(b) for b, _ in h["labels"]], device=DEV)
            self.ic[idx, :k] = torch.tensor([self.transforms.index(c) for _, c in h["labels"]], device=DEV)
            self.tasks[task] = (idx, ext)
        # Contiguous group ids, so every parameter belongs to a nonempty group.
        self.groups = {k: torch.unique(v, return_inverse=True)[1] for k, v in
                       {"source_n": self.source * 5 + self.n, "n": self.n, "source": self.source,
                        "all": torch.zeros_like(self.n)}.items()}

    def predict(self, config: dict, params: dict) -> torch.Tensor:
        logp = torch.where(torch.isfinite(self.logw0), self.logw0, torch.zeros_like(self.logw0))
        if config["alpha"]:
            logp = torch.exp(params["log_alpha"][self.groups[config["alpha"]]])[:, None] * logp
        if config["prior"] == "shape":
            logp = logp + params["a"][self.ia] + params["c"][self.ic]
        if config["prior"] == "rank1":
            w = params["w"][self.groups["source_n"]][:, None]
            logp = logp + params["a"][self.ia] + params["c"][self.ic] + w * (params["a1"][self.ia] + params["c1"][self.ic])
        beta = torch.exp(params["log_beta"][self.groups["source_n"]])[:, None]
        logw = torch.where(torch.isfinite(self.logw0), logp + beta * self.loglik, -torch.inf)
        weights = torch.softmax(logw, 1)  # [rows, hypotheses]
        pred = torch.zeros_like(self.target)
        for idx, ext in self.tasks.values():
            pred[idx] = weights[idx, : ext.shape[0]] @ ext
        return pred


def init_params(batch: Batch, config: dict) -> dict:
    zeros = lambda k: torch.zeros(k, dtype=DT, device=DEV)
    params = {"log_beta": zeros(int(batch.groups["source_n"].max()) + 1)}
    if config["alpha"]:
        params["log_alpha"] = zeros(int(batch.groups[config["alpha"]].max()) + 1)
    if config["prior"]:
        params |= {"a": zeros(len(batch.bases)), "c": zeros(len(batch.transforms))}
    if config["prior"] == "rank1":
        # w = v = 0 is a saddle point; a fixed random w breaks the symmetry.
        w = 0.5 * torch.randn(len(params["log_beta"]), generator=torch.Generator().manual_seed(0), dtype=DT)
        params |= {"a1": zeros(len(batch.bases)), "c1": zeros(len(batch.transforms)), "w": w.to(DEV)}
    return params


def fit(batch: Batch, config: dict, train: torch.Tensor) -> dict:
    """L-BFGS on mean per-row MSE over training rows plus the anchors and shape ridges."""
    params = {k: v.requires_grad_(True) for k, v in init_params(batch, config).items()}
    mask = batch.mask & train[:, None]
    keep = mask.any(1)
    opt = torch.optim.LBFGS(list(params.values()), max_iter=300, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        sq = ((batch.predict(config, params) - batch.target) ** 2 * mask).sum(1) / mask.sum(1).clamp(min=1)
        loss = sq[keep].mean() + REG * (params["log_beta"] ** 2).mean()
        if config["alpha"]:
            loss = loss + config["alpha_reg"] * (params["log_alpha"] ** 2).mean()
        for k in ("a", "c", "a1", "c1"):
            if k in params:
                loss = loss + SHAPE_REG * (params[k] ** 2).mean()
        if "w" in params:
            loss = loss + REG * (params["w"] ** 2).mean()
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(closure)
    return {k: v.detach() for k, v in params.items()}


if __name__ == "__main__":
    main()
