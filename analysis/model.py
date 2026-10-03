"""Bayesian concept learning model (Tenenbaum 1999) and derived fits.

Computes p(y in C | X) for all y in {1..100} given observed examples X,
using hypothesis averaging with the size principle.

  p(y in C | X) = sum_{h: X subset h, y in h} w(h)
                  ------------------------------------
                  sum_{h: X subset h}               w(h)

  where w(h) = p(h) / |h|^n   (prior * likelihood up to normalization)

Also provides α-β parameterized fits: w(h|X) ∝ p(h)^α × (1/|h|^n)^β
for decomposing LLM behavior into prior vs size-principle components.

Internal arrays (extensions, priors, sizes) are stored as GPU tensors.
Public methods return GPU tensors; callers convert at save/plot boundaries.
"""

from __future__ import annotations

import numpy as np
import torch

from tasks.hypotheses import Hypothesis
from tasks.stimuli import StimulusSet

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _erlang_pdf(s: np.ndarray, sigma: float) -> np.ndarray:
    """Erlang(k=2) unnormalized density: f(s) = s * exp(-s / theta), theta = sigma/2."""
    theta = sigma / 2.0
    return s * np.exp(-s / theta)


def _consistent_mask(extensions: torch.Tensor, examples: list[int]) -> torch.Tensor:
    """Boolean mask of hypotheses consistent with all examples."""
    idx = torch.tensor([x - 1 for x in examples], dtype=torch.long, device=extensions.device)
    return extensions[:, idx].all(dim=1)


def _hypothesis_weights(
    extensions: torch.Tensor,
    sizes: torch.Tensor,
    priors: torch.Tensor,
    examples: list[int],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Normalized hypothesis weights: w(h) ∝ p(h)^α × (1/|h|^n)^β for consistent h.

    Returns tensor of length n_hyps. Zeros when no hypotheses are consistent.
    """
    consistent = _consistent_mask(extensions, examples)
    n_hyps = extensions.shape[0]
    device = extensions.device

    if not consistent.any():
        weights = torch.zeros(n_hyps, dtype=torch.float64, device=device)
        if torch.is_tensor(alpha) or torch.is_tensor(beta):
            alpha_t = torch.as_tensor(alpha, dtype=torch.float64, device=device)
            beta_t = torch.as_tensor(beta, dtype=torch.float64, device=device)
            return weights + (alpha_t + beta_t) * 0.0
        return weights

    n = len(examples)
    log_sizes = torch.log(sizes)

    beta_t = torch.as_tensor(beta, dtype=torch.float64, device=device)

    # Scalar alpha=0 is a public likelihood-only ablation contract.
    # Tensor alpha comes from positive log-parameter optimization, so keep the
    # differentiable prior term active for autograd.
    if torch.is_tensor(alpha) or alpha > 0:
        alpha_t = torch.as_tensor(alpha, dtype=torch.float64, device=device)
        mask = consistent & (priors > 0)
        log_w = alpha_t * torch.log(priors[mask]) - beta_t * n * log_sizes[mask]
    else:
        mask = consistent
        log_w = -beta_t * n * log_sizes[mask]

    weights = torch.zeros(n_hyps, dtype=torch.float64, device=device)
    weights[mask] = torch.exp(log_w - log_w.max())
    weights /= weights.sum()
    return weights


class NumberGameModel:
    """Bayesian number game model with configurable lambda and sigma."""

    def __init__(self, rules: list[Hypothesis], intervals: list[Hypothesis],
                 lam: float = 2/3, sigma: float = 10.0, max_integer: int = 100):
        self.max_integer = max_integer
        domain = range(1, max_integer + 1)

        self.n_math = len(rules)
        prior_per_math = lam / self.n_math

        # Interval priors: Erlang-weighted by size
        interval_sizes = np.array([sum(1 for x in domain if h.predicate(x)) for h in intervals], dtype=np.float64)
        erlang_weights = _erlang_pdf(interval_sizes, sigma)
        erlang_weights /= erlang_weights.sum()
        erlang_weights *= (1 - lam)

        # Build parallel arrays: math rules (uniform prior) then intervals (Erlang prior)
        all_hyps = rules + intervals
        all_priors = [prior_per_math] * len(rules) + list(erlang_weights)

        self.names: list[str] = [h.name for h in all_hyps]
        self.descriptions: list[str] = [h.description for h in all_hyps]

        # extensions[i] is a boolean mask of length max_integer (index 0 = number 1)
        ext_np = np.zeros((len(all_hyps), max_integer), dtype=bool)
        for i, h in enumerate(all_hyps):
            for x in domain:
                if h.predicate(x):
                    ext_np[i, x - 1] = True

        self.extensions = torch.tensor(ext_np, dtype=torch.bool, device=_DEVICE)
        self.priors = torch.tensor(all_priors, dtype=torch.float64, device=_DEVICE)
        self.sizes = self.extensions.sum(dim=1).to(torch.float64)
        self._posterior_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._predict_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._math_rule_cache: dict[tuple[int, ...], dict[str, float]] = {}
        self._top_order_cache: dict[tuple[int, ...], torch.Tensor] = {}

    def _posteriors(self, examples: list[int]) -> torch.Tensor:
        """Full posterior over all hypotheses given examples."""
        key = tuple(examples)
        if key not in self._posterior_cache:
            self._posterior_cache[key] = _hypothesis_weights(
                self.extensions, self.sizes, self.priors, examples
            )
        return self._posterior_cache[key]

    def predict(self, examples: list[int]) -> torch.Tensor:
        """Compute p(y in C | X) for y = 1..max_integer.

        Args:
            examples: list of observed positive examples (integers 1-max_integer).

        Returns:
            Tensor of shape (max_integer,) with probabilities (GPU).
        """
        key = tuple(examples)
        if key not in self._predict_cache:
            self._predict_cache[key] = self._posteriors(examples) @ self.extensions.to(torch.float64)
        return self._predict_cache[key]

    def top_hypotheses(self, examples: list[int], k: int = 10) -> list[tuple[str, float, torch.Tensor]]:
        """Return top-k hypotheses by posterior: (name, posterior, extension_mask)."""
        key = tuple(examples)
        posteriors = self._posteriors(examples)
        if key not in self._top_order_cache:
            self._top_order_cache[key] = torch.argsort(posteriors, descending=True)
        top_idx = self._top_order_cache[key][:k]
        return [(self.names[int(i)], float(posteriors[i]), self.extensions[i].clone()) for i in top_idx]

    def math_rule_posteriors(self, examples: list[int]) -> dict[str, float]:
        """Posterior probabilities for math-rule hypotheses + 'other' (intervals).

        Keyed by description (human-readable) so results align with LLM
        hypothesis_eval output, which uses the same description strings.
        """
        key = tuple(examples)
        if key not in self._math_rule_cache:
            posteriors = self._posteriors(examples)
            result = {self.descriptions[i]: float(posteriors[i]) for i in range(self.n_math)}
            result["other"] = 1.0 - sum(result.values())
            self._math_rule_cache[key] = result
        return dict(self._math_rule_cache[key])



def parameterized_predict(
    extensions: torch.Tensor,
    sizes: torch.Tensor,
    priors: torch.Tensor,
    examples: list[int],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute p(y∈C|X) with w(h) ∝ p(h)^α × (1/|h|^n)^β.

    Returns (max_integer,) tensor of generalization probabilities (GPU).
    """
    weights = _hypothesis_weights(extensions, sizes, priors, examples, alpha, beta)
    return weights @ extensions.to(torch.float64)


def _alpha_beta_targets(
    model: NumberGameModel,
    llm_preds: dict[str, np.ndarray],
    stimuli: list[StimulusSet],
) -> list[tuple[NumberGameModel, StimulusSet, torch.Tensor, torch.Tensor]]:
    """Collect valid alpha/beta fit targets for one model spec."""
    device = model.extensions.device
    targets = []
    for stimulus in stimuli:
        if stimulus.name not in llm_preds:
            continue
        values = np.asarray(llm_preds[stimulus.name], dtype=float)
        if values.ndim != 1 or values.size != model.extensions.shape[1]:
            continue
        target = torch.tensor(values, dtype=torch.float64, device=device)
        mask = torch.isfinite(target)
        if mask.any():
            targets.append((model, stimulus, target, mask))
    return targets


ALPHA_BETA_REG_LAMBDA = 0.01
AlphaBetaFitSpec = tuple["NumberGameModel", dict[str, np.ndarray], list[StimulusSet]]


def _fit_alpha_beta_targets(
    targets: list[tuple[NumberGameModel, StimulusSet, torch.Tensor, torch.Tensor]],
    reg_lambda: float = ALPHA_BETA_REG_LAMBDA,
) -> dict:
    """Fit one positive (alpha,beta) pair over pre-built target records.

    Adds a Gaussian prior on (log α, log β) anchored at 0 (α=β=1) with weight
    `reg_lambda` to keep the optimizer from running off to extreme values when
    a small/underdetermined stimulus group leaves a flat loss valley. The
    reported best_loss is data-only MSE (excludes the regularizer).
    """
    device = targets[0][0].extensions.device
    raw_params = torch.zeros(2, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw_params], max_iter=100, line_search_fn="strong_wolfe")

    def data_loss():
        alpha, beta = torch.exp(raw_params)
        losses = []
        for model, stimulus, target, mask in targets:
            pred = parameterized_predict(model.extensions, model.sizes, model.priors, stimulus.examples, alpha, beta)
            losses.append(((pred[mask] - target[mask]) ** 2).mean())
        return torch.stack(losses).mean()

    def squared_correlation():
        alpha, beta = torch.exp(raw_params)
        preds, observed = [], []
        for model, stimulus, target, mask in targets:
            pred = parameterized_predict(model.extensions, model.sizes, model.priors, stimulus.examples, alpha, beta)
            preds.append(pred[mask].reshape(-1))
            observed.append(target[mask].reshape(-1))
        pred_vec = torch.cat(preds)
        target_vec = torch.cat(observed)
        pred_centered = pred_vec - pred_vec.mean()
        target_centered = target_vec - target_vec.mean()
        denom = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
        corr = torch.nan if denom <= 0 else (pred_centered * target_centered).sum() / denom
        return corr ** 2

    def closure():
        optimizer.zero_grad()
        loss = data_loss() + reg_lambda * (raw_params ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        alpha, beta = torch.exp(raw_params)
        best_loss = data_loss()
        r2 = squared_correlation()
    return {
        "best_alpha": float(alpha),
        "best_beta": float(beta),
        "best_loss": float(best_loss),
        "r2": float(r2),
        "n_stimuli": len(targets),
    }


def fit_alpha_beta(specs: list[AlphaBetaFitSpec]) -> dict | None:
    """Joint (alpha, beta) fit via torch L-BFGS pooling all target specs.

    Returns:
        {"best_alpha", "best_beta", "best_loss", "r2", "n_stimuli"}, or None with no targets.
    """
    targets = []
    for model, llm_preds, stimuli in specs:
        targets.extend(_alpha_beta_targets(model, llm_preds, stimuli))

    if not targets:
        return None
    return _fit_alpha_beta_targets(targets)


def compute_alpha_beta_ablations(
    model: NumberGameModel,
    llm_preds: dict[str, np.ndarray],
    stimuli: list[StimulusSet],
    best_alpha: float,
    best_beta: float,
) -> dict:
    """Compute ablation overlays/MSEs for a fitted (α,β) solution."""
    device = model.extensions.device
    stimulus_names = [c.name for c in stimuli if c.name in llm_preds]
    n_stimuli = len(stimulus_names)

    # Plot overlays include best_fit; MSE reporting uses fit["best_loss"].
    ablation_ab = {
        "prior_only":      (1.0, 0.0),
        "likelihood_only": (0.0, 1.0),
        "best_fit":        (best_alpha, best_beta),
        "full_bayesian":   (1.0, 1.0),
    }
    mse_keys = ("full_bayesian", "prior_only", "likelihood_only")
    ablation_preds = {}
    ablation_mse = {k: 0.0 for k in mse_keys}

    for stimulus in stimuli:
        if stimulus.name not in llm_preds:
            continue
        llm = torch.tensor(np.asarray(llm_preds[stimulus.name]), dtype=torch.float64, device=device)
        preds = {
            k: parameterized_predict(model.extensions, model.sizes, model.priors, stimulus.examples, a, b)
            for k, (a, b) in ablation_ab.items()
        }
        ablation_preds[stimulus.name] = {k: v.cpu().numpy() for k, v in preds.items()}
        mask = ~torch.isnan(llm)
        n_valid = mask.sum().to(llm.dtype).clamp(min=1)
        for k in mse_keys:
            sq = torch.where(mask, (preds[k] - llm) ** 2, torch.zeros_like(preds[k]))
            ablation_mse[k] += float(sq.sum() / n_valid)
    for k in ablation_mse:
        ablation_mse[k] /= max(n_stimuli, 1)

    return {
        "ablation_preds": ablation_preds,
        "ablation_mse": ablation_mse,
    }
