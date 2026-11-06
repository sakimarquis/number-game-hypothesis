"""Bayesian concept learning model (Tenenbaum 1999) and derived fits.

Computes p(y in C | X) for all y in {1..100} given observed examples X,
using hypothesis averaging with the size principle.

  p(y in C | X) = sum_{h: X subset h, y in h} w(h)
                  ------------------------------------
                  sum_{h: X subset h}               w(h)

  where w(h) = p(h) / |h|^n   (prior * likelihood up to normalization)

Also provides size-principle fits: w(h|X) ∝ p(h) × (1/|h|^n)^β under a fixed
prior p(h), the pooled prior of analysis.prior_pooling for model and human fits.

Internal arrays (extensions, priors, sizes) are stored as GPU tensors.
Public methods return GPU tensors; callers convert at save/plot boundaries.
"""

from __future__ import annotations

import numpy as np
import torch

from tasks.hypotheses import Hypothesis
from tasks.stimuli import StimulusSet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    Returns tensor of length n_hyps. The full-domain interval is always consistent.
    """
    consistent = _consistent_mask(extensions, examples)
    n_hyps = extensions.shape[0]
    device = extensions.device
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
                 lam: float = 2/3, sigma: float = 10.0, domain: int = 100):
        numbers = range(1, domain + 1)

        self.n_math = len(rules)
        prior_per_math = lam / self.n_math

        # Interval priors: Erlang-weighted by size
        interval_sizes = np.array([sum(1 for x in numbers if h.predicate(x)) for h in intervals], dtype=np.float64)
        erlang_weights = _erlang_pdf(interval_sizes, sigma)
        erlang_weights /= erlang_weights.sum()
        erlang_weights *= (1 - lam)

        # Build parallel arrays: math rules (uniform prior) then intervals (Erlang prior)
        all_hyps = rules + intervals
        all_priors = [prior_per_math] * len(rules) + list(erlang_weights)

        self.names: list[str] = [h.name for h in all_hyps]
        self.descriptions: list[str] = [h.description for h in all_hyps]

        # extensions[i] is a boolean mask of length domain (index 0 = number 1)
        ext_np = np.zeros((len(all_hyps), domain), dtype=bool)
        for i, h in enumerate(all_hyps):
            for x in numbers:
                if h.predicate(x):
                    ext_np[i, x - 1] = True

        self.extensions = torch.tensor(ext_np, dtype=torch.bool, device=DEVICE)
        self.priors = torch.tensor(all_priors, dtype=torch.float64, device=DEVICE)
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
        """Compute p(y in C | X) for y = 1..domain.

        Args:
            examples: list of observed positive examples (integers 1..domain).

        Returns:
            Tensor of shape (domain,) with probabilities (GPU).
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

    def rule_posteriors(self, examples: list[int]) -> dict[str, float]:
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

    Returns (domain,) tensor of generalization probabilities (GPU).
    """
    weights = _hypothesis_weights(extensions, sizes, priors, examples, alpha, beta)
    return weights @ extensions.to(torch.float64)


def _beta_targets(
    model: NumberGameModel,
    priors: torch.Tensor,
    llm_preds: dict[str, np.ndarray],
    stimuli: list[StimulusSet],
) -> list[tuple[NumberGameModel, torch.Tensor, StimulusSet, torch.Tensor, torch.Tensor]]:
    """Collect valid size-principle fit targets for one model spec."""
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
            targets.append((model, priors, stimulus, target, mask))
    return targets


ALPHA_BETA_REG_LAMBDA = 0.01
# (Bayesian model, hypothesis prior weights, predictions by stimulus, stimuli)
BetaFitSpec = tuple["NumberGameModel", torch.Tensor, dict[str, np.ndarray], list[StimulusSet]]


def _fit_beta_targets(targets, reg_lambda: float = ALPHA_BETA_REG_LAMBDA) -> dict:
    """Fit one positive β over pre-built target records, each with its own fixed prior.

    A Gaussian prior on log β anchored at 0 (β=1) with weight `reg_lambda` keeps small groups from
    running off along a flat loss valley. The reported mse is data-only (excludes the regularizer).
    """
    device = targets[0][0].extensions.device
    log_beta = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_beta], max_iter=100, line_search_fn="strong_wolfe")

    def predictions():
        beta = torch.exp(log_beta)
        return [
            (parameterized_predict(model.extensions, model.sizes, priors, stimulus.examples, 1.0, beta)[mask], target[mask])
            for model, priors, stimulus, target, mask in targets
        ]

    def data_loss():
        return torch.stack([((pred - target) ** 2).mean() for pred, target in predictions()]).mean()

    def closure():
        optimizer.zero_grad()
        loss = data_loss() + reg_lambda * log_beta ** 2
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        pairs = predictions()
        mse = torch.stack([((pred - target) ** 2).mean() for pred, target in pairs]).mean()
        pred_vec = torch.cat([pred for pred, _ in pairs])
        target_vec = torch.cat([target for _, target in pairs])
        pred_centered = pred_vec - pred_vec.mean()
        target_centered = target_vec - target_vec.mean()
        denom = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
        r2 = torch.nan if denom <= 0 else ((pred_centered * target_centered).sum() / denom) ** 2
        beta = torch.exp(log_beta)
    return {"beta": float(beta), "mse": float(mse), "r2": float(r2), "n_stimuli": len(targets)}


def fit_beta(specs: list[BetaFitSpec]) -> dict | None:
    """Joint size-principle β fit via torch L-BFGS pooling all target specs.

    Returns:
        {"beta", "mse", "r2", "n_stimuli"}, or None with no targets.
    """
    targets = [target for spec in specs for target in _beta_targets(*spec)]
    return _fit_beta_targets(targets) if targets else None


def fit_beta_with_ablations(
    model: NumberGameModel,
    priors: torch.Tensor,
    llm_preds: dict[str, np.ndarray],
    stimuli: list[StimulusSet],
) -> dict:
    """Fit one stimulus group, adding ablation overlays and MSEs at the fitted β.

    Bayesian and likelihood-only curves use the model's own prior (likelihood-only flattens it);
    prior-only and best-fit curves use `priors`.
    """
    fit = fit_beta([(model, priors, llm_preds, stimuli)])
    device = model.extensions.device
    n_stimuli = sum(c.name in llm_preds for c in stimuli)

    # Plot overlays include best_fit; MSE reporting uses fit["mse"].
    ablation_args = {
        "prior_only":      (priors, 1.0, 0.0),
        "likelihood_only": (model.priors, 0.0, 1.0),
        "best_fit":        (priors, 1.0, fit["beta"]),
        "full_bayesian":   (model.priors, 1.0, 1.0),
    }
    mse_keys = ("full_bayesian", "prior_only", "likelihood_only")
    ablation_preds = {}
    ablation_mse = {k: 0.0 for k in mse_keys}

    for stimulus in stimuli:
        if stimulus.name not in llm_preds:
            continue
        llm = torch.tensor(np.asarray(llm_preds[stimulus.name]), dtype=torch.float64, device=device)
        preds = {
            k: parameterized_predict(model.extensions, model.sizes, p, stimulus.examples, a, b)
            for k, (p, a, b) in ablation_args.items()
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
        **fit,
        "ablation_preds": ablation_preds,
        "ablation_mse": ablation_mse,
    }
