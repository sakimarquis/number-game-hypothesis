"""Pure computation functions for LLM-Bayesian comparison metrics.

Divergence, overlap, entropy, and correlation measures used across analysis modules.
"""

import numpy as np
from scipy.spatial.distance import jensenshannon

HYPOTHESIS_TYPES = ("rule", "interval", "other")


def shannon_entropy(values, *, base: float = np.e, empty: float = float("nan")) -> float:
    """Entropy of the normalized positive mass in `values`; `empty` when there is none."""
    arr = np.asarray([float(v) for v in values], dtype=float)
    arr = arr[arr > 0.0]
    total = arr.sum()
    if total <= 0.0:
        return empty
    p = arr / total
    return float(-(p * np.log(p)).sum() / np.log(base))


def hypothesis_type(hyp: dict) -> str:
    """Pinned `type` of one structured hypothesis dict; anything unpinned is 'other'."""
    typ = hyp.get("type")
    return typ if typ in HYPOTHESIS_TYPES else "other"


def type_proportions(kind_weights) -> dict[str, float]:
    """Normalize positive (kind, weight) mass into rule/interval/other proportions."""
    props = dict.fromkeys(HYPOTHESIS_TYPES, 0.0)
    for kind, weight in kind_weights:
        if float(weight) > 0.0:
            props[kind if kind in HYPOTHESIS_TYPES else "other"] += float(weight)
    total = sum(props.values())
    return {key: value / total for key, value in props.items()} if total > 0.0 else props


def compute_hypothesis_generation_metrics(
    gen_results: dict[str, list[dict]],
    stimulus_names: list[str],
) -> dict:
    """Quantitative metrics for structured hypothesis generation results.

    Confidences are per-stimulus sum-normalized (treated as q(h|D)) for all
    weighted metrics. Raw `mean_confidence` is kept as a self-reporting diagnostic.
    """
    stimuli = {}
    for name in stimulus_names:
        hyps = gen_results.get(name, [])
        if not hyps:
            continue
        types = [hypothesis_type(h) for h in hyps]
        counts = {typ: types.count(typ) for typ in HYPOTHESIS_TYPES}
        confs = np.array([h["confidence"] for h in hyps], dtype=float)
        total_conf = confs.sum()
        stimuli[name] = {
            "n_hypotheses": len(hyps),
            "type_counts": counts,
            "type_proportions": {typ: count / len(hyps) for typ, count in counts.items()},
            "confidence_weighted_types": type_proportions(zip(types, confs)),
            "mean_confidence": float(confs.mean()),
            "top1_confidence_norm": float(confs.max() / total_conf) if total_conf > 0 else 0.0,
            "confidence_entropy": shannon_entropy(confs, empty=0.0),
        }
    return {"stimuli": stimuli}


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL(p || q) over arrays, both normalized first; q is floor-smoothed.

    Used wherever a privileged reference (Bayesian, Human) sits on the p side;
    direction-asymmetric on purpose.
    """
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        return float("nan")
    p_norm = p / p_sum
    q_smooth = q / q_sum + eps
    q_smooth /= q_smooth.sum()
    mask = p_norm > 0
    return float(np.sum(p_norm[mask] * np.log(p_norm[mask] / q_smooth[mask])))


def jensen_shannon_distance(p, q) -> float:
    """Jensen-Shannon distance (base 2, in [0, 1]) between two arrays.

    Use when both sides are estimates with no privileged reference. Returns
    NaN if either side has zero mass — "no signal", not "maximally divergent".
    """
    pa = np.asarray(p, dtype=float)
    qa = np.asarray(q, dtype=float)
    p_sum, q_sum = pa.sum(), qa.sum()
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    return float(jensenshannon(pa / p_sum, qa / q_sum, base=2.0))


def compute_topk_overlap(llm_topk: list, bayesian_topk: list) -> float:
    """Jaccard overlap between two ranked lists."""
    s1, s2 = set(llm_topk), set(bayesian_topk)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, NaN for fewer than two points or constant vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2)))
    return float(np.sum(x_centered * y_centered) / denom)


def _positive_topk_indices(values: np.ndarray, k: int) -> list[int]:
    """Top-k 1-based indices among strictly positive entries only."""
    positive = np.flatnonzero(values > 0)
    order = np.argsort(-values[positive], kind="stable")
    return (positive[order[:min(k, positive.size)]] + 1).tolist()


def compute_topk_overlap_vectors(p: np.ndarray, q: np.ndarray, k: int = 10) -> float:
    """Jaccard overlap between the top-k positive-support indices of two vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    k = min(k, p.size, q.size)
    return compute_topk_overlap(_positive_topk_indices(p, k), _positive_topk_indices(q, k))
