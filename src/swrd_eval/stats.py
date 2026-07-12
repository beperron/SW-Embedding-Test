"""Statistics: bootstrap CIs, paired Wilcoxon with Holm-Bonferroni correction."""
from __future__ import annotations
import numpy as np
from scipy import stats as ss


def bootstrap_ci(per_query: np.ndarray, resamples: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI over per-query scores. Returns (mean, lo, hi)."""
    per_query = np.asarray(per_query, dtype=float)
    n = len(per_query)
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    means = per_query[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(per_query.mean()), float(lo), float(hi)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value (a vs b over aligned queries)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    try:
        return float(ss.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def holm_correction(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj.tolist()
