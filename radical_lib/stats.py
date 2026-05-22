"""
Statistical primitives used across the project. Every paper number flows
through these functions.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.stats import ttest_ind, spearmanr


# ── effect size ──────────────────────────────────────────────────────────────
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Cohen's d with pooled (unbiased) standard deviation."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    sp = np.sqrt(((len(a) - 1) * s1 + (len(b) - 1) * s2) / (len(a) + len(b) - 2))
    if sp == 0:
        return 0.0
    return float((a.mean() - b.mean()) / sp)


# ── inferential ──────────────────────────────────────────────────────────────
def welch_t(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    t, p = ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def bootstrap_ci_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, np.ndarray]:
    """Percentile bootstrap CI for mean(a) − mean(b)."""
    rng = rng or np.random.default_rng(42)
    a = np.asarray(a)
    b = np.asarray(b)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        diffs[i] = ai.mean() - bi.mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), diffs


def permutation_test_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 1000,
    rng: np.random.Generator | None = None,
    one_sided: bool = True,
) -> Tuple[float, float, np.ndarray]:
    """One-sided permutation p (mean(a) - mean(b) > 0) with continuity correction.

    Returns (p, observed_diff, null_distribution).
    """
    rng = rng or np.random.default_rng(42)
    a = np.asarray(a)
    b = np.asarray(b)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        rng.shuffle(pooled)
        null[i] = pooled[:n_a].mean() - pooled[n_a:].mean()
    if one_sided:
        p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return float(p), float(observed), null


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Vectorized Holm–Bonferroni step-down. Returns adjusted p-values."""
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = min(p[idx] * (n - rank), 1.0)
        running_max = max(running_max, candidate)
        adj[idx] = running_max
    return adj


# ── representational similarity analysis ────────────────────────────────────
def rsa_spearman(
    embedding_rdm: np.ndarray, model_rdm: np.ndarray
) -> Tuple[float, float]:
    """Spearman correlation between the upper-triangles of two RDMs.

    embedding_rdm and model_rdm must be square and same shape. The diagonal
    is ignored. Returns (rho, p).
    """
    n = embedding_rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    rho, p = spearmanr(embedding_rdm[iu], model_rdm[iu])
    return float(rho), float(p)
