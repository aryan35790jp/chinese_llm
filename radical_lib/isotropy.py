"""
Anisotropy correction for contextual embeddings.

Why this exists:
    Raw transformer embeddings live in a narrow cone (Ethayarajh 2019).
    Cosine similarities are inflated by a global mean direction and a
    small number of dominant principal components. Reporting raw
    cosines confounds anisotropy with semantic structure.

Method (Mu & Viswanath 2018, "All-but-the-Top"):
    1. Subtract the per-coordinate mean across the corpus
    2. Divide by per-coordinate std (optional but standard)
    3. Project out the top-k principal components (default k=2)

We expose:
    fit_isotropy(X)              -> Isotropy params
    apply_isotropy(X, params)    -> standardized X
    cosine_isotropic(X)          -> N×N cosine matrix on the corrected X
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IsotropyParams:
    mean: np.ndarray
    std: np.ndarray
    components: np.ndarray  # k×D top principal components
    k: int


def fit_isotropy(
    X: np.ndarray, k: int = 2, standardize: bool = True
) -> IsotropyParams:
    """Fit anisotropy-correction parameters on X (N×D)."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean
    if standardize:
        std = Xc.std(axis=0)
        std[std == 0] = 1.0
    else:
        std = np.ones_like(mean)
    Xs = Xc / std

    # Top-k principal components via SVD on centered+scaled matrix
    if k > 0:
        # economy SVD of (N×D); right singular vectors give principal axes
        _, _, vt = np.linalg.svd(Xs, full_matrices=False)
        components = vt[:k]  # k × D
    else:
        components = np.zeros((0, X.shape[1]), dtype=np.float64)

    return IsotropyParams(mean=mean, std=std, components=components, k=k)


def apply_isotropy(X: np.ndarray, params: IsotropyParams) -> np.ndarray:
    """Apply previously-fit isotropy correction."""
    X = np.asarray(X, dtype=np.float64)
    Xs = (X - params.mean) / params.std
    if params.k > 0:
        # remove projection onto top-k components
        proj = Xs @ params.components.T  # N × k
        Xs = Xs - proj @ params.components
    return Xs


def cosine_isotropic(
    X: np.ndarray, params: Optional[IsotropyParams] = None, k: int = 2
) -> np.ndarray:
    """Return the N×N cosine similarity matrix on isotropy-corrected X."""
    if params is None:
        params = fit_isotropy(X, k=k)
    Xc = apply_isotropy(X, params)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = Xc / norms
    return Xn @ Xn.T
