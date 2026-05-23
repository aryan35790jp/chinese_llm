"""
cluster_robust_regression.py — fix the pair non-independence problem.

Reviewer objection (correct):
    "When you draw 200,000 pairs from 6,306 characters, each character
    appears in ~63 pairs on average. Your observations are not
    independent, your standard errors are wrong, your significance is
    overstated."

This script re-runs the variance decomposition for every model with
*two-way cluster-robust standard errors* (Cameron, Gelbach, & Miller
2011), clustering on both characters in each pair. The point estimate
of beta is unchanged from naive OLS; the standard errors widen,
*correctly*, to reflect that we have many fewer effectively-independent
observations than naive pair counts suggest.

We also report:
    - effective sample size per cluster level (= number of unique chars)
    - cluster-robust 95% CI for each beta
    - the Cameron-Miller "design effect" ratio: how much wider the
      cluster-robust SE is than naive OLS SE

This is the statistically defensible version of `cooccurrence_baseline`
and the version a senior reviewer at ACL/EMNLP/NeurIPS will require.

Output:
    results/variance_decomposition_clustered.csv
        rows = (model, predictor, beta, se_naive, se_clustered,
                t_clustered, p_clustered, ci_lo, ci_hi, design_effect,
                n_pairs, n_unique_chars)

Runtime: ~5 minutes total for 8-10 models on local CPU.
Uses cached sparse PMI matrix; does NOT re-stream Wikipedia.

Depends on: extract_embeddings.py, isotropy_correction.py,
            cooccurrence_baseline.py (must have run once to populate
            cache/char_ppmi.npz)
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    CACHE_DIR,
    RESULTS_DIR,
    set_seed,
    load_radical_dataset,
    list_available_models,
    list_available_layers,
)
from radical_lib.embeddings import model_tag  # noqa: E402

set_seed()
ISO_DIR = CACHE_DIR / "embeddings_iso"
PMI_PATH = CACHE_DIR / "char_ppmi.npz"
N_PAIRS = 200_000


def two_way_cluster_robust_se(
    X: np.ndarray, y: np.ndarray, beta: np.ndarray,
    cluster_a: np.ndarray, cluster_b: np.ndarray
) -> np.ndarray:
    """Cameron, Gelbach, Miller (2011) two-way cluster-robust SE.

    For each pair (i, j) we have two cluster IDs (one for each char).
    The variance estimator is:
        V_AB = V_A + V_B - V_{A∩B}
    where V_C is the cluster-robust variance estimator with one-way
    clustering on C.

    Returns SE vector (one per coefficient).
    """
    n, k = X.shape
    resid = y - X @ beta
    # XtX inverse — same for all three terms
    XtX_inv = np.linalg.pinv(X.T @ X)

    def one_way_meat(cluster: np.ndarray) -> np.ndarray:
        """Compute the meat S = sum_g (X_g' u_g)(X_g' u_g)' for cluster groups."""
        S = np.zeros((k, k))
        # group rows by cluster id
        order = np.argsort(cluster)
        sorted_cluster = cluster[order]
        sorted_X = X[order]
        sorted_u = resid[order]
        # find boundaries
        boundaries = np.concatenate(
            [[0], np.where(np.diff(sorted_cluster) != 0)[0] + 1, [n]]
        )
        for k_idx in range(len(boundaries) - 1):
            lo, hi = boundaries[k_idx], boundaries[k_idx + 1]
            Xg = sorted_X[lo:hi]
            ug = sorted_u[lo:hi]
            xtu = Xg.T @ ug
            S += np.outer(xtu, xtu)
        return S

    # cluster on a, on b, and on the intersection
    S_a = one_way_meat(cluster_a)
    S_b = one_way_meat(cluster_b)
    # intersection: encode as combined cluster id
    a_max = cluster_a.max() + 1
    cluster_ab = cluster_a * a_max + cluster_b
    S_ab = one_way_meat(cluster_ab)
    S = S_a + S_b - S_ab
    V = XtX_inv @ S @ XtX_inv
    # Cameron-Miller small-sample correction (G/(G-1) per dimension)
    G_a = len(np.unique(cluster_a))
    G_b = len(np.unique(cluster_b))
    G_min = min(G_a, G_b)
    correction = G_min / max(G_min - 1, 1)
    se = np.sqrt(np.maximum(np.diag(V) * correction, 0))
    return se


def cosine_for_pairs(X: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return (Xn[i_idx] * Xn[j_idx]).sum(axis=1)


def load_iso_last_char(model_id: str) -> Tuple[np.ndarray, int]:
    layers = list_available_layers(model_id)
    if not layers:
        raise FileNotFoundError(model_id)
    L = max(layers)
    path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_char.npy"
    if not path.exists():
        path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_mean.npy"
    return np.load(path), L


def main():
    if not PMI_PATH.exists():
        print(f"[fatal] {PMI_PATH} missing — run cooccurrence_baseline.py first.")
        sys.exit(1)
    ppmi = load_npz(PMI_PATH)

    df = load_radical_dataset()
    n_chars = len(df)
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    radical = df[rad_col].astype(int).to_numpy()
    freq = (df["frequency_proxy"].astype(float).to_numpy()
            if "frequency_proxy" in df.columns else np.arange(n_chars).astype(float))
    strokes = (df["stroke_count"].astype(float).to_numpy()
               if "stroke_count" in df.columns else np.zeros(n_chars))

    # Sample 200k pairs (same RNG as cooccurrence_baseline so results align)
    rng = np.random.default_rng(42)
    iu_i, iu_j = np.triu_indices(n_chars, k=1)
    sel = rng.choice(len(iu_i), size=min(N_PAIRS, len(iu_i)), replace=False)
    i_idx = iu_i[sel]
    j_idx = iu_j[sel]

    same_rad = (radical[i_idx] == radical[j_idx]).astype(np.float64)
    ppmi_vals = np.asarray(ppmi[i_idx, j_idx]).flatten()
    freq_diff = np.abs(freq[i_idx] - freq[j_idx])
    stroke_diff = np.abs(strokes[i_idx] - strokes[j_idx])

    X_raw = np.column_stack([same_rad, ppmi_vals, freq_diff, stroke_diff])
    X_mean = X_raw.mean(axis=0)
    X_std = np.maximum(X_raw.std(axis=0), 1e-12)
    Xs = (X_raw - X_mean) / X_std
    X = np.column_stack([np.ones(len(Xs)), Xs])
    pred_names = ["intercept", "same_radical", "ppmi", "freq_diff", "stroke_diff"]

    rows = []
    for model_id in list_available_models():
        try:
            Xemb, _ = load_iso_last_char(model_id)
        except FileNotFoundError:
            continue
        y = cosine_for_pairs(Xemb, i_idx, j_idx).astype(np.float64)

        # Naive OLS
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        n_obs, k = X.shape
        sigma2 = float(np.dot(resid, resid)) / max(n_obs - k, 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_naive = np.sqrt(np.maximum(sigma2 * np.diag(XtX_inv), 0))

        # Cluster-robust SE on (i_idx, j_idx)
        se_clustered = two_way_cluster_robust_se(X, y, beta, i_idx, j_idx)

        # Effective sample (Cameron-Miller G_min)
        G_min = min(len(np.unique(i_idx)), len(np.unique(j_idx)))
        df_resid = G_min - k
        if df_resid <= 0:
            df_resid = max(n_obs - k, 1)

        for p_idx, name in enumerate(pred_names):
            if name == "intercept":
                continue
            t_clust = beta[p_idx] / max(se_clustered[p_idx], 1e-12)
            p_clust = 2.0 * float(student_t.sf(abs(t_clust), df=df_resid))
            ci_lo = float(beta[p_idx] - 1.96 * se_clustered[p_idx])
            ci_hi = float(beta[p_idx] + 1.96 * se_clustered[p_idx])
            de = (float(se_clustered[p_idx]) /
                  max(float(se_naive[p_idx]), 1e-12))
            rows.append({
                "model": model_id,
                "predictor": name,
                "beta": float(beta[p_idx]),
                "se_naive": float(se_naive[p_idx]),
                "se_clustered": float(se_clustered[p_idx]),
                "t_clustered": float(t_clust),
                "p_clustered": p_clust,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "design_effect": float(de),
                "n_pairs": int(n_obs),
                "n_unique_chars": int(G_min),
                "df_resid": int(df_resid),
            })
        print(f"  {model_id:<48s}  done; design_effect range = "
              f"{min(de for de in [r['design_effect'] for r in rows[-4:]]):.2f}–"
              f"{max(de for de in [r['design_effect'] for r in rows[-4:]]):.2f}x")

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "variance_decomposition_clustered.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
