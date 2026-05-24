"""
nonparametric_stats.py — reviewer-proofing the headline numbers.

A senior reviewer will object to Cohen's d as the only effect size for
embedding cosines (heavy-tailed; pair non-independence). This script adds:

    1. Wendt's r (= rank-biserial correlation) on intra vs inter cosines —
       a scale-invariant non-parametric effect size.
    2. Character-clustered bootstrap CI: resamples *characters* (with all
       their pairs), not pairs themselves, to honour pair non-independence.
    3. For the variance decomposition: Spearman rank correlation between
       predicted (regression-fitted) cosine and observed cosine. This is
       scale-invariant complement to partial R².

Operates only on already-extracted embeddings (cache/embeddings_iso/).
Fully local. ~5 min CPU.

Output:
    results/nonparametric_cohesion.csv
        rows = (model, layer, pool, iso, n_intra, n_inter, cohens_d,
                wendt_r, char_boot_lo, char_boot_hi)
    results/variance_decomposition_rank.csv
        rows = (model, predictor, partial_R2, spearman_rho_pred_obs)

Depends on: extract_embeddings.py, isotropy_correction.py
"""
from __future__ import annotations
import itertools
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    CACHE_DIR,
    RESULTS_DIR,
    set_seed,
    load_radical_dataset,
    list_available_models,
    list_available_layers,
    cohens_d,
)
from radical_lib.embeddings import model_tag  # noqa: E402

set_seed()
ISO_DIR = CACHE_DIR / "embeddings_iso"


# ── Wendt's r (rank-biserial; non-parametric) ──────────────────────────────
def wendt_r(intra: np.ndarray, inter: np.ndarray) -> float:
    """Rank-biserial correlation:
        r = 1 − 2·U / (n1·n2)
    where U is the Mann-Whitney U.
    Scale-invariant; range [-1, +1]; analog to Cohen's d for non-normal data.
    """
    intra = np.asarray(intra, dtype=np.float64)
    inter = np.asarray(inter, dtype=np.float64)
    n1, n2 = len(intra), len(inter)
    if n1 == 0 or n2 == 0:
        return float("nan")
    # Combined ranking
    combined = np.concatenate([intra, inter])
    ranks = pd.Series(combined).rank().to_numpy()
    R1 = ranks[:n1].sum()
    U = R1 - n1 * (n1 + 1) / 2
    return float(1.0 - 2.0 * U / (n1 * n2))


# ── Character-clustered bootstrap ──────────────────────────────────────────
def char_clustered_bootstrap_ci(
    char_radical: Dict[str, int],
    radical_groups: Dict[int, List[str]],
    sim: np.ndarray,
    char_idx: Dict[str, int],
    n_boot: int = 200,
    pairs_per_radical: int = 50,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, np.ndarray]:
    """For each bootstrap iteration:
       - resample characters with replacement (preserving radical labels);
       - rebuild intra/inter pair pools from the resampled set;
       - record Δ (mean intra − mean inter).

    This honours pair non-independence: a popular character that participates
    in many pairs gets resampled with replacement just like any other char.
    """
    rng = rng or np.random.default_rng(42)
    chars = list(char_radical.keys())
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        # resample chars with replacement, stratified by radical
        # (each radical's group is bootstrapped within itself)
        boot_groups: Dict[int, List[str]] = {}
        for rad, group in radical_groups.items():
            boot_groups[rad] = list(rng.choice(group, size=len(group), replace=True))
        intra: List[float] = []
        inter: List[float] = []
        py_rng = random.Random(int(rng.integers(0, 2**31)))
        for rad, group in boot_groups.items():
            if len(group) < 2:
                continue
            pairs = list(itertools.combinations(group, 2))
            if len(pairs) > pairs_per_radical:
                pairs = py_rng.sample(pairs, pairs_per_radical)
            for a, b_ in pairs:
                intra.append(sim[char_idx[a], char_idx[b_]])
            others = [c for c in chars if char_radical[c] != rad]
            if not others:
                continue
            for _ in range(pairs_per_radical):
                a = py_rng.choice(group)
                b_ = py_rng.choice(others)
                inter.append(sim[char_idx[a], char_idx[b_]])
        if intra and inter:
            deltas[b] = np.mean(intra) - np.mean(inter)
        else:
            deltas[b] = np.nan
    valid = deltas[~np.isnan(deltas)]
    if len(valid) < 2:
        return float("nan"), float("nan"), valid
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return float(lo), float(hi), valid


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return Xn @ Xn.T


def load_iso_last_char(model_id: str):
    layers = list_available_layers(model_id)
    if not layers:
        raise FileNotFoundError(model_id)
    L = max(layers)
    path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_char.npy"
    if not path.exists():
        path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_mean.npy"
    return np.load(path), L


def sample_intra_inter(
    chars: List[str], char_idx: Dict[str, int],
    char_radical: Dict[str, int],
    radical_groups: Dict[int, List[str]],
    sim: np.ndarray,
    pairs_per_radical: int = 50,
):
    rng = random.Random(42)
    intra: List[float] = []
    inter: List[float] = []
    for rad, group in radical_groups.items():
        if len(group) < 2:
            continue
        pairs = list(itertools.combinations(group, 2))
        if len(pairs) > pairs_per_radical:
            pairs = rng.sample(pairs, pairs_per_radical)
        for a, b in pairs:
            intra.append(sim[char_idx[a], char_idx[b]])
        others = [c for c in chars if char_radical[c] != rad]
        for _ in range(pairs_per_radical):
            a = rng.choice(group)
            b = rng.choice(others)
            inter.append(sim[char_idx[a], char_idx[b]])
    return np.array(intra), np.array(inter)


def main():
    df = load_radical_dataset()
    chars = df["char"].tolist()
    char_idx = {c: i for i, c in enumerate(chars)}
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_radical = dict(zip(df["char"], df[rad_col].astype(int)))
    radical_groups: Dict[int, List[str]] = {}
    for c, r in char_radical.items():
        radical_groups.setdefault(r, []).append(c)
    radical_groups = {r: g for r, g in radical_groups.items() if len(g) >= 20}

    # ── 1. Non-parametric cohesion stats per model ─────────────────────────
    rows = []
    for model_id in list_available_models():
        try:
            X, L = load_iso_last_char(model_id)
        except FileNotFoundError:
            continue
        sim = cosine_matrix(X)
        intra, inter = sample_intra_inter(chars, char_idx, char_radical, radical_groups, sim)
        d = cohens_d(intra, inter)
        wr = wendt_r(intra, inter)
        # Character-clustered bootstrap (200 reps; ~30 sec per model on CPU)
        lo, hi, _ = char_clustered_bootstrap_ci(
            char_radical, radical_groups, sim, char_idx,
            n_boot=200, pairs_per_radical=50,
            rng=np.random.default_rng(42),
        )
        rows.append({
            "model": model_id,
            "layer": int(L),
            "pool": "char",
            "iso": 1,
            "n_intra": len(intra),
            "n_inter": len(inter),
            "cohens_d": float(d),
            "wendt_r": float(wr),
            "char_boot_lo": float(lo),
            "char_boot_hi": float(hi),
        })
        print(f"  {model_id:<48s}  d={d:+.3f}  r_w={wr:+.3f}  "
              f"char-boot Δ ∈ [{lo:+.4f}, {hi:+.4f}]")

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "nonparametric_cohesion.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows to {out_path}")

    # ── 2. Variance-decomposition rank correlation ─────────────────────────
    # The reviewer's complaint about partial R² being absolute-tiny is partly
    # answered by reporting Spearman rank corr between predicted cosine
    # (from the OLS fit) and observed cosine. Scale-invariant.
    vd_path = RESULTS_DIR / "variance_decomposition.csv"
    if not vd_path.exists():
        print("[skip] variance_decomposition.csv missing")
        return
    vd = pd.read_csv(vd_path)

    # We'll re-run a tiny in-memory regression per model to compute
    # predicted-vs-observed Spearman. Re-use the same predictor sample we
    # built in cooccurrence_baseline. If char_ppmi.npz isn't on disk, skip.
    ppmi_path = CACHE_DIR / "char_ppmi.npz"
    if not ppmi_path.exists():
        print("[skip] char_ppmi.npz missing — skipping rank-corr step")
        return
    from scipy.sparse import load_npz
    ppmi = load_npz(ppmi_path)

    rows_rank = []
    rng = np.random.default_rng(42)
    iu_i, iu_j = np.triu_indices(len(chars), k=1)
    sel = rng.choice(len(iu_i), size=min(50_000, len(iu_i)), replace=False)
    i_idx = iu_i[sel]
    j_idx = iu_j[sel]
    same_rad = (df[rad_col].astype(int).to_numpy()[i_idx] ==
                 df[rad_col].astype(int).to_numpy()[j_idx]).astype(np.float64)
    freq = (df["frequency_proxy"].astype(float).to_numpy()
            if "frequency_proxy" in df.columns else np.arange(len(chars)).astype(float))
    strokes = (df["stroke_count"].astype(float).to_numpy()
               if "stroke_count" in df.columns else np.zeros(len(chars)))
    freq_diff = np.abs(freq[i_idx] - freq[j_idx])
    stroke_diff = np.abs(strokes[i_idx] - strokes[j_idx])
    ppmi_vals = np.asarray(ppmi[i_idx, j_idx]).flatten()

    Xpred = np.column_stack([same_rad, ppmi_vals, freq_diff, stroke_diff])
    Xpred = (Xpred - Xpred.mean(axis=0)) / np.maximum(Xpred.std(axis=0), 1e-12)
    Xpred = np.column_stack([np.ones(len(Xpred)), Xpred])

    for model_id in list_available_models():
        try:
            Xemb, _ = load_iso_last_char(model_id)
        except FileNotFoundError:
            continue
        Xn = Xemb / np.maximum(np.linalg.norm(Xemb, axis=1, keepdims=True), 1e-12)
        cos = (Xn[i_idx] * Xn[j_idx]).sum(axis=1).astype(np.float64)
        beta, _, _, _ = np.linalg.lstsq(Xpred, cos, rcond=None)
        pred = Xpred @ beta
        rho, p = spearmanr(pred, cos)
        rows_rank.append({
            "model": model_id,
            "spearman_rho_pred_obs": float(rho),
            "spearman_p": float(p),
            "n_pairs": int(len(cos)),
        })
        print(f"  {model_id:<48s}  Spearman(predicted, observed) = {rho:+.3f}")

    pd.DataFrame(rows_rank).to_csv(
        RESULTS_DIR / "variance_decomposition_rank.csv", index=False
    )


if __name__ == "__main__":
    main()
