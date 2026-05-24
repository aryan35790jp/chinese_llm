"""
cooccurrence_baseline.py — variance decomposition that turns the paper from
"rigorous null result" into a positive contribution.

Question:
    The radical-cohesion effect at corpus scale (Cohen's d ~ 0.06–0.14 in the
    original paper) is consistent with multiple causes:
        (a) shared semantic content
        (b) shared distributional context
        (c) tokenizer/frequency artifacts
        (d) actual orthographic/form encoding
    A single regression that includes all four predictors lets us read off
    how much of the cohesion is explained by each.

Pipeline:
    1. Stream Wikipedia zh, build character co-occurrence with window ±5.
    2. Compute PPMI(c1, c2) = max(0, log P(c1, c2) / (P(c1) P(c2))).
    3. Sample up to 200k character pairs (i, j), i < j.
       For each pair record:
            same_radical    ∈ {0, 1}
            ppmi            float
            freq_diff       |rank(c_i) - rank(c_j)|
            stroke_diff     |strokes(c_i) - strokes(c_j)|
            bert_cosine     last-layer isotropy-corrected cosine
    4. Fit OLS:  bert_cosine ~ same_radical + ppmi + freq_diff + stroke_diff
    5. Report coefficients, standard errors, partial R² for each predictor.

Output:
    cache/char_ppmi.npz, cache/char_counts.npy   (corpus statistics)
    results/variance_decomposition.csv
        rows = (model, predictor, beta, se, t, p, partial_R2, full_R2, n_pairs)

Runtime:
    PMI build (first run): 30 min on Wikipedia stream + char tokenize
    Regression:             5 min per model
RAM peak: 8 GB (sparse co-occurrence matrix on 6,306 chars).

Depends on: extract_embeddings.py, isotropy_correction.py, dataset_builder.py
            internet on first run for the Wikipedia stream
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, dok_matrix, save_npz, load_npz

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
COUNTS_PATH = CACHE_DIR / "char_counts.npy"
N_SENTENCES_DEFAULT = 1_000_000
WINDOW = 5


# ── 1. corpus → co-occurrence ────────────────────────────────────────────────
def stream_wiki_sentences(n_sentences: int):
    """Stream raw text from wikimedia/wikipedia 20231101.zh.

    Falls back gracefully if `datasets` or the network is unavailable.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[warn] `datasets` not installed; cannot stream Wikipedia.")
        return

    try:
        ds = load_dataset(
            "wikimedia/wikipedia", "20231101.zh", streaming=True, split="train"
        )
    except Exception as e:
        print(f"[warn] couldn't stream Wikipedia zh: {e}")
        return

    n_yielded = 0
    for row in ds:
        text = row.get("text", "")
        for sentence in text.split("\n"):
            if len(sentence) < 3:
                continue
            yield sentence
            n_yielded += 1
            if n_yielded >= n_sentences:
                return


def build_cooccurrence(
    chars: list[str], n_sentences: int = N_SENTENCES_DEFAULT, window: int = WINDOW
) -> Tuple[csr_matrix, np.ndarray]:
    """Build a (PPMI sparse matrix, char counts) pair restricted to `chars`."""
    char_idx = {c: i for i, c in enumerate(chars)}
    n = len(chars)

    counts = np.zeros(n, dtype=np.int64)
    co = dok_matrix((n, n), dtype=np.int64)

    n_done = 0
    for sentence in stream_wiki_sentences(n_sentences):
        idx_seq = [char_idx[c] for c in sentence if c in char_idx]
        if len(idx_seq) < 2:
            continue
        for i, ci in enumerate(idx_seq):
            counts[ci] += 1
            lo = max(0, i - window)
            hi = min(len(idx_seq), i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                co[ci, idx_seq[j]] += 1
        n_done += 1
        if n_done % 50_000 == 0:
            print(f"  processed {n_done} sentences, total chars seen={counts.sum()}")

    co_csr = co.tocsr()
    print(f"  total sentences: {n_done}; co-occ entries: {co_csr.nnz}")

    # PPMI (Levy & Goldberg 2014):
    #   PMI(i, j) = log( #(i, j) * D / ( #(i) * #(j) ) )
    # where D is the total number of (target, context) pairs observed.
    # All four quantities are now in the *same* base measure, removing the
    # subtle inconsistency in the previous version that mixed P(i) over
    # total chars seen and P(i,j) over total co-occurrences.
    rows, cols = co_csr.nonzero()
    co_data = co_csr.data.astype(np.float64)
    D = float(co_csr.sum())  # total (target, context) pair count
    if D <= 0:
        return csr_matrix(co_csr.shape, dtype=np.float32), counts

    c_i = counts[rows].astype(np.float64)
    c_j = counts[cols].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Safe-log; the smoothing 1e-30 only kicks in for the truly empty
        # rows we already excluded by iterating co_csr.nonzero().
        ratio = (co_data * D) / np.maximum(c_i * c_j, 1.0)
        pmi = np.log(np.maximum(ratio, 1e-30))
    pmi_vals = np.maximum(pmi, 0.0).astype(np.float32)

    ppmi = csr_matrix((pmi_vals, (rows, cols)), shape=(n, n))
    return ppmi, counts


# ── 2. per-pair predictors ──────────────────────────────────────────────────
def build_predictors(
    df: pd.DataFrame, ppmi: csr_matrix, max_pairs: int = 200_000
) -> pd.DataFrame:
    """Long-format dataframe of pair-level predictors. Subsamples uniformly
    from the upper triangle for tractability."""
    n = len(df)
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    radical = df[rad_col].astype(int).to_numpy()
    freq = (df["frequency_proxy"].astype(float).to_numpy()
            if "frequency_proxy" in df.columns else np.arange(n).astype(float))
    strokes = (df["stroke_count"].astype(float).to_numpy()
               if "stroke_count" in df.columns else np.zeros(n))

    rng = np.random.default_rng(42)
    iu_i, iu_j = np.triu_indices(n, k=1)
    n_total = len(iu_i)
    if n_total > max_pairs:
        sel = rng.choice(n_total, size=max_pairs, replace=False)
        i_idx = iu_i[sel]
        j_idx = iu_j[sel]
    else:
        i_idx = iu_i
        j_idx = iu_j

    # Sparse lookup. ppmi[i_idx, j_idx] returns a 1-row matrix; flatten.
    ppmi_arr = np.asarray(ppmi[i_idx, j_idx]).flatten()

    same_rad = (radical[i_idx] == radical[j_idx]).astype(np.int8)
    freq_diff = np.abs(freq[i_idx] - freq[j_idx])
    stroke_diff = np.abs(strokes[i_idx] - strokes[j_idx])

    return pd.DataFrame({
        "i": i_idx,
        "j": j_idx,
        "same_radical": same_rad,
        "ppmi": ppmi_arr,
        "freq_diff": freq_diff,
        "stroke_diff": stroke_diff,
    })


# ── 3. per-model regression ─────────────────────────────────────────────────
def load_iso_last_char(model_id: str) -> Tuple[np.ndarray, int]:
    layers = list_available_layers(model_id)
    if not layers:
        raise FileNotFoundError(model_id)
    L = max(layers)
    path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_char.npy"
    if not path.exists():
        path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_mean.npy"
    return np.load(path), L


def cosine_for_pairs(X: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    A = Xn[i_idx]
    B = Xn[j_idx]
    return (A * B).sum(axis=1)


def _ols_with_se(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Plain OLS via least squares with standard errors and R².

    Returns (beta, se, r_squared). X must already include the intercept column.
    Implementation:
        β = (X'X)^-1 X'y
        residuals = y - X β
        sigma² = RSS / (n - k)
        Var(β) = sigma² (X'X)^-1
        SE = sqrt(diag Var(β))
    """
    n, k = X.shape
    # solve via lstsq (more stable than (X'X)^-1)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    rss = float((resid ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else 0.0

    if n > k:
        sigma2 = rss / (n - k)
        # (X'X)^-1 — use pinv for safety
        xtx_inv = np.linalg.pinv(X.T @ X)
        var_beta = sigma2 * xtx_inv
        se = np.sqrt(np.maximum(np.diag(var_beta), 0))
    else:
        se = np.full_like(beta, np.nan)

    return beta, se, r2


def fit_regression(predictors: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """Fit y = β·X with a numpy-only OLS so we get standard errors and
    partial R² (full R² minus R² of model omitting the predictor)."""
    from scipy.stats import t as student_t

    y = predictors["bert_cosine"].to_numpy()
    X_cols = ["same_radical", "ppmi", "freq_diff", "stroke_diff"]
    X = predictors[X_cols].to_numpy().astype(np.float64)

    # Standardize so coefficients are directly comparable.
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xs = (X - X_mean) / X_std

    n = len(y)
    Xs_c = np.column_stack([np.ones(n), Xs])
    beta_full, se_full, r2_full = _ols_with_se(Xs_c, y)

    rows = []
    for k, name in enumerate(X_cols):
        keep = [i for i in range(len(X_cols)) if i != k]
        Xk = X[:, keep]
        Xk_std = Xk.std(axis=0)
        Xk_std[Xk_std == 0] = 1.0
        Xk_norm = (Xk - Xk.mean(axis=0)) / Xk_std
        Xk_c = np.column_stack([np.ones(n), Xk_norm])
        _, _, r2_partial = _ols_with_se(Xk_c, y)
        partial_r2 = r2_full - r2_partial

        # t-test on coefficient k+1 (skip intercept at index 0)
        b = beta_full[k + 1]
        s = se_full[k + 1]
        if s > 0:
            t = b / s
            p = 2.0 * float(student_t.sf(abs(t), df=max(n - len(X_cols) - 1, 1)))
        else:
            t = float("nan")
            p = float("nan")

        rows.append({
            "model": model_id,
            "predictor": name,
            "beta": float(b),
            "se": float(s),
            "t": float(t),
            "p": float(p),
            "partial_R2": float(partial_r2),
            "full_R2": float(r2_full),
            "n_pairs": int(n),
        })
    return pd.DataFrame(rows)


# ── 4. main ─────────────────────────────────────────────────────────────────
def main(n_sentences: int = N_SENTENCES_DEFAULT, max_pairs: int = 200_000):
    df = load_radical_dataset()
    chars = df["char"].tolist()

    if PMI_PATH.exists() and COUNTS_PATH.exists():
        print(f"Loading cached PPMI from {PMI_PATH}")
        ppmi = load_npz(PMI_PATH)
        counts = np.load(COUNTS_PATH)
    else:
        print(f"Building PPMI from {n_sentences} Wikipedia zh sentences …")
        ppmi, counts = build_cooccurrence(chars, n_sentences=n_sentences)
        save_npz(PMI_PATH, ppmi)
        np.save(COUNTS_PATH, counts)
        print(f"Saved PPMI → {PMI_PATH}")

    print("Building per-pair predictors …")
    predictors = build_predictors(df, ppmi, max_pairs=max_pairs)
    print(f"  pairs: {len(predictors)}")

    all_rows = []
    for model_id in list_available_models():
        try:
            X, L = load_iso_last_char(model_id)
        except FileNotFoundError:
            continue
        cos = cosine_for_pairs(X, predictors["i"].to_numpy(), predictors["j"].to_numpy())
        local = predictors.copy()
        local["bert_cosine"] = cos
        try:
            res = fit_regression(local, model_id)
        except Exception as e:
            print(f"[warn] regression failed for {model_id}: {e}")
            continue
        all_rows.append(res)
        print(f"  {model_id:60s}  full R²={res['full_R2'].iloc[0]:.4f}")

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(RESULTS_DIR / "variance_decomposition.csv", index=False)
        print(f"\nWrote {len(out)} rows to results/variance_decomposition.csv")
    else:
        # write an empty file with headers so downstream scripts don't crash
        pd.DataFrame(columns=[
            "model", "predictor", "beta", "se", "t", "p",
            "partial_R2", "full_R2", "n_pairs",
        ]).to_csv(RESULTS_DIR / "variance_decomposition.csv", index=False)
        print("[warn] no models had cached embeddings; wrote empty file.")


if __name__ == "__main__":
    main()
