"""
layer_wise_analysis.py — the centerpiece experiment.

For every (model, layer, pool, isotropy) combination, compute:
    - intra-radical vs inter-radical cosine similarity
    - Welch's t-test
    - permutation test (1000 shuffles)
    - bootstrap 95% CI for the mean difference
    - Cohen's d
    - RSA: Spearman correlation between embedding RDM and same-radical RDM

Output:
    results/layer_wise.csv
        columns: model, layer, pool, iso, intra_mean, inter_mean, delta,
                 cohens_d, p_welch, p_perm, ci_lower, ci_upper,
                 rsa_rho, rsa_p, n_intra, n_inter

This is the file behind the "x=layer, y=Cohen's d, one line per model"
figure that becomes the paper's centerpiece.

Memory strategy:
    We never hold a full 6306×6306 cosine matrix for more than one
    (model, layer, pool, iso) combination at a time. After each
    combination we throw away the matrix and only keep the scalar
    statistics. RAM stays under 8 GB.

Runtime: ~1 hour for 11 models × 13 layers × 2 pools (char, mean) × 2
isotropy settings (raw, corrected). The bottleneck is the bootstrap
inside each cell; we vectorize it.

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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    CACHE_DIR,
    RESULTS_DIR,
    set_seed,
    load_radical_dataset,
    radical_groups,
    list_available_models,
    list_available_layers,
    load_layer_embeddings,
    cohens_d,
    welch_t,
    bootstrap_ci_diff,
    permutation_test_diff,
    rsa_spearman,
)
from radical_lib.embeddings import model_tag  # noqa: E402
from scripts.new.config import (  # noqa: E402
    MAX_PAIRS_PER_RADICAL,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
)

set_seed()
RNG = np.random.default_rng(42)

ISO_DIR = CACHE_DIR / "embeddings_iso"


def load_layer(model_id: str, layer: int, pool: str, iso: bool) -> np.ndarray:
    """Load either raw or isotropy-corrected embeddings for a (model, layer, pool)."""
    if not iso:
        return load_layer_embeddings(model_id, layer, pool=pool)
    path = ISO_DIR / model_tag(model_id) / f"layer{layer:02d}_{pool}.npy"
    if not path.exists():
        raise FileNotFoundError(f"isotropy-corrected file missing: {path}")
    return np.load(path)


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Row-normalized X X^T. Rows of X must align with the dataset's char order."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T


def sample_pairs(
    chars: List[str],
    char_to_radical: Dict[str, int],
    groups: Dict[int, List[str]],
    sim: np.ndarray,
    char_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample MAX_PAIRS_PER_RADICAL intra and inter pairs per radical."""
    intra: List[float] = []
    inter: List[float] = []
    for rad, rad_chars in groups.items():
        pairs = list(itertools.combinations(rad_chars, 2))
        if len(pairs) > MAX_PAIRS_PER_RADICAL:
            pairs = random.sample(pairs, MAX_PAIRS_PER_RADICAL)
        for a, b in pairs:
            intra.append(sim[char_idx[a], char_idx[b]])

        others = [c for c in chars if char_to_radical[c] != rad]
        for _ in range(MAX_PAIRS_PER_RADICAL):
            a = random.choice(rad_chars)
            b = random.choice(others)
            inter.append(sim[char_idx[a], char_idx[b]])
    return np.asarray(intra), np.asarray(inter)


def rsa_value(sim: np.ndarray, char_to_radical_arr: np.ndarray) -> Tuple[float, float]:
    """RSA: Spearman corr between embedding similarity and the binary
    same-radical / different-radical RDM. We use *similarity* on both sides
    so the sign is interpretable (positive ρ = same-radical pairs are more
    similar)."""
    n = sim.shape[0]
    # Subsample to keep this tractable: 2000 chars → ~2M off-diagonal pairs
    if n > 2000:
        idx = RNG.choice(n, size=2000, replace=False)
        sim = sim[np.ix_(idx, idx)]
        labels = char_to_radical_arr[idx]
    else:
        labels = char_to_radical_arr
    iu = np.triu_indices(sim.shape[0], k=1)
    sim_vec = sim[iu]
    same = (labels[iu[0]] == labels[iu[1]]).astype(np.float64)
    return rsa_spearman(sim_vec.reshape(-1, 1) @ np.array([[1.0]]),  # noqa: E501
                        same.reshape(-1, 1) @ np.array([[1.0]]))     # noqa: E501


def main():
    df = load_radical_dataset()
    chars = df["char"].tolist()
    char_idx = {c: i for i, c in enumerate(chars)}
    char_to_radical = dict(zip(df["char"], df["radical_number"] if "radical_number" in df.columns else df["radical"]))
    groups = radical_groups()
    char_to_radical_arr = np.array([char_to_radical[c] for c in chars])

    rows = []
    models = list_available_models()
    if not models:
        print("[fatal] no cached embeddings. Run extract_embeddings.py first.")
        sys.exit(1)

    for model_id in models:
        layers = list_available_layers(model_id)
        if not layers:
            continue
        print(f"\n=== {model_id}  layers={layers} ===")
        for layer in tqdm(layers, desc=f"{model_id}"):
            for pool in ("char", "mean"):
                for iso in (False, True):
                    try:
                        X = load_layer(model_id, layer, pool, iso)
                    except FileNotFoundError:
                        continue
                    sim = cosine_matrix(X)

                    # Reset the RNG so pair sampling is identical across cells
                    random.seed(42)

                    intra, inter = sample_pairs(chars, char_to_radical, groups, sim, char_idx)
                    t, p_w = welch_t(intra, inter)
                    d = cohens_d(intra, inter)
                    lo, hi, _ = bootstrap_ci_diff(intra, inter, n_boot=N_BOOTSTRAP, rng=np.random.default_rng(42))
                    p_perm, observed, _ = permutation_test_diff(
                        intra, inter, n_perm=N_PERMUTATIONS, rng=np.random.default_rng(42)
                    )

                    # RSA against same-radical RDM (subsampled to 2000 chars)
                    n = sim.shape[0]
                    sub_idx = RNG.choice(n, size=min(n, 2000), replace=False)
                    sub_sim = sim[np.ix_(sub_idx, sub_idx)]
                    sub_labels = char_to_radical_arr[sub_idx]
                    same_rdm = (sub_labels[:, None] == sub_labels[None, :]).astype(np.float64)
                    rho, p_rsa = rsa_spearman(sub_sim, same_rdm)

                    rows.append({
                        "model": model_id,
                        "layer": layer,
                        "pool": pool,
                        "iso": int(iso),
                        "intra_mean": float(intra.mean()),
                        "inter_mean": float(inter.mean()),
                        "delta": float(intra.mean() - inter.mean()),
                        "cohens_d": d,
                        "p_welch": p_w,
                        "p_perm": p_perm,
                        "ci_lower": lo,
                        "ci_upper": hi,
                        "rsa_rho": rho,
                        "rsa_p": p_rsa,
                        "n_intra": len(intra),
                        "n_inter": len(inter),
                    })

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "layer_wise.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}")
    print(out.groupby(["model", "iso"])["cohens_d"].agg(["max", "idxmax"]))


if __name__ == "__main__":
    main()
