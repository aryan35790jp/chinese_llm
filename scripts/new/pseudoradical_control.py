"""
pseudoradical_control.py — null partition control.

Reviewer question this answers:
    "How do you know the radical effect is specific to *Kangxi radicals*,
    and not just an artifact of partitioning 6,306 chars into 68 groups
    of any kind?"

Method:
    1. For each model: take the same isotropy-corrected last-layer
       embeddings used in layer_wise_analysis.
    2. Generate B random partitions of the dataset, each preserving the
       *exact* group-size distribution of the real radicals (so the
       baseline is matched on group sizes — without this match, the
       baseline cohesion would look artificially high in tiny groups).
    3. For each random partition compute the same Cohen's d as
       layer_wise_analysis would on real radicals.
    4. Compare the *real* d against this null distribution. The
       reported empirical p is:
            p_pseudo = (#{d_random ≥ d_real} + 1) / (B + 1)
       This is a stronger control than the per-pair label permutation in
       layer_wise_analysis — that one preserves marginal distribution
       only, while this preserves the joint group-size structure.

Output:
    results/pseudoradical_control.csv
        rows = (model, d_real, d_random_mean, d_random_p95, p_pseudo,
                n_partitions)

Runtime: ~10 min for B=200 partitions × 11 models. CPU only.
Depends on: extract_embeddings.py, isotropy_correction.py,
            layer_wise_analysis.py (uses its CSV for d_real)
"""
from __future__ import annotations
import itertools
import os
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
    list_available_models,
    list_available_layers,
    cohens_d,
)
from radical_lib.embeddings import model_tag  # noqa: E402
from scripts.new.config import MAX_PAIRS_PER_RADICAL  # noqa: E402

set_seed()
ISO_DIR = CACHE_DIR / "embeddings_iso"
B_DEFAULT = 200


def load_iso_last_char(model_id: str) -> Tuple[np.ndarray, int]:
    layers = list_available_layers(model_id)
    if not layers:
        raise FileNotFoundError(model_id)
    L = max(layers)
    path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_char.npy"
    if not path.exists():
        path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_mean.npy"
    return np.load(path), L


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return Xn @ Xn.T


def cohesion_d(sim: np.ndarray, partition: Dict[int, List[int]]) -> float:
    """Mirror layer_wise_analysis sampling logic, returning Cohen's d."""
    intra: List[float] = []
    inter: List[float] = []
    rng = random.Random(42)
    chars_idx = [i for cs in partition.values() for i in cs]
    for label, idxs in partition.items():
        pairs = list(itertools.combinations(idxs, 2))
        if len(pairs) > MAX_PAIRS_PER_RADICAL:
            pairs = rng.sample(pairs, MAX_PAIRS_PER_RADICAL)
        for a, b in pairs:
            intra.append(sim[a, b])
        others = [i for i in chars_idx if i not in set(idxs)]
        for _ in range(MAX_PAIRS_PER_RADICAL):
            a = rng.choice(idxs)
            b = rng.choice(others)
            inter.append(sim[a, b])
    return cohens_d(np.array(intra), np.array(inter))


def make_size_matched_partition(
    n_chars: int, group_sizes: List[int], rng: np.random.Generator
) -> Dict[int, List[int]]:
    """Assign chars to fake groups whose sizes match the real radical sizes."""
    indices = np.arange(n_chars)
    rng.shuffle(indices)
    partition: Dict[int, List[int]] = {}
    cursor = 0
    for k, sz in enumerate(group_sizes):
        partition[k] = indices[cursor:cursor + sz].tolist()
        cursor += sz
    return partition


def main(B: int = None):
    if B is None:
        # honour RADICAL_PSEUDO_B for fast Colab runs (e.g. =100 cuts time in half)
        B = int(os.environ.get("RADICAL_PSEUDO_B", B_DEFAULT))
    df = load_radical_dataset()
    chars = df["char"].tolist()
    n = len(chars)
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col].astype(int)))

    # real partition (only radicals with ≥ 20)
    real_groups: Dict[int, List[int]] = {}
    for i, c in enumerate(chars):
        real_groups.setdefault(char_to_radical[c], []).append(i)
    real_groups = {r: idxs for r, idxs in real_groups.items() if len(idxs) >= 20}
    group_sizes = [len(v) for v in real_groups.values()]

    rows = []
    for model_id in list_available_models():
        try:
            X, L = load_iso_last_char(model_id)
        except FileNotFoundError:
            continue
        sim = cosine_matrix(X)
        d_real = cohesion_d(sim, real_groups)
        rng = np.random.default_rng(42)
        d_random = []
        for b in tqdm(range(B), desc=f"{model_id}"):
            part = make_size_matched_partition(n, group_sizes, rng)
            d_random.append(cohesion_d(sim, part))
        d_random = np.array(d_random)
        p_pseudo = (np.sum(d_random >= d_real) + 1) / (B + 1)

        rows.append({
            "model": model_id,
            "layer": int(L),
            "d_real": float(d_real),
            "d_random_mean": float(d_random.mean()),
            "d_random_std": float(d_random.std(ddof=1)),
            "d_random_p95": float(np.percentile(d_random, 95)),
            "p_pseudo": float(p_pseudo),
            "n_partitions": int(B),
        })
        print(f"  {model_id}: d_real={d_real:.3f}, d_rand={d_random.mean():.3f}±{d_random.std():.3f}, "
              f"p={p_pseudo:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "pseudoradical_control.csv", index=False)
    print(f"\nWrote {len(out)} rows.")


if __name__ == "__main__":
    main()
