"""
phonetic_vs_semantic_radicals.py — does the radical signal differ depending
on the role the radical plays?

Splits chars into:
    - role = "semantic"   (Kangxi radical is the semantic component of a
                          phonosemantic char, ~85% of CJK)
    - role = "identity"   (the char IS the radical, e.g. 水 for radical 水)
    - role = "unknown"    (radical didn't match any IDS component, often
                          due to glyph-variant fragmentation like 氵↔水)

For each role-restricted subset we re-run the corpus-scale cohesion test:
    intra-radical vs inter-radical cosine on the last layer of every model
    (isotropy-corrected, char-pool).

We expect:
    semantic-role chars → effect comparable to the corpus-wide effect
    identity chars      → not very meaningful (identity == radical itself)

This is meant to *probe* whether the corpus-scale cohesion effect is driven
specifically by chars where the radical historically marked meaning, or
whether it shows up everywhere. The paper currently has no result on this.

Output:
    results/phonetic_vs_semantic_radicals.csv
        rows = (model, role, n_chars, intra_mean, inter_mean, delta,
                cohens_d, p_perm, ci_lower, ci_upper)

Depends on: classify_liushu.py (for the role column), extract_embeddings,
            isotropy_correction
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
    list_available_models,
    list_available_layers,
    cohens_d,
    bootstrap_ci_diff,
    permutation_test_diff,
)
from radical_lib.embeddings import model_tag  # noqa: E402
from scripts.new.config import MAX_PAIRS_PER_RADICAL, N_PERMUTATIONS  # noqa: E402

set_seed()
ISO_DIR = CACHE_DIR / "embeddings_iso"


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


def sample_within_role(
    chars: List[str],
    char_idx: Dict[str, int],
    char_to_radical: Dict[str, int],
    sim: np.ndarray,
    role_keep: set,
    role_map: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample intra/inter pairs *restricted to chars whose role is in
    `role_keep`*. The radical-grouping logic mirrors layer_wise_analysis."""
    chars_role = [c for c in chars if role_map.get(c, "unknown") in role_keep]
    if len(chars_role) < 50:
        return np.array([]), np.array([])

    groups: Dict[int, List[str]] = {}
    for c in chars_role:
        groups.setdefault(char_to_radical[c], []).append(c)
    groups = {r: cs for r, cs in groups.items() if len(cs) >= 5}
    if not groups:
        return np.array([]), np.array([])

    intra: List[float] = []
    inter: List[float] = []
    rng = random.Random(42)
    for rad, rad_chars in groups.items():
        pairs = list(itertools.combinations(rad_chars, 2))
        if len(pairs) > MAX_PAIRS_PER_RADICAL:
            pairs = rng.sample(pairs, MAX_PAIRS_PER_RADICAL)
        for a, b in pairs:
            intra.append(sim[char_idx[a], char_idx[b]])
        others = [c for c in chars_role if char_to_radical[c] != rad]
        for _ in range(MAX_PAIRS_PER_RADICAL):
            a = rng.choice(rad_chars)
            b = rng.choice(others)
            inter.append(sim[char_idx[a], char_idx[b]])
    return np.array(intra), np.array(inter)


def main():
    df = load_radical_dataset()
    if "radical_role" not in df.columns:
        print("[fatal] data/radical_dataset.csv has no `radical_role` column.")
        print("        Run scripts/new/classify_liushu.py first.")
        sys.exit(1)

    chars = df["char"].tolist()
    char_idx = {c: i for i, c in enumerate(chars)}
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col]))
    role_map = dict(zip(df["char"], df["radical_role"]))

    role_subsets = {
        "semantic":  {"semantic"},
        "identity":  {"identity"},
        "unknown":   {"unknown"},
        "non_identity": {"semantic", "unknown"},
        "all":       {"semantic", "identity", "unknown"},
    }

    rows = []
    for model_id in list_available_models():
        try:
            X, L = load_iso_last_char(model_id)
        except FileNotFoundError:
            continue
        sim = cosine_matrix(X)
        print(f"\n[{model_id}]  layer={L}")
        for role_name, role_set in tqdm(role_subsets.items(), desc=model_id):
            intra, inter = sample_within_role(
                chars, char_idx, char_to_radical, sim, role_set, role_map
            )
            if len(intra) < 20 or len(inter) < 20:
                continue
            d = cohens_d(intra, inter)
            lo, hi, _ = bootstrap_ci_diff(intra, inter, rng=np.random.default_rng(42))
            p_perm, _, _ = permutation_test_diff(
                intra, inter, n_perm=N_PERMUTATIONS, rng=np.random.default_rng(42)
            )
            n_chars_in_role = int(sum(1 for c in chars if role_map.get(c) in role_set))
            rows.append({
                "model": model_id,
                "role": role_name,
                "layer": L,
                "n_chars": n_chars_in_role,
                "n_intra": len(intra),
                "n_inter": len(inter),
                "intra_mean": float(intra.mean()),
                "inter_mean": float(inter.mean()),
                "delta": float(intra.mean() - inter.mean()),
                "cohens_d": d,
                "ci_lower": lo,
                "ci_upper": hi,
                "p_perm": p_perm,
            })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "phonetic_vs_semantic_radicals.csv", index=False)
    print(f"\nWrote {len(out)} rows.")
    if not out.empty:
        print(out.groupby(["model", "role"])["cohens_d"].mean().unstack())


if __name__ == "__main__":
    main()
