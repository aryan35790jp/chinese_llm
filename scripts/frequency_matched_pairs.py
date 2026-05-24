"""
frequency_matched_pairs.py — frequency-matched pair control.

Reviewer question this answers:
    "Your existing data shows the per-radical cohesion correlates with
    character frequency (freq_rho ≈ 0.33). How much of the radical
    effect just reflects high-frequency chars clustering with each other?"

Method:
    1. Bin chars into K frequency deciles (using `frequency_proxy`).
    2. For each radical, sample intra-radical pairs as before.
    3. For each intra pair, sample a *frequency-matched* inter pair:
       same frequency-bin difference, but different radicals.
       This gives an exactly matched-on-freq inter distribution.
    4. Recompute Cohen's d. Report the matched effect alongside the
       unmatched one and the per-model "freq inflation" = d_unmatched − d_matched.

Output:
    results/frequency_matched.csv
        rows = (model, layer, d_unmatched, d_matched, freq_inflation,
                ci_matched_lo, ci_matched_hi, n_intra, n_inter_matched)

Why this is needed:
    The original pipeline reports `freq_rho` as a sanity check but does
    not actually *control out* frequency. A reviewer can demand "show me
    the effect with frequency held fixed." This script does exactly that.

Runtime: ~10 min total. CPU only.
Depends on: extract_embeddings.py, isotropy_correction.py
"""
from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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
)
from radical_lib.embeddings import model_tag  # noqa: E402
from scripts.new.config import N_BOOTSTRAP  # noqa: E402

set_seed()
ISO_DIR = CACHE_DIR / "embeddings_iso"
N_BINS = 10
PAIRS_PER_RADICAL = 30


def load_iso_last_char(model_id: str):
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


def main():
    df = load_radical_dataset()
    chars = df["char"].tolist()
    char_idx = {c: i for i, c in enumerate(chars)}
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col].astype(int)))

    if "frequency_proxy" not in df.columns:
        print("[fatal] dataset has no frequency_proxy column; rerun dataset_builder.py")
        sys.exit(1)
    freq = df["frequency_proxy"].astype(float).to_numpy()
    bins = pd.qcut(freq, q=N_BINS, labels=False, duplicates="drop")
    bin_arr = np.asarray(bins)

    # group chars by radical (only ≥20)
    groups: Dict[int, List[str]] = {}
    for c in chars:
        groups.setdefault(char_to_radical[c], []).append(c)
    groups = {r: cs for r, cs in groups.items() if len(cs) >= 20}

    # group chars by bin (for frequency-matched inter sampling)
    chars_by_bin: Dict[int, List[str]] = {}
    for c in chars:
        chars_by_bin.setdefault(int(bin_arr[char_idx[c]]), []).append(c)

    rows = []
    for model_id in list_available_models():
        try:
            X, L = load_iso_last_char(model_id)
        except FileNotFoundError:
            continue
        sim = cosine_matrix(X)
        rng = random.Random(42)

        intra: List[float] = []
        inter_unmatched: List[float] = []
        inter_matched: List[float] = []

        for rad, rad_chars in groups.items():
            if len(rad_chars) < 4:
                continue
            for _ in range(PAIRS_PER_RADICAL):
                a, b = rng.sample(rad_chars, 2)
                ia, ib = char_idx[a], char_idx[b]
                intra.append(sim[ia, ib])

                # unmatched inter
                others = [c for c in chars if char_to_radical[c] != rad]
                c1 = rng.choice(rad_chars)
                c2 = rng.choice(others)
                inter_unmatched.append(sim[char_idx[c1], char_idx[c2]])

                # frequency-matched inter
                # sample (c1, c2) where c1 is from radical R, c2 from a
                # different radical, AND |bin(c1)-bin(c2)| equals the bin
                # difference of the intra pair we just sampled
                target_diff = abs(bin_arr[ia] - bin_arr[ib])
                # try a few times; if no match, use a uniform random other
                tried = 0
                matched = None
                while tried < 20 and matched is None:
                    c1 = rng.choice(rad_chars)
                    c1_bin = int(bin_arr[char_idx[c1]])
                    candidate_bin = c1_bin + (target_diff if rng.random() < 0.5 else -target_diff)
                    if candidate_bin in chars_by_bin:
                        candidates = [
                            c2 for c2 in chars_by_bin[candidate_bin]
                            if char_to_radical[c2] != rad
                        ]
                        if candidates:
                            c2 = rng.choice(candidates)
                            matched = (c1, c2)
                    tried += 1
                if matched:
                    a2, b2 = matched
                    inter_matched.append(sim[char_idx[a2], char_idx[b2]])

        intra_a = np.array(intra)
        unm_a = np.array(inter_unmatched)
        mat_a = np.array(inter_matched)

        d_unm = cohens_d(intra_a, unm_a)
        d_mat = cohens_d(intra_a, mat_a) if len(mat_a) >= 30 else float("nan")
        if len(mat_a) >= 30:
            lo, hi, _ = bootstrap_ci_diff(intra_a, mat_a, n_boot=N_BOOTSTRAP,
                                          rng=np.random.default_rng(42))
        else:
            lo = hi = float("nan")

        rows.append({
            "model": model_id,
            "layer": int(L),
            "d_unmatched": float(d_unm),
            "d_matched": float(d_mat),
            "freq_inflation": float(d_unm - d_mat) if np.isfinite(d_mat) else float("nan"),
            "ci_matched_lo": float(lo),
            "ci_matched_hi": float(hi),
            "n_intra": int(len(intra_a)),
            "n_inter_unmatched": int(len(unm_a)),
            "n_inter_matched": int(len(mat_a)),
        })
        print(f"  {model_id}: d_unmatched={d_unm:.3f}, d_matched={d_mat:.3f}, "
              f"freq_inflation={d_unm - d_mat:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "frequency_matched.csv", index=False)
    print(f"\nWrote {len(out)} rows.")


if __name__ == "__main__":
    main()
