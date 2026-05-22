"""
activation_patching.py — geometric ablation, the causal cousin of
orthographic_arithmetic.

Honest framing first:
    Full activation patching (Vig 2020; Meng 2022) needs a downstream
    task whose output we can measure. Chinese character cohesion has no
    such natural task at the level we're studying — there's no "next
    character" prediction we care about that would surface a clean
    causal effect of swapping one char's representation for another.

    What we CAN do is a clean *geometric* intervention. For every pair
    of radicals (S, T):
        1. Compute the "S direction" in embedding space:
              μ_S = mean of all S-radical chars
              r_S = (μ_S - global mean), unit-normalized
           Same for T.
        2. For chars c with radical S, project out r_S and add r_T at
           the same scale → "patched" representation c'.
        3. Cosine-rank c' against the full embedding. What fraction of
           the top-k retrieved chars belong to radical T?

    A high "swap success rate" means the radical direction is
    decodable AND modifiable by a linear patch. A low rate means it
    isn't a clean compositional axis.

We run this as a sanity-check baseline against orthographic_arithmetic;
they should agree qualitatively.

Output:
    results/activation_patching.csv
        rows = (model, src_radical, tgt_radical, n_src_chars,
                top10_target_rate, baseline_rate, lift)

Runtime: ~10 min. CPU only.
Depends on: extract_embeddings.py, isotropy_correction.py
"""
from __future__ import annotations
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
)
from radical_lib.embeddings import model_tag  # noqa: E402

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


def radical_directions(
    X: np.ndarray, char_to_radical: Dict[str, int], chars: List[str],
    min_size: int = 10
) -> Dict[int, np.ndarray]:
    """{radical: unit vector pointing from global mean to radical mean}."""
    global_mean = X.mean(axis=0)
    by_rad: Dict[int, List[int]] = {}
    for i, c in enumerate(chars):
        by_rad.setdefault(char_to_radical[c], []).append(i)

    dirs: Dict[int, np.ndarray] = {}
    for rad, idx in by_rad.items():
        if len(idx) < min_size:
            continue
        mu = X[idx].mean(axis=0)
        d = mu - global_mean
        n = np.linalg.norm(d)
        if n > 0:
            dirs[rad] = d / n
    return dirs


def main(top_k: int = 10, max_src_chars: int = 30):
    df = load_radical_dataset()
    chars = df["char"].tolist()
    char_idx = {c: i for i, c in enumerate(chars)}
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col].astype(int)))
    n_chars = len(chars)

    rows = []
    for model_id in list_available_models():
        try:
            X, L = load_iso_last_char(model_id)
        except FileNotFoundError:
            continue

        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        dirs = radical_directions(X, char_to_radical, chars)
        if len(dirs) < 2:
            continue

        by_rad: Dict[int, List[str]] = {}
        for c in chars:
            r = char_to_radical[c]
            if r in dirs:
                by_rad.setdefault(r, []).append(c)

        radicals = list(dirs.keys())
        print(f"\n[{model_id}]  layer {L}, {len(radicals)} radicals patchable")

        for src in radicals:
            src_chars = by_rad.get(src, [])
            if len(src_chars) < 5:
                continue
            for tgt in radicals:
                if src == tgt:
                    continue
                target_count = len(by_rad.get(tgt, []))
                baseline = target_count / n_chars

                top10_hits = 0
                n_eff = 0
                for c in src_chars[:max_src_chars]:
                    e = X[char_idx[c]]
                    proj_src = (e @ dirs[src]) * dirs[src]
                    # patched = remove src component, add tgt with same scale
                    e_patched = e - proj_src + dirs[tgt] * np.linalg.norm(proj_src)
                    en = e_patched / max(np.linalg.norm(e_patched), 1e-12)
                    sims = Xn @ en
                    sims[char_idx[c]] = -np.inf  # exclude self
                    order = np.argsort(-sims)[:top_k]
                    n_eff += top_k
                    for idx in order:
                        if char_to_radical[chars[idx]] == tgt:
                            top10_hits += 1

                top10_rate = top10_hits / n_eff if n_eff else 0.0
                lift = (top10_rate / baseline) if baseline > 0 else float("nan")
                rows.append({
                    "model": model_id,
                    "src_radical": int(src), "tgt_radical": int(tgt),
                    "n_src_chars": min(max_src_chars, len(src_chars)),
                    "top10_target_rate": top10_rate,
                    "baseline_rate": float(baseline),
                    "lift": float(lift) if np.isfinite(lift) else float("nan"),
                })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "activation_patching.csv", index=False)
    print(f"\nWrote {len(rows)} rows.")


if __name__ == "__main__":
    main()
