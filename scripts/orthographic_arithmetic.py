"""
orthographic_arithmetic.py — Mikolov-style compositionality test for CJK.

Question (Mikolov et al. 2013, but for radicals):
    Does the model encode characters as the sum of their components?
    If so, vector arithmetic should retrieve semantically related characters:
        E(河) − E(氵) + E(火)  ≈  E(?)
    where ? should be a fire-related character.

For each pair of "anchor radicals" (R1, R2) chosen from a curated list of
radicals whose unified-CJK glyph is in our dataset:
    Pick c sharing radical R1.
    Query: q = E(c) − E(glyph(R1)) + E(glyph(R2))
    Cosine-rank q against all chars; exclude c, glyph(R1), glyph(R2).
    Score: how many of the top-k retrieved chars share radical R2?

Reported metric per (model, R1, R2):
    - top-1 target rate         (top-1 retrieved has radical R2?)
    - top-10 target rate        (mean over k=10)
    - baseline rate              (|R2| / |dataset|, i.e. random retrieval)
    - lift = top10 / baseline    (>1 means radical-arithmetic is informative)

Why this matters:
    Behavioral, model-agnostic, no fine-tuning. If lift ≫ 1 we have
    evidence that radicals are encoded *additively* in embedding space —
    a stronger claim than "characters with the same radical happen to be
    close." If lift ≈ 1, then the radical signal is not compositional and
    the semantic-only account is reinforced.

Output:
    results/orthographic_arithmetic.csv
        rows = (model, anchor_r1, r1_num, anchor_r2, r2_num, n_trials,
                top1_target_rate, top10_target_rate, baseline_rate, lift_top10)
    results/orthographic_arithmetic_summary.csv
        rows = (model, n_pairs, mean_top10_rate, mean_baseline, mean_lift)

Runtime: ~5 min on cached embeddings. CPU only.
Depends on: extract_embeddings.py, isotropy_correction.py
"""
from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Dict, List

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


# Anchor radicals: kangxi number → its unified-CJK glyph (the form that
# typically appears in our dataset). We use unified-CJK forms (not the
# Kangxi-radical block U+2F00) because that's what the dataset uses.
ANCHOR_RADICALS: Dict[int, str] = {
    85:  "水",
    86:  "火",
    75:  "木",
    167: "金",
    61:  "心",
    30:  "口",
    149: "言",
    196: "鸟",
    195: "鱼",
    140: "艸",
    46:  "山",
    112: "石",
    74:  "月",
    102: "田",
    109: "目",
}


def load_iso_last_char(model_id: str):
    layers = list_available_layers(model_id)
    if not layers:
        raise FileNotFoundError(model_id)
    L = max(layers)
    path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_char.npy"
    if not path.exists():
        path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_mean.npy"
    return np.load(path), L


def main(n_trials_per_pair: int = 30, top_k: int = 10):
    df = load_radical_dataset()
    chars = df["char"].tolist()
    char_idx = {c: i for i, c in enumerate(chars)}
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col].astype(int)))

    # Restrict anchors to ones whose glyph is in the dataset
    anchors_present = {n: g for n, g in ANCHOR_RADICALS.items() if g in char_idx}
    print(f"Anchors present in dataset: {list(anchors_present.values())}  "
          f"({len(anchors_present)} of {len(ANCHOR_RADICALS)})")

    radical_chars: Dict[int, List[str]] = {}
    for c, r in char_to_radical.items():
        radical_chars.setdefault(r, []).append(c)

    rows = []
    summary_rows = []

    for model_id in list_available_models():
        try:
            X, L = load_iso_last_char(model_id)
        except FileNotFoundError:
            continue

        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        print(f"\n[{model_id}]  layer {L}, shape {X.shape}")

        all_top10_rates = []
        all_baseline_rates = []

        for r1, glyph1 in anchors_present.items():
            for r2, glyph2 in anchors_present.items():
                if r1 == r2:
                    continue
                src_chars = [c for c in radical_chars.get(r1, []) if c != glyph1]
                if len(src_chars) < 5 or len(radical_chars.get(r2, [])) < 5:
                    continue

                rng = random.Random(hash((model_id, r1, r2)) & 0xFFFFFFFF)
                trials = rng.sample(src_chars, min(n_trials_per_pair, len(src_chars)))

                top1_hits = 0
                top10_hits = 0
                for c in trials:
                    q = X[char_idx[c]] - X[char_idx[glyph1]] + X[char_idx[glyph2]]
                    qn = q / max(np.linalg.norm(q), 1e-12)
                    sims = Xn @ qn
                    exclude = {char_idx[c], char_idx[glyph1], char_idx[glyph2]}
                    order = np.argsort(-sims)
                    order = [i for i in order if i not in exclude]
                    top10 = order[:top_k]
                    top1 = order[0]
                    if char_to_radical[chars[top1]] == r2:
                        top1_hits += 1
                    n_top10 = sum(1 for i in top10 if char_to_radical[chars[i]] == r2)
                    top10_hits += n_top10

                n_eff = len(trials) * top_k
                top10_rate = top10_hits / n_eff if n_eff else 0.0
                baseline = len(radical_chars.get(r2, [])) / len(chars)
                rows.append({
                    "model": model_id,
                    "anchor_r1": glyph1, "r1_num": int(r1),
                    "anchor_r2": glyph2, "r2_num": int(r2),
                    "n_trials": len(trials),
                    "top1_target_rate": top1_hits / len(trials) if trials else 0.0,
                    "top10_target_rate": top10_rate,
                    "baseline_rate": float(baseline),
                    "lift_top10": (top10_rate / baseline) if baseline > 0 else float("nan"),
                })
                all_top10_rates.append(top10_rate)
                all_baseline_rates.append(baseline)

        if all_top10_rates:
            tr = np.array(all_top10_rates)
            br = np.maximum(np.array(all_baseline_rates), 1e-12)
            summary_rows.append({
                "model": model_id,
                "n_pairs": len(tr),
                "mean_top10_rate": float(tr.mean()),
                "mean_baseline": float(np.array(all_baseline_rates).mean()),
                "mean_lift": float((tr / br).mean()),
            })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "orthographic_arithmetic.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        RESULTS_DIR / "orthographic_arithmetic_summary.csv", index=False
    )
    print(f"\nWrote {len(rows)} pair-rows and {len(summary_rows)} summary rows.")
    if summary_rows:
        sm = pd.DataFrame(summary_rows)
        print(sm[["model", "mean_top10_rate", "mean_baseline", "mean_lift"]].to_string(index=False))


if __name__ == "__main__":
    main()
