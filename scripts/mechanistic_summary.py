"""
mechanistic_summary.py — turn the 22,780 raw activation-patching trials
into a per-model summary of cross-radical retrieval lift.

What's in activation_patching.csv:
    For each pair of *distinct* radicals (R_src, R_tgt) and 30 source
    chars in R_src, we measured how often a top-10 nearest-neighbour
    retrieval from src ends up in R_tgt vs. random expectation. So all
    rows are cross-radical (the script never logs R_src==R_tgt).
    `lift` is observed_rate / baseline_rate, where lift = 1 is random.

What this tells us mechanistically:
    A model that has learnt anything at all about radical structure will
    have lift > 1 on average across cross-radical retrieval, because the
    retrieval space is biased: retrieving from char(R_src) lands in
    *similar* chars, and "similar" can correlate with shared radical for
    semantically-loaded radicals. The size of the off-diagonal lift
    quantifies how much radical-aligned retrieval the model's
    representations enable.

We summarise per model:
    - mean cross-radical retrieval lift
    - 95% bootstrap CI on the mean (resampling rows)
    - share of (R_src, R_tgt) pairs with lift > 1

Output:
    results/activation_patching_summary.csv

Runtime: ~5 sec on CPU.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import RESULTS_DIR, set_seed  # noqa: E402

set_seed()
B = 5000


def per_model(group: pd.DataFrame, rng: np.random.Generator) -> dict:
    lifts = group["lift"].to_numpy()
    if len(lifts) == 0:
        return {}
    obs_mean = float(np.mean(lifts))
    share_above_1 = float(np.mean(lifts > 1.0))
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(lifts), size=len(lifts))
        boot[b] = lifts[idx].mean()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_cross_lift": obs_mean,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "share_above_chance": share_above_1,
        "n_pairs": int(len(lifts)),
    }


def main():
    src = RESULTS_DIR / "activation_patching.csv"
    if not src.exists():
        print(f"[fatal] {src} not found")
        sys.exit(1)
    df = pd.read_csv(src)
    if df.empty:
        print("[fatal] activation_patching.csv is empty")
        sys.exit(1)

    rng = np.random.default_rng(42)
    rows = []
    for model, sub in df.groupby("model"):
        res = per_model(sub, rng)
        if res:
            res["model"] = model
            rows.append(res)
    out = pd.DataFrame(rows).sort_values("mean_cross_lift", ascending=False)
    out = out[["model", "mean_cross_lift", "ci_lo", "ci_hi",
               "share_above_chance", "n_pairs"]]
    out_path = RESULTS_DIR / "activation_patching_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
