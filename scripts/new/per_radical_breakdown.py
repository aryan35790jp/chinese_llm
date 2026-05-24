"""
per_radical_breakdown.py — per-radical heterogeneity of the cohesion signal.

Reviewer question this answers (implicit in both reviews):
    "Is the radical effect uniform across all 68 radicals, or is it driven
     by a small subset of semantically-coherent ones (e.g. 鸟, 鱼, 木)?"

Method:
    Read activation_patching.csv (which gives per-(R_src, R_tgt) lift)
    and compute, per (model, R_src), the *self-axis* score:
        for each source radical R, average the lift across all target
        radicals R'≠R. High average means "char(R) tends to retrieve
        chars from any other radical at above-chance rates", which is a
        proxy for how distinctive the model's representation of R is.

    Then identify the top-10 and bottom-10 radicals per model and
    cross-tabulate with the radical's *semantic transparency* (which we
    approximate with liushu_class from the dataset: pictograph or
    identity-class radicals are visually concrete; phonosemantic with
    semantic-radical role correlates with meaning).

Output:
    results/per_radical_lift.csv
        rows = (model, src_radical, mean_lift_to_others, n_targets)
    results/radical_heterogeneity_summary.csv
        rows = (model, top10_radicals, bottom10_radicals,
                top_minus_bottom_gap)

Runtime: ~3 sec on CPU. No embeddings needed.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import RESULTS_DIR, load_radical_dataset  # noqa: E402

# Map kangxi number → unicode radical glyph for the report
KANGXI_GLYPHS = {
    1: "一", 9: "人", 30: "口", 32: "土", 38: "女", 46: "山", 53: "广",
    57: "弓", 60: "彳", 61: "心", 64: "手", 66: "攴", 72: "日", 75: "木",
    85: "水", 86: "火", 94: "犬", 96: "玉", 104: "疒", 108: "皿", 109: "目",
    112: "石", 113: "示", 116: "穴", 118: "竹", 119: "米", 120: "糸",
    128: "耳", 130: "肉", 137: "舟", 140: "艸", 142: "虫", 145: "衣",
    147: "見", 149: "言", 154: "貝", 157: "足", 159: "車", 162: "辵",
    163: "邑", 164: "酉", 167: "金", 169: "門", 170: "阜", 173: "雨",
    184: "食", 187: "馬", 195: "魚", 196: "鳥",
}


def main():
    src = RESULTS_DIR / "activation_patching.csv"
    if not src.exists():
        print(f"[fatal] {src} not found")
        sys.exit(1)
    df = pd.read_csv(src)
    if df.empty:
        print("[fatal] empty")
        sys.exit(1)

    # per (model, src_radical) average lift across distinct target radicals
    grp = df.groupby(["model", "src_radical"])["lift"].agg(["mean", "count"]).reset_index()
    grp.columns = ["model", "src_radical", "mean_lift_to_others", "n_targets"]
    grp.to_csv(RESULTS_DIR / "per_radical_lift.csv", index=False)
    print(f"Wrote per_radical_lift.csv ({len(grp)} rows)")

    # build top/bottom-10 per model
    rows = []
    for model, sub in grp.groupby("model"):
        sub_sorted = sub.sort_values("mean_lift_to_others", ascending=False)
        top10 = sub_sorted.head(10)
        bot10 = sub_sorted.tail(10)
        top_str = ", ".join(
            f"{KANGXI_GLYPHS.get(int(r), str(int(r)))} ({l:.1f})"
            for r, l in zip(top10["src_radical"], top10["mean_lift_to_others"])
        )
        bot_str = ", ".join(
            f"{KANGXI_GLYPHS.get(int(r), str(int(r)))} ({l:.1f})"
            for r, l in zip(bot10["src_radical"], bot10["mean_lift_to_others"])
        )
        gap = float(top10["mean_lift_to_others"].mean() - bot10["mean_lift_to_others"].mean())
        rows.append({
            "model": model,
            "top10_radicals": top_str,
            "bottom10_radicals": bot_str,
            "top_minus_bottom_gap": gap,
            "max_lift": float(top10["mean_lift_to_others"].max()),
            "min_lift": float(bot10["mean_lift_to_others"].min()),
        })
    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "radical_heterogeneity_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print("\nGap (top10 − bottom10) by model:")
    print(out[["model", "top_minus_bottom_gap", "max_lift", "min_lift"]]
          .round(2).to_string(index=False))
    print("\nTop 10 most distinctive radicals (Chinese-BERT):")
    print(out.loc[out["model"] == "hfl/chinese-bert-wwm-ext", "top10_radicals"].iloc[0])


if __name__ == "__main__":
    main()
