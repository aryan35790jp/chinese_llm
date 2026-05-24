"""
downstream_validation.py — connect geometry to behavior.

Question:
    Cohesion measurements are about embedding *geometry*. Does that
    geometry actually predict task performance? Two ways to ask:

    (a) Per-pair: does cosine similarity between two CJK characters
        correlate with their similarity score in a Chinese word-similarity
        dataset like PKU-500?
    (b) Per-radical: do characters from high-cohesion radicals show
        higher mean human-similarity scores in the dataset?

We use:
    PKU-500 (Wang et al. 2017), or fall back to an embedded mini-version
    if the file isn't available. PKU-500 is word-pair scored; for each
    bigram pair (w1, w2) where every char is in our dataset, we compute
    a model "word similarity" as the mean pairwise char-cosine.

Pipeline:
    1. Load PKU-500 from data/pku500.txt; otherwise use the fallback set.
    2. For each model and each word pair: compute mean pairwise cosine.
    3. Spearman + Kendall correlation with human scores.
    4. Per-radical breakdown: which radicals' chars correlate best with
       human judgment?

Output:
    results/downstream_validation.csv
        rows = (model, n_pairs, layer, spearman_rho, spearman_p,
                kendall_tau, kendall_p)
    results/downstream_per_radical.csv
        rows = (model, radical, n_pairs, mean_human_score, mean_model_cos)

Runtime: ~5 min. CPU only.
Depends on: extract_embeddings.py, isotropy_correction.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    RESULTS_DIR,
    set_seed,
    load_radical_dataset,
    list_available_models,
    list_available_layers,
)
from radical_lib.embeddings import model_tag  # noqa: E402

set_seed()
ISO_DIR = CACHE_DIR / "embeddings_iso"


# ── 1. dataset loaders ──────────────────────────────────────────────────────
def load_pku500() -> pd.DataFrame:
    """Load PKU-500 if available, else return an embedded fallback set."""
    paths = [
        DATA_DIR / "pku500.txt",
        DATA_DIR / "pku_simlex.txt",
        DATA_DIR / "downstream" / "pku500.txt",
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(
                p, sep=r"\s+", header=None,
                names=["w1", "w2", "score"], engine="python"
            )
            print(f"Loaded {len(df)} pairs from {p}")
            return df

    print("[warn] no PKU-500 file found; using embedded fallback (~40 pairs)")
    fallback = [
        ("河流", "江河", 8.5), ("湖泊", "海洋", 7.0), ("书本", "书籍", 9.0),
        ("机器", "机械", 8.5), ("飞机", "汽车", 5.0), ("学习", "教育", 7.5),
        ("音乐", "歌曲", 8.0), ("电脑", "计算", 6.5), ("跑步", "运动", 7.0),
        ("食物", "饭菜", 8.0), ("猫咪", "狗狗", 6.0), ("国家", "城市", 6.0),
        ("学生", "学校", 7.0), ("医生", "医院", 7.5), ("夏天", "冬天", 4.0),
        ("睡觉", "休息", 7.0), ("吃饭", "饮食", 7.5), ("说话", "讲话", 9.0),
        ("时间", "时刻", 8.0), ("地方", "地点", 8.5), ("看到", "看见", 9.0),
        ("天空", "云彩", 6.0), ("大海", "波浪", 6.5), ("森林", "树林", 8.5),
        ("钢铁", "金属", 8.0), ("石头", "岩石", 8.5), ("木头", "树木", 7.5),
        ("山峰", "山顶", 8.5), ("身体", "肉体", 7.5), ("心情", "情绪", 8.0),
        ("声音", "音响", 7.0), ("眼睛", "目光", 6.5), ("手指", "手掌", 7.0),
        ("武器", "兵器", 8.5), ("房屋", "建筑", 7.0), ("车辆", "汽车", 8.0),
        ("衣服", "服装", 8.5), ("食品", "粮食", 7.5), ("水果", "果实", 7.5),
        ("颜色", "色彩", 8.5), ("数字", "数目", 8.0),
    ]
    return pd.DataFrame(fallback, columns=["w1", "w2", "score"])


def load_iso_last_char(model_id: str) -> Tuple[np.ndarray, int]:
    layers = list_available_layers(model_id)
    if not layers:
        raise FileNotFoundError(model_id)
    L = max(layers)
    path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_char.npy"
    if not path.exists():
        path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_mean.npy"
    return np.load(path), L


def word_pair_cosine(
    w1: str, w2: str, X: np.ndarray, char_idx: dict
) -> float:
    """Mean pairwise cosine across chars(w1) × chars(w2)."""
    a = [char_idx[c] for c in w1 if c in char_idx]
    b = [char_idx[c] for c in w2 if c in char_idx]
    if not a or not b:
        return float("nan")
    A = X[a]
    B = X[b]
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return float((A @ B.T).mean())


def main():
    df = load_radical_dataset()
    chars = df["char"].tolist()
    char_idx = {c: i for i, c in enumerate(chars)}
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col].astype(int)))

    pairs = load_pku500()

    rows = []
    radical_rows = []

    for model_id in list_available_models():
        try:
            X, L = load_iso_last_char(model_id)
        except FileNotFoundError:
            continue

        valid_pairs = []
        cosines = []
        scores = []
        radicals_per_pair = []
        for _, row in pairs.iterrows():
            w1, w2, sc = row["w1"], row["w2"], row["score"]
            cos = word_pair_cosine(w1, w2, X, char_idx)
            if np.isnan(cos):
                continue
            valid_pairs.append((w1, w2))
            cosines.append(cos)
            scores.append(sc)
            first_char = next((c for c in w1 if c in char_idx), None)
            radicals_per_pair.append(char_to_radical.get(first_char, -1))

        if len(valid_pairs) < 5:
            print(f"[skip] {model_id}: only {len(valid_pairs)} valid pairs")
            continue

        rho, p_rho = spearmanr(cosines, scores)
        tau, p_tau = kendalltau(cosines, scores)
        rows.append({
            "model": model_id, "n_pairs": len(valid_pairs), "layer": L,
            "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "kendall_tau": float(tau), "kendall_p": float(p_tau),
        })
        print(f"  {model_id:60s}  ρ={rho:.3f}  τ={tau:.3f}  n={len(valid_pairs)}")

        rdf = pd.DataFrame({
            "radical": radicals_per_pair, "score": scores, "cos": cosines,
        })
        for rad, g in rdf.groupby("radical"):
            if rad < 0 or len(g) < 3:
                continue
            radical_rows.append({
                "model": model_id, "radical": int(rad), "n_pairs": len(g),
                "mean_human_score": float(g["score"].mean()),
                "mean_model_cos": float(g["cos"].mean()),
            })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "downstream_validation.csv", index=False)
    pd.DataFrame(radical_rows).to_csv(RESULTS_DIR / "downstream_per_radical.csv", index=False)
    print(f"\nWrote {len(rows)} model rows and {len(radical_rows)} per-radical rows.")


if __name__ == "__main__":
    main()
