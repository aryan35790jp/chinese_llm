from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# ══════════════════════════════════════════════
# Controlled experiment: same semantic field, different radicals
# Across FOUR semantic fields — isolates radical signal from semantics
# ══════════════════════════════════════════════
semantic_fields = {
    "animals_犭": {
        "same_radical":  ["狗","狼","狐","猫","狮"],   # share 犭 (dog radical)
        "diff_radical":  ["虎","熊","牛","羊","马"],   # animals WITHOUT 犭
        "radical": "犭",
    },
    "water_氵": {
        "same_radical":  ["河","湖","洋","泪","洗"],   # share 氵 (water radical)
        "diff_radical":  ["水","冰","雨","霜","泉"],   # water-related WITHOUT 氵
        "radical": "氵",
    },
    "wood_木": {
        "same_radical":  ["林","松","柳","桥","棉"],   # share 木 (wood radical)
        "diff_radical":  ["竹","藤","草","芦","茎"],   # plant-related WITHOUT 木
        "radical": "木",
    },
    "metal_钅": {
        "same_radical":  ["铁","铜","锡","钢","银"],   # share 钅 (metal radical)
        "diff_radical":  ["金","玉","石","矿","宝"],   # material-related WITHOUT 钅
        "radical": "钅",
    },
}

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    mean_diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return mean_diff / pooled_std


def run_experiment(model_name, label):
    print(f"\n{'#'*60}")
    print(f"  {label}  ({model_name})")
    print(f"{'#'*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Collect ALL unique characters across all fields
    all_chars = []
    for field in semantic_fields.values():
        all_chars.extend(field["same_radical"])
        all_chars.extend(field["diff_radical"])
    all_chars = list(dict.fromkeys(all_chars))  # deduplicate, preserve order

    # Extract embeddings (attention-mask-weighted mean pooling)
    all_embeddings = {}
    inputs = tokenizer(all_chars, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1)
    embs = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
    for j, c in enumerate(all_chars):
        all_embeddings[c] = embs[j].numpy()

    emb_list = [all_embeddings[c] for c in all_chars]
    sim = cosine_similarity(emb_list)
    idx = {c: i for i, c in enumerate(all_chars)}

    # Per-field analysis
    field_results = []
    pooled_intra = []
    pooled_cross = []

    for fname, field in semantic_fields.items():
        same = field["same_radical"]
        diff = field["diff_radical"]

        # Intra-radical: pairs within same_radical group
        intra = [sim[idx[a]][idx[b]] for a, b in itertools.combinations(same, 2)]
        # Cross: same_radical char ↔ diff_radical char
        cross = [sim[idx[a]][idx[b]] for a in same for b in diff]

        pooled_intra.extend(intra)
        pooled_cross.extend(cross)

        t, p = ttest_ind(intra, cross, equal_var=False)
        d = cohens_d(intra, cross)

        print(f"\n  Field: {fname}")
        print(f"    Same-radical ({field['radical']}↔{field['radical']}): {np.mean(intra):.4f}  (n={len(intra)})")
        print(f"    Cross ({field['radical']}↔other):     {np.mean(cross):.4f}  (n={len(cross)})")
        print(f"    p = {p:.6f}   d = {d:.4f}")

        field_results.append({
            "field": fname, "radical": field["radical"],
            "intra_mean": np.mean(intra), "cross_mean": np.mean(cross),
            "n_intra": len(intra), "n_cross": len(cross),
            "p": p, "d": d,
        })

    # Pooled across all 4 fields
    pooled_intra = np.array(pooled_intra)
    pooled_cross = np.array(pooled_cross)
    t_pool, p_pool = ttest_ind(pooled_intra, pooled_cross, equal_var=False)
    d_pool = cohens_d(pooled_intra, pooled_cross)

    # Permutation test on pooled data
    observed = np.mean(pooled_intra) - np.mean(pooled_cross)
    combined = np.concatenate([pooled_intra, pooled_cross])
    n_intra = len(pooled_intra)
    perm_diffs = []
    for _ in range(5000):
        np.random.shuffle(combined)
        perm_diffs.append(np.mean(combined[:n_intra]) - np.mean(combined[n_intra:]))
    perm_diffs = np.array(perm_diffs)
    perm_p = (np.sum(perm_diffs >= observed) + 1) / (5001)

    print(f"\n  {'='*50}")
    print(f"  POOLED ACROSS {len(semantic_fields)} SEMANTIC FIELDS ({label})")
    print(f"  {'='*50}")
    print(f"  Same-radical:  {np.mean(pooled_intra):.4f} (n={len(pooled_intra)})")
    print(f"  Cross-radical: {np.mean(pooled_cross):.4f} (n={len(pooled_cross)})")
    print(f"  Welch t = {t_pool:.4f}   p = {p_pool:.6f}   d = {d_pool:.4f}")
    print(f"  Permutation p = {perm_p:.4f} (5000 shuffles)")

    return {
        "label": label,
        "field_results": field_results,
        "pooled_intra_mean": np.mean(pooled_intra),
        "pooled_cross_mean": np.mean(pooled_cross),
        "pooled_p": p_pool,
        "pooled_d": d_pool,
        "pooled_perm_p": perm_p,
        "pooled_intra": pooled_intra,
        "pooled_cross": pooled_cross,
        "perm_diffs": perm_diffs,
    }


# ══════════════════════════════════════════════
# Run both models
# ══════════════════════════════════════════════
results = []
for model_name, label in [
    ("bert-base-multilingual-cased", "mBERT"),
    ("hfl/chinese-bert-wwm-ext",     "Chinese-BERT"),
]:
    results.append(run_experiment(model_name, label))

# ══════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════
os.makedirs("results", exist_ok=True)
rows = []
for r in results:
    for fr in r["field_results"]:
        rows.append({
            "Model": r["label"], "Field": fr["field"], "Radical": fr["radical"],
            "Intra": round(fr["intra_mean"], 4), "Cross": round(fr["cross_mean"], 4),
            "p": fr["p"], "d": round(fr["d"], 4),
        })
    rows.append({
        "Model": r["label"], "Field": "POOLED", "Radical": "all",
        "Intra": round(r["pooled_intra_mean"], 4),
        "Cross": round(r["pooled_cross_mean"], 4),
        "p": r["pooled_p"], "d": round(r["pooled_d"], 4),
    })
pd.DataFrame(rows).to_csv("results/semantic_control_results.csv", index=False)

for r in results:
    tag = r["label"].lower().replace("-", "_")
    np.save(f"results/{tag}_semantic_control_intra.npy", r["pooled_intra"])
    np.save(f"results/{tag}_semantic_control_cross.npy", r["pooled_cross"])
    np.save(f"results/{tag}_semantic_control_perm.npy", r["perm_diffs"])

# ══════════════════════════════════════════════
# Figure — Semantic control forest plot
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for idx_m, r in enumerate(results):
    ax = axes[idx_m]
    fields = r["field_results"]
    labels = [f["field"].split("_")[0] for f in fields] + ["POOLED"]
    d_vals = [f["d"] for f in fields] + [r["pooled_d"]]
    colors = ["#2196F3"]*len(fields) + ["#E91E63"]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, d_vals, color=colors, edgecolor="black", height=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    for i, (bar, dv) in enumerate(zip(bars, d_vals)):
        ax.text(bar.get_width() + 0.02 if dv >= 0 else bar.get_width() - 0.02,
                bar.get_y() + bar.get_height()/2,
                f"d={dv:.3f}", va="center", ha="left" if dv >= 0 else "right",
                fontsize=9, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Cohen's d", fontsize=11)
    ax.set_title(f"{r['label']}\nPooled p={r['pooled_p']:.4f}, perm p={r['pooled_perm_p']:.4f}",
                 fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Semantic Control: Radical Effect Within Same Semantic Field\n"
             "(4 fields, same-radical vs cross-radical, semantics held constant)",
             fontsize=12, fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig("figures/semantic_control.png", dpi=200, bbox_inches="tight")
print("\n[Figure saved to figures/semantic_control.png]")

print(f"\n{'='*60}")
print("  FINAL RESULTS")
print(f"{'='*60}")
for r in results:
    print(f"\n  {r['label']}:")
    print(f"    Pooled same-radical = {r['pooled_intra_mean']:.4f}")
    print(f"    Pooled cross-radical = {r['pooled_cross_mean']:.4f}")
    print(f"    p = {r['pooled_p']:.6f}  d = {r['pooled_d']:.4f}")
    print(f"    Permutation p = {r['pooled_perm_p']:.4f}")

