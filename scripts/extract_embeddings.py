from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════
# Character groups (constant across all models)
# ══════════════════════════════════════════════
groups = {
    "water":  ["河","海","湖","洋","泪","洗"],   # 氵 radical
    "animal": ["猫","狗","狼","狐","狮"],         # 犭 radical
    "wood":   ["林","松","柳","桥","棉"],         # 木 radical
    "fire":   ["灯","烧","炎","烟"],             # 火 radical
    "object": ["椅","桌","车","船","刀"],         # mixed
}

control_groups = {
    "water_semantic_no_radical": ["水","冰","雨","雪","霜","泉"],
    "animal_semantic_no_radical": ["鸟","鱼","马","虎","蛇","鹿"],
}

# Flatten + deduplicate
chars = []
for cl in groups.values():
    chars.extend(cl)
for cl in control_groups.values():
    chars.extend(cl)
seen = set()
chars_deduped = []
for c in chars:
    if c not in seen:
        chars_deduped.append(c)
        seen.add(c)
chars = chars_deduped

# ══════════════════════════════════════════════
# Analysis functions
# ══════════════════════════════════════════════
def cohens_d(a, b):
    mean_diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    return mean_diff / pooled_std

def run_analysis(model_name, label):
    print("\n" + "#"*60)
    print(f"  MODEL: {label}")
    print(f"  ({model_name})")
    print("#"*60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_embedding(char):
        inputs = tokenizer(char, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.squeeze().numpy()

    embeddings = [get_embedding(c) for c in chars]
    sim_matrix = cosine_similarity(embeddings)

    # ── Intra-group ──
    intra_scores = {}
    intra_values = []
    for group, char_list in groups.items():
        sims = []
        for a, b in itertools.combinations(char_list, 2):
            i, j = chars.index(a), chars.index(b)
            s = sim_matrix[i][j]
            sims.append(s)
            intra_values.append(s)
        intra_scores[group] = np.mean(sims)

    avg_intra = np.mean(list(intra_scores.values()))

    print(f"\nINTRA-GROUP (same radical):")
    for g, v in intra_scores.items():
        print(f"  {g:>8s}: {v:.4f}")
    print(f"  >>> AVG INTRA = {avg_intra:.4f}")

    # ── Inter-group ──
    inter_values = []
    group_names = list(groups.keys())
    for i in range(len(group_names)):
        for j in range(i+1, len(group_names)):
            g1 = groups[group_names[i]]
            g2 = groups[group_names[j]]
            for c1 in g1:
                for c2 in g2:
                    i1, i2 = chars.index(c1), chars.index(c2)
                    inter_values.append(sim_matrix[i1][i2])

    avg_inter = np.mean(inter_values)
    print(f"\nINTER-GROUP (diff radical):")
    print(f"  >>> AVG INTER = {avg_inter:.4f}")

    # ── Control ──
    control_scores = {}
    for group, char_list in control_groups.items():
        sims = []
        for a, b in itertools.combinations(char_list, 2):
            i, j = chars.index(a), chars.index(b)
            sims.append(sim_matrix[i][j])
        control_scores[group] = np.mean(sims)

    avg_control = np.mean(list(control_scores.values()))
    print(f"\nCONTROL (same semantic, diff radical):")
    for g, v in control_scores.items():
        print(f"  {g}: {v:.4f}")
    print(f"  >>> AVG CONTROL = {avg_control:.4f}")

    # ── Critical comparison ──
    print(f"\nCRITICAL COMPARISON:")
    print(f"  Radical water (氵) = {intra_scores['water']:.4f}  |  Control water = {control_scores['water_semantic_no_radical']:.4f}")
    print(f"  Radical animal (犭) = {intra_scores['animal']:.4f}  |  Control animal = {control_scores['animal_semantic_no_radical']:.4f}")

    # ── t-test ──
    t_stat, p_value = ttest_ind(intra_values, inter_values)
    d = cohens_d(intra_values, inter_values)

    print(f"\n{'='*50}")
    print(f"  STATISTICS")
    print(f"{'='*50}")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  p-value     = {p_value:.6f}")
    print(f"  Cohen's d   = {d:.4f}")
    print(f"  n_intra     = {len(intra_values)}")
    print(f"  n_inter     = {len(inter_values)}")

    return {
        "label": label,
        "avg_intra": avg_intra,
        "avg_inter": avg_inter,
        "avg_control": avg_control,
        "p_value": p_value,
        "cohens_d": d,
        "intra_scores": intra_scores,
        "control_scores": control_scores,
    }

# ══════════════════════════════════════════════
# Run both models
# ══════════════════════════════════════════════
print(f"Total characters: {len(chars)}")
print("Characters:", chars)

results = []
models = [
    ("bert-base-multilingual-cased", "mBERT"),
    ("hfl/chinese-bert-wwm-ext",     "Chinese-BERT"),
]

for model_name, label in models:
    r = run_analysis(model_name, label)
    results.append(r)

# ══════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
for r in results:
    print(f"\n  {r['label']}:")
    print(f"    avg intra = {r['avg_intra']:.4f}")
    print(f"    avg inter = {r['avg_inter']:.4f}")
    print(f"    p = {r['p_value']:.6f}")
    print(f"    d = {r['cohens_d']:.4f}")

# ══════════════════════════════════════════════
# Figure — Comparative bar chart
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for idx, r in enumerate(results):
    ax = axes[idx]
    labels = ["Intra-radical", "Inter-radical", "Control\n(semantic)"]
    values = [r["avg_intra"], r["avg_inter"], r["avg_control"]]
    colors = ["#2196F3", "#9E9E9E", "#FF9800"]

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title(f"{r['label']}\np={r['p_value']:.4f}, d={r['cohens_d']:.2f}", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Average Cosine Similarity", fontsize=12)

fig.suptitle("Radical Compositionality in Embedding Space", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/intra_vs_inter_similarity.png", dpi=150, bbox_inches="tight")
print("\n[Figure saved to figures/intra_vs_inter_similarity.png]")
