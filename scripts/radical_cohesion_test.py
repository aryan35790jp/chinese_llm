"""
Radical Cohesion Test — Canonical Experiment
For each radical: avg similarity within radical vs similarity to random characters.
Across thousands of pairs, with t-test + Cohen's d.
Bootstrap CI, permutation test, radical-size bias check.
Run on mBERT and Chinese-BERT.
"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import ttest_ind, spearmanr, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import itertools
import os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ═══════════════════════════════════════════
# Load dataset
# ═══════════════════════════════════════════
df = pd.read_csv("data/radical_dataset.csv")
chars = df["char"].tolist()
radicals = df["radical"].tolist()
char_to_radical = dict(zip(chars, radicals))

print(f"Dataset: {len(chars)} characters, {len(set(radicals))} radicals")

# ═══════════════════════════════════════════
# Group characters by radical
# Keep only radicals with >= 20 chars (already filtered, but double check)
# ═══════════════════════════════════════════
radical_groups = {}
for char, rad in char_to_radical.items():
    radical_groups.setdefault(rad, []).append(char)

radical_groups = {r: cs for r, cs in radical_groups.items() if len(cs) >= 20}
print(f"Radicals for analysis: {len(radical_groups)}")

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    mean_diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return mean_diff / pooled_std

def run_cohesion_test(model_name, label):
    print(f"\n{'#'*60}")
    print(f"  {label}  ({model_name})")
    print(f"{'#'*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # ── Extract all embeddings ──
    print("Extracting embeddings...")
    all_embeddings = {}
    batch_size = 64
    for i in tqdm(range(0, len(chars), batch_size)):
        batch_chars = chars[i:i+batch_size]
        inputs = tokenizer(batch_chars, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        embs = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        for j, c in enumerate(batch_chars):
            all_embeddings[c] = embs[j].numpy()

    emb_matrix = np.stack([all_embeddings[c] for c in chars])
    sim_matrix = cosine_similarity(emb_matrix)
    dist_matrix = euclidean_distances(emb_matrix)
    euclid_sim_matrix = 1.0 / (1.0 + dist_matrix)
    char_idx = {c: i for i, c in enumerate(chars)}

    # ── Compute intra-radical and inter-radical similarities ──
    print("Computing similarities...")
    intra_sims = []
    inter_sims = []
    euclid_intra_sims = []
    euclid_inter_sims = []
    per_radical_cohesion = {}  # radical → mean intra similarity
    per_radical_cohesion_euclid = {}

    MAX_PAIRS_PER_RADICAL = 50
    for rad, rad_chars in radical_groups.items():
        pairs = list(itertools.combinations(rad_chars, 2))
        if len(pairs) > MAX_PAIRS_PER_RADICAL:
            sampled = random.sample(pairs, MAX_PAIRS_PER_RADICAL)
        else:
            sampled = pairs
        rad_sims = []
        rad_sims_e = []
        for a, b in sampled:
            s = sim_matrix[char_idx[a]][char_idx[b]]
            s_e = euclid_sim_matrix[char_idx[a]][char_idx[b]]
            intra_sims.append(s)
            euclid_intra_sims.append(s_e)
            rad_sims.append(s)
            rad_sims_e.append(s_e)
        per_radical_cohesion[rad] = np.mean(rad_sims)
        per_radical_cohesion_euclid[rad] = np.mean(rad_sims_e)

    for rad, rad_chars in radical_groups.items():
        other_chars = [c for c in chars if char_to_radical[c] != rad]
        for _ in range(MAX_PAIRS_PER_RADICAL):
            a = random.choice(rad_chars)
            b = random.choice(other_chars)
            inter_sims.append(sim_matrix[char_idx[a]][char_idx[b]])
            euclid_inter_sims.append(euclid_sim_matrix[char_idx[a]][char_idx[b]])

    intra_sims = np.array(intra_sims)
    inter_sims = np.array(inter_sims)
    euclid_intra_sims = np.array(euclid_intra_sims)
    euclid_inter_sims = np.array(euclid_inter_sims)

    # ── Cosine statistics ──
    t_stat, p_value = ttest_ind(intra_sims, inter_sims, equal_var=False)
    d = cohens_d(intra_sims, inter_sims)

    # ── Euclidean statistics ──
    t_stat_e, p_value_e = ttest_ind(euclid_intra_sims, euclid_inter_sims, equal_var=False)
    d_e = cohens_d(euclid_intra_sims, euclid_inter_sims)

    print(f"\n{'='*50}")
    print(f"  RESULTS: {label}")
    print(f"{'='*50}")
    print(f"\n  --- Cosine Similarity ---")
    print(f"  Intra-radical similarity: {np.mean(intra_sims):.4f} (std={np.std(intra_sims):.4f}, n={len(intra_sims)})")
    print(f"  Inter-radical similarity: {np.mean(inter_sims):.4f} (std={np.std(inter_sims):.4f}, n={len(inter_sims)})")
    print(f"  Difference:              {np.mean(intra_sims) - np.mean(inter_sims):.4f}")
    print(f"  t-statistic:             {t_stat:.4f}")
    print(f"  p-value:                 {p_value:.2e}")
    print(f"  Cohen's d:               {d:.4f}")
    print(f"\n  --- Euclidean Similarity [1/(1+d)] ---")
    print(f"  Intra-radical similarity: {np.mean(euclid_intra_sims):.4f} (std={np.std(euclid_intra_sims):.4f}, n={len(euclid_intra_sims)})")
    print(f"  Inter-radical similarity: {np.mean(euclid_inter_sims):.4f} (std={np.std(euclid_inter_sims):.4f}, n={len(euclid_inter_sims)})")
    print(f"  Difference:              {np.mean(euclid_intra_sims) - np.mean(euclid_inter_sims):.4f}")
    print(f"  t-statistic:             {t_stat_e:.4f}")
    print(f"  p-value:                 {p_value_e:.2e}")
    print(f"  Cohen's d:               {d_e:.4f}")

    # ═══════════════════════════════════════════
    # ROBUSTNESS #1 — Bootstrap 95% CI (Cosine)
    # ═══════════════════════════════════════════
    print(f"\n  Bootstrap CI — Cosine (1000 iterations)...")
    n_boot = 1000
    boot_diffs = []
    for _ in range(n_boot):
        intra_sample = np.random.choice(intra_sims, size=len(intra_sims), replace=True)
        inter_sample = np.random.choice(inter_sims, size=len(inter_sims), replace=True)
        boot_diffs.append(np.mean(intra_sample) - np.mean(inter_sample))
    boot_diffs = np.array(boot_diffs)
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    print(f"  Mean diff = {np.mean(boot_diffs):.4f}")
    print(f"  95% CI    = [{ci_low:.4f}, {ci_high:.4f}]")

    # ═══════════════════════════════════════════
    # ROBUSTNESS #1b — Bootstrap 95% CI (Euclidean)
    # ═══════════════════════════════════════════
    print(f"\n  Bootstrap CI — Euclidean (1000 iterations)...")
    boot_diffs_e = []
    for _ in range(n_boot):
        intra_sample_e = np.random.choice(euclid_intra_sims, size=len(euclid_intra_sims), replace=True)
        inter_sample_e = np.random.choice(euclid_inter_sims, size=len(euclid_inter_sims), replace=True)
        boot_diffs_e.append(np.mean(intra_sample_e) - np.mean(inter_sample_e))
    boot_diffs_e = np.array(boot_diffs_e)
    ci_low_e, ci_high_e = np.percentile(boot_diffs_e, [2.5, 97.5])
    print(f"  Mean diff = {np.mean(boot_diffs_e):.4f}")
    print(f"  95% CI    = [{ci_low_e:.4f}, {ci_high_e:.4f}]")

    # ═══════════════════════════════════════════
    # ROBUSTNESS #2 — Permutation Test
    # ═══════════════════════════════════════════
    print(f"\n  Permutation test (1000 iterations)...")
    observed_diff = np.mean(intra_sims) - np.mean(inter_sims)
    all_sims = np.concatenate([intra_sims, inter_sims])
    n_intra = len(intra_sims)
    n_perm = 1000
    perm_diffs = []
    for _ in range(n_perm):
        np.random.shuffle(all_sims)
        perm_diff = np.mean(all_sims[:n_intra]) - np.mean(all_sims[n_intra:])
        perm_diffs.append(perm_diff)
    perm_diffs = np.array(perm_diffs)
    perm_p = (np.sum(perm_diffs >= observed_diff) + 1) / (n_perm + 1)  # corrected estimator
    print(f"  Observed diff  = {observed_diff:.4f}")
    print(f"  Permutation p  = {perm_p:.4f}")
    print(f"  (proportion of shuffled diffs >= observed, with continuity correction)")
    # Restore original order after shuffle
    all_sims_orig = np.concatenate([intra_sims, inter_sims])

    # ═══════════════════════════════════════════
    # ROBUSTNESS #2b — Permutation Test (Euclidean)
    # ═══════════════════════════════════════════
    print(f"\n  Permutation test — Euclidean (1000 iterations)...")
    observed_diff_e = np.mean(euclid_intra_sims) - np.mean(euclid_inter_sims)
    all_sims_e = np.concatenate([euclid_intra_sims, euclid_inter_sims])
    n_intra_e = len(euclid_intra_sims)
    perm_diffs_e = []
    for _ in range(n_perm):
        np.random.shuffle(all_sims_e)
        perm_diffs_e.append(np.mean(all_sims_e[:n_intra_e]) - np.mean(all_sims_e[n_intra_e:]))
    perm_diffs_e = np.array(perm_diffs_e)
    perm_p_e = (np.sum(perm_diffs_e >= observed_diff_e) + 1) / (n_perm + 1)
    print(f"  Observed diff  = {observed_diff_e:.4f}")
    print(f"  Permutation p  = {perm_p_e:.4f}")

    # ═══════════════════════════════════════════
    # ROBUSTNESS #3 — Radical Size Bias Check
    # ═══════════════════════════════════════════
    rad_sizes = []
    rad_cohesions = []
    for rad in radical_groups:
        rad_sizes.append(len(radical_groups[rad]))
        rad_cohesions.append(per_radical_cohesion[rad])
    rho, rho_p = spearmanr(rad_sizes, rad_cohesions)
    print(f"\n  Radical size bias check:")
    print(f"  Spearman rho   = {rho:.4f}")
    print(f"  p-value        = {rho_p:.4f}")
    if abs(rho) < 0.3:
        print(f"  → Weak/no correlation. No size bias detected.")
    else:
        print(f"  → Moderate correlation. Report in paper.")

    # ═══════════════════════════════════════════
    # ROBUSTNESS #4 — Frequency Confound Check
    # Use tokenizer vocab rank as character frequency proxy
    # ═══════════════════════════════════════════
    print(f"\n  Frequency confound check (vocab rank proxy)...")
    vocab = tokenizer.get_vocab()
    char_ranks = []
    char_cohesions_freq = []
    for rad, rad_chars in radical_groups.items():
        avg_rank = np.mean([vocab.get(c, len(vocab)) for c in rad_chars])
        char_ranks.append(avg_rank)
        char_cohesions_freq.append(per_radical_cohesion[rad])
    freq_rho, freq_rho_p = spearmanr(char_ranks, char_cohesions_freq)
    print(f"  Spearman rho (vocab rank vs cohesion) = {freq_rho:.4f}")
    print(f"  p-value = {freq_rho_p:.4f}")
    if abs(freq_rho) < 0.3:
        print(f"  → Weak/no frequency bias detected.")
    else:
        print(f"  → Moderate frequency correlation. Report in paper.")

    return {
        "label": label,
        "model_name": model_name,
        # Cosine
        "intra_mean": np.mean(intra_sims),
        "inter_mean": np.mean(inter_sims),
        "intra_std": np.std(intra_sims),
        "inter_std": np.std(inter_sims),
        "intra_sims": intra_sims,
        "inter_sims": inter_sims,
        "p_value": p_value,
        "cohens_d": d,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "boot_diffs": boot_diffs,
        # Euclidean
        "euclid_intra_mean": np.mean(euclid_intra_sims),
        "euclid_inter_mean": np.mean(euclid_inter_sims),
        "euclid_intra_std": np.std(euclid_intra_sims),
        "euclid_inter_std": np.std(euclid_inter_sims),
        "euclid_intra_sims": euclid_intra_sims,
        "euclid_inter_sims": euclid_inter_sims,
        "euclid_p_value": p_value_e,
        "euclid_cohens_d": d_e,
        "euclid_ci_low": ci_low_e,
        "euclid_ci_high": ci_high_e,
        "euclid_boot_diffs": boot_diffs_e,
        # Shared
        "perm_p": perm_p,
        "perm_diffs": perm_diffs,
        "observed_diff": observed_diff,
        "perm_p_euclid": perm_p_e,
        "perm_diffs_euclid": perm_diffs_e,
        "observed_diff_euclid": observed_diff_e,
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "freq_rho": freq_rho,
        "freq_rho_p": freq_rho_p,
        "per_radical_cohesion": per_radical_cohesion,
        "sim_matrix": sim_matrix,
        "rad_sizes": rad_sizes,
        "rad_cohesions": rad_cohesions,
    }

# ═══════════════════════════════════════════
# Run both models
# ═══════════════════════════════════════════
results = []
for model_name, label in [
    ("bert-base-multilingual-cased", "mBERT"),
    ("hfl/chinese-bert-wwm-ext",     "Chinese-BERT"),
]:
    results.append(run_cohesion_test(model_name, label))

# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print("  FINAL SUMMARY")
print(f"{'='*60}")
for r in results:
    print(f"\n  {r['label']}:")
    print(f"    --- Cosine ---")
    print(f"    Intra-radical  = {r['intra_mean']:.4f}")
    print(f"    Inter-radical  = {r['inter_mean']:.4f}")
    print(f"    p              = {r['p_value']:.2e}")
    print(f"    d              = {r['cohens_d']:.4f}")
    print(f"    95% CI         = [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    print(f"    --- Euclidean [1/(1+d)] ---")
    print(f"    Intra-radical  = {r['euclid_intra_mean']:.4f}")
    print(f"    Inter-radical  = {r['euclid_inter_mean']:.4f}")
    print(f"    p              = {r['euclid_p_value']:.2e}")
    print(f"    d              = {r['euclid_cohens_d']:.4f}")
    print(f"    95% CI         = [{r['euclid_ci_low']:.4f}, {r['euclid_ci_high']:.4f}]")
    print(f"    --- Shared ---")
    print(f"    Perm p (cos)   = {r['perm_p']:.4f}")
    print(f"    Perm p (euc)   = {r['perm_p_euclid']:.4f}")
    print(f"    Size bias rho  = {r['spearman_rho']:.4f} (p={r['spearman_p']:.4f})")
    print(f"    Freq bias rho  = {r['freq_rho']:.4f} (p={r['freq_rho_p']:.4f})")

# ═══════════════════════════════════════════
# Save raw data artifacts
# ═══════════════════════════════════════════
os.makedirs("results", exist_ok=True)
for r in results:
    tag = r["label"].lower().replace("-", "_")
    # results/ (canonical)
    np.save(f"results/{tag}_similarity_matrix.npy", r["sim_matrix"])
    np.save(f"results/{tag}_intra_pairs.npy", r["intra_sims"])
    np.save(f"results/{tag}_inter_pairs.npy", r["inter_sims"])
    np.save(f"results/{tag}_permutation_scores.npy", r["perm_diffs"])
    np.save(f"results/{tag}_euclid_permutation_scores.npy", r["perm_diffs_euclid"])
    np.save(f"results/{tag}_bootstrap.npy", r["boot_diffs"])
    np.save(f"results/{tag}_euclid_intra_pairs.npy", r["euclid_intra_sims"])
    np.save(f"results/{tag}_euclid_inter_pairs.npy", r["euclid_inter_sims"])
    np.save(f"results/{tag}_euclid_bootstrap.npy", r["euclid_boot_diffs"])
    np.save(f"results/{tag}_rad_sizes.npy", np.array(r["rad_sizes"]))
    np.save(f"results/{tag}_rad_cohesions.npy", np.array(r["rad_cohesions"]))

# Save summary table with Holm-Bonferroni correction
all_p_values = []
for r in results:
    all_p_values.extend([r["p_value"], r["euclid_p_value"]])

# Holm-Bonferroni correction
sorted_indices = np.argsort(all_p_values)
holm_adjusted = np.zeros(len(all_p_values))
for rank, idx in enumerate(sorted_indices):
    holm_adjusted[idx] = min(all_p_values[idx] * (len(all_p_values) - rank), 1.0)
# Ensure monotonicity
for rank in range(1, len(sorted_indices)):
    idx = sorted_indices[rank]
    prev_idx = sorted_indices[rank - 1]
    holm_adjusted[idx] = max(holm_adjusted[idx], holm_adjusted[prev_idx])

summary_df = pd.DataFrame([{
    "Model": r["label"],
    "Cos_Intra": round(r["intra_mean"], 4),
    "Cos_Inter": round(r["inter_mean"], 4),
    "Cos_Diff": round(r["intra_mean"] - r["inter_mean"], 4),
    "Cos_p": r["p_value"],
    "Cos_p_holm": holm_adjusted[i*2],
    "Cos_d": round(r["cohens_d"], 4),
    "Cos_CI_lo": round(r["ci_low"], 4),
    "Cos_CI_hi": round(r["ci_high"], 4),
    "Euc_Intra": round(r["euclid_intra_mean"], 4),
    "Euc_Inter": round(r["euclid_inter_mean"], 4),
    "Euc_Diff": round(r["euclid_intra_mean"] - r["euclid_inter_mean"], 4),
    "Euc_p": r["euclid_p_value"],
    "Euc_p_holm": holm_adjusted[i*2 + 1],
    "Euc_d": round(r["euclid_cohens_d"], 4),
    "Euc_CI_lo": round(r["euclid_ci_low"], 4),
    "Euc_CI_hi": round(r["euclid_ci_high"], 4),
    "Perm_p_cos": r["perm_p"],
    "Perm_p_euc": r["perm_p_euclid"],
    "Size_rho": round(r["spearman_rho"], 4),
    "Size_p": round(r["spearman_p"], 4),
    "Freq_rho": round(r["freq_rho"], 4),
    "Freq_p": round(r["freq_rho_p"], 4),
} for i, r in enumerate(results)])
summary_df.to_csv("results/main_results.csv", index=False)

print("\n[Raw data saved to results/]")
print("[Table saved to results/main_results.csv]")


# ═══════════════════════════════════════════
# FIGURE 1 — Combined density plot (cosine + euclidean)
# ═══════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, r in enumerate(results):
    # Top row: Cosine
    ax = axes[0][idx]
    ax.hist(r["intra_sims"], bins=60, alpha=0.6, color="#2196F3",
            label=f"Same radical (μ={r['intra_mean']:.3f})", density=True)
    ax.hist(r["inter_sims"], bins=60, alpha=0.6, color="#E91E63",
            label=f"Diff radical (μ={r['inter_mean']:.3f})", density=True)
    ax.set_xlabel("Cosine Similarity", fontsize=10)
    ax.set_title(f"{r['label']} — Cosine\np={r['p_value']:.2e}, d={r['cohens_d']:.2f}, "
                 f"CI=[{r['ci_low']:.3f},{r['ci_high']:.3f}]", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom row: Euclidean
    ax = axes[1][idx]
    ax.hist(r["euclid_intra_sims"], bins=60, alpha=0.6, color="#4CAF50",
            label=f"Same radical (μ={r['euclid_intra_mean']:.3f})", density=True)
    ax.hist(r["euclid_inter_sims"], bins=60, alpha=0.6, color="#FF9800",
            label=f"Diff radical (μ={r['euclid_inter_mean']:.3f})", density=True)
    ax.set_xlabel("Euclidean Similarity [1/(1+d)]", fontsize=10)
    ax.set_title(f"{r['label']} — Euclidean\np={r['euclid_p_value']:.2e}, d={r['euclid_cohens_d']:.2f}, "
                 f"CI=[{r['euclid_ci_low']:.3f},{r['euclid_ci_high']:.3f}]", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0][0].set_ylabel("Density", fontsize=10)
axes[1][0].set_ylabel("Density", fontsize=10)
fig.suptitle("Radical Cohesion in Embedding Space\n(6,306 characters, 68 radicals — Cosine vs Euclidean)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/radical_cohesion_density.png", dpi=200, bbox_inches="tight")
print("[Figure saved to figures/radical_cohesion_density.png]")

# ═══════════════════════════════════════════
# FIGURE 2 — Bar chart with CI error bars (cosine)
# ═══════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(8, 5))
x = np.arange(len(results))
width = 0.3
intra_vals = [r["intra_mean"] for r in results]
inter_vals = [r["inter_mean"] for r in results]
bars1 = ax2.bar(x - width/2, intra_vals, width, label="Intra-radical",
                color="#2196F3", edgecolor="black")
bars2 = ax2.bar(x + width/2, inter_vals, width, label="Inter-radical",
                color="#E91E63", edgecolor="black")
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_ylabel("Average Cosine Similarity", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels([r["label"] for r in results], fontsize=12)
ax2.legend(fontsize=11)
ax2.set_ylim(0, 1.0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_title("Radical Cohesion Test (corpus-backed)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/radical_cohesion_bars.png", dpi=200, bbox_inches="tight")
print("[Figure saved to figures/radical_cohesion_bars.png]")

# ═══════════════════════════════════════════
# FIGURE 3 — Effect size comparison (Cosine vs Euclidean)
# ═══════════════════════════════════════════
fig_es, ax_es = plt.subplots(figsize=(8, 5))
models = [r["label"] for r in results]
cos_d = [r["cohens_d"] for r in results]
euc_d = [r["euclid_cohens_d"] for r in results]
cos_ci = [(r["ci_low"], r["ci_high"]) for r in results]
euc_ci = [(r["euclid_ci_low"], r["euclid_ci_high"]) for r in results]

x_es = np.arange(len(results))
w = 0.3
# Cosine bars
cos_lo = [r["cohens_d"] - 0 for r in results]  # Cohen's d point estimate
bars_cos = ax_es.bar(x_es - w/2, cos_d, w, label="Cosine d",
                     color="#2196F3", edgecolor="black")
# Euclidean bars
bars_euc = ax_es.bar(x_es + w/2, euc_d, w, label="Euclidean d",
                     color="#4CAF50", edgecolor="black")
for i, bar in enumerate(bars_cos):
    ax_es.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
               f"{cos_d[i]:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
for i, bar in enumerate(bars_euc):
    ax_es.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
               f"{euc_d[i]:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_es.set_ylabel("Cohen's d (effect size)", fontsize=12)
ax_es.set_xticks(x_es)
ax_es.set_xticklabels(models, fontsize=12)
ax_es.legend(fontsize=11)
ax_es.spines["top"].set_visible(False)
ax_es.spines["right"].set_visible(False)
ax_es.set_title("Effect Size: Cosine vs Euclidean Metric", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/effect_size_comparison.png", dpi=200, bbox_inches="tight")
print("[Figure saved to figures/effect_size_comparison.png]")

# ═══════════════════════════════════════════
# FIGURE 3 — Permutation test distribution
# ═══════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
for idx, r in enumerate(results):
    ax = axes3[idx]
    ax.hist(r["perm_diffs"], bins=50, alpha=0.7, color="#9E9E9E",
            label="Shuffled diffs", density=True)
    ax.axvline(r["observed_diff"], color="#E91E63", linewidth=2, linestyle="--",
               label=f"Observed ({r['observed_diff']:.4f})")
    ax.set_xlabel("Mean Difference (intra − inter)", fontsize=11)
    ax.set_title(f"{r['label']} — Permutation Test\nperm p = {r['perm_p']:.4f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes3[0].set_ylabel("Density", fontsize=11)
fig3.suptitle("Permutation Test: Is Radical Cohesion Real?",
              fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/permutation_test.png", dpi=200, bbox_inches="tight")
print("[Figure saved to figures/permutation_test.png]")

# ═══════════════════════════════════════════
# FIGURE 4 — Radical size vs cohesion (bias check)
# ═══════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
for idx, r in enumerate(results):
    ax = axes4[idx]
    ax.scatter(r["rad_sizes"], r["rad_cohesions"], alpha=0.6, color="#2196F3", edgecolor="black", s=40)
    ax.set_xlabel("Radical Group Size (# characters)", fontsize=11)
    ax.set_ylabel("Mean Intra-radical Similarity", fontsize=11)
    ax.set_title(f"{r['label']}\nSpearman ρ = {r['spearman_rho']:.3f} (p={r['spearman_p']:.3f})", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig4.suptitle("Radical Size Bias Check", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/radical_size_bias.png", dpi=200, bbox_inches="tight")
print("[Figure saved to figures/radical_size_bias.png]")

# ═══════════════════════════════════════════
# FIGURE 5 — Bootstrap distribution
# ═══════════════════════════════════════════
fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
for idx, r in enumerate(results):
    # Top row: Cosine bootstrap
    ax = axes5[0][idx]
    ax.hist(r["boot_diffs"], bins=50, alpha=0.7, color="#2196F3", density=True)
    ax.axvline(0, color="black", linewidth=1, linestyle=":")
    ax.axvline(r["ci_low"], color="#E91E63", linewidth=1.5, linestyle="--",
               label=f"95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    ax.axvline(r["ci_high"], color="#E91E63", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Bootstrap Δ (intra − inter)", fontsize=10)
    ax.set_title(f"{r['label']} — Cosine Bootstrap", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom row: Euclidean bootstrap
    ax = axes5[1][idx]
    ax.hist(r["euclid_boot_diffs"], bins=50, alpha=0.7, color="#4CAF50", density=True)
    ax.axvline(0, color="black", linewidth=1, linestyle=":")
    ax.axvline(r["euclid_ci_low"], color="#FF9800", linewidth=1.5, linestyle="--",
               label=f"95% CI [{r['euclid_ci_low']:.4f}, {r['euclid_ci_high']:.4f}]")
    ax.axvline(r["euclid_ci_high"], color="#FF9800", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Bootstrap Δ (intra − inter)", fontsize=10)
    ax.set_title(f"{r['label']} — Euclidean Bootstrap", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes5[0][0].set_ylabel("Density", fontsize=10)
axes5[1][0].set_ylabel("Density", fontsize=10)
fig5.suptitle("Bootstrap Distributions of Radical Cohesion Difference",
              fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/bootstrap_distribution.png", dpi=200, bbox_inches="tight")
print("[Figure saved to figures/bootstrap_distribution.png]")

print(f"\n{'='*60}")
print("  ALL DONE — artifacts saved, figures generated.")
print(f"{'='*60}")
