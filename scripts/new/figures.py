"""
figures.py — every paper figure, in one place.

Reads the CSVs and .npy artifacts produced by the rest of the pipeline
and writes the figures into figures/ at 300 DPI.

Figures produced (each as PNG and PDF):

    fig_layer_wise_d.png/pdf
        x = layer, y = Cohen's d.  one line per model.
        char-pool, isotropy-corrected.
        This is the centerpiece result.

    fig_layer_wise_rsa.png/pdf
        same but y = RSA Spearman ρ.

    fig_per_radical_box.png/pdf
        boxplot of per-radical intra cosine, sorted by median.
        one panel per model.

    fig_semantic_control_forest.png/pdf
        forest plot of per-field Cohen's d (semantic control).

    fig_phonetic_vs_semantic.png/pdf
        bars: d for semantic vs identity vs unknown role.

    fig_cross_script_japanese.png/pdf
        bars: d for Chinese models on Japanese kanji subset, vs Japanese
        models on the same subset.

    fig_glyph_comparison.png/pdf
        side-by-side: d_corpus vs d_semantic_ctrl, colored by model_class.

    fig_scaling.png/pdf
        scatter: params (M) on log axis vs Cohen's d.

    fig_downstream_correlation.png/pdf
        scatter: model_cos vs human_score per word pair, plus per-radical means.

    fig_variance_decomposition.png/pdf
        stacked horizontal bars: partial R² of each predictor per model.
        This is the "money figure" that lets reviewers see at a glance
        how much of the radical effect is semantics, distributional,
        frequency, form.

    fig_orthographic_arithmetic.png/pdf
        bar plot of mean lift per model.

    fig_pca_umap_projections.png/pdf
        PCA + UMAP scatter of last-layer embeddings, colored by top-12
        radicals.

Some figures will be skipped silently if the upstream CSV doesn't exist.
That's intentional — the script always produces what it can.

Runtime: ~10 min total. CPU only.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    RESULTS_DIR,
    set_seed,
    setup_publication_style,
    save_fig,
    load_radical_dataset,
    list_available_models,
    list_available_layers,
)
from radical_lib.embeddings import model_tag  # noqa: E402
from radical_lib.core import CACHE_DIR  # noqa: E402

set_seed()
setup_publication_style()
ISO_DIR = CACHE_DIR / "embeddings_iso"


def maybe_read(name: str) -> Optional[pd.DataFrame]:
    p = RESULTS_DIR / name
    if not p.exists():
        print(f"[skip] {name} not found")
        return None
    return pd.read_csv(p)


# ── 1. centerpiece: layer-wise d ────────────────────────────────────────────
def fig_layer_wise_d():
    import matplotlib.pyplot as plt
    df = maybe_read("layer_wise.csv")
    if df is None or df.empty:
        return
    df = df[(df["pool"] == "char") & (df["iso"] == 1)].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model_id, g in df.groupby("model"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["cohens_d"], marker="o", label=model_id, linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Hidden layer")
    ax.set_ylabel("Cohen's d  (intra − inter cosine)")
    ax.set_title("Radical-aligned cohesion across layers (char-pool, isotropy-corrected)")
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.85)
    save_fig(fig, "fig_layer_wise_d")


def fig_layer_wise_rsa():
    import matplotlib.pyplot as plt
    df = maybe_read("layer_wise.csv")
    if df is None or df.empty:
        return
    df = df[(df["pool"] == "char") & (df["iso"] == 1)].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model_id, g in df.groupby("model"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["rsa_rho"], marker="s", label=model_id, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Hidden layer")
    ax.set_ylabel("RSA Spearman ρ  (embedding ~ same-radical)")
    ax.set_title("Layer-wise representational alignment with the same-radical RDM")
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.85)
    save_fig(fig, "fig_layer_wise_rsa")


# ── 2. semantic control forest plot ─────────────────────────────────────────
def fig_semantic_control_forest():
    import matplotlib.pyplot as plt
    df = maybe_read("expanded_semantic_control.csv")
    if df is None or df.empty:
        return

    models = sorted(df["model"].unique())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(3.0 * n_models, 6.5),
                             sharey=True, squeeze=False)
    for ax, model_id in zip(axes[0], models):
        sub = df[df["model"] == model_id].sort_values("cohens_d")
        y = np.arange(len(sub))
        ax.errorbar(
            sub["cohens_d"], y,
            xerr=0,  # CI not propagated in current CSV; see fig later
            fmt="o", color="#2563eb", markersize=4, linewidth=1.0,
        )
        ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["field"].tolist(), fontsize=7)
        ax.set_xlabel("Cohen's d (within-field)")
        ax.set_title(model_id, fontsize=8)
    axes[0][0].set_ylabel("semantic field")
    fig.suptitle("Semantic control: same-radical vs cross-radical d, per field")
    save_fig(fig, "fig_semantic_control_forest")


# ── 3. phonetic vs semantic radicals ────────────────────────────────────────
def fig_phonetic_vs_semantic():
    import matplotlib.pyplot as plt
    df = maybe_read("phonetic_vs_semantic_radicals.csv")
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.0))
    pivot = df.pivot_table(index="role", columns="model", values="cohens_d", aggfunc="mean")
    pivot = pivot.reindex(["semantic", "identity", "unknown", "non_identity", "all"]).dropna(how="all")
    pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Radical role")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Cohesion by radical role (semantic / identity / unknown)")
    ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.85)
    save_fig(fig, "fig_phonetic_vs_semantic")


# ── 4. cross-script japanese ────────────────────────────────────────────────
def fig_cross_script_japanese():
    import matplotlib.pyplot as plt
    df = maybe_read("cross_script_japanese.csv")
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.0))
    df_sorted = df.sort_values(["language", "cohens_d"])
    colors = {"ja": "#dc2626", "zh-on-kanji-subset": "#2563eb"}
    cols = [colors.get(lang, "#888") for lang in df_sorted["language"]]
    ax.barh(df_sorted["model"], df_sorted["cohens_d"], color=cols, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Cohen's d  (intra − inter)")
    ax.set_title("Cross-script: Chinese vs Japanese models on the kanji subset")
    save_fig(fig, "fig_cross_script_japanese")


# ── 5. glyph comparison ─────────────────────────────────────────────────────
def fig_glyph_comparison():
    import matplotlib.pyplot as plt
    df = maybe_read("glyph_comparison.csv")
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.0))
    width = 0.4
    df = df.sort_values(["model_class", "d_corpus"])
    x = np.arange(len(df))
    ax.bar(x - width / 2, df["d_corpus"], width, label="d_corpus",
           color="#2563eb", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, df["d_semantic_ctrl"], width, label="d_semantic_ctrl",
           color="#16a34a", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_id"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Cohen's d")
    ax.set_title("Form-specific component: d_corpus − d_semantic_ctrl")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.legend()
    save_fig(fig, "fig_glyph_comparison")


# ── 6. scaling ──────────────────────────────────────────────────────────────
def fig_scaling():
    import matplotlib.pyplot as plt
    df = maybe_read("scaling.csv")
    if df is None or df.empty:
        return
    df = df.dropna(subset=["params_M"])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.scatter(df["params_M"], df["last_layer_d"], s=70, color="#2563eb",
               edgecolor="black", linewidth=0.7)
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r["params_M"], r["last_layer_d"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log scale)")
    ax.set_ylabel("Last-layer Cohen's d")
    ax.set_title("Scaling: does radical signal grow with model size?")
    save_fig(fig, "fig_scaling")


# ── 7. downstream correlation ───────────────────────────────────────────────
def fig_downstream_correlation():
    import matplotlib.pyplot as plt
    df = maybe_read("downstream_per_radical.csv")
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for model_id, g in df.groupby("model"):
        ax.scatter(g["mean_human_score"], g["mean_model_cos"],
                   alpha=0.6, label=model_id, s=30)
    ax.set_xlabel("Mean PKU-500 human score (per radical)")
    ax.set_ylabel("Mean model cosine (per radical)")
    ax.set_title("Per-radical alignment with human similarity")
    ax.legend(fontsize=7, framealpha=0.85)
    save_fig(fig, "fig_downstream_correlation")


# ── 8. variance decomposition ───────────────────────────────────────────────
def fig_variance_decomposition():
    import matplotlib.pyplot as plt
    df = maybe_read("variance_decomposition.csv")
    if df is None or df.empty:
        return
    pivot = df.pivot_table(
        index="model", columns="predictor", values="partial_R2", aggfunc="mean"
    ).fillna(0)
    cols_order = ["same_radical", "ppmi", "freq_diff", "stroke_diff"]
    pivot = pivot[[c for c in cols_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(7.5, 0.45 * max(3, len(pivot)) + 1))
    pivot.plot(kind="barh", stacked=True, ax=ax, edgecolor="black", linewidth=0.4,
               color=["#2563eb", "#16a34a", "#f59e0b", "#dc2626"])
    ax.set_xlabel("Partial R² of cosine-similarity regression")
    ax.set_title("Variance decomposition of last-layer pairwise cosine")
    ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, "fig_variance_decomposition")


# ── 9. orthographic arithmetic ──────────────────────────────────────────────
def fig_orthographic_arithmetic():
    import matplotlib.pyplot as plt
    df = maybe_read("orthographic_arithmetic_summary.csv")
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    df = df.sort_values("mean_lift")
    ax.barh(df["model"], df["mean_lift"], color="#2563eb",
            edgecolor="black", linewidth=0.5)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", label="random baseline (lift=1)")
    ax.set_xlabel("Mean retrieval lift (top-10 target rate / chance)")
    ax.set_title("Orthographic arithmetic: E(c) − E(R₁) + E(R₂) → R₂?")
    ax.legend(fontsize=7)
    save_fig(fig, "fig_orthographic_arithmetic")


# ── 10. PCA + UMAP ──────────────────────────────────────────────────────────
def fig_pca_umap_projections():
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    df = load_radical_dataset()
    chars = df["char"].tolist()
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col].astype(int)))

    # Pick top-12 radicals by frequency
    counts = pd.Series([char_to_radical[c] for c in chars]).value_counts()
    top12 = counts.head(12).index.tolist()
    keep_idx = [i for i, c in enumerate(chars) if char_to_radical[c] in top12]
    keep_chars = [chars[i] for i in keep_idx]
    keep_rads = [char_to_radical[c] for c in keep_chars]
    palette = plt.cm.tab20(np.linspace(0, 1, len(top12)))
    rad_to_color = {r: palette[k] for k, r in enumerate(top12)}
    cols = [rad_to_color[r] for r in keep_rads]

    models = list_available_models()
    if not models:
        return

    # We use the *first* model for the visualization; you can manually
    # rerun with another model by editing this.
    target = models[0]
    layers = list_available_layers(target)
    if not layers:
        return
    L = max(layers)
    path = ISO_DIR / model_tag(target) / f"layer{L:02d}_char.npy"
    if not path.exists():
        path = ISO_DIR / model_tag(target) / f"layer{L:02d}_mean.npy"
    if not path.exists():
        return
    X = np.load(path)[keep_idx]

    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X)

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.1)
        Xu = reducer.fit_transform(X)
        n_panels = 2
    except Exception:
        Xu = None
        n_panels = 1

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)
    axes[0][0].scatter(Xp[:, 0], Xp[:, 1], c=cols, s=8, alpha=0.7, edgecolors="none")
    axes[0][0].set_title(f"{target}  PCA (last layer, char-pool)")
    axes[0][0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    axes[0][0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    if Xu is not None:
        axes[0][1].scatter(Xu[:, 0], Xu[:, 1], c=cols, s=8, alpha=0.7, edgecolors="none")
        axes[0][1].set_title(f"{target}  UMAP")
    save_fig(fig, "fig_pca_umap_projections")


def main():
    fig_layer_wise_d()
    fig_layer_wise_rsa()
    fig_semantic_control_forest()
    fig_phonetic_vs_semantic()
    fig_cross_script_japanese()
    fig_glyph_comparison()
    fig_scaling()
    fig_downstream_correlation()
    fig_variance_decomposition()
    fig_orthographic_arithmetic()
    fig_pca_umap_projections()
    print("\nfigures done")


if __name__ == "__main__":
    main()
