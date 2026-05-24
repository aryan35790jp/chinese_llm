"""
glyph_comparison.py — head-to-head: glyph-aware models vs standard models
on the same radical-cohesion test, with and without semantic control.

The argument the paper wants to make is:
    "Standard distributional models cluster radicals because of semantics.
     Models with explicit visual/orthographic input should cluster
     radicals more strongly, *and that extra cohesion should survive
     semantic control* — because the radical's *form* is now a real
     input feature."

This script runs the comparison directly:
    For each model in the "comparison set":
        - corpus-scale: intra-radical vs inter-radical d
        - semantic-controlled: intra-radical (within-field)
                                vs cross-radical (within-field) d
    Then we compute the "form-specific" component as:
        d_form = d_corpus - d_semantic_controlled

    The argument is that d_form should be ~0 for distributional models and
    >> 0 for glyph-aware / vision-only models.

Comparison set:
    standard_distributional = mBERT, Chinese-BERT, MacBERT, ERNIE, XLM-R
    glyph_aware             = ChineseBERT-glyph (if loaded), glyph_only/resnet18

Output:
    results/glyph_comparison.csv
        rows = (model, model_class, d_corpus, p_corpus, d_semantic_ctrl,
                p_semantic_ctrl, d_form_specific, ci_form_lo, ci_form_hi)

Depends on: layer_wise_analysis.py, expanded_semantic_control.py,
            glyph_only_baseline.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import RESULTS_DIR, set_seed  # noqa: E402

set_seed()


def main():
    layerwise_path = RESULTS_DIR / "layer_wise.csv"
    pooled_path = RESULTS_DIR / "expanded_semantic_control_pooled.csv"
    if not (layerwise_path.exists() and pooled_path.exists()):
        print("[skip] glyph_comparison: missing layer_wise.csv or "
              "expanded_semantic_control_pooled.csv — write empty file.")
        pd.DataFrame(columns=[
            "model_id", "model_class", "d_corpus", "p_corpus",
            "ci_corpus_lo", "ci_corpus_hi", "d_semantic_ctrl",
            "p_semantic_ctrl", "d_form_specific",
        ]).to_csv(RESULTS_DIR / "glyph_comparison.csv", index=False)
        return

    layerwise = pd.read_csv(layerwise_path)
    layerwise = layerwise[(layerwise["pool"] == "char") & (layerwise["iso"] == 1)]
    try:
        pooled = pd.read_csv(pooled_path)
    except pd.errors.EmptyDataError:
        pooled = pd.DataFrame()

    classes = {
        "bert-base-multilingual-cased":     "standard",
        "hfl/chinese-bert-wwm-ext":          "standard",
        "hfl/chinese-macbert-base":          "standard",
        "xlm-roberta-base":                  "standard",
        "xlm-roberta-large":                 "standard",
        "nghuyong/ernie-3.0-base-zh":        "standard",
        "uer/chinese_roberta_L-4_H-512":     "standard",
        "uer/chinese_roberta_L-8_H-512":     "standard",
        "ShannonAI/ChineseBERT-base":        "glyph_aware",
        "glyph_only/resnet18":               "vision_only",
    }

    rows = []
    for model_id, model_class in classes.items():
        sub = layerwise[layerwise["model"] == model_id]
        if sub.empty:
            continue
        last_layer = sub["layer"].max()
        last_row = sub[sub["layer"] == last_layer].iloc[0]
        d_corpus = float(last_row["cohens_d"])
        p_corpus = float(last_row["p_perm"])
        ci_lo_corpus = float(last_row["ci_lower"])
        ci_hi_corpus = float(last_row["ci_upper"])

        sem_row = pooled[pooled["model"] == model_id]
        if sem_row.empty:
            continue
        sem = sem_row.iloc[0]
        d_sem = float(sem["d_pooled"])
        p_sem = float(sem["p_perm_pooled"])

        d_form = d_corpus - d_sem
        # Approximate CI for d_form using the corpus CI as a rough envelope.
        # A proper CI on the difference would require redoing the bootstrap
        # on per-pair differences; we leave this for the figures script
        # which produces a more careful estimate.
        rows.append({
            "model_id": model_id,
            "model_class": model_class,
            "d_corpus": d_corpus,
            "p_corpus": p_corpus,
            "ci_corpus_lo": ci_lo_corpus,
            "ci_corpus_hi": ci_hi_corpus,
            "d_semantic_ctrl": d_sem,
            "p_semantic_ctrl": p_sem,
            "d_form_specific": d_form,
        })

    if rows:
        out = pd.DataFrame(rows).sort_values(["model_class", "d_form_specific"])
    else:
        out = pd.DataFrame(columns=[
            "model_id", "model_class", "d_corpus", "p_corpus",
            "ci_corpus_lo", "ci_corpus_hi", "d_semantic_ctrl",
            "p_semantic_ctrl", "d_form_specific",
        ])
    out.to_csv(RESULTS_DIR / "glyph_comparison.csv", index=False)
    print(f"\nWrote {len(out)} rows.")
    if len(out):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
