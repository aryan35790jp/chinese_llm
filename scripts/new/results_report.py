"""
results_report.py — auto-generate the Phase 3 analysis from CSVs.

When the pipeline finishes, this script reads every results/*.csv and
writes a single markdown report (results/_REPORT.md) that contains:

    1. Headline numbers — last-layer Cohen's d per model, isotropy on/off
    2. Centerpiece interpretation — which layers peak; cross-model patterns
    3. Semantic control — pooled effect across 20+ fields
    4. Variance decomposition — partial R² per predictor, ranked
    5. Pseudoradical null — is the effect specific to Kangxi?
    6. Frequency-matched effect — what survives after matching freq?
    7. Random-init noise floor — d on untrained models
    8. Cross-script — Japanese results
    9. Glyph comparison — d_form_specific by model class
    10. Probing — radical-class accuracy vs semantic-field accuracy
    11. Orthographic arithmetic — mean retrieval lift
    12. Anomaly flags — anything that violates expected patterns

Plus a "BLOCKERS" section listing every CSV that didn't get produced
or had unexpected content, so you (the human) know what to re-run.

Usage:
    python scripts/new/results_report.py
    cat results/_REPORT.md
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import RESULTS_DIR  # noqa: E402


def maybe_read(name: str) -> Optional[pd.DataFrame]:
    p = RESULTS_DIR / name
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p)
        return df if not df.empty else None
    except (pd.errors.EmptyDataError, Exception):
        return None


def fmt_num(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{digits}f}"


def section(title: str) -> str:
    return f"\n## {title}\n\n"


def render_table(df: pd.DataFrame, cols: list, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    df = df[cols].copy()
    if len(df) > max_rows:
        df = df.head(max_rows)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(
            fmt_num(row[c]) if isinstance(row[c], (int, float, np.floating)) else str(row[c])
            for c in cols
        ) + " |")
    return "\n".join(out) + "\n"


def headline(out: list, blockers: list):
    df = maybe_read("layer_wise.csv")
    out.append("## 1. Headline numbers (last layer, char-pool, isotropy-corrected)\n")
    if df is None:
        out.append("_layer_wise.csv missing — run `layer_wise_analysis.py`_\n")
        blockers.append("layer_wise.csv")
        return
    iso = df[(df["pool"] == "char") & (df["iso"] == 1)]
    last = (iso.sort_values("layer").groupby("model").tail(1)
                .sort_values("cohens_d", ascending=False))
    cols = ["model", "layer", "cohens_d", "delta", "p_perm", "ci_lower", "ci_upper", "rsa_rho"]
    out.append(render_table(last, cols))


def layerwise_analysis(out: list):
    df = maybe_read("layer_wise.csv")
    if df is None:
        return
    iso = df[(df["pool"] == "char") & (df["iso"] == 1)].copy()
    iso = iso.dropna(subset=["cohens_d"])
    if iso.empty:
        out.append(section("2. Layer-wise interpretation"))
        out.append("_no rows with valid cohens_d_\n")
        return
    out.append(section("2. Layer-wise interpretation"))
    peak_idx = iso.groupby("model")["cohens_d"].idxmax().dropna()
    if peak_idx.empty:
        out.append("_no peak rows_\n")
        return
    peak = iso.loc[peak_idx][["model", "layer", "cohens_d"]]
    peak = peak.rename(columns={"layer": "peak_layer", "cohens_d": "peak_d"})
    out.append("Peak layer per model:\n\n")
    out.append(render_table(peak.sort_values("peak_d", ascending=False),
                             ["model", "peak_layer", "peak_d"]))
    out.append("\n**Interpretation prompts** (read these before writing the paper):\n")
    out.append("- Are most peaks in the *middle* layers (suggests semantic encoding) "
               "or the *last* layer (suggests task-specific tuning)?\n")
    out.append("- Does the peak layer scale with model depth — i.e. d the same "
               "fraction-of-depth across XLM-R-base (12L) and XLM-R-large (24L)?\n")
    out.append("- Is the embedding layer (layer 0, static lookup) already showing "
               "non-zero d? If yes: tokenizer-level signal is leaking in.\n")


def semantic_control(out: list, blockers: list):
    pooled = maybe_read("expanded_semantic_control_pooled.csv")
    out.append(section("3. Semantic control (pooled across 20+ fields)"))
    if pooled is None:
        out.append("_pooled CSV missing_\n")
        blockers.append("expanded_semantic_control_pooled.csv")
        return
    cols = ["model", "n_fields", "intra_pooled", "cross_pooled",
            "delta_pooled", "d_pooled", "p_perm_pooled", "n_intra", "n_cross"]
    out.append(render_table(pooled.sort_values("d_pooled", ascending=False), cols))
    out.append("\n**Decision prompt:** if `d_pooled` is statistically zero for "
               "*every* standard model and only non-zero for glyph-aware / vision-only, "
               "the paper's headline becomes the variance decomposition — "
               "**radical structure in distributional models is fully accounted for "
               "by semantics, but glyph-aware models retain a form-specific residual.**\n")


def variance_decomposition(out: list, blockers: list):
    vd = maybe_read("variance_decomposition.csv")
    out.append(section("4. Variance decomposition  *[the money figure]*"))
    if vd is None:
        out.append("_missing — run cooccurrence_baseline.py_\n")
        blockers.append("variance_decomposition.csv")
        return
    pivot = vd.pivot_table(index="model", columns="predictor",
                           values="partial_R2", aggfunc="mean").fillna(0)
    pivot = pivot.round(4)

    # Render manually (avoids the optional `tabulate` dependency that
    # pandas.to_markdown requires).
    cols = list(pivot.columns)
    out.append("| model | " + " | ".join(cols) + " |\n")
    out.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|\n")
    for idx, row in pivot.iterrows():
        out.append(f"| {idx} | " + " | ".join(fmt_num(row[c]) for c in cols) + " |\n")

    out.append("\n**Reading guide:**\n")
    out.append("- column `same_radical` partial R² ≫ others → orthographic encoding wins\n")
    out.append("- column `ppmi` partial R² ≫ others → distributional dominates (the most likely outcome)\n")
    out.append("- column `freq_diff` partial R² substantial → much of d is frequency-driven\n")
    out.append("- column `stroke_diff` substantial only in glyph-aware → form-specific signal lives here\n")


def pseudoradical(out: list):
    pr = maybe_read("pseudoradical_control.csv")
    out.append(section("5. Pseudoradical null (size-matched random partitions)"))
    if pr is None:
        out.append("_missing — run pseudoradical_control.py_\n")
        return
    cols = ["model", "d_real", "d_random_mean", "d_random_p95", "p_pseudo", "n_partitions"]
    out.append(render_table(pr.sort_values("p_pseudo"), cols))
    out.append("\n**Specificity check:** if `p_pseudo` < 0.05, the radical signal is "
               "specific to Kangxi categories, not just any partition into 68 groups.\n")


def freq_matched(out: list):
    fm = maybe_read("frequency_matched.csv")
    out.append(section("6. Frequency-matched effect"))
    if fm is None:
        out.append("_missing — run frequency_matched_pairs.py_\n")
        return
    cols = ["model", "d_unmatched", "d_matched", "freq_inflation",
            "ci_matched_lo", "ci_matched_hi"]
    out.append(render_table(fm.sort_values("freq_inflation", ascending=False), cols))
    out.append("\n**Interpretation:** `freq_inflation = d_unmatched − d_matched`. "
               "Large freq_inflation means a substantial chunk of the apparent radical "
               "effect is just frequency-correlation. The residual `d_matched` is the "
               "honest effect after holding character frequency fixed.\n")


def random_init(out: list):
    ri = maybe_read("random_init_baseline.csv")
    out.append(section("7. Random-init noise floor"))
    if ri is None:
        out.append("_missing — run random_init_baseline.py_\n")
        return
    cols = ["model", "d_random_init", "p_perm", "intra_mean", "inter_mean"]
    out.append(render_table(ri, cols))
    out.append("\n**Interpretation:** if `d_random_init` ≈ 0, our reported effects "
               "come from training, not architecture. If non-zero, the architecture "
               "itself encodes radical structure even before training — which would "
               "be a separate finding.\n")


def cross_script(out: list):
    cs = maybe_read("cross_script_japanese.csv")
    out.append(section("8. Cross-script Japanese"))
    if cs is None:
        out.append("_missing — run cross_script_japanese.py_\n")
        return
    out.append(render_table(cs.sort_values("cohens_d", ascending=False),
               ["model", "language", "n_chars", "cohens_d", "p_perm"]))


def glyph(out: list):
    gc = maybe_read("glyph_comparison.csv")
    out.append(section("9. Glyph-aware vs standard"))
    if gc is None:
        out.append("_missing — run glyph_comparison.py_\n")
        return
    cols = ["model_id", "model_class", "d_corpus", "d_semantic_ctrl", "d_form_specific"]
    out.append(render_table(gc.sort_values("d_form_specific", ascending=False), cols))
    out.append("\n**Decision prompt:** is `d_form_specific` substantially larger for "
               "`vision_only`/`glyph_aware` than for `standard`? If yes, the form-specific "
               "residual is real — paper main contribution.\n")


def probing(out: list):
    pr = maybe_read("probing.csv")
    out.append(section("10. Probing classifier"))
    if pr is None:
        out.append("_missing — run probing_classifier.py_\n")
        return
    iso_char = pr[(pr["pool"] == "char") & (pr["iso"] == 1)]
    last = (iso_char.sort_values("layer").groupby(["model", "probe"]).tail(1))
    cols = ["model", "probe", "layer", "macro_f1", "accuracy", "balanced_accuracy", "baseline"]
    out.append(render_table(last.sort_values(["model", "probe"]), cols))


def orth_arith(out: list):
    oa = maybe_read("orthographic_arithmetic_summary.csv")
    out.append(section("11. Orthographic arithmetic (Mikolov-style)"))
    if oa is None:
        out.append("_missing — run orthographic_arithmetic.py_\n")
        return
    out.append(render_table(oa.sort_values("mean_lift", ascending=False),
               ["model", "n_pairs", "mean_top10_rate", "mean_baseline", "mean_lift"]))
    out.append("\n**Decision prompt:** lift ≈ 1 → no compositionality. "
               "lift > 2 → meaningful linear composition.\n")


def anomalies(out: list):
    out.append(section("12. Anomaly flags"))
    flags = []

    lw = maybe_read("layer_wise.csv")
    if lw is not None:
        iso = lw[(lw["pool"] == "char") & (lw["iso"] == 1)]
        zero_layer = iso[iso["layer"] == 0]
        for _, row in zero_layer.iterrows():
            if row["cohens_d"] > 0.05:
                flags.append(f"⚠ {row['model']} has d={row['cohens_d']:.3f} at layer 0 "
                             f"(static embeddings) — tokenizer-level signal leaking in.")

    fm = maybe_read("frequency_matched.csv")
    if fm is not None:
        for _, row in fm.iterrows():
            if np.isfinite(row["freq_inflation"]) and row["freq_inflation"] > 0.05:
                flags.append(
                    f"⚠ {row['model']}: large freq_inflation={row['freq_inflation']:.3f} — "
                    f"a meaningful share of d is frequency-driven."
                )

    pr = maybe_read("pseudoradical_control.csv")
    if pr is not None:
        for _, row in pr.iterrows():
            if row["p_pseudo"] > 0.05:
                flags.append(
                    f"⚠ {row['model']}: p_pseudo={row['p_pseudo']:.3f} > 0.05 — "
                    f"effect is *not* specific to Kangxi radicals."
                )

    ri = maybe_read("random_init_baseline.csv")
    if ri is not None:
        for _, row in ri.iterrows():
            if row["d_random_init"] > 0.05:
                flags.append(
                    f"⚠ {row['model']}: d_random_init={row['d_random_init']:.3f} > 0.05 — "
                    f"untrained model already shows radical-aligned cohesion."
                )

    if flags:
        for f in flags:
            out.append(f"- {f}\n")
    else:
        out.append("None detected.\n")


def cloze(out: list):
    df = maybe_read("radical_cloze_summary.csv")
    out.append(section("13. Radical cloze probe (LM behavior)"))
    if df is None:
        out.append("_missing — run radical_cloze_probe.py_\n")
        return
    cols = ["model", "family", "mean_delta", "mean_top1_rate", "mean_mrr", "n_fields"]
    out.append(render_table(df.sort_values("mean_delta", ascending=False), cols))
    out.append("\n**Decision prompt:** if `mean_delta` (target log-prob − distractor log-prob) "
               "is large for radical-aware models and small/negative for radical-naive ones, "
               "the geometric findings translate into observable LM behavior.\n")


def main():
    out = []
    blockers = []
    out.append("# Phase 3 auto-analysis report\n\n")
    out.append("Generated by `scripts/new/results_report.py`. "
               "Re-run after every pipeline run; this file is regenerated, not appended.\n")

    headline(out, blockers)
    layerwise_analysis(out)
    semantic_control(out, blockers)
    variance_decomposition(out, blockers)
    pseudoradical(out)
    freq_matched(out)
    random_init(out)
    cross_script(out)
    glyph(out)
    probing(out)
    orth_arith(out)
    anomalies(out)
    cloze(out)

    if blockers:
        out.append(section("BLOCKERS"))
        out.append("The following expected outputs are missing or empty. "
                   "Phase 3 / Phase 4 cannot be finalized until they are produced:\n\n")
        for b in blockers:
            out.append(f"- `results/{b}`\n")

    report = "".join(out)
    out_path = RESULTS_DIR / "_REPORT.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}  ({len(report)} chars)")
    if blockers:
        print(f"⚠ {len(blockers)} blockers — run the missing scripts.")


if __name__ == "__main__":
    main()
