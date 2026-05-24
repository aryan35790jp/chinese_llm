"""Audit every CSV in results/ and report row counts + presence of load-bearing columns."""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

EXPECT = {
    "layer_wise.csv":                       ["cohens_d", "rsa_rho"],
    "expanded_semantic_control.csv":        ["cohens_d", "p_perm"],
    "expanded_semantic_control_pooled.csv": ["d_pooled", "p_perm_pooled"],
    "glyph_comparison.csv":                 ["d_form_specific"],
    "random_init_baseline.csv":             ["d_random_init"],
    "pseudoradical_control.csv":            ["p_pseudo", "d_real"],
    "frequency_matched.csv":                ["freq_inflation", "d_matched"],
    "cross_script_japanese.csv":            ["cohens_d", "p_perm"],
    "orthographic_arithmetic_summary.csv":  ["mean_lift"],
    "orthographic_arithmetic.csv":          ["lift_top10"],
    "phonetic_vs_semantic_radicals.csv":    ["cohens_d"],
    "scaling.csv":                          ["last_layer_d"],
    "downstream_validation.csv":            ["spearman_rho"],
    "downstream_per_radical.csv":           [],
    "tokenization_audit_summary.csv":       ["coverage"],
    "tokenization_audit.csv":               [],
    "probing.csv":                          ["macro_f1"],
    "activation_patching.csv":              ["lift"],
    "variance_decomposition.csv":           ["partial_R2", "full_R2"],
    "main_results.csv":                     [],
    "semantic_control_results.csv":         [],
}

print(f"\n{'file':<44} {'rows':>6} {'cols':>5}  status")
print("-" * 78)

missing, bad = [], []
for name, cols_expected in sorted(EXPECT.items()):
    p = RESULTS / name
    if not p.exists() or p.stat().st_size == 0:
        print(f"{name:<44} {'—':>6} {'—':>5}  MISSING")
        missing.append(name)
        continue
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"{name:<44} {'?':>6} {'?':>5}  READ-FAIL: {e}")
        bad.append(name)
        continue
    n_rows = len(df)
    n_cols = len(df.columns)
    issues = []
    if n_rows == 0:
        issues.append("empty")
    for c in cols_expected:
        if c not in df.columns:
            issues.append(f"missing-col:{c}")
    status = "OK" if not issues else " ".join(issues)
    print(f"{name:<44} {n_rows:>6} {n_cols:>5}  {status}")
    if issues:
        bad.append(name)

print()
print(f"missing: {missing}")
print(f"bad:     {bad}")

# Also list anything in results/ that we *don't* expect
all_csvs = {p.name for p in RESULTS.glob("*.csv")}
unknown = sorted(all_csvs - set(EXPECT.keys()))
if unknown:
    print(f"\nunexpected CSVs in results/: {unknown}")
else:
    print("\n(no unexpected CSVs)")

# also check figures
print(f"\nfigures: {len(list((ROOT / 'figures').glob('*.png')))} PNG, "
      f"{len(list((ROOT / 'figures').glob('*.pdf')))} PDF")

print(f"data/radical_dataset.csv rows: {len(pd.read_csv(ROOT / 'data' / 'radical_dataset.csv'))}")
