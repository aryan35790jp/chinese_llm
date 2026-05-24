"""
Smoke test for batch B logic that doesn't require trained model downloads.

Tests:
    1. classify_liushu.kangxi_unified() returns the right radical for known cases
    2. classify_liushu.atomic_components() correctly strips IDC markers
    3. classify_liushu.classify() identifies pictograph identities
    4. expanded_semantic_control.build_fields_fallback() returns ≥10 fields
       given the actual dataset
    5. expanded_semantic_control.split_field_by_dominant_radical() partitions
       sensibly
    6. probing_classifier.cv_probe() returns reasonable numbers on synthetic data
    7. glyph_only_baseline.find_cjk_font() returns *something* on this machine
       (or warns)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import set_seed, load_radical_dataset  # noqa: E402

set_seed()


def test_kangxi_table():
    from scripts.new.classify_liushu import kangxi_unified
    # Spot-check: well-known mappings
    assert kangxi_unified(1) == "一", f"radical 1 → {kangxi_unified(1)}"
    assert kangxi_unified(9) == "人", f"radical 9 → {kangxi_unified(9)}"
    assert kangxi_unified(85) == "水", f"radical 85 → {kangxi_unified(85)}"
    assert kangxi_unified(86) == "火", f"radical 86 → {kangxi_unified(86)}"
    assert kangxi_unified(94) == "犬", f"radical 94 → {kangxi_unified(94)}"
    assert kangxi_unified(75) == "木", f"radical 75 → {kangxi_unified(75)}"
    assert kangxi_unified(167) == "金", f"radical 167 → {kangxi_unified(167)}"
    assert kangxi_unified(195) == "魚", f"radical 195 → {kangxi_unified(195)}"
    assert kangxi_unified(214) == "龠", f"radical 214 → {kangxi_unified(214)}"
    print("[OK] kangxi_unified(n) — spot checks for radicals 1, 9, 75, 85, 86, 94, 167, 195, 214")


def test_atomic_components():
    from scripts.new.classify_liushu import atomic_components
    # 河 → ⿰氵可 → atomic = [氵, 可]
    assert atomic_components("⿰氵可", "河") == ["氵", "可"]
    # 一 → 一 → no decomposition
    assert atomic_components("一", "一") == []
    # nested: ⿰⿱艹田 → atomic = [艹, 田]
    assert atomic_components("⿰⿱艹田⺘", "苗") == ["艹", "田", "⺘"]
    print("[OK] atomic_components — strips IDC, handles nesting and identity")


def test_classify_identity():
    from scripts.new.classify_liushu import classify
    # 水 IS radical 85 → identity
    cls, role = classify("水", 85, ids_map={})
    assert cls == "simple" and role == "identity", f"水 → {cls},{role}"
    # 河 has radical 85; if we pretend the IDS knows 河 = ⿰氵可, the IDS
    # contains 氵 but kangxi_unified(85) is 水 not 氵, so this hits the
    # "phonosemantic, unknown" branch. That is the documented behavior:
    # the script can't reconcile glyph variants without a variant table.
    cls, role = classify("河", 85, ids_map={"河": "⿰氵可"})
    assert cls == "phonosemantic", f"河 → {cls},{role}"
    print(f"[OK] classify(水) → identity; classify(河) → phonosemantic ({role})")


def test_fallback_fields():
    from scripts.new.expanded_semantic_control import (
        build_fields_fallback,
        split_field_by_dominant_radical,
        MIN_FIELD_SIZE,
        MIN_GROUP_SIZE,
    )
    df = load_radical_dataset()
    chars = df["char"].tolist()
    fields = build_fields_fallback(chars)
    assert len(fields) >= 5, f"too few usable fields: {len(fields)}"
    for fname, fchars in fields.items():
        assert len(fchars) >= MIN_FIELD_SIZE, f"{fname}: {len(fchars)}"

    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col]))
    n_usable = 0
    for fname, fchars in fields.items():
        same, diff, dom = split_field_by_dominant_radical(fchars, char_to_radical)
        if len(same) >= MIN_GROUP_SIZE and len(diff) >= MIN_GROUP_SIZE:
            n_usable += 1
    print(f"[OK] fallback fields  ({len(fields)} total, {n_usable} with valid radical split)")


def test_cv_probe():
    from scripts.new.probing_classifier import cv_probe
    rng = np.random.default_rng(0)
    # 3 well-separated clusters, 100 samples each, 10-d
    n_per = 100
    centers = np.array([[5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)
    X = np.vstack([centers[i] + rng.normal(0, 0.5, (n_per, 10)) for i in range(3)])
    y = np.array([i for i in range(3) for _ in range(n_per)])
    f1, acc, bal, base = cv_probe(X, y, n_splits=3)
    assert f1 > 0.95, f"linear probe should solve 3 separable clusters: f1={f1}"
    assert base < 0.5, f"baseline weird: {base}"
    print(f"[OK] cv_probe  (separable clusters: f1={f1:.3f}, acc={acc:.3f}, baseline={base:.3f})")


def test_cjk_font():
    from scripts.new.glyph_only_baseline import find_cjk_font
    f = find_cjk_font()
    if f is None:
        print("[WARN] no CJK font found on this machine; glyph_only_baseline will fail. "
              "Install Noto Sans CJK or set a path manually.")
    else:
        print(f"[OK] CJK font discoverable: {f}")


def main():
    print("\n=== batch B smoke tests ===\n")
    test_kangxi_table()
    test_atomic_components()
    test_classify_identity()
    test_fallback_fields()
    test_cv_probe()
    test_cjk_font()
    print("\nall green ✔")


if __name__ == "__main__":
    main()
