"""
Smoke test for the new controls (pseudoradical, freq-matched) and the
report generator. random_init_baseline is import-only (it instantiates
HF models which we don't want in a smoke test).
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from radical_lib import (  # noqa: E402
    CACHE_DIR,
    RESULTS_DIR,
    DATA_DIR,
    set_seed,
    save_layer_embeddings,
    fit_isotropy,
    apply_isotropy,
    load_radical_dataset,
)
from radical_lib.embeddings import model_tag, model_dir  # noqa: E402

set_seed()
FAKE_MODEL = "smoketest/fakebert_d"
ISO_DIR = CACHE_DIR / "embeddings_iso"


def plant_fake_cache(chars, rad_y):
    rng = np.random.default_rng(0)
    D = 64
    centroids = {int(r): rng.normal(size=D) * 3 for r in np.unique(rad_y)}
    X = np.stack([centroids[int(r)] + rng.normal(scale=0.5, size=D) for r in rad_y]).astype(np.float32)
    save_layer_embeddings(FAKE_MODEL, 12, X, chars, pool="char")
    iso_dir = ISO_DIR / model_tag(FAKE_MODEL)
    iso_dir.mkdir(parents=True, exist_ok=True)
    params = fit_isotropy(X, k=2)
    Xc = apply_isotropy(X, params).astype(np.float32)
    np.save(iso_dir / "layer12_char.npy", Xc)


def cleanup():
    if model_dir(FAKE_MODEL).exists():
        shutil.rmtree(model_dir(FAKE_MODEL))
    iso = ISO_DIR / model_tag(FAKE_MODEL)
    if iso.exists():
        shutil.rmtree(iso)


def test_pseudoradical():
    from scripts.new import pseudoradical_control as pr
    original_lam = pr.list_available_models
    pr.list_available_models = lambda: [FAKE_MODEL]
    try:
        pr.main(B=20)  # tiny B for smoke
    finally:
        pr.list_available_models = original_lam
    out = pd.read_csv(RESULTS_DIR / "pseudoradical_control.csv")
    assert len(out) >= 1, "pseudoradical wrote no rows"
    assert "p_pseudo" in out.columns
    # synthetic data has perfect clustering, so d_real should dominate
    assert (out["d_real"] > out["d_random_mean"]).all()
    print(f"[OK] pseudoradical  ({len(out)} row(s), d_real > d_random_mean as expected)")


def test_freq_matched():
    from scripts.new import frequency_matched_pairs as fm
    original_lam = fm.list_available_models
    fm.list_available_models = lambda: [FAKE_MODEL]
    try:
        fm.main()
    finally:
        fm.list_available_models = original_lam
    out = pd.read_csv(RESULTS_DIR / "frequency_matched.csv")
    assert len(out) >= 1
    assert "freq_inflation" in out.columns
    print(f"[OK] frequency_matched  ({len(out)} rows)")


def test_results_report():
    from scripts.new import results_report as rr
    rr.main()
    p = RESULTS_DIR / "_REPORT.md"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "Phase 3 auto-analysis report" in text
    assert "## 1. Headline numbers" in text
    print(f"[OK] results_report  ({len(text)} chars)")


def test_random_init_imports():
    """Just import — instantiating real HF models is too heavy for a smoke test."""
    from scripts.new import random_init_baseline as ri  # noqa: F401
    print("[OK] random_init_baseline imports clean")


def main():
    print("\n=== batch D smoke tests ===\n")
    df = load_radical_dataset()
    chars = df["char"].tolist()
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    rad_y = df[rad_col].astype(int).to_numpy()

    # Ensure the dataset has frequency_proxy + role columns. If the user
    # hasn't run dataset_builder yet, the canonical CSV is the original
    # 2-column version; we transiently augment it for the smoke test.
    needs_fp = "frequency_proxy" not in df.columns
    needs_role = "radical_role" not in df.columns
    backup = None
    if needs_fp or needs_role:
        backup = DATA_DIR / "radical_dataset.csv.bak_smoke_d"
        shutil.copy(DATA_DIR / "radical_dataset.csv", backup)
        rng = np.random.default_rng(0)
        if needs_fp:
            df["frequency_proxy"] = rng.integers(100, 25000, size=len(df))
        if needs_role:
            roles = rng.choice(["semantic", "identity", "unknown"],
                               size=len(df), p=[0.7, 0.1, 0.2])
            df["radical_role"] = roles
            df["liushu_class"] = ["phonosemantic" if r == "semantic" else "simple"
                                   for r in roles]
        df.to_csv(DATA_DIR / "radical_dataset.csv", index=False)
        load_radical_dataset.cache_clear()

    cleanup()
    plant_fake_cache(chars, rad_y)
    try:
        test_pseudoradical()
        test_freq_matched()
        test_results_report()
        test_random_init_imports()
        print("\nall green ✔")
    finally:
        cleanup()
        if backup is not None:
            shutil.move(str(backup), str(DATA_DIR / "radical_dataset.csv"))
            load_radical_dataset.cache_clear()


if __name__ == "__main__":
    main()
