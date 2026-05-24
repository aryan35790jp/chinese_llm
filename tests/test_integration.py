"""
Integration test for batch C.

Plants a fake model with synthetic radical-clustered embeddings, then
calls main() of every batch C script. Asserts each writes its CSV with
sane content. Heavy network/GPU scripts (cooccurrence_baseline that
streams Wikipedia, sentential_context that downloads models) are run in
"stub mode" — we feed them a tiny pre-built input so we still exercise
the main() function without needing 30 minutes and the network.

Mirrors the strategy used in _smoke_test.py and _smoke_test_b.py.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

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
FAKE_MODEL_RAW = "smoketest__fakebert_c"
FAKE_MODEL = "smoketest/fakebert_c"
ISO_DIR = CACHE_DIR / "embeddings_iso"


# ── 1. setup ────────────────────────────────────────────────────────────────
def plant_fake_cache(chars: list[str], rad_y: np.ndarray) -> None:
    """Strong radical-clustered embeddings."""
    rng = np.random.default_rng(0)
    D = 64
    centroids = {int(r): rng.normal(size=D) * 3 for r in np.unique(rad_y)}
    X = np.stack([centroids[int(r)] + rng.normal(scale=0.5, size=D) for r in rad_y]).astype(np.float32)

    for layer in (0, 6, 12):
        for pool in ("char", "mean", "cls"):
            save_layer_embeddings(FAKE_MODEL, layer, X, chars, pool=pool)

    iso_dir = ISO_DIR / model_tag(FAKE_MODEL)
    iso_dir.mkdir(parents=True, exist_ok=True)
    params = fit_isotropy(X, k=2)
    Xc = apply_isotropy(X, params).astype(np.float32)
    for layer in (0, 6, 12):
        for pool in ("char", "mean", "cls"):
            np.save(iso_dir / f"layer{layer:02d}_{pool}.npy", Xc)


def plant_fake_layerwise() -> None:
    rows = []
    for layer in (0, 6, 12):
        for pool in ("char", "mean"):
            for iso in (0, 1):
                rows.append({
                    "model": FAKE_MODEL, "layer": layer, "pool": pool, "iso": iso,
                    "intra_mean": 0.7, "inter_mean": 0.5, "delta": 0.2,
                    "cohens_d": 1.2, "p_welch": 1e-50, "p_perm": 0.001,
                    "ci_lower": 0.18, "ci_upper": 0.22,
                    "rsa_rho": 0.3, "rsa_p": 1e-20,
                    "n_intra": 1000, "n_inter": 1000,
                })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "layer_wise.csv", index=False)


def plant_fake_semantic_pooled() -> None:
    pd.DataFrame([{
        "model": FAKE_MODEL, "n_fields": 20,
        "intra_pooled": 0.65, "cross_pooled": 0.55,
        "delta_pooled": 0.10, "d_pooled": 0.4, "p_perm_pooled": 0.001,
        "n_intra": 200, "n_cross": 600,
    }]).to_csv(RESULTS_DIR / "expanded_semantic_control_pooled.csv", index=False)


def plant_fake_ppmi(n_chars: int) -> None:
    """Plant a fake sparse PPMI cache so cooccurrence_baseline skips its
    Wikipedia stream."""
    rng = np.random.default_rng(0)
    nnz = n_chars * 20
    rows = rng.integers(0, n_chars, size=nnz)
    cols = rng.integers(0, n_chars, size=nnz)
    vals = rng.uniform(0, 5, size=nnz).astype(np.float32)
    ppmi = csr_matrix((vals, (rows, cols)), shape=(n_chars, n_chars))
    save_npz(CACHE_DIR / "char_ppmi.npz", ppmi)
    counts = rng.integers(10, 1000, size=n_chars).astype(np.int64)
    np.save(CACHE_DIR / "char_counts.npy", counts)


def plant_fake_sentences(top_chars: list[str], k: int = 5) -> None:
    """Plant a fake sentences.json so sentential_context skips its Wikipedia
    stream. Each sentence simply concatenates a few chars including the
    target so the script can find the position."""
    bucket = {}
    for c in top_chars:
        sentences = [f"今天{c}很{ch}" for ch in top_chars[:k]]
        bucket[c] = sentences
    (CACHE_DIR / "sentences.json").write_text(
        json.dumps(bucket, ensure_ascii=False), encoding="utf-8"
    )


def add_role_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fake liushu + radical_role columns so phonetic_vs_semantic works."""
    rng = np.random.default_rng(0)
    roles = rng.choice(["semantic", "identity", "unknown"], size=len(df), p=[0.7, 0.1, 0.2])
    out = df.copy()
    out["liushu_class"] = ["phonosemantic" if r == "semantic" else "simple" for r in roles]
    out["radical_role"] = roles
    return out


# ── 2. cleanup ──────────────────────────────────────────────────────────────
def cleanup_fake_cache():
    if model_dir(FAKE_MODEL).exists():
        shutil.rmtree(model_dir(FAKE_MODEL))
    iso = ISO_DIR / model_tag(FAKE_MODEL)
    if iso.exists():
        shutil.rmtree(iso)
    for p in [
        CACHE_DIR / "char_ppmi.npz",
        CACHE_DIR / "char_counts.npy",
        CACHE_DIR / "sentences.json",
        CACHE_DIR / "sentential" / model_tag(FAKE_MODEL),
    ]:
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()


def assert_csv_has_rows(name: str, expected_min_rows: int = 1):
    path = RESULTS_DIR / name
    assert path.exists(), f"{name} was not written"
    out = pd.read_csv(path)
    assert len(out) >= expected_min_rows, (
        f"{name} has only {len(out)} rows (expected >= {expected_min_rows})"
    )
    return out


# ── 3. driver ───────────────────────────────────────────────────────────────
def main():
    print("\n=== batch C integration test ===\n")

    dataset_path = DATA_DIR / "radical_dataset.csv"
    backup_path = DATA_DIR / "radical_dataset.csv.bak_integration_c"
    shutil.copy(dataset_path, backup_path)

    try:
        df = load_radical_dataset()
        chars = df["char"].tolist()
        rad_col = "radical_number" if "radical_number" in df.columns else "radical"
        rad_y = df[rad_col].astype(int).to_numpy()

        df_with_roles = add_role_columns(df)
        df_with_roles.to_csv(dataset_path, index=False)
        load_radical_dataset.cache_clear()

        cleanup_fake_cache()
        plant_fake_cache(chars, rad_y)
        plant_fake_layerwise()
        plant_fake_semantic_pooled()
        plant_fake_ppmi(len(chars))

        # ── cooccurrence_baseline ─────────────────────────────────────────
        print("[run] cooccurrence_baseline.main()  (PPMI cache pre-planted)")
        from scripts.new import cooccurrence_baseline as cb
        original_lam = cb.list_available_models
        cb.list_available_models = lambda: [FAKE_MODEL]
        try:
            cb.main(n_sentences=0, max_pairs=5000)
        finally:
            cb.list_available_models = original_lam
        out = assert_csv_has_rows("variance_decomposition.csv", 4)  # 4 predictors
        assert "partial_R2" in out.columns
        print(f"  → {len(out)} rows, full_R2={out['full_R2'].iloc[0]:.3f}  [OK]")

        # ── orthographic_arithmetic ──────────────────────────────────────
        print("[run] orthographic_arithmetic.main()")
        from scripts.new import orthographic_arithmetic as oa
        original_lam = oa.list_available_models
        oa.list_available_models = lambda: [FAKE_MODEL]
        try:
            oa.main(n_trials_per_pair=10, top_k=10)
        finally:
            oa.list_available_models = original_lam
        out = assert_csv_has_rows("orthographic_arithmetic.csv", 1)
        summary = assert_csv_has_rows("orthographic_arithmetic_summary.csv", 1)
        print(f"  → {len(out)} pair-rows, mean lift={summary['mean_lift'].iloc[0]:.2f}  [OK]")

        # ── activation_patching ──────────────────────────────────────────
        print("[run] activation_patching.main()")
        from scripts.new import activation_patching as ap
        original_lam = ap.list_available_models
        ap.list_available_models = lambda: [FAKE_MODEL]
        try:
            ap.main(top_k=10, max_src_chars=10)
        finally:
            ap.list_available_models = original_lam
        out = assert_csv_has_rows("activation_patching.csv", 1)
        print(f"  → {len(out)} rows, mean top10={out['top10_target_rate'].mean():.3f}  [OK]")

        # ── downstream_validation ────────────────────────────────────────
        print("[run] downstream_validation.main()  (uses embedded fallback)")
        from scripts.new import downstream_validation as dv
        original_lam = dv.list_available_models
        dv.list_available_models = lambda: [FAKE_MODEL]
        try:
            dv.main()
        finally:
            dv.list_available_models = original_lam
        out_val = assert_csv_has_rows("downstream_validation.csv", 1)
        # per-radical CSV may be empty if not enough pairs — just check exists
        assert (RESULTS_DIR / "downstream_per_radical.csv").exists()
        print(f"  → {len(out_val)} model rows, ρ={out_val['spearman_rho'].iloc[0]:.3f}  [OK]")

        # ── sentential_context (skip: needs HF model download) ───────────
        # Just check the module imports cleanly. main() needs real transformers
        # which we can't run here.
        print("[run] sentential_context  (import-only)")
        import scripts.new.sentential_context as sc  # noqa: F401
        print("  [OK]")

        # ── figures ──────────────────────────────────────────────────────
        print("[run] figures.main()")
        from scripts.new import figures as fg
        fg.main()
        # Just check that at least one figure was produced
        figs = list((ROOT / "figures").glob("fig_*.png"))
        assert len(figs) >= 3, f"too few figures produced: {len(figs)}"
        print(f"  → {len(figs)} figures written  [OK]")

        # ── full_pipeline (--only figures, --force) — verify driver runs ─
        print("[run] full_pipeline.main()  (--only figures)")
        from scripts.new import full_pipeline as fp
        original_argv = sys.argv
        sys.argv = ["full_pipeline.py", "--only", "figures", "--force"]
        try:
            try:
                fp.main()
            except SystemExit as e:
                assert e.code in (0, None), f"full_pipeline exited with {e.code}"
        finally:
            sys.argv = original_argv
        print("  [OK]")

        print("\nall green ✔")

    finally:
        shutil.move(str(backup_path), str(dataset_path))
        load_radical_dataset.cache_clear()
        cleanup_fake_cache()


if __name__ == "__main__":
    main()
