"""
End-to-end smoke test for batch A.

Synthesizes fake embeddings, runs them through:
    - the cache layer (save / load)
    - isotropy correction
    - all stats primitives (cohens_d, welch_t, bootstrap, permutation, holm, rsa)
    - the layer_wise_analysis pair-sampling pipeline

This catches every kind of silent bug (off-by-one, dtype, NaN, empty array,
column-name mismatch) without needing a single transformer download.

Run:
    venv/Scripts/python.exe scripts/new/_smoke_test.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    set_seed,
    load_radical_dataset,
    radical_groups,
    save_layer_embeddings,
    load_layer_embeddings,
    list_available_models,
    list_available_layers,
    fit_isotropy,
    apply_isotropy,
    cosine_isotropic,
    cohens_d,
    welch_t,
    bootstrap_ci_diff,
    permutation_test_diff,
    rsa_spearman,
    holm_bonferroni,
)


# ── 1. stats primitives ─────────────────────────────────────────────────────
def test_stats():
    rng = np.random.default_rng(0)
    a = rng.normal(0.6, 0.1, 500)
    b = rng.normal(0.5, 0.1, 1000)

    d = cohens_d(a, b)
    assert 0.7 < d < 1.3, f"cohens_d sanity: {d}"

    t, p = welch_t(a, b)
    assert p < 1e-10, f"welch_t p too high: {p}"

    lo, hi, dist = bootstrap_ci_diff(a, b, n_boot=200, rng=rng)
    assert lo > 0 and hi > 0, f"bootstrap CI should exclude 0: [{lo}, {hi}]"
    assert dist.shape == (200,), f"dist shape: {dist.shape}"

    p_perm, obs, null = permutation_test_diff(a, b, n_perm=500, rng=rng)
    assert p_perm < 0.01, f"perm p too high: {p_perm}"
    assert null.shape == (500,)

    adj = holm_bonferroni([0.001, 0.04, 0.5])
    # 0.001*3=0.003, 0.04*2=0.08, 0.5*1=0.5
    assert np.allclose(adj, [0.003, 0.08, 0.5], atol=1e-6), f"holm: {adj}"

    # RSA: identical RDMs should give rho = 1
    M = rng.normal(size=(50, 50))
    M = (M + M.T) / 2
    rho, _ = rsa_spearman(M, M)
    assert rho > 0.99, f"RSA self-corr: {rho}"

    print("[OK] stats primitives")


# ── 2. isotropy ─────────────────────────────────────────────────────────────
def test_isotropy():
    rng = np.random.default_rng(0)
    # synth: heavily anisotropic embedding cone
    # smaller per-sample noise + larger common offset → high mean cosine
    base = rng.normal(size=(200, 50)) * 0.3
    cone_dir = rng.normal(size=50)
    cone_dir /= np.linalg.norm(cone_dir)
    X = base + 8.0 * cone_dir[None, :]

    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    raw_cos = Xn @ Xn.T
    iu = np.triu_indices(len(X), k=1)
    raw_mean_off = raw_cos[iu].mean()

    iso_cos = cosine_isotropic(X, k=2)
    iso_mean_off = iso_cos[iu].mean()

    assert raw_mean_off > 0.7, f"sanity: anisotropic raw mean cosine = {raw_mean_off}"
    # corrected should be close to 0 — within 0.05 is generous
    assert abs(iso_mean_off) < 0.05, f"isotropy correction failed: {iso_mean_off}"

    # round-trip
    params = fit_isotropy(X, k=2)
    X2 = apply_isotropy(X, params)
    assert X2.shape == X.shape

    print(f"[OK] isotropy  (raw mean off-diag={raw_mean_off:.3f}  → corrected={iso_mean_off:.3f})")


# ── 3. cache layout ─────────────────────────────────────────────────────────
def test_cache():
    df = load_radical_dataset()
    chars = df["char"].tolist()[:100]
    fake_id = "smoketest/fake-model"
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(chars), 32)).astype(np.float32)

    save_layer_embeddings(fake_id, layer=3, embeddings=emb, chars=chars, pool="mean")
    save_layer_embeddings(fake_id, layer=3, embeddings=emb, chars=chars, pool="char")
    save_layer_embeddings(fake_id, layer=3, embeddings=emb, chars=chars, pool="cls")

    loaded = load_layer_embeddings(fake_id, 3, pool="mean")
    assert np.allclose(loaded, emb), "round-trip failed"

    models = list_available_models()
    assert fake_id in models, f"discovery failed: {models}"
    layers = list_available_layers(fake_id)
    assert 3 in layers, f"layer discovery: {layers}"

    # cleanup
    from radical_lib.embeddings import model_dir
    import shutil
    shutil.rmtree(model_dir(fake_id))
    print("[OK] embeddings cache (save / load / discover)")


# ── 4. radical group sampling (mirrors layer_wise_analysis) ─────────────────
def test_radical_pairs():
    df = load_radical_dataset()
    groups = radical_groups()
    assert len(groups) > 0, "no radical groups loaded"

    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col]))
    chars = df["char"].tolist()

    # fake similarity: identity + small noise
    n = len(chars)
    sim = np.eye(n) + np.random.default_rng(0).normal(0, 0.01, (n, n))

    import itertools
    import random
    random.seed(42)
    intra, inter = [], []
    for rad, rad_chars in list(groups.items())[:5]:
        for a, b in itertools.islice(itertools.combinations(rad_chars, 2), 10):
            intra.append(sim[chars.index(a), chars.index(b)])
        others = [c for c in chars if char_to_radical[c] != rad]
        for _ in range(10):
            a = random.choice(rad_chars)
            b = random.choice(others)
            inter.append(sim[chars.index(a), chars.index(b)])

    assert len(intra) == 50 and len(inter) == 50
    print(f"[OK] radical-pair sampling  (n_intra={len(intra)}, n_inter={len(inter)})")


# ── 5. config sanity ────────────────────────────────────────────────────────
def test_config():
    from scripts.new.config import (
        MODELS,
        chinese_models,
        japanese_models,
        scaling_models,
    )
    assert len(MODELS) >= 11
    assert len(japanese_models()) >= 1
    assert len(chinese_models()) >= 6
    assert len(scaling_models()) >= 4
    # all hf_ids unique
    ids = [m.hf_id for m in MODELS]
    assert len(ids) == len(set(ids)), "duplicate hf_id"
    print(f"[OK] config  ({len(MODELS)} models, {len(scaling_models())} in scaling subset)")


def main():
    set_seed()
    print("\n=== batch A smoke tests ===\n")
    test_config()
    test_stats()
    test_isotropy()
    test_cache()
    test_radical_pairs()
    print("\nall green ✔")


if __name__ == "__main__":
    main()
