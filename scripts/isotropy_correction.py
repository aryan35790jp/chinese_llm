"""
isotropy_correction.py — fit and apply Mu & Viswanath (2018) anisotropy
correction to every cached layer of every model, and save the corrected
matrices alongside the raw ones.

Why this script exists:
    Raw transformer embeddings live in a narrow cone. Cosine similarities
    are inflated by a global mean direction and ~2 dominant principal
    components. Reporting raw cosines confounds anisotropy with semantic
    structure. Every paper number from this point onward should use the
    isotropy-corrected embeddings.

We save the corrected embeddings as a separate cache layout so you can
diff raw vs corrected at any time:

    cache/embeddings_iso/{model_tag}/layer{NN}_{pool}.npy

Runtime: ~10 minutes total (it's just SVDs on N×D matrices). RAM: <8 GB.

Depends on: extract_embeddings.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    CACHE_DIR,
    set_seed,
    fit_isotropy,
    apply_isotropy,
    list_available_models,
    list_available_layers,
    load_layer_embeddings,
)
from radical_lib.embeddings import model_tag  # noqa: E402
from scripts.new.config import ISOTROPY_K  # noqa: E402

set_seed()

ISO_DIR = CACHE_DIR / "embeddings_iso"
ISO_DIR.mkdir(parents=True, exist_ok=True)


def main():
    models = list_available_models()
    if not models:
        print("[fatal] no embeddings cached yet. Run extract_embeddings.py first.")
        sys.exit(1)

    for model_id in models:
        layers = list_available_layers(model_id)
        if not layers:
            continue
        out_dir = ISO_DIR / model_tag(model_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {model_id}  layers={layers} ===")
        for L in layers:
            for pool in ("mean", "char", "cls"):
                src = load_layer_embeddings(model_id, L, pool=pool)
                params = fit_isotropy(src, k=ISOTROPY_K)
                Xc = apply_isotropy(src, params)
                out_path = out_dir / f"layer{L:02d}_{pool}.npy"
                np.save(out_path, Xc.astype(np.float32))
            print(f"  layer {L:02d} done")


if __name__ == "__main__":
    main()
