"""
Cache layout for layer-wise embeddings.

We deliberately store one .npy per (model, layer, pool) so that downstream
scripts can stream-load only what they need rather than holding 11 models
× 13 layers × 6306 × 768 floats in memory.

Filename convention:
    cache/embeddings/{model_tag}/layer{NN}_{pool}.npy
    cache/embeddings/{model_tag}/charlist.txt   ← row order, one char per line

`pool` ∈ {mean, char, cls} — corresponds to attention-mask-weighted mean,
the character-token position only, and [CLS] respectively.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np

from .core import CACHE_DIR


EMBEDDING_DIR = CACHE_DIR / "embeddings"
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)


def model_tag(model_id: str) -> str:
    """Filesystem-safe tag for a HuggingFace model id.

    Only `/` needs escaping — hyphens, dots, and underscores are all valid
    on every filesystem we care about. Keeping hyphens unchanged means
    `list_available_models()` can losslessly recover the original id by
    reversing the single `/`<->`__` substitution.
    """
    return model_id.replace("/", "__")


def model_dir(model_id: str) -> Path:
    d = EMBEDDING_DIR / model_tag(model_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def embedding_path(model_id: str, layer: int, pool: str = "mean") -> Path:
    return model_dir(model_id) / f"layer{layer:02d}_{pool}.npy"


def save_layer_embeddings(
    model_id: str,
    layer: int,
    embeddings: np.ndarray,
    chars: List[str],
    pool: str = "mean",
) -> None:
    """Save `embeddings` (N×D) and the char order. Char order written once per model."""
    np.save(embedding_path(model_id, layer, pool), embeddings.astype(np.float32))
    charlist = model_dir(model_id) / "charlist.txt"
    if not charlist.exists():
        charlist.write_text("\n".join(chars), encoding="utf-8")


def load_layer_embeddings(
    model_id: str, layer: int, pool: str = "mean"
) -> np.ndarray:
    return np.load(embedding_path(model_id, layer, pool))


def list_available_models() -> List[str]:
    """Return model_ids that have at least one cached layer file (any pool).

    Robust to partial extraction: if a model's `mean` pool failed to save
    but `char` did, we still surface it. The caller can then check
    pool-specific availability with `embedding_path()`.
    """
    out: List[str] = []
    if not EMBEDDING_DIR.exists():
        return out
    for sub in EMBEDDING_DIR.iterdir():
        if sub.is_dir() and any(sub.glob("layer*.npy")):
            out.append(sub.name.replace("__", "/"))
    return out


def list_available_layers(
    model_id: str, pool: str | None = None
) -> List[int]:
    """Return cached layer indices for `model_id`.

    If `pool` is None (default), returns layers that have *any* pool cached.
    If `pool` is given, returns only layers that have that specific pool.

    The "any-pool" default makes the function robust to partial extraction:
    if a model's `mean` pool failed to save but `char` did, we still
    surface the layers and the caller can fall back to the available pool.
    """
    d = model_dir(model_id)
    if pool is None:
        glob_pat = "layer*.npy"
    else:
        glob_pat = f"layer*_{pool}.npy"
    layers: List[int] = []
    for p in d.glob(glob_pat):
        try:
            layers.append(int(p.stem.split("_")[0].replace("layer", "")))
        except ValueError:
            continue
    return sorted(set(layers))
