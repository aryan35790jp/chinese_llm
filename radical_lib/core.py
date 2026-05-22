"""
Project-wide constants, seed control, device detection.
"""
from __future__ import annotations
import os
import random
from pathlib import Path

import numpy as np

# ── project paths ────────────────────────────────────────────────────────────
# We resolve paths relative to this file's location so the library works
# regardless of where the calling script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

for _d in (CACHE_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set every RNG used in the pipeline. Call this at the top of every script."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """Return a torch.device choosing cuda > mps > cpu. Lazy-imports torch."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
