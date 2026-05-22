"""
radical_lib — shared utilities for the Radical-Aligned Structure project.

This package centralizes everything that is used by more than one script:
    - device + seed management
    - embedding loaders (cache-aware)
    - statistics (Cohen's d, bootstrap CI, permutation, Welch, RSA)
    - isotropy correction (Mu & Viswanath; mean-centering + std + top-k PC removal)
    - plotting style
    - dataset accessors

Designed so any script can do `from radical_lib import ...` and get
consistent, identically-seeded behavior across the whole pipeline.
"""

from .core import (
    set_seed,
    get_device,
    PROJECT_ROOT,
    CACHE_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    DATA_DIR,
)
from .data import (
    load_radical_dataset,
    radical_groups,
    char_to_radical,
    char_index,
)
from .embeddings import (
    embedding_path,
    save_layer_embeddings,
    load_layer_embeddings,
    list_available_models,
    list_available_layers,
)
from .stats import (
    cohens_d,
    welch_t,
    bootstrap_ci_diff,
    permutation_test_diff,
    rsa_spearman,
    holm_bonferroni,
)
from .isotropy import (
    fit_isotropy,
    apply_isotropy,
    cosine_isotropic,
)
from .plotting import (
    setup_publication_style,
    save_fig,
)

__all__ = [
    "set_seed",
    "get_device",
    "PROJECT_ROOT",
    "CACHE_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "DATA_DIR",
    "load_radical_dataset",
    "radical_groups",
    "char_to_radical",
    "char_index",
    "embedding_path",
    "save_layer_embeddings",
    "load_layer_embeddings",
    "list_available_models",
    "list_available_layers",
    "cohens_d",
    "welch_t",
    "bootstrap_ci_diff",
    "permutation_test_diff",
    "rsa_spearman",
    "holm_bonferroni",
    "fit_isotropy",
    "apply_isotropy",
    "cosine_isotropic",
    "setup_publication_style",
    "save_fig",
]
