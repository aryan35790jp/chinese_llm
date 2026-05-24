"""
config.py — single source of truth for the expanded experiment.

Every other script imports from here. If you want to add or skip a model,
edit MODELS. If you want a different sentence corpus, edit SENTENCE_CORPUS.

This file deliberately does no heavy work and has no top-level side effects
beyond defining constants, so it is cheap to import everywhere.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    label: str               # short human label used in figures
    family: str              # bert | xlm-r | macbert | ernie | uer | chinese-bert | glyph | ja-char | ja-subword
    size: str                # tiny | small | base | large
    n_layers: Optional[int] = None  # set after load if unknown
    trust_remote_code: bool = False
    notes: str = ""


# ── Models in the study ─────────────────────────────────────────────────────
# Order here is the order they appear in figures. Skip ones you don't want
# by commenting them out — every downstream script iterates this list.
#
# Sizing for free Colab T4 (16 GB VRAM, fp16):
#   <1B params  → batch 256, fits easily
#   1.5–3B      → batch 32–64, fp16, fits
#   ≥7B         → won't fit unquantized; we don't include any
MODELS: List[ModelSpec] = [
    # ── Multilingual encoders ───────────────────────────────────────────────
    ModelSpec("bert-base-multilingual-cased",      "mBERT",            "bert",          "base",  notes="multilingual baseline"),
    ModelSpec("xlm-roberta-base",                  "XLM-R-base",       "xlm-r",         "base",  notes="cross-lingual contrast"),
    ModelSpec("xlm-roberta-large",                 "XLM-R-large",      "xlm-r",         "large", notes="multilingual scaling"),
    # ── Chinese-specialized encoders ────────────────────────────────────────
    ModelSpec("hfl/chinese-bert-wwm-ext",          "Chinese-BERT",     "chinese-bert",  "base",  notes="WWM Chinese"),
    ModelSpec("hfl/chinese-macbert-base",          "MacBERT",          "macbert",       "base",  notes="MLM-as-correction"),
    ModelSpec("nghuyong/ernie-3.0-base-zh",        "ERNIE-3.0",        "ernie",         "base",  trust_remote_code=True),
    # ── UER scaling subset (Chinese, varying depth) ─────────────────────────
    ModelSpec("uer/chinese_roberta_L-4_H-512",     "UER-tiny",         "uer",           "tiny",  notes="scaling bottom"),
    ModelSpec("uer/chinese_roberta_L-8_H-512",     "UER-small",        "uer",           "small"),
    # ── Glyph-aware (skipped on transformers 5.x; ResNet baseline replaces) ─
    ModelSpec("ShannonAI/ChineseBERT-base",        "ChineseBERT-glyph","glyph",         "base",  trust_remote_code=True,
              notes="glyph-aware; may fail to load — fallback is glyph_only_baseline.py"),
    # ── Cross-script Japanese ───────────────────────────────────────────────
    ModelSpec("cl-tohoku/bert-base-japanese-char-v3", "JP-BERT-char",  "ja-char",       "base",  notes="cross-script"),
    ModelSpec("cl-tohoku/bert-base-japanese-v3",   "JP-BERT-sub",      "ja-subword",    "base",  notes="cross-script ablation"),
    # ── Modern decoder LLMs (fp16 on T4) ────────────────────────────────────
    ModelSpec("Qwen/Qwen2.5-1.5B",                 "Qwen2.5-1.5B",     "qwen",          "small", notes="modern decoder LLM, batch 64"),
    ModelSpec("Qwen/Qwen2.5-3B",                   "Qwen2.5-3B",       "qwen",          "base",  notes="modern decoder LLM, batch 32"),
    # ── Production retrieval encoder (contrastively trained) ────────────────
    ModelSpec("BAAI/bge-large-zh-v1.5",            "BGE-large-zh",     "bge",           "large", notes="task-tuned Chinese retrieval"),
]


# Convenience splits ----------------------------------------------------------
def chinese_models() -> List[ModelSpec]:
    """Models that primarily target Chinese or are multilingual including Chinese."""
    chinese_families = {"bert", "chinese-bert", "macbert", "xlm-r", "ernie", "uer",
                         "glyph", "qwen", "bge"}
    return [m for m in MODELS if m.family in chinese_families]


def japanese_models() -> List[ModelSpec]:
    return [m for m in MODELS if m.family.startswith("ja-")]


def scaling_models() -> List[ModelSpec]:
    """The scaling-trajectory subset (Chinese-only, varying size)."""
    return [
        m for m in MODELS
        if m.family in {"uer", "chinese-bert", "macbert", "xlm-r"}
        and m.label in {"UER-tiny", "UER-small", "Chinese-BERT", "MacBERT",
                        "XLM-R-base", "XLM-R-large"}
    ]


# ── Fast subset ────────────────────────────────────────────────────────────
# Eight models that cover every reviewer category. Use this when you want
# results in ~75 minutes on free Colab T4 instead of many hours for the full
# 14-model run.
#
# Reviewer categories covered:
#   - multilingual encoder: mBERT
#   - Chinese-specialized encoder: Chinese-BERT
#   - cross-lingual scaling contrast: XLM-R-base
#   - cross-script Japanese: JP-BERT-char
#   - modern decoder LLM (small): Qwen2.5-1.5B
#   - modern decoder LLM (medium): Qwen2.5-3B
#   - production retrieval encoder: BGE-large-zh-v1.5
#   - pure-vision baseline: glyph_only/resnet18 (built locally)
FAST_MODEL_LABELS = {
    "mBERT",
    "Chinese-BERT",
    "MacBERT",            # ADDED for n>=4 Chinese-specialized
    "ERNIE-3.0",          # ADDED — Chinese-specific, with knowledge prior
    "XLM-R-base",
    "JP-BERT-char",
    "Qwen2.5-1.5B",
    "Qwen2.5-3B",
    "BGE-large-zh",
    # plus glyph_only/resnet18 from glyph_only_baseline.py
}

# Full-layer override: even in fast mode, extract every layer for these models.
# Used to generate the full-resolution layer-wise figure for the paper.
LAYER_OVERRIDE_LABELS = {
    "mBERT",
    "Chinese-BERT",
}


def fast_models() -> List[ModelSpec]:
    """Return the fast-subset models for --fast mode."""
    return [m for m in MODELS if m.label in FAST_MODEL_LABELS]


def get_active_models() -> List[ModelSpec]:
    """Return MODELS or FAST subset depending on the RADICAL_FAST env var.

    Set RADICAL_FAST=1 to use the fast subset across all scripts.
    Default behaviour is the full model list.
    """
    import os
    if os.environ.get("RADICAL_FAST", "").strip() in ("1", "true", "yes"):
        return fast_models()
    return MODELS


# ── Per-model batch-size override ───────────────────────────────────────────
# Larger LLMs need smaller batches to fit in 16 GB VRAM at fp16.
def batch_size_for(spec: ModelSpec, default_fast: int = 256, default_full: int = 64) -> int:
    """Return per-model batch size considering VRAM constraints on T4."""
    # Decoder LLMs in the 1–3B range
    if spec.family == "qwen":
        if spec.size == "small":   # 1.5B
            return 64
        if spec.size == "base":    # 3B
            return 32
        return 16                  # any larger Qwen we add later
    # Production retrieval encoder
    if spec.family == "bge" and spec.size == "large":
        return 128
    # XLM-R-large
    if spec.family == "xlm-r" and spec.size == "large":
        return 128
    return default_fast


# ── Statistical defaults ────────────────────────────────────────────────────
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
N_PERMUTATIONS_SEMANTIC = 5000  # we have less data so we permute harder
ALPHA = 0.05
MIN_RADICAL_SIZE = 20
MAX_PAIRS_PER_RADICAL = 50      # subsample for the corpus-scale comparison

# ── Isotropy defaults ───────────────────────────────────────────────────────
ISOTROPY_K = 2  # number of top principal components to project out

# ── Sentential context ──────────────────────────────────────────────────────
SENTENCE_CORPUS = "wikimedia/wikipedia"
SENTENCE_CORPUS_CONFIG = "20231101.zh"
SENTENCE_TOP_N_CHARS = 500
SENTENCES_PER_CHAR = 20

# ── Layers ──────────────────────────────────────────────────────────────────
# We extract output_hidden_states, which yields embedding-layer + n_layers.
# Set INCLUDE_LAYER_0 = True to also analyze the embedding layer (useful for
# isolating the static/lookup signal vs contextual signal).
INCLUDE_LAYER_0 = True

# In FAST mode we sample N layers evenly spaced from 0..n_layers, instead of
# extracting every layer. The layer-wise figure becomes 5 points per model
# instead of 13, but the trend is preserved and the run is 4× faster.
LAYER_SAMPLE_COUNT_FAST = 5

# In FAST mode we crank up the inference batch size and use bf16/fp16 if the
# device supports it. These knobs are read by extract_embeddings.py.
FAST_BATCH_SIZE = 256
FULL_BATCH_SIZE = 64
FAST_DTYPE = "bfloat16"   # falls back to float16 if device doesn't support bf16
