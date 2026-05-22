"""
extract_embeddings.py — extract layer-wise character embeddings for every
configured model.

For each model and each of its hidden layers (including layer 0 = embedding
layer), we save three N×D matrices:

    layer{NN}_mean.npy   attention-mask-weighted mean over [CLS] char [SEP]
    layer{NN}_char.npy   the character-token position only (preferred for
                         single-character analysis; this is what the
                         theoretically cleanest analysis uses)
    layer{NN}_cls.npy    the [CLS] token only (for ablation)

Outputs land in cache/embeddings/{model_tag}/. Each model creates a
charlist.txt that records the row-order of every layer file for that model.

Why save all three:
    The paper's main result will use the char-position embedding because
    it isolates the character's representation, but reviewers always ask
    about [CLS] and mean-pooled, so we cache them once.

Why per-layer files:
    Loading every layer for every model into memory at once is ~25 GB.
    Per-layer files let downstream scripts stream what they need.

Runtime:
    A100: 3–4 hours total for all 11 models.
    T4 (free Colab): ~15 hours; you may want to comment out the large
        models (XLM-R-large, ChineseBERT-glyph) on the first pass.
    CPU: 30–40 hours. Don't.

RAM peak: ~16 GB (the largest single model + a layer's worth of activations).

This script is checkpointed — if a model has all layers cached already,
it is skipped. Safe to interrupt and rerun.
"""
from __future__ import annotations
import sys
import time
import traceback
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    set_seed,
    get_device,
    load_radical_dataset,
    save_layer_embeddings,
    embedding_path,
)
from radical_lib.embeddings import model_dir  # noqa: E402
from scripts.new.config import (  # noqa: E402
    INCLUDE_LAYER_0,
    LAYER_SAMPLE_COUNT_FAST,
    FAST_BATCH_SIZE,
    FULL_BATCH_SIZE,
    FAST_DTYPE,
    get_active_models,
)

set_seed()


def _is_fast_mode() -> bool:
    import os
    return os.environ.get("RADICAL_FAST", "").strip() in ("1", "true", "yes")


def _select_layers(n_layers: int) -> List[int]:
    """Pick which layers to extract.

    Full mode: every layer (0..n_layers if INCLUDE_LAYER_0 else 1..n_layers).
    Fast mode: ~5 layers evenly spaced including last and (optionally) 0.
    """
    full = list(range(0 if INCLUDE_LAYER_0 else 1, n_layers + 1))
    if not _is_fast_mode() or len(full) <= LAYER_SAMPLE_COUNT_FAST:
        return full
    # Evenly spaced indices into `full`, always including endpoints
    idx = np.linspace(0, len(full) - 1, LAYER_SAMPLE_COUNT_FAST).round().astype(int)
    return [full[i] for i in sorted(set(idx))]


def _select_dtype():
    """Pick the lowest-precision dtype the device supports for inference."""
    if not _is_fast_mode():
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    if FAST_DTYPE == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def already_cached(model_id: str, n_layers: int) -> bool:
    """Skip a model if all expected layer files exist (uses fast/full layer set)."""
    layers = _select_layers(n_layers)
    for L in layers:
        for pool in ("mean", "char", "cls"):
            if not embedding_path(model_id, L, pool).exists():
                return False
    return True


def char_token_span(input_ids: torch.Tensor, tokenizer, target_char: str) -> tuple[int, int]:
    """Return (start, end_exclusive) of the token span that encodes `target_char`.

    For tokenizers that map a single CJK char to one token, this is (1, 2)
    (after [CLS]). For SentencePiece-style tokenizers (XLM-R) a single CJK
    char can be split into multiple subwords like ['▁', '河'] or ['河', '##er'];
    in that case we return the full span so the caller can pool over it.

    Strategy:
        Skip leading special tokens. Then walk forward until we have
        consumed enough subword pieces to reconstruct `target_char`.
        Subword markers ('##', '▁', '_') are stripped before matching.
    """
    special = set(tokenizer.all_special_ids)
    ids = input_ids.tolist()
    # find first non-special
    start = None
    for i, tid in enumerate(ids):
        if tid not in special:
            start = i
            break
    if start is None:
        return 1, 2

    # walk until we've covered target_char
    decoded = ""
    end = start
    for j in range(start, len(ids)):
        if ids[j] in special:
            break
        piece = tokenizer.convert_ids_to_tokens(ids[j])
        # normalize subword markers
        clean = piece.lstrip("▁").lstrip("_").lstrip("##")
        decoded += clean
        end = j + 1
        if target_char in decoded:
            break
    return start, end


def char_token_position(input_ids: torch.Tensor, tokenizer) -> int:
    """Backward-compatible single-position helper. Prefer char_token_span."""
    span = char_token_span(input_ids, tokenizer, "")
    return span[0]


@torch.no_grad()
def extract_for_model(spec, chars: List[str], device: torch.device,
                      batch_size: int | None = None) -> None:
    """Extract and cache the configured layers × all pools for a single model.

    In RADICAL_FAST=1 mode this samples ~5 layers and uses bf16 inference,
    making the run ~12× faster than the full extraction.
    """
    if batch_size is None:
        batch_size = FAST_BATCH_SIZE if _is_fast_mode() else FULL_BATCH_SIZE
    print(f"\n=== {spec.label} ({spec.hf_id}) ===  fast={_is_fast_mode()}  bs={batch_size}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id, trust_remote_code=spec.trust_remote_code
        )
        model = AutoModel.from_pretrained(
            spec.hf_id,
            trust_remote_code=spec.trust_remote_code,
            output_hidden_states=True,
            torch_dtype=_select_dtype(),
        )
    except Exception as e:
        print(f"[skip] failed to load: {e}")
        traceback.print_exc()
        return

    model.eval()
    model.to(device)

    # Determine layer count from a tiny forward pass
    sample = tokenizer("测", return_tensors="pt").to(device)
    out = model(**sample)
    n_layers = len(out.hidden_states) - 1  # excluding embedding layer
    layer_range = _select_layers(n_layers)
    print(f"  hidden states available: {n_layers + 1}; extracting layers: {layer_range}")

    if already_cached(spec.hf_id, n_layers):
        print("  all layers cached — skipping")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return

    # Pre-allocate buffers per layer × pool
    hidden_dim = out.hidden_states[-1].shape[-1]
    n_chars = len(chars)
    buffers = {
        (L, pool): np.zeros((n_chars, hidden_dim), dtype=np.float32)
        for L in layer_range
        for pool in ("mean", "char", "cls")
    }

    # Batched forward passes
    t0 = time.time()
    for start in tqdm(range(0, n_chars, batch_size), desc=f"  {spec.label}"):
        batch = chars[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        result = model(**enc)
        hidden = result.hidden_states  # tuple of (n_layers+1) tensors B×T×D
        attn = enc["attention_mask"].unsqueeze(-1).float()  # B×T×1

        # find char-token *spans* per row in the batch — handles XLM-R's
        # SentencePiece multi-subword tokenization fairly.
        char_spans = [
            char_token_span(enc["input_ids"][b], tokenizer, batch[b])
            for b in range(enc["input_ids"].shape[0])
        ]

        for L in layer_range:
            h = hidden[L]
            # mean pool (mask-weighted)
            mean_pool = (h * attn).sum(dim=1) / attn.sum(dim=1)
            # char-position pool: average over the whole char span
            char_pool_rows = []
            for b in range(h.shape[0]):
                s, e = char_spans[b]
                char_pool_rows.append(h[b, s:e].mean(dim=0))
            char_pool = torch.stack(char_pool_rows)
            # CLS pool — position 0
            cls_pool = h[:, 0, :]

            for pool, vec in (("mean", mean_pool), ("char", char_pool), ("cls", cls_pool)):
                # Cast to fp32 before numpy — bf16 doesn't have a numpy dtype.
                buffers[(L, pool)][start:start + len(batch)] = (
                    vec.detach().to(torch.float32).cpu().numpy().astype(np.float32)
                )

    # save
    for (L, pool), arr in buffers.items():
        save_layer_embeddings(spec.hf_id, L, arr, chars, pool=pool)

    elapsed = time.time() - t0
    print(f"  cached {len(layer_range)} layers × 3 pools in {elapsed:.1f}s")
    print(f"  → {model_dir(spec.hf_id)}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    df = load_radical_dataset()
    chars = df["char"].tolist()
    print(f"Loaded {len(chars)} characters from radical_dataset.csv")

    device = get_device()
    print(f"Device: {device}, fast_mode={_is_fast_mode()}")
    if device.type == "cpu":
        print("[warn] running on CPU. This will take many hours even in fast mode.")

    for spec in get_active_models():
        try:
            extract_for_model(spec, chars, device)
        except Exception:
            print(f"[error] extraction failed for {spec.label} — continuing.")
            traceback.print_exc()


if __name__ == "__main__":
    main()
