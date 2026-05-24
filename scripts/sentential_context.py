"""
sentential_context.py — radical cohesion in real sentence context.

The original paper extracts embeddings for each character in isolation
(`[CLS] char [SEP]`). That's the "default" representation but it doesn't
say anything about how characters cluster *as used in real text*.

This script:
    1. Picks the top N most-frequent characters in our dataset
       (most reliably encoded as single tokens).
    2. For each, draws K random sentences containing that character
       from a Wikipedia zh sample.
    3. Runs each sentence through every Chinese model and extracts the
       hidden state at *the position of the target character* in each
       layer.
    4. Averages over K occurrences → one in-context embedding per char
       per (model, layer).
    5. Reruns the same intra-vs-inter radical cohesion test on these
       in-context embeddings, and reports d compared to the isolated-
       character baseline.

We expect:
    - The intra-radical effect to *shrink* in context (chars in
      sentences are more individuated by their neighbors than by their
      script structure)
    - Or to disappear entirely, which would indicate the isolated-char
      effect was an artifact of single-token preprocessing

Output:
    cache/sentences.json                          gathered sentence pool
    cache/sentential/{model_tag}/layer{L}.npy     per-layer in-context embeddings
    results/sentential_cohesion.csv
        rows = (model, layer, mode, cohens_d, p_perm)
        mode ∈ {isolated, in_context}

Heads-up:
    Streams Wikipedia zh on first run. ~30 min just to gather sentences.
    Per-model extraction is ~10 min on GPU.

Runtime: ~90 min total for all Chinese models. Colab Pro recommended.
RAM: 16 GB during sentence-batched forward.
Depends on: extract_embeddings.py (for the isolated baseline)
"""
from __future__ import annotations
import itertools
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    CACHE_DIR,
    RESULTS_DIR,
    set_seed,
    get_device,
    load_radical_dataset,
    cohens_d,
    permutation_test_diff,
    load_layer_embeddings,
)
from radical_lib.embeddings import model_tag  # noqa: E402
from scripts.new.config import (  # noqa: E402
    chinese_models,
    SENTENCE_CORPUS,
    SENTENCE_CORPUS_CONFIG,
    SENTENCE_TOP_N_CHARS,
    SENTENCES_PER_CHAR,
    MAX_PAIRS_PER_RADICAL,
    N_PERMUTATIONS,
)

set_seed()
SENT_CACHE = CACHE_DIR / "sentences.json"
SENTENTIAL_CACHE = CACHE_DIR / "sentential"
SENTENTIAL_CACHE.mkdir(parents=True, exist_ok=True)


# ── 1. sentence pool ────────────────────────────────────────────────────────
def gather_sentences(target_chars: set[str], k_per_char: int) -> Dict[str, List[str]]:
    """Stream Wikipedia zh until each target char has `k_per_char` sentences.
    Caches to disk so subsequent runs reuse the pool."""
    if SENT_CACHE.exists():
        import json
        cached = json.loads(SENT_CACHE.read_text(encoding="utf-8"))
        if all(len(cached.get(c, [])) >= k_per_char for c in target_chars):
            print(f"Using cached sentences from {SENT_CACHE}")
            return {c: cached[c][:k_per_char] for c in target_chars}

    try:
        from datasets import load_dataset
    except ImportError:
        print("[fatal] `datasets` not installed.")
        return {}

    print(f"Streaming {SENTENCE_CORPUS}/{SENTENCE_CORPUS_CONFIG} …")
    try:
        ds = load_dataset(
            SENTENCE_CORPUS, SENTENCE_CORPUS_CONFIG, streaming=True, split="train"
        )
    except Exception as e:
        print(f"[fatal] could not stream: {e}")
        return {}

    bucket: Dict[str, List[str]] = {c: [] for c in target_chars}
    seen = 0
    for row in ds:
        text = row.get("text", "")
        for s in text.split("\n"):
            s = s.strip()
            if not (10 <= len(s) <= 100):
                continue
            seen += 1
            for c in target_chars:
                if c in s and len(bucket[c]) < k_per_char:
                    bucket[c].append(s)
            if seen % 50_000 == 0:
                done = sum(1 for c in target_chars if len(bucket[c]) >= k_per_char)
                print(f"  scanned {seen} sentences, {done}/{len(target_chars)} chars done")
            if all(len(bucket[c]) >= k_per_char for c in target_chars):
                break
        if all(len(bucket[c]) >= k_per_char for c in target_chars):
            break

    import json
    SENT_CACHE.write_text(json.dumps(bucket, ensure_ascii=False), encoding="utf-8")
    print(f"Cached → {SENT_CACHE}")
    return bucket


# ── 2. in-context extraction ────────────────────────────────────────────────
def extract_in_context(spec, sentences: Dict[str, List[str]], chars: List[str]) -> Dict[int, np.ndarray]:
    """Return {layer: N×D in-context-mean embedding}. Rows for chars without
    sentences remain at zero. Cached to disk per model.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    cache_dir = SENTENTIAL_CACHE / model_tag(spec.hf_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    layers_cached = sorted(int(p.stem.replace("layer", ""))
                           for p in cache_dir.glob("layer*.npy"))
    if layers_cached:
        print(f"  using cached layers {layers_cached}")
        return {L: np.load(cache_dir / f"layer{L:02d}.npy") for L in layers_cached}

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=spec.trust_remote_code
    )
    model = AutoModel.from_pretrained(
        spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
        output_hidden_states=True,
    ).to(device).eval()

    char_idx = {c: i for i, c in enumerate(chars)}
    n = len(chars)

    with torch.no_grad():
        sample = tokenizer("测试", return_tensors="pt").to(device)
        out = model(**sample)
        n_layers = len(out.hidden_states)
        D = out.hidden_states[-1].shape[-1]

    accum = {L: np.zeros((n, D), dtype=np.float64) for L in range(n_layers)}
    counts = np.zeros(n, dtype=np.int32)

    with torch.no_grad():
        for c, sents in tqdm(sentences.items(), desc=spec.label):
            if c not in char_idx:
                continue
            row = char_idx[c]
            for s in sents:
                if c not in s:
                    continue
                enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=128).to(device)
                out = model(**enc)
                input_ids = enc["input_ids"][0]
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                # find first token equal to (or ending in) c
                positions = [t for t, tok in enumerate(tokens) if tok == c or tok.endswith(c)]
                if not positions:
                    continue
                pos = positions[0]
                for L, h in enumerate(out.hidden_states):
                    accum[L][row] += h[0, pos].detach().cpu().numpy()
                counts[row] += 1

    nonzero_mask = counts > 0
    for L in range(n_layers):
        accum[L][nonzero_mask] /= counts[nonzero_mask][:, None]

    for L in range(n_layers):
        np.save(cache_dir / f"layer{L:02d}.npy", accum[L].astype(np.float32))
    return {L: accum[L].astype(np.float32) for L in range(n_layers)}


# ── 3. cohesion test for in-context vs isolated ─────────────────────────────
def cohesion(emb: np.ndarray, char_to_radical: Dict[str, int],
             chars: List[str], char_idx: Dict[str, int]) -> tuple[float, float]:
    """Sample intra/inter pairs (mirroring layer_wise_analysis) and return
    (cohens_d, p_perm). Returns NaNs if too few valid chars."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    valid = (norms.flatten() > 0)
    if valid.sum() < 30:
        return float("nan"), float("nan")

    Xn = emb / np.maximum(norms, 1e-12)
    sim_full = Xn @ Xn.T

    groups: Dict[int, List[str]] = {}
    for c, i in char_idx.items():
        if not valid[i]:
            continue
        groups.setdefault(char_to_radical[c], []).append(c)
    groups = {r: cs for r, cs in groups.items() if len(cs) >= 5}
    if not groups:
        return float("nan"), float("nan")

    intra: List[float] = []
    inter: List[float] = []
    rng = random.Random(42)
    chars_valid = [c for c in chars if valid[char_idx[c]]]
    for rad, rad_chars in groups.items():
        pairs = list(itertools.combinations(rad_chars, 2))
        if len(pairs) > MAX_PAIRS_PER_RADICAL:
            pairs = rng.sample(pairs, MAX_PAIRS_PER_RADICAL)
        for a, b in pairs:
            intra.append(sim_full[char_idx[a], char_idx[b]])
        others = [c for c in chars_valid if char_to_radical[c] != rad]
        for _ in range(MAX_PAIRS_PER_RADICAL):
            a = rng.choice(rad_chars)
            b = rng.choice(others)
            inter.append(sim_full[char_idx[a], char_idx[b]])

    if len(intra) < 10 or len(inter) < 10:
        return float("nan"), float("nan")
    a = np.array(intra)
    b = np.array(inter)
    d = cohens_d(a, b)
    p, _, _ = permutation_test_diff(a, b, n_perm=N_PERMUTATIONS,
                                     rng=np.random.default_rng(42))
    return d, p


def main():
    df = load_radical_dataset()
    chars = df["char"].tolist()
    char_idx = {c: i for i, c in enumerate(chars)}
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col].astype(int)))

    # Pick top-N most-frequent chars for in-context analysis
    if "frequency_proxy" in df.columns:
        top_chars = df.sort_values("frequency_proxy")["char"].head(SENTENCE_TOP_N_CHARS).tolist()
    else:
        top_chars = chars[:SENTENCE_TOP_N_CHARS]

    target_set = set(top_chars)
    sentences = gather_sentences(target_set, SENTENCES_PER_CHAR)
    if not sentences:
        print("[fatal] no sentences gathered.")
        sys.exit(1)

    rows = []
    for spec in chinese_models():
        print(f"\n=== {spec.label} ({spec.hf_id}) ===")
        try:
            in_ctx = extract_in_context(spec, sentences, chars)
        except Exception as e:
            print(f"  [skip] {e}")
            continue

        for L in sorted(in_ctx.keys()):
            d_ctx, p_ctx = cohesion(in_ctx[L], char_to_radical, chars, char_idx)
            rows.append({
                "model": spec.hf_id, "layer": L, "mode": "in_context",
                "cohens_d": d_ctx, "p_perm": p_ctx,
            })
            try:
                iso = load_layer_embeddings(spec.hf_id, L, pool="char")
                d_iso, p_iso = cohesion(iso, char_to_radical, chars, char_idx)
                rows.append({
                    "model": spec.hf_id, "layer": L, "mode": "isolated",
                    "cohens_d": d_iso, "p_perm": p_iso,
                })
            except FileNotFoundError:
                pass

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "sentential_cohesion.csv", index=False)
    print(f"\nWrote {len(out)} rows.")


if __name__ == "__main__":
    main()
