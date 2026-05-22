"""
cross_script_japanese.py — cross-script replication on Japanese kanji.

If the radical-cohesion effect is real and is mediated by semantics
(rather than orthography), then the same effect should appear in *Japanese*
text models scoring *Japanese* kanji, since Japanese inherits the same
Kangxi radical system but uses the characters in a different
distributional environment.

If the effect is *stronger* in Japanese, that's interesting (different
distributional channel still preserves the radical–meaning correlation).
If *weaker*, that suggests Chinese-specific co-occurrence statistics are
doing some of the work.

Pipeline:
    1. Build a kanji dataset:
        - intersect 6,306 chars with the Japanese Joyo+JIS-208 union
        - require single-token in both Japanese tokenizers we use
    2. Reuse the cached layers from extract_embeddings.py for the
       Japanese models. (The chars overlap with the Chinese set, so
       the embedding rows are already present; we just subset rows.)
    3. Run the same intra/inter cohesion test on this subset.
    4. Compare: Chinese embedding on these same chars vs Japanese.

Output:
    results/cross_script_japanese.csv
        rows = (model_id, language, n_chars, intra_mean, inter_mean,
                delta, cohens_d, p_perm)

Depends on: extract_embeddings.py
"""
from __future__ import annotations
import itertools
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    CACHE_DIR,
    RESULTS_DIR,
    set_seed,
    load_radical_dataset,
    list_available_layers,
    cohens_d,
    permutation_test_diff,
)
from radical_lib.embeddings import model_tag  # noqa: E402
from scripts.new.config import (  # noqa: E402
    MAX_PAIRS_PER_RADICAL,
    N_PERMUTATIONS,
    japanese_models,
    chinese_models,
)

set_seed()
ISO_DIR = CACHE_DIR / "embeddings_iso"


def load_iso_last_char(model_id: str) -> Tuple[np.ndarray, int]:
    layers = list_available_layers(model_id)
    if not layers:
        raise FileNotFoundError(model_id)
    L = max(layers)
    path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_char.npy"
    if not path.exists():
        path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_mean.npy"
    return np.load(path), L


JOYO_LOCAL = Path(__file__).resolve().parents[2] / "data" / "joyo_kanji.txt"
JOYO_URL = "https://raw.githubusercontent.com/fasiha/joyo/master/joyo.txt"


def joyo_kanji_set() -> set[str]:
    """Load the 2,136 Joyo kanji from a local file; download once on first run.

    Reviewer note: the previous fallback used "every char the Japanese
    char-tokenizer accepts as a single token", which lets through almost
    all 6,306 chars and makes the cross-script comparison meaningless
    (we'd be testing the *same* chars under two tokenizers, not a
    proper Japanese-restricted set). This stricter filter scopes the
    comparison to a real Joyo subset.
    """
    if JOYO_LOCAL.exists() and JOYO_LOCAL.stat().st_size > 1000:
        text = JOYO_LOCAL.read_text(encoding="utf-8")
        return set(c for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
    try:
        import urllib.request
        print(f"Downloading Joyo kanji list to {JOYO_LOCAL} …")
        JOYO_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(JOYO_URL, JOYO_LOCAL)
        text = JOYO_LOCAL.read_text(encoding="utf-8")
        return set(c for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
    except Exception as e:
        print(f"[warn] could not download Joyo list: {e}")
        return set()


def japanese_chars_via_tokenizer(chars: List[str]) -> set[str]:
    """Return chars that the Japanese char-tokenizer treats as a single token."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char-v3")
    except Exception as e:
        print(f"[warn] couldn't load JP tokenizer: {e}")
        return set()
    out = set()
    for c in chars:
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1 and tok.convert_ids_to_tokens(ids[0]) not in (tok.unk_token, "[UNK]"):
            out.add(c)
    return out


def sample_intra_inter(
    chars: List[str],
    char_idx: Dict[str, int],
    char_to_radical: Dict[str, int],
    sim: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    groups: Dict[int, List[str]] = {}
    for c in chars:
        groups.setdefault(char_to_radical[c], []).append(c)
    groups = {r: cs for r, cs in groups.items() if len(cs) >= 5}
    intra, inter = [], []
    rng = random.Random(42)
    for rad, rad_chars in groups.items():
        pairs = list(itertools.combinations(rad_chars, 2))
        if len(pairs) > MAX_PAIRS_PER_RADICAL:
            pairs = rng.sample(pairs, MAX_PAIRS_PER_RADICAL)
        for a, b in pairs:
            intra.append(sim[char_idx[a], char_idx[b]])
        others = [c for c in chars if char_to_radical[c] != rad]
        for _ in range(MAX_PAIRS_PER_RADICAL):
            a = rng.choice(rad_chars)
            b = rng.choice(others)
            inter.append(sim[char_idx[a], char_idx[b]])
    return np.array(intra), np.array(inter)


def cosine_block(X: np.ndarray, idx: List[int]) -> np.ndarray:
    Xs = X[idx]
    Xs = Xs / np.maximum(np.linalg.norm(Xs, axis=1, keepdims=True), 1e-12)
    return Xs @ Xs.T


def main():
    df = load_radical_dataset()
    chars = df["char"].tolist()
    char_idx_global = {c: i for i, c in enumerate(chars)}
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col]))

    # Identify the kanji subset we care about. Prefer Joyo (real
    # Japanese-restricted set); fall back to tokenizer coverage with a
    # loud warning, since that fallback weakens the cross-script claim.
    kanji = joyo_kanji_set()
    if not kanji:
        print("[warn] no Joyo list — falling back to tokenizer coverage. "
              "This weakens the cross-script comparison; install or download "
              "data/joyo_kanji.txt for a stricter subset.")
        kanji = japanese_chars_via_tokenizer(chars)
    if not kanji:
        print("[fatal] could not assemble a Japanese kanji subset.")
        sys.exit(1)
    kanji_in_dataset = [c for c in chars if c in kanji]
    print(f"Japanese-relevant kanji in dataset: {len(kanji_in_dataset)} "
          f"(of {len(kanji)} Joyo)")

    rows = []

    # 1. Each Japanese model on the kanji subset
    for spec in japanese_models():
        try:
            X, L = load_iso_last_char(spec.hf_id)
        except FileNotFoundError:
            print(f"[skip] {spec.label}: no embeddings cached")
            continue
        # subset rows to kanji
        local_idx = {c: char_idx_global[c] for c in kanji_in_dataset}
        sub_global_idx = list(local_idx.values())
        Xs = X[sub_global_idx]
        # build a local sim matrix
        Xs_n = Xs / np.maximum(np.linalg.norm(Xs, axis=1, keepdims=True), 1e-12)
        sim_local = Xs_n @ Xs_n.T
        local_char_idx = {c: i for i, c in enumerate(kanji_in_dataset)}
        intra, inter = sample_intra_inter(
            kanji_in_dataset, local_char_idx, char_to_radical, sim_local
        )
        if len(intra) < 20:
            continue
        d = cohens_d(intra, inter)
        p, _, _ = permutation_test_diff(
            intra, inter, n_perm=N_PERMUTATIONS, rng=np.random.default_rng(42)
        )
        rows.append({
            "model": spec.label, "model_id": spec.hf_id,
            "language": "ja", "n_chars": len(kanji_in_dataset),
            "n_intra": len(intra), "n_inter": len(inter),
            "intra_mean": float(intra.mean()), "inter_mean": float(inter.mean()),
            "delta": float(intra.mean() - inter.mean()),
            "cohens_d": d, "p_perm": p,
        })

    # 2. Each Chinese model on the *same* kanji subset for direct comparison
    for spec in chinese_models():
        try:
            X, L = load_iso_last_char(spec.hf_id)
        except FileNotFoundError:
            continue
        sub_global_idx = [char_idx_global[c] for c in kanji_in_dataset]
        Xs = X[sub_global_idx]
        Xs_n = Xs / np.maximum(np.linalg.norm(Xs, axis=1, keepdims=True), 1e-12)
        sim_local = Xs_n @ Xs_n.T
        local_char_idx = {c: i for i, c in enumerate(kanji_in_dataset)}
        intra, inter = sample_intra_inter(
            kanji_in_dataset, local_char_idx, char_to_radical, sim_local
        )
        if len(intra) < 20:
            continue
        d = cohens_d(intra, inter)
        p, _, _ = permutation_test_diff(
            intra, inter, n_perm=N_PERMUTATIONS, rng=np.random.default_rng(42)
        )
        rows.append({
            "model": spec.label, "model_id": spec.hf_id,
            "language": "zh-on-kanji-subset", "n_chars": len(kanji_in_dataset),
            "n_intra": len(intra), "n_inter": len(inter),
            "intra_mean": float(intra.mean()), "inter_mean": float(inter.mean()),
            "delta": float(intra.mean() - inter.mean()),
            "cohens_d": d, "p_perm": p,
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "cross_script_japanese.csv", index=False)
    print(f"\nWrote {len(out)} rows.")
    if not out.empty:
        print(out[["model", "language", "cohens_d", "p_perm"]].to_string(index=False))


if __name__ == "__main__":
    main()
