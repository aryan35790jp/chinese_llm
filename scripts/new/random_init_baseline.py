"""
random_init_baseline.py — what's the radical signal in an *untrained* model?

Reviewer question this answers:
    "Is your radical signal a property of *training* on language, or
    just a property of the architecture / random init?"

Method:
    1. For one or two representative architectures (Chinese-BERT and
       XLM-R-base), instantiate the model with random weights (no
       pre-training).
    2. Pass each character through and extract the same char-pool
       embeddings from the last layer.
    3. Run the same intra-vs-inter cohesion test as layer_wise_analysis.
    4. Report Cohen's d as the noise floor.

Expected:
    d ≈ 0 (no signal). Anything else means the architecture is
    surprisingly informative even before training, which would be a
    finding in itself.

Output:
    results/random_init_baseline.csv
        rows = (model, layer, d_random_init, p_perm, intra_mean, inter_mean)

Runtime: ~10 min. CPU OK; GPU faster.
Depends on: data/radical_dataset.csv only (no cached embeddings; we
            instantiate fresh random-weight models).
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
    RESULTS_DIR,
    set_seed,
    get_device,
    load_radical_dataset,
    cohens_d,
    permutation_test_diff,
    fit_isotropy,
    apply_isotropy,
)
from scripts.new.config import (  # noqa: E402
    MAX_PAIRS_PER_RADICAL,
    N_PERMUTATIONS,
    ISOTROPY_K,
)

set_seed()

# We probe two architectures. We use AutoConfig+AutoModel(config) which
# constructs the network with random weights but reuses the tokenizer
# from the trained release (so the input encoding is comparable).
TARGET_ARCHITECTURES = [
    ("hfl/chinese-bert-wwm-ext", "Chinese-BERT (random init)"),
    ("xlm-roberta-base",          "XLM-R-base (random init)"),
]


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return Xn @ Xn.T


def sample_pairs(
    chars: List[str], char_idx: Dict[str, int],
    char_to_radical: Dict[str, int], sim: np.ndarray
):
    groups: Dict[int, List[str]] = {}
    for c in chars:
        groups.setdefault(char_to_radical[c], []).append(c)
    groups = {r: cs for r, cs in groups.items() if len(cs) >= 20}
    intra: List[float] = []
    inter: List[float] = []
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


def main():
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    df = load_radical_dataset()
    chars = df["char"].tolist()
    char_idx = {c: i for i, c in enumerate(chars)}
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col].astype(int)))

    device = get_device()
    rows = []
    for hf_id, label in TARGET_ARCHITECTURES:
        print(f"\n=== {label} ({hf_id}) ===")
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            config = AutoConfig.from_pretrained(hf_id)
            config.output_hidden_states = True
            # Construct model with random weights — DOES NOT load pretrained
            torch.manual_seed(42)
            model = AutoModel.from_config(config).to(device).eval()
        except Exception as e:
            print(f"  [skip] could not instantiate: {e}")
            continue

        # extract last-layer char-pool embeddings
        D = config.hidden_size
        feats = np.zeros((len(chars), D), dtype=np.float32)
        with torch.no_grad():
            for i in tqdm(range(0, len(chars), 32), desc=label):
                batch = chars[i:i + 32]
                enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
                out = model(**enc)
                hidden = out.hidden_states[-1]
                # mask-weighted mean pool
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                feats[i:i + len(batch)] = pooled.cpu().numpy().astype(np.float32)

        # isotropy correction (so the noise floor is on the same scale)
        params = fit_isotropy(feats, k=ISOTROPY_K)
        feats_iso = apply_isotropy(feats, params)
        sim = cosine_matrix(feats_iso)

        intra, inter = sample_pairs(chars, char_idx, char_to_radical, sim)
        d = cohens_d(intra, inter)
        p, _, _ = permutation_test_diff(intra, inter, n_perm=N_PERMUTATIONS,
                                         rng=np.random.default_rng(42))
        rows.append({
            "model": label,
            "model_id": hf_id,
            "layer": int(config.num_hidden_layers),
            "d_random_init": float(d),
            "p_perm": float(p),
            "intra_mean": float(intra.mean()),
            "inter_mean": float(inter.mean()),
            "n_intra": int(len(intra)),
            "n_inter": int(len(inter)),
        })
        print(f"  d_random_init = {d:.4f}, p_perm = {p:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "random_init_baseline.csv", index=False)
    print(f"\nWrote {len(out)} rows.")


if __name__ == "__main__":
    main()
