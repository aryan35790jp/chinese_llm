"""
tokenization_audit.py — produce the per-model tokenization coverage table
for the appendix.

This is a fast sanity-check script: for every model in config.MODELS and
every character in radical_dataset.csv, record whether the character
encodes to a single, non-[UNK] token. Outputs:

    results/tokenization_audit.csv      one row per (model, char)
    results/tokenization_audit_summary.csv   one row per model

Runtime: ~3 minutes on CPU. RAM: <2 GB.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import RESULTS_DIR, load_radical_dataset, set_seed
from scripts.new.config import get_active_models  # noqa: E402

set_seed()


def main():
    df = load_radical_dataset()
    chars = df["char"].tolist()

    rows = []
    for spec in get_active_models():
        print(f"[audit] {spec.label}  ({spec.hf_id})")
        try:
            tok = AutoTokenizer.from_pretrained(
                spec.hf_id, trust_remote_code=spec.trust_remote_code
            )
        except Exception as e:
            print(f"  [skip] tokenizer load failed: {e}")
            continue
        unk = tok.unk_token
        for c in chars:
            ids = tok.encode(c, add_special_tokens=False)
            n = len(ids)
            tok_str = tok.convert_ids_to_tokens(ids[0]) if n >= 1 else ""
            rows.append({
                "model": spec.label,
                "char": c,
                "n_tokens": n,
                "first_token": tok_str,
                "is_single": int(n == 1 and tok_str != unk),
                "is_unk": int(n >= 1 and tok_str == unk),
            })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "tokenization_audit.csv", index=False)

    summary = out.groupby("model").agg(
        n_chars=("char", "size"),
        n_single_token=("is_single", "sum"),
        n_unk=("is_unk", "sum"),
        n_multitok=("n_tokens", lambda s: int((s > 1).sum())),
    )
    summary["coverage"] = summary["n_single_token"] / summary["n_chars"]
    summary.to_csv(RESULTS_DIR / "tokenization_audit_summary.csv")
    print("\n=== summary ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
