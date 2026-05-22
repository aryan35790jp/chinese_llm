"""
dataset_builder.py — rebuild data/radical_dataset.csv from Unihan.

Improvements over the original prepare_final_dataset.py:
    - codepoint, kangxi_radical, radical_number, group_size columns
    - frequency_proxy column (Chinese-BERT vocab rank inverse-rank)
    - stroke_count from kTotalStrokes (used for confound checks)
    - liushu_class placeholder (filled later by classify_liushu.py)
    - radical_role placeholder (semantic | phonetic | unknown)
    - tokenizer-coverage filter that requires single-token encoding in
      *all* configured Chinese-text models (not just Chinese-BERT)

Output columns:
    char, codepoint, kangxi_radical, radical_number, group_size,
    frequency_proxy, stroke_count, liushu_class, radical_role

Runtime: ~2 minutes on CPU. The slow step is loading tokenizers once each.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

# Make the local package importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import DATA_DIR, set_seed
from scripts.new.config import MODELS, MIN_RADICAL_SIZE, chinese_models  # noqa: E402

set_seed()

UNIHAN_IRG = DATA_DIR / "unihan" / "Unihan_IRGSources.txt"
UNIHAN_DICT = DATA_DIR / "unihan" / "Unihan_DictionaryLikeData.txt"


# ── 1. parse Unihan ─────────────────────────────────────────────────────────
def parse_kRSUnicode() -> dict:
    """Map char → kangxi radical number from kRSUnicode (e.g. '85.5' → 85)."""
    radical_map: dict = {}
    with open(UNIHAN_IRG, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("U+") or "kRSUnicode" not in line:
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            codepoint = parts[0]
            radical_info = parts[2]
            try:
                ch = chr(int(codepoint[2:], 16))
            except ValueError:
                continue
            radical = radical_info.split(".")[0].rstrip("'")
            try:
                radical_map.setdefault(ch, int(radical))
            except ValueError:
                continue
    return radical_map


def parse_stroke_counts() -> dict:
    """Map char → total stroke count from kTotalStrokes."""
    strokes: dict = {}
    if not UNIHAN_DICT.exists():
        # kTotalStrokes lives in DictionaryLikeData but Unihan layout has shifted
        # over the years — try IRG too as a fallback.
        return strokes
    for path in (UNIHAN_DICT,):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.startswith("U+") or "kTotalStrokes" not in line:
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                try:
                    ch = chr(int(parts[0][2:], 16))
                    val = parts[2].split()[0]  # may be e.g. "9 8"
                    strokes[ch] = int(val)
                except ValueError:
                    continue
    return strokes


# Kangxi radical number → its canonical visual form (the actual radical char).
# We store this so figures can show the radical glyph rather than just an int.
KANGXI_NUM_TO_CHAR = {i + 1: chr(0x2F00 + i) for i in range(214)}


# ── 2. tokenizer coverage filter ─────────────────────────────────────────────
def is_single_token(tokenizer, ch: str) -> bool:
    """True iff `ch` encodes to exactly one token (excluding special tokens) and
    that token is not [UNK]."""
    enc = tokenizer.encode(ch, add_special_tokens=False)
    if len(enc) != 1:
        return False
    tok = tokenizer.convert_ids_to_tokens(enc[0])
    return tok not in (tokenizer.unk_token, "[UNK]", "<unk>")


def build_coverage_table(chars: list[str]) -> pd.DataFrame:
    """For every Chinese-text model, mark which chars tokenize to a single token.

    We require single-token coverage in *all* of the standard Chinese models
    (not the glyph-aware one and not Japanese — those are handled separately).
    """
    targets = [
        m for m in chinese_models()
        if m.family in {"bert", "chinese-bert", "macbert", "xlm-r", "ernie", "uer"}
    ]
    rows = {}
    for spec in targets:
        try:
            tok = AutoTokenizer.from_pretrained(
                spec.hf_id, trust_remote_code=spec.trust_remote_code
            )
        except Exception as e:
            print(f"[warn] could not load tokenizer for {spec.hf_id}: {e}")
            continue
        rows[spec.label] = [is_single_token(tok, c) for c in chars]
    return pd.DataFrame(rows, index=chars)


# ── 3. main ─────────────────────────────────────────────────────────────────
def main():
    print("Parsing Unihan kRSUnicode...")
    radical_map = parse_kRSUnicode()
    print(f"  total chars with radical info: {len(radical_map)}")

    print("Parsing Unihan kTotalStrokes...")
    strokes = parse_stroke_counts()
    print(f"  total chars with stroke counts: {len(strokes)}")

    # CJK Unified Ideographs basic block (matches the original pipeline's scope)
    candidates = [
        ch for ch, _ in radical_map.items()
        if 0x4E00 <= ord(ch) <= 0x9FFF
    ]
    print(f"  chars in CJK Unified (U+4E00–U+9FFF): {len(candidates)}")

    print("Computing tokenizer coverage across Chinese models...")
    coverage = build_coverage_table(candidates)
    if coverage.empty:
        print("[fatal] no tokenizers loaded. Check internet / cache.")
        sys.exit(1)
    keep_mask = coverage.all(axis=1)
    kept = coverage.index[keep_mask].tolist()
    print(f"  chars single-token in *all* configured Chinese models: {len(kept)}")
    coverage.to_csv(DATA_DIR / "tokenization_coverage.csv")

    # frequency proxy: rank in Chinese-BERT vocab (lower rank = more frequent)
    bert_tok = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    vocab = bert_tok.get_vocab()
    n_vocab = len(vocab)

    # Build dataframe
    rows = []
    for ch in kept:
        rad = radical_map[ch]
        rows.append({
            "char": ch,
            "codepoint": f"U+{ord(ch):04X}",
            "kangxi_radical": KANGXI_NUM_TO_CHAR.get(rad, ""),
            "radical_number": rad,
            "frequency_proxy": vocab.get(ch, n_vocab),
            "stroke_count": strokes.get(ch, -1),
            "liushu_class": "unknown",   # filled by classify_liushu.py
            "radical_role": "unknown",   # filled by classify_liushu.py
        })
    df = pd.DataFrame(rows)

    # Drop radicals with fewer than MIN_RADICAL_SIZE characters
    counts = df.groupby("radical_number").size()
    big_enough = counts[counts >= MIN_RADICAL_SIZE].index
    df = df[df["radical_number"].isin(big_enough)].copy()

    # Add group_size column
    df["group_size"] = df["radical_number"].map(df["radical_number"].value_counts())

    # Order: by radical_number then by frequency_proxy (most frequent first)
    df = df.sort_values(["radical_number", "frequency_proxy"]).reset_index(drop=True)

    # Maintain backward compatibility with old code that used 'radical' column
    df["radical"] = df["radical_number"]

    out = DATA_DIR / "radical_dataset.csv"
    df.to_csv(out, index=False)
    print(f"\nFinal dataset: {len(df)} chars, {df['radical_number'].nunique()} radicals")
    print(f"Saved to {out}")

    # summary CSV (per-radical)
    summary = df.groupby("radical_number").agg(
        kangxi_radical=("kangxi_radical", "first"),
        count=("char", "size"),
        sample_chars=("char", lambda x: "".join(x.head(5))),
    ).sort_values("count", ascending=False)
    summary.to_csv(DATA_DIR / "radical_summary.csv")
    print(f"Per-radical summary: {DATA_DIR / 'radical_summary.csv'}")


if __name__ == "__main__":
    main()
