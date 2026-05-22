"""
Dataset accessors. One canonical view of `data/radical_dataset.csv`.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, List

import pandas as pd

from .core import DATA_DIR


@lru_cache(maxsize=1)
def load_radical_dataset() -> pd.DataFrame:
    """Load the canonical character-radical dataframe.

    Columns expected:
        char, radical, codepoint, kangxi_radical, radical_number,
        group_size, frequency_proxy, stroke_count, liushu_class,
        radical_role  (semantic|phonetic|unknown)

    Falls back gracefully if newer columns are missing — older versions
    of the CSV from the original pipeline still load.
    """
    df = pd.read_csv(DATA_DIR / "radical_dataset.csv")
    # Normalize: the original file had only `char, radical`.
    if "kangxi_radical" not in df.columns and "radical" in df.columns:
        df["kangxi_radical"] = df["radical"]
    return df


@lru_cache(maxsize=1)
def char_to_radical() -> Dict[str, int]:
    df = load_radical_dataset()
    return dict(zip(df["char"], df["radical"]))


@lru_cache(maxsize=1)
def char_index() -> Dict[str, int]:
    """Return {char: row_index_in_dataset}. This is the canonical row order."""
    df = load_radical_dataset()
    return {c: i for i, c in enumerate(df["char"].tolist())}


@lru_cache(maxsize=1)
def radical_groups(min_size: int = 20) -> Dict[int, List[str]]:
    """Return {radical_number: [chars]} for radicals with at least `min_size` members."""
    df = load_radical_dataset()
    groups: Dict[int, List[str]] = {}
    for char, rad in zip(df["char"], df["radical"]):
        groups.setdefault(int(rad), []).append(char)
    return {r: cs for r, cs in groups.items() if len(cs) >= min_size}
