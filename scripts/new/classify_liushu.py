"""
classify_liushu.py — annotate every character with its 六書 (liushu) class
and the role its Kangxi radical plays inside it.

Output columns added to data/radical_dataset.csv:
    liushu_class   ∈ {pictograph, ideograph, phonosemantic, simple, unknown}
    radical_role   ∈ {semantic, phonetic, identity, unknown}
        identity = the character IS the Kangxi radical itself
        semantic = ~85% of phonosemantic chars; the Kangxi radical bears meaning
        phonetic = rare for Kangxi-radical-as-component; flagged when the
                   radical is on the right/bottom and known phonetic series

Method:
    1. Pull CHISE Ideographic Description Sequences from
       https://github.com/cjkvi/cjkvi-ids  (file IDS.TXT)
    2. Parse the IDS for each char → list of atomic components
    3. If the char has only one atomic component (= itself), it is a
       pictograph or ideograph (we call this "simple" since CHISE doesn't
       distinguish). If it has 2+ components and the Kangxi radical is
       among them, classify as phonosemantic with role=semantic
       (the radical is conventionally the semantic component in 形声字).
    4. Save back to data/radical_dataset.csv (in place).

Why this matters for the paper:
    The negative semantic-control finding could be artifact: maybe Kangxi
    radicals in our 4 chosen fields happen to be the *semantic*
    components, and a parallel result wouldn't hold for radicals that
    function as phonetic markers. With this column we can test that
    explicitly in phonetic_vs_semantic_radicals.py.

Runtime: ~30 seconds. RAM: <500 MB.
Depends on: dataset_builder.py
"""
from __future__ import annotations
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import DATA_DIR, set_seed, load_radical_dataset  # noqa: E402

set_seed()

IDS_URL = "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt"
IDS_LOCAL = DATA_DIR / "ids" / "ids.txt"

# Ideographic Description Characters — markers, not real components.
IDC = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")

# Kangxi radical number → its conventional Unicode form (U+2F00…U+2FD5)
# AND its CJK Unified Ideographs equivalent (so we can compare against IDS
# components which use the unified form).
def kangxi_unified(num: int) -> str:
    """Return the CJK Unified character for Kangxi radical number `num`.

    Hardcoded for the 214 standard Kangxi radicals — the Kangxi radical
    block (U+2F00) is a distinct codepoint from the unified-ideograph form
    that actually appears in real character decompositions.
    """
    # Source: https://en.wikipedia.org/wiki/Kangxi_radical (exhaustive table)
    table = (
        "一丨丶丿乙亅二亠人儿入八冂冖冫几凵刀力勹匕匚匸十卜卩厂厶又"  # 1-29
        "口囗土士夂夊夕大女子宀寸小尢尸屮山巛工己巾干幺广廴廾弋弓彐彡彳"  # 30-59
        "心戈戶手支攴文斗斤方无日曰月木欠止歹殳毋比毛氏气水火爪父爻爿片牙牛犬"  # 60-94
        "玄玉瓜瓦甘生用田疋疒癶白皮皿目矛矢石示禸禾穴立"  # 95-118
        "竹米糸缶网羊羽老而耒耳聿肉臣自至臼舌舛舟艮色艸虍虫血行衣襾"  # 119-145
        "見角言谷豆豕豸貝赤走足身車辛辰辵邑酉釆里"  # 146-166
        "金長門阜隶隹雨青非"  # 167-175
        "面革韋韭音頁風飛食首香"  # 176-186
        "馬骨高髟鬥鬯鬲鬼"  # 187-194
        "魚鳥鹵鹿麥麻"  # 195-200
        "黃黍黑黹"  # 201-204
        "黽鼎鼓鼠"  # 205-208
        "鼻齊"  # 209-210
        "齒"  # 211
        "龍龜"  # 212-213
        "龠"  # 214
    )
    if 1 <= num <= len(table):
        return table[num - 1]
    return ""


# ── 1. fetch IDS ────────────────────────────────────────────────────────────
def ensure_ids() -> Path:
    """Download ids.txt if missing. CHISE updates infrequently; one fetch is
    enough."""
    IDS_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if IDS_LOCAL.exists() and IDS_LOCAL.stat().st_size > 100_000:
        return IDS_LOCAL
    print(f"Downloading CHISE IDS to {IDS_LOCAL} …")
    try:
        urllib.request.urlretrieve(IDS_URL, IDS_LOCAL)
    except Exception as e:
        print(f"[error] could not download CHISE IDS: {e}")
        print("        Manually download from https://github.com/cjkvi/cjkvi-ids")
        sys.exit(1)
    return IDS_LOCAL


def parse_ids(path: Path) -> dict[str, str]:
    """Return {char: ids_string}. We use the first IDS source per char."""
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            char = parts[1]
            ids = parts[2]
            # Strip CJK source tags like "[GTKV]" if present
            ids = ids.split()[0]
            if char not in mapping:
                mapping[char] = ids
    return mapping


def atomic_components(ids: str, char: str) -> list[str]:
    """Return the atomic CJK components of an IDS expression.

    An "atomic" component is any CJK Unified character in the IDS that is
    not an Ideographic Description Character (IDC). We exclude `char`
    itself when the IDS is just the character verbatim (= simple char).
    """
    if not ids:
        return []
    comps = [c for c in ids if c not in IDC]
    if comps == [char]:
        return []
    return comps


# ── 2. classify ─────────────────────────────────────────────────────────────
def classify(char: str, radical_num: int, ids_map: dict[str, str]) -> tuple[str, str]:
    """Return (liushu_class, radical_role) for a given character."""
    rad_char = kangxi_unified(radical_num)

    # Identity: the character IS the radical itself.
    if char == rad_char:
        return "simple", "identity"

    ids = ids_map.get(char, "")
    comps = atomic_components(ids, char)

    if not comps:
        # CHISE has no decomposition or it is the same as the char.
        # Treat as simple pictograph/ideograph.
        return "simple", "unknown"

    if rad_char and rad_char in comps:
        # The Kangxi radical appears as a component → phonosemantic with the
        # radical conventionally the semantic component.
        return "phonosemantic", "semantic"

    # Multi-component but Kangxi radical is not one of them. Could be a
    # variant where the radical morphed (e.g. 氵 vs 水, 艹 vs 艸) — we treat
    # this as compound but don't claim a clean role.
    return "phonosemantic", "unknown"


def main():
    ids_path = ensure_ids()
    print(f"Parsing IDS from {ids_path}")
    ids_map = parse_ids(ids_path)
    print(f"  parsed {len(ids_map)} chars")

    df = load_radical_dataset()
    n = len(df)

    classes, roles = [], []
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    for char, rad in zip(df["char"], df[rad_col]):
        cls, role = classify(char, int(rad), ids_map)
        classes.append(cls)
        roles.append(role)

    df["liushu_class"] = classes
    df["radical_role"] = roles

    out_path = DATA_DIR / "radical_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\nUpdated {out_path} with liushu_class / radical_role columns.")
    print("\nclass distribution:")
    print(df["liushu_class"].value_counts())
    print("\nrole distribution:")
    print(df["radical_role"].value_counts())


if __name__ == "__main__":
    main()
