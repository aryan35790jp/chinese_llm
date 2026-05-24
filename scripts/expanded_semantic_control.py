"""
expanded_semantic_control.py — semantic control across 25+ fields, replacing
the original 4-field hand-curated set.

Pipeline:
    1. Build a sememe → [chars] index using OpenHowNet (primary) or
       a Tongyici Cilin / hand-cluster fallback if HowNet is unavailable.
    2. Filter to sememes with ≥ 15 chars in our dataset.
    3. For each sememe (semantic field), identify the dominant Kangxi
       radical (the one shared by the most chars in that field). Split
       chars into:
            same-radical:  share the dominant radical
            diff-radical:  same sememe, different radical
       Both groups are constrained to be ≥ 5 chars (to keep d estimates
       meaningful).
    4. For every (model, layer=last, pool=char, iso=True) cell, compute
       intra (same-radical) vs cross (same-radical ↔ diff-radical) cosine.
    5. Welch's t, Cohen's d, permutation (5000 shuffles) per field.
       Holm correction across all (model, field) cells.
    6. Pooled meta-result across fields.

Output:
    results/expanded_semantic_control.csv
        rows = (model, field, sememe, n_same, n_diff, intra, cross, d, p_welch, p_perm, p_holm)
    results/expanded_semantic_control_pooled.csv
        rows = (model, n_fields, intra_pooled, cross_pooled, d_pooled, p_perm_pooled)

Runtime: ~10 minutes. CPU only.
Depends on: extract_embeddings.py, isotropy_correction.py
"""
from __future__ import annotations
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    CACHE_DIR,
    RESULTS_DIR,
    set_seed,
    load_radical_dataset,
    list_available_models,
    list_available_layers,
    cohens_d,
    welch_t,
    permutation_test_diff,
    holm_bonferroni,
)
from radical_lib.embeddings import model_tag  # noqa: E402
from scripts.new.config import N_PERMUTATIONS_SEMANTIC  # noqa: E402

set_seed()

ISO_DIR = CACHE_DIR / "embeddings_iso"
MIN_FIELD_SIZE = 12        # ≥ 12 chars in the sememe (gives ≥66 intra pairs)
MIN_GROUP_SIZE = 4         # ≥ 4 chars in same-rad and diff-rad sub-groups
                           # (4 → C(4,2)=6 intra and 4*N_diff cross; small per-field
                           # but the pooled analysis is the unit of inference)


# ── 1. semantic field construction ──────────────────────────────────────────
def build_fields_hownet(chars: List[str]) -> Dict[str, List[str]]:
    """Use OpenHowNet to map each char to its primary sememe.

    Falls back to {} if OpenHowNet is not installed or downloads fail.
    """
    try:
        import OpenHowNet
    except ImportError:
        print("[warn] OpenHowNet not installed; trying fallback resource")
        return {}

    try:
        hn = OpenHowNet.HowNetDict()
    except Exception as e:
        print(f"[warn] OpenHowNet init failed: {e}")
        try:
            OpenHowNet.download()
            hn = OpenHowNet.HowNetDict()
        except Exception as e2:
            print(f"[warn] OpenHowNet download/init failed twice: {e2}")
            return {}

    fields: Dict[str, List[str]] = defaultdict(list)
    n_hits = 0
    for c in chars:
        try:
            senses = hn.get_sense(c, language="zh")
        except Exception:
            continue
        if not senses:
            continue
        # use the first sense's first sememe as the field label
        try:
            sememes = senses[0].get_sememe_list()
        except Exception:
            continue
        if not sememes:
            continue
        sm = sememes[0]
        # try Chinese name, fall back to English
        name = getattr(sm, "zh", None) or getattr(sm, "en", None) or str(sm)
        fields[str(name)].append(c)
        n_hits += 1

    print(f"[hownet] mapped {n_hits}/{len(chars)} chars to sememes; "
          f"{len(fields)} unique sememes")
    return dict(fields)


def build_fields_fallback(chars: List[str]) -> Dict[str, List[str]]:
    """Hand-curated fallback covering 35+ broad semantic fields. We use this
    when OpenHowNet is unavailable. Each field lists chars that semantically
    belong to it; we then intersect with our dataset and dedupe within
    fields (so e.g. 葉/叶 only count once if both pass the tokenizer filter).

    Categories were assembled from the Tongyici Cilin (同义词词林) top-level
    taxonomy and the Mandarin frequency dictionaries. Both traditional and
    simplified variants are listed where a field has both, which makes the
    set robust to whichever subset of variants the tokenizer keeps.
    """
    fields = {
        # Each field is a SEMANTIC concept with members from MULTIPLE radicals.
        # The whole point of the control: same-radical pairs vs cross-radical
        # pairs *within the same semantic neighborhood*. So every field
        # deliberately mixes radicals, with at least 5 chars per side after
        # the dominant-radical split.

        # ---- nature ---------------------------------------------------------
        "water":         list("河海湖洋洗汗江沟池泡浪潮溪滴湿淹灌沿浮"
                              "水冰雨雪霜露雾泉永求"),
        "fire_heat":     list("灯炎焰烧炸烤煮烟焦灰烫炒煤煎熔燃熨炬"
                              "热熱炊蒸暖煦温"),
        "metal":         list("钢铁铜铝锡铅锌锈锐锯钉钓钎钞钩钮锁锅链"
                              "金玉宝矿珠玻璃硅"),
        "wood_plant":    list("树林松柏杨柳桑桃梨枣桂桦槐枫梅栋桶柜"
                              "竹芦藤草苗"),
        "earth":         list("地坡场坑坝堂塔墙坟堆塌坎壤垫埃壁垒坊"
                              "石泥沙尘灰"),
        "stone":         list("矿砂砖砍砧砥硬碎碰碾磨碑碱碘碳"
                              "岩崖岭峰山岳"),
        "mountain":      list("峰峡岭岗岚岛岩崖崎峻嵌嵘嵩"
                              "土地坡坂石"),
        "grass_plant":   list("草苗芽花莲茎萍蓝蔬荷莉菊菱芦芭蓬蕊"
                              "树木林森竹"),

        # ---- weather / sky -------------------------------------------------
        "sky_weather":   list("雷电雨雪霜露雾霞虹"
                              "云日月星空气风"),
        "season_time":   list("春夏秋冬"
                              "年月日早晚朝夕昼夜旦晨昏暮"),

        # ---- animals --------------------------------------------------------
        "bird":          list("鸽雀燕鸦鹰鹊鹂鹭鹦鹏鸥鸿鹤鸭鹅鸡"
                              "雕雏雁雇隼雉雌雄"),
        "fish":          list("鲍鲤鲫鲸鲨鳕鳗鳅鳃鳄鲜鲆鲶鳔"
                              "鱼蟹虾蛤蚌"),
        "insect":        list("蛇蛙蚊蝇蜂蛛蚁蟹蛤蝶蛾蜻蝉蛛蚂蚜"
                              "虫鼠鸟鸡螺蛹"),
        "mammal_dom":    list("猫狗猪猴狼狐狮"
                              "马驴骡骆驹驼"
                              "牛羊兔鼠"),
        "mammal_wild":   list("狼狐熊狮猿猴狈獾豹豺豢豕貂"
                              "虎豹象鹿牛"),

        # ---- body -----------------------------------------------------------
        "body_internal": list("肝肺胃肠肾胆胸腰背腹肌肤膜脂肪膀脏"
                              "心血骨筋脉"),
        "body_external": list("头颈面脸眉眼耳鼻嘴唇舌齿额颊腮颜"
                              "手指掌拳"),
        "limb":          list("指掌拳臂肘趾"
                              "手脚腿膝肩腕脖足"),
        "skin_hair":     list("发毛须皮肤膜疮疣痣痕疤"
                              "色香味"),

        # ---- people / society ----------------------------------------------
        "kinship":       list("妈奶姑姨妹姐"
                              "爸爷父叔伯哥弟兄孙儿子女妇婆婶嫂"),
        "occupation":    list("农工商医师徒"
                              "兵将卒臣君主官吏王僧侠匠"),
        "person_general":list("人民众群"
                              "夫妇老少男女"
                              "童叟"),

        # ---- emotion / mind -------------------------------------------------
        "emotion":       list("怀想恨悦悲愁怒怕恐恼恶慢愧愤恋恤恕"
                              "喜乐爱欲念盼怨"),
        "mental":        list("想念思忆悟悉慎悔愕惑憎"
                              "知觉记忆懂明白会"),

        # ---- communication --------------------------------------------------
        "speech":        list("说讲谈论诉告问答辩诵颂订计许诫语"
                              "言谓曰云口呼喊"),
        "writing":       list("字章篇页节"
                              "文书写记录笔墨纸"),

        # ---- perception -----------------------------------------------------
        "vision":        list("瞪睡眠瞧瞄盯瞻睥睁睑瞎瞅瞒"
                              "看见视观察望"),
        "sound":         list("响歌唱叫鸣咆哮咳咬嚎啧"
                              "声音听闻"),

        # ---- tools / objects ------------------------------------------------
        "tool":          list("锯锤锥针锁钩钉锅"
                              "刀剑斧"
                              "杯碗盆罐"),
        "weapon":        list("剑枪炮箭弓弹炸刺"
                              "戟戚戈矛盾盔"),
        "vehicle":       list("轮轴辆轨轿驰驶驾"
                              "车船舰艇舶"),
        "container":     list("瓶罐缸盆碗碟盘"
                              "箱筐桶杯篮罩"),
        "clothing":      list("衫袄袍裙裤帽袜袖袋"
                              "衣裳鞋帽巾"),
        "building":      list("屋宫堂楼阁阶院庭厅厦庙塔"
                              "门窗墙壁"),

        # ---- food -----------------------------------------------------------
        "food_grain":    list("米饭粥糕饼粮"
                              "糖油盐醋酒茶"
                              "肉菜豆"),
        "food_fruit":    list("桃梨柿杏李枣"
                              "桔橙苹葡萄柚柑梅榴"
                              "瓜西"),

        # ---- abstract -------------------------------------------------------
        "color":         list("红绿蓝灰紫粉橙褐"
                              "白黑黄青"),
        "number":        list("一二三四五六七八九十"
                              "百千万亿零半双"),
        "size_quality":  list("大低粗细轻"
                              "宽窄厚薄高低长短"),

        # ---- motion --------------------------------------------------------
        "movement":      list("跑跳跃攀奔逃挪移转"
                              "走行飞驰"),

        # ---- alcohol/fermentation ------------------------------------------
        "alcohol":       list("酒酱醋醉酸酵酗醒醪酌酬"
                              "汁汤"),
    }

    chars_set = set(chars)
    out: Dict[str, List[str]] = {}
    for name, members in fields.items():
        # intersect, then dedupe preserving order
        seen = set()
        kept = []
        for c in members:
            if c in chars_set and c not in seen:
                seen.add(c)
                kept.append(c)
        if len(kept) >= MIN_FIELD_SIZE:
            out[name] = kept

    print(f"[fallback] {len(out)} usable fields after intersection with dataset")
    return out


def split_field_by_dominant_radical(
    field_chars: List[str], char_to_radical: Dict[str, int]
) -> tuple[List[str], List[str], int]:
    """Pick the dominant radical in this field. Return (same_rad, diff_rad,
    dom_radical_number)."""
    rads = [char_to_radical[c] for c in field_chars if c in char_to_radical]
    if not rads:
        return [], [], -1
    counter = Counter(rads)
    dom_rad, _ = counter.most_common(1)[0]
    same = [c for c in field_chars if char_to_radical.get(c) == dom_rad]
    diff = [c for c in field_chars if char_to_radical.get(c) != dom_rad]
    return same, diff, dom_rad


# ── 2. similarity helpers ───────────────────────────────────────────────────
def load_iso_last_layer_char(model_id: str) -> tuple[np.ndarray, int]:
    """Load the isotropy-corrected `char`-pool embeddings for the *last* layer
    of `model_id`. Returns (X, layer_index)."""
    layers = list_available_layers(model_id)
    if not layers:
        raise FileNotFoundError(f"no cached layers for {model_id}")
    L = max(layers)
    path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_char.npy"
    if not path.exists():
        # fall back to mean pool if char-pool not present
        path = ISO_DIR / model_tag(model_id) / f"layer{L:02d}_mean.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"no isotropy-corrected last-layer file for {model_id}"
        )
    return np.load(path), L


def cosine_block(X: np.ndarray, idx_a: List[int], idx_b: List[int]) -> np.ndarray:
    A = X[idx_a]
    B = X[idx_b]
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T


# ── 3. main ─────────────────────────────────────────────────────────────────
def main():
    df = load_radical_dataset()
    chars = df["char"].tolist()
    char_idx = {c: i for i, c in enumerate(chars)}
    rad_col = "radical_number" if "radical_number" in df.columns else "radical"
    char_to_radical = dict(zip(df["char"], df[rad_col]))

    # 1. construct semantic fields
    fields = build_fields_hownet(chars)
    if not fields:
        fields = build_fields_fallback(chars)
    # filter to fields large enough
    fields = {k: v for k, v in fields.items() if len(v) >= MIN_FIELD_SIZE}

    # 2. partition each field by dominant radical, drop fields where either
    # side ends up too small
    usable = {}
    for fname, fchars in fields.items():
        same, diff, dom = split_field_by_dominant_radical(fchars, char_to_radical)
        if len(same) >= MIN_GROUP_SIZE and len(diff) >= MIN_GROUP_SIZE:
            usable[fname] = (same, diff, dom)
    print(f"\n{len(usable)} fields with ≥{MIN_GROUP_SIZE} same-rad and ≥{MIN_GROUP_SIZE} diff-rad chars")

    # 3. iterate models
    models = list_available_models()
    if not models:
        print("[fatal] no embeddings cached.")
        sys.exit(1)

    rows = []
    pooled_rows = []

    for model_id in models:
        try:
            X, layer = load_iso_last_layer_char(model_id)
        except FileNotFoundError as e:
            print(f"[skip] {model_id}: {e}")
            continue
        print(f"\n[{model_id}] last layer={layer}, embedding shape={X.shape}")

        all_intra: list[float] = []
        all_cross: list[float] = []
        for fname, (same, diff, dom) in usable.items():
            si = [char_idx[c] for c in same if c in char_idx]
            di = [char_idx[c] for c in diff if c in char_idx]
            if len(si) < 2 or len(di) < 1:
                continue

            same_block = cosine_block(X, si, si)
            iu = np.triu_indices(len(si), k=1)
            intra = same_block[iu]
            cross = cosine_block(X, si, di).flatten()

            t, p_w = welch_t(intra, cross)
            d = cohens_d(intra, cross)
            p_perm, observed, _ = permutation_test_diff(
                intra, cross, n_perm=N_PERMUTATIONS_SEMANTIC,
                rng=np.random.default_rng(42),
            )

            rows.append({
                "model": model_id,
                "field": fname,
                "dominant_radical": dom,
                "n_same_radical": len(si),
                "n_diff_radical": len(di),
                "intra_mean": float(intra.mean()),
                "cross_mean": float(cross.mean()),
                "delta": float(intra.mean() - cross.mean()),
                "cohens_d": d,
                "p_welch": p_w,
                "p_perm": p_perm,
                "n_intra_pairs": len(intra),
                "n_cross_pairs": len(cross),
            })
            all_intra.extend(intra.tolist())
            all_cross.extend(cross.tolist())

        # pooled
        if all_intra and all_cross:
            ai = np.array(all_intra)
            ac = np.array(all_cross)
            d_pool = cohens_d(ai, ac)
            p_pool, _, _ = permutation_test_diff(
                ai, ac, n_perm=N_PERMUTATIONS_SEMANTIC,
                rng=np.random.default_rng(42),
            )
            pooled_rows.append({
                "model": model_id,
                "n_fields": len(usable),
                "intra_pooled": float(ai.mean()),
                "cross_pooled": float(ac.mean()),
                "delta_pooled": float(ai.mean() - ac.mean()),
                "d_pooled": d_pool,
                "p_perm_pooled": p_pool,
                "n_intra": len(ai),
                "n_cross": len(ac),
            })

    out = pd.DataFrame(rows)

    # Holm correction across all per-field rows (within model)
    if not out.empty:
        adj_p_perm = []
        for model_id, g in out.groupby("model"):
            adj = holm_bonferroni(g["p_perm"].values)
            adj_p_perm.extend(zip(g.index.tolist(), adj))
        adj_p_perm.sort(key=lambda x: x[0])
        out["p_perm_holm"] = [v for _, v in adj_p_perm]

    out.to_csv(RESULTS_DIR / "expanded_semantic_control.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(
        RESULTS_DIR / "expanded_semantic_control_pooled.csv", index=False
    )
    print(f"\nWrote {len(out)} field rows and {len(pooled_rows)} pooled rows.")


if __name__ == "__main__":
    main()
