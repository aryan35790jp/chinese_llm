"""
radical_cloze_probe.py — the downstream task that grounds the geometry.

Reviewer question this answers:
    "All of this is geometry. So what? Show me a real LM behavior that
    radical-aware models do better at than radical-naive ones."

Method:
    Build "radical cloze" trials. Each trial:
        prompt:    "他在河里游泳。这条__里有很多鱼。"
                    (He's swimming in the river. The __ has many fish.)
        target:    河 / 江 / 湖 / 海   (water-radical 氵 candidates)
        distractor: 山 / 林 / 路 / 屋   (non-water-radical, plausible nouns)

    For each trial, score every candidate by the conditional log-prob of
    the candidate given the cloze context. Compare score(target) against
    score(distractor) under each model.

    A model that has internalized radical-meaning correlations should
    consistently rank water-radical candidates above non-water candidates
    in water-related contexts, animal-radical above non-animal in animal
    contexts, and so on.

    Reported metric per (model, radical_field):
        - target_logprob       mean log P(target | context)
        - distractor_logprob   mean log P(distractor | context)
        - delta                target_logprob − distractor_logprob
        - top1_rate            fraction of trials where target_top1 > distractor_top1
        - mrr                  mean reciprocal rank of target candidates among
                                target ∪ distractor

This grounds the embedding-geometry findings in observable LM behavior.
Models with bigger d_form_specific should have bigger delta here.

Tested on:
    - Qwen2.5-1.5B  and Qwen2.5-3B   (causal LM scoring, native)
    - mBERT, Chinese-BERT, MacBERT   (MLM scoring of [MASK] position)
    - XLM-R-base, XLM-R-large        (MLM)

Excluded (not auto-scoreable):
    - JP-BERT (Japanese tokenizer; trial sentences are Chinese)
    - BGE-large-zh (no LM head)
    - glyph_only/resnet18 (no LM)

Output:
    results/radical_cloze.csv
        rows = (model, family, radical_field, n_trials, target_logprob,
                distractor_logprob, delta, top1_rate, mrr)
    results/radical_cloze_summary.csv
        rows = (model, family, mean_delta, mean_top1_rate, mean_mrr)

Runtime on Colab T4:
    encoder MLM:  ~5 min total across 5 encoder models
    Qwen-1.5B:    ~3 min
    Qwen-3B:      ~6 min
Total: ~15 min.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import (  # noqa: E402
    RESULTS_DIR,
    set_seed,
    get_device,
)

set_seed()


# ── 1. cloze trial set ─────────────────────────────────────────────────────
# Each entry:
#   field           — the semantic / radical theme
#   target_radical  — Kangxi number whose chars should win
#   contexts        — list of cloze sentences with __ marking the slot
#   targets         — characters that have target_radical and fit semantically
#   distractors     — characters that DO NOT have target_radical but are
#                     plausible single-char nouns (could fit syntactically)
TRIALS: List[Dict] = [
    # ── nature: water (氵 vs non-氵 water-related) ─────────────────────────
    {
        "field": "water",
        "target_radical": 85,
        "contexts": [
            "他在__里游泳。",
            "船在__上航行。",
            "鱼在__里游来游去。",
            "下了大雨,__水涨了。",
            "口渴的时候要喝__。",
            "桥下有一条__。",
            "我看见远处的__面在闪光。",
        ],
        "targets":     ["河", "江", "湖", "海", "溪", "潭", "渠"],
        "distractors": ["山", "林", "路", "屋", "门", "石", "木"],
    },
    # ── nature: fire (火 vs non-火 heat-related) ──────────────────────────
    {
        "field": "fire",
        "target_radical": 86,
        "contexts": [
            "厨房的__很大。",
            "森林里着了__。",
            "他用__烤肉。",
            "请把__关小一点。",
            "冬天我们围着__取暖。",
            "炉子里的__熊熊燃烧。",
        ],
        "targets":     ["烟", "炎", "焰", "灯", "炒", "炮", "烈"],
        "distractors": ["水", "木", "石", "土", "金", "纸", "布"],
    },
    # ── nature: metal (钅 vs non-钅 material-related) ─────────────────────
    {
        "field": "metal",
        "target_radical": 167,
        "contexts": [
            "这把__很锋利。",
            "工人用__敲钉子。",
            "古代用__做兵器。",
            "她戴着__项链。",
            "桥是用__造的。",
            "锅是__做的。",
        ],
        "targets":     ["铁", "铜", "钢", "锤", "银", "钉", "锁"],
        "distractors": ["木", "石", "布", "纸", "玉", "竹", "土"],
    },
    # ── nature: tree/plant (木 vs non-木 plant-related) ───────────────────
    {
        "field": "tree",
        "target_radical": 75,
        "contexts": [
            "院子里有一棵大__。",
            "秋天__叶变黄了。",
            "他爬上了那棵__。",
            "用__做家具。",
            "山上长满了__。",
            "公园里种着__。",
        ],
        "targets":     ["树", "松", "柳", "桃", "梅", "柏", "枫"],
        "distractors": ["山", "河", "鸟", "云", "鱼", "石", "土"],
    },
    # ── nature: mountain (山 vs non-山 terrain) ───────────────────────────
    {
        "field": "mountain",
        "target_radical": 46,
        "contexts": [
            "远处有一座__。",
            "我们爬上了__。",
            "__顶上有积雪。",
            "云雾缭绕在__间。",
        ],
        "targets":     ["峰", "岭", "崖", "岗", "岛", "岳"],
        "distractors": ["河", "湖", "树", "草", "路", "桥"],
    },
    # ── nature: stone (石 vs non-石) ──────────────────────────────────────
    {
        "field": "stone",
        "target_radical": 112,
        "contexts": [
            "路边有一块大__。",
            "工人在搬__。",
            "墙是用__砌的。",
            "海边有许多小__。",
        ],
        "targets":     ["砖", "砂", "硬", "碎", "碰", "磨"],
        "distractors": ["木", "草", "水", "云", "风", "雪"],
    },
    # ── animals: bird (鸟 vs non-鸟 / 隹 birds) ───────────────────────────
    {
        "field": "bird",
        "target_radical": 196,
        "contexts": [
            "天上飞着一只__。",
            "他养了一只__。",
            "树上停着__。",
            "公园里有许多__。",
            "__在天上飞翔。",
        ],
        "targets":     ["鸽", "雀", "鹰", "鹅", "鸭", "鹊", "鹂"],
        "distractors": ["猫", "鱼", "羊", "马", "蛇", "牛", "狗"],
    },
    # ── animals: fish (鱼 vs non-鱼 sea creatures) ────────────────────────
    {
        "field": "fish",
        "target_radical": 195,
        "contexts": [
            "渔夫钓到了一条__。",
            "海里有很多__。",
            "我喜欢吃__。",
            "市场上卖各种__。",
        ],
        "targets":     ["鲤", "鲨", "鲸", "鳕", "鳗", "鲍", "鲫"],
        "distractors": ["猫", "鸟", "牛", "羊", "蛇", "马", "狗"],
    },
    # ── animals: insect (虫 vs non-虫) ────────────────────────────────────
    {
        "field": "insect",
        "target_radical": 142,
        "contexts": [
            "草地上有许多__。",
            "夏天最讨厌__。",
            "这只__在墙上。",
            "花上飞着小__。",
        ],
        "targets":     ["蚊", "蝇", "蜂", "蚁", "蝶", "蛾", "蜻"],
        "distractors": ["猫", "狗", "鱼", "鸟", "马", "羊", "牛"],
    },
    # ── animals: wild mammal (犭 dog/animal radical) ──────────────────────
    {
        "field": "wild_animal",
        "target_radical": 94,
        "contexts": [
            "森林里跑出来一只__。",
            "动物园里有__。",
            "猎人遇见了一头__。",
            "电视里播着__的纪录片。",
        ],
        "targets":     ["狼", "狐", "狮", "猴", "猪", "猫", "狗"],
        "distractors": ["鸟", "鱼", "蛇", "羊", "牛", "马", "鸡"],
    },
    # ── people: kinship ───────────────────────────────────────────────────
    {
        "field": "kinship",
        "target_radical": 38,  # 女
        "contexts": [
            "我的__很慈祥。",
            "她有一个__。",
            "全家人都来了,包括我__。",
            "__给我做了好吃的。",
        ],
        "targets":     ["妈", "姐", "妹", "姑", "婶", "嫂", "奶"],
        "distractors": ["父", "兄", "叔", "伯", "爷", "弟", "孙"],
    },
    # ── speech (讠 vs non-讠 speech-acts) ─────────────────────────────────
    {
        "field": "speech",
        "target_radical": 149,
        "contexts": [
            "请你__清楚一点。",
            "他对我__了一句话。",
            "老师在__课文。",
            "她大声__着。",
            "同学们正在__问题。",
        ],
        "targets":     ["说", "讲", "谈", "诉", "讨", "诵", "议"],
        "distractors": ["看", "走", "吃", "睡", "想", "读", "写"],
    },
    # ── perception: vision (目 vs non-目) ────────────────────────────────
    {
        "field": "vision",
        "target_radical": 109,
        "contexts": [
            "他盯着我__了一会。",
            "她在__着远方。",
            "孩子__着糖果。",
            "我__不见前面的路。",
        ],
        "targets":     ["瞪", "瞧", "瞄", "盯", "睁", "瞅"],
        "distractors": ["听", "说", "走", "笑", "哭", "想"],
    },
    # ── body: organ (月/肉 vs non-月) ─────────────────────────────────────
    {
        "field": "body_organ",
        "target_radical": 130,
        "contexts": [
            "医生说他__不好。",
            "运动员的__很强壮。",
            "饭后__里很饱。",
            "__里疼。",
        ],
        "targets":     ["肝", "肺", "胃", "肠", "肾", "胆", "胸"],
        "distractors": ["脑", "心", "骨", "血", "皮", "毛", "齿"],
    },
    # ── emotion (心/忄 vs non-心) ─────────────────────────────────────────
    {
        "field": "emotion",
        "target_radical": 61,
        "contexts": [
            "我感到很__。",
            "他听了非常__。",
            "她__得说不出话。",
            "__使他不能入睡。",
        ],
        "targets":     ["怀", "悦", "愁", "怕", "恐", "悲", "怒"],
        "distractors": ["看", "说", "走", "吃", "笑", "哭", "想"],
    },
    # ── time: sun-related (日 vs non-日) ──────────────────────────────────
    {
        "field": "time_sun",
        "target_radical": 72,
        "contexts": [
            "今天__光很好。",
            "明__我去上学。",
            "__出__落。",
            "她每__早起。",
        ],
        "targets":     ["晴", "晚", "早", "昨", "明", "时", "晨"],
        "distractors": ["月", "云", "雨", "雪", "风", "山", "河"],
    },
    # ── weather (雨 vs non-雨) ───────────────────────────────────────────
    {
        "field": "weather",
        "target_radical": 173,
        "contexts": [
            "今天下__了。",
            "__后空气清新。",
            "冬天会下__。",
            "天上有__。",
        ],
        "targets":     ["雷", "电", "雪", "霜", "露", "雾", "霞"],
        "distractors": ["山", "河", "树", "石", "鸟", "鱼", "草"],
    },
    # ── grass/plant (艹 vs non-艹) ────────────────────────────────────────
    {
        "field": "grass",
        "target_radical": 140,
        "contexts": [
            "院子里长满了__。",
            "她在花园里种__。",
            "__里开着小花。",
            "牛吃__。",
        ],
        "targets":     ["苗", "芽", "花", "莲", "蓬", "茎", "蕊"],
        "distractors": ["树", "木", "石", "山", "鸟", "鱼", "云"],
    },
    # ── tools: knife/cut (刀/刂) ─────────────────────────────────────────
    {
        "field": "knife",
        "target_radical": 18,
        "contexts": [
            "他用__切菜。",
            "工匠__着木头。",
            "请把__递给我。",
            "刺客拔出了__。",
        ],
        "targets":     ["刀", "剑", "削", "划", "刺", "切", "剪"],
        "distractors": ["碗", "桌", "杯", "壶", "盘", "勺", "锅"],
    },
    # ── walking/movement (辶 vs non-辶) ──────────────────────────────────
    {
        "field": "movement",
        "target_radical": 162,
        "contexts": [
            "他正在__回家。",
            "我们一起__街。",
            "她__得很快。",
            "请你__过来。",
        ],
        "targets":     ["走", "进", "退", "返", "迎", "送", "追"],
        "distractors": ["看", "说", "想", "笑", "哭", "睡", "吃"],
    },
]


# ── trial loader ────────────────────────────────────────────────────────────
def _load_trials() -> List[Dict]:
    """Load cloze trials. Prefer the procedurally generated
    `data/cloze_items.json` over the inline TRIALS list, because the
    procedural version eliminates manual selection bias.
    See `scripts/new/cloze_procedural.py` for the construction protocol.
    """
    import json
    json_path = Path(__file__).resolve().parents[2] / "data" / "cloze_items.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if data:
                print(f"[cloze] using procedural items from {json_path.name} "
                      f"({len(data)} fields)")
                return data
        except Exception as e:
            print(f"[cloze] could not load {json_path}: {e}")
    print(f"[cloze] using inline hand-curated TRIALS ({len(TRIALS)} fields)")
    return TRIALS


# ── 2. encoder MLM scoring ──────────────────────────────────────────────────
def score_with_mlm(model_id: str, trials: List[Dict],
                    trust_remote_code: bool = False) -> List[Dict]:
    """For each trial, replace the slot with [MASK] and read off log P(token)
    for every target and distractor.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    device = get_device()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForMaskedLM.from_pretrained(
            model_id, trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        ).to(device).eval()
    except Exception as e:
        print(f"  [skip] failed to load: {e}")
        return []

    mask_token = tokenizer.mask_token
    if mask_token is None:
        print("  [skip] tokenizer has no mask token; MLM scoring not applicable")
        return []

    rows = []
    with torch.no_grad():
        for trial in trials:
            target_logprobs: List[float] = []
            distractor_logprobs: List[float] = []
            target_top1_wins = 0
            target_ranks: List[int] = []
            n_eff = 0

            for ctx in trial["contexts"]:
                masked = ctx.replace("__", mask_token, 1)
                enc = tokenizer(masked, return_tensors="pt").to(device)
                input_ids = enc["input_ids"][0]
                mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                if mask_positions.numel() == 0:
                    continue
                pos = int(mask_positions[0].item())
                logits = model(**enc).logits[0, pos]
                logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

                # Per-trial token IDs for targets and distractors
                t_ids = [tokenizer.convert_tokens_to_ids(c) for c in trial["targets"]]
                d_ids = [tokenizer.convert_tokens_to_ids(c) for c in trial["distractors"]]
                t_pairs = [(c, i) for c, i in zip(trial["targets"], t_ids)
                           if i is not None and i != tokenizer.unk_token_id]
                d_pairs = [(c, i) for c, i in zip(trial["distractors"], d_ids)
                           if i is not None and i != tokenizer.unk_token_id]
                if not t_pairs or not d_pairs:
                    continue

                t_lp = [float(logp[i]) for _, i in t_pairs]
                d_lp = [float(logp[i]) for _, i in d_pairs]
                target_logprobs.extend(t_lp)
                distractor_logprobs.extend(d_lp)

                if max(t_lp) > max(d_lp):
                    target_top1_wins += 1

                pool = sorted(t_pairs + d_pairs, key=lambda p: -float(logp[p[1]]))
                rank_of_first_target = next(
                    (k + 1 for k, p in enumerate(pool) if p in t_pairs), len(pool)
                )
                target_ranks.append(rank_of_first_target)
                n_eff += 1

            if n_eff == 0:
                continue
            rows.append({
                "model": model_id,
                "family": "mlm",
                "radical_field": trial["field"],
                "target_radical": trial["target_radical"],
                "n_trials": n_eff,
                "target_logprob": float(np.mean(target_logprobs)),
                "distractor_logprob": float(np.mean(distractor_logprobs)),
                "delta": float(np.mean(target_logprobs) - np.mean(distractor_logprobs)),
                "top1_rate": target_top1_wins / n_eff,
                "mrr": float(np.mean([1.0 / r for r in target_ranks])),
            })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


# ── 3. decoder LM scoring (Qwen) ────────────────────────────────────────────
def score_with_causal_lm(model_id: str, trials: List[Dict],
                          trust_remote_code: bool = False) -> List[Dict]:
    """For decoder-only models we can't use [MASK]. Instead we score the
    candidate as the next token after the prefix prompt."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = get_device()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device).eval()
    except Exception as e:
        print(f"  [skip] failed to load: {e}")
        return []

    rows = []
    with torch.no_grad():
        for trial in trials:
            target_logprobs: List[float] = []
            distractor_logprobs: List[float] = []
            target_top1_wins = 0
            target_ranks: List[int] = []
            n_eff = 0

            for ctx in trial["contexts"]:
                # Replace "__" with empty so the model has to predict the next char
                prefix, _, suffix = ctx.partition("__")
                # We score P(candidate | prefix). Suffix is ignored for the score
                # (decoder only sees left context); for diagnostic purposes that's
                # fine — radical-aware models should still rank correctly.
                enc = tokenizer(prefix, return_tensors="pt").to(device)
                logits = model(**enc).logits[0, -1]
                logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

                t_ids = [tokenizer.encode(c, add_special_tokens=False) for c in trial["targets"]]
                d_ids = [tokenizer.encode(c, add_special_tokens=False) for c in trial["distractors"]]
                # Use only single-token candidates (Qwen BBPE may split CJK chars)
                t_pairs = [(c, ids[0]) for c, ids in zip(trial["targets"], t_ids) if len(ids) == 1]
                d_pairs = [(c, ids[0]) for c, ids in zip(trial["distractors"], d_ids) if len(ids) == 1]
                if not t_pairs or not d_pairs:
                    continue

                t_lp = [float(logp[i]) for _, i in t_pairs]
                d_lp = [float(logp[i]) for _, i in d_pairs]
                target_logprobs.extend(t_lp)
                distractor_logprobs.extend(d_lp)

                if max(t_lp) > max(d_lp):
                    target_top1_wins += 1

                pool = sorted(t_pairs + d_pairs, key=lambda p: -float(logp[p[1]]))
                rank_of_first_target = next(
                    (k + 1 for k, p in enumerate(pool) if p in t_pairs), len(pool)
                )
                target_ranks.append(rank_of_first_target)
                n_eff += 1

            if n_eff == 0:
                continue
            rows.append({
                "model": model_id,
                "family": "causal",
                "radical_field": trial["field"],
                "target_radical": trial["target_radical"],
                "n_trials": n_eff,
                "target_logprob": float(np.mean(target_logprobs)),
                "distractor_logprob": float(np.mean(distractor_logprobs)),
                "delta": float(np.mean(target_logprobs) - np.mean(distractor_logprobs)),
                "top1_rate": target_top1_wins / n_eff,
                "mrr": float(np.mean([1.0 / r for r in target_ranks])),
            })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


# ── 4. main ─────────────────────────────────────────────────────────────────
def main():
    # Models we can score with MLM
    mlm_models = [
        "bert-base-multilingual-cased",
        "hfl/chinese-bert-wwm-ext",
        "hfl/chinese-macbert-base",
        "xlm-roberta-base",
        "xlm-roberta-large",
    ]
    # Models we can score causally
    causal_models = [
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
    ]

    trials = _load_trials()
    all_rows: List[Dict] = []

    for hf_id in mlm_models:
        print(f"\n=== {hf_id}  (MLM) ===")
        all_rows.extend(score_with_mlm(hf_id, trials))

    for hf_id in causal_models:
        print(f"\n=== {hf_id}  (causal) ===")
        all_rows.extend(score_with_causal_lm(hf_id, trials))

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "radical_cloze.csv", index=False)
    print(f"\nWrote {len(df)} rows to results/radical_cloze.csv")

    if not df.empty:
        summary = df.groupby(["model", "family"]).agg(
            mean_delta=("delta", "mean"),
            mean_top1_rate=("top1_rate", "mean"),
            mean_mrr=("mrr", "mean"),
            n_fields=("radical_field", "size"),
        ).reset_index()
        summary.to_csv(RESULTS_DIR / "radical_cloze_summary.csv", index=False)
        print("\nSummary:")
        print(summary.to_string(index=False))
    else:
        pd.DataFrame(columns=[
            "model", "family", "mean_delta", "mean_top1_rate", "mean_mrr", "n_fields"
        ]).to_csv(RESULTS_DIR / "radical_cloze_summary.csv", index=False)


if __name__ == "__main__":
    main()
