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
    {
        "field": "water",
        "target_radical": 85,  # 氵
        "contexts": [
            "他在__里游泳。",          # He swims in the __
            "船在__上航行。",          # The ship sails on the __
            "鱼在__里游来游去。",      # Fish swim in the __
            "下了大雨,__水涨了。",     # After heavy rain, the __ rose
            "口渴的时候要喝__。",      # When thirsty, drink __
        ],
        "targets": ["河", "江", "湖", "海", "溪"],
        "distractors": ["山", "林", "路", "屋", "门"],
    },
    {
        "field": "fire",
        "target_radical": 86,  # 火
        "contexts": [
            "厨房的__很大。",          # The kitchen's __ is big
            "森林里着了__。",           # A __ broke out in the forest
            "他用__烤肉。",            # He grilled meat with __
            "请把__关小一点。",        # Please turn the __ down
        ],
        "targets": ["烟", "炎", "焰", "灯", "炒"],
        "distractors": ["水", "木", "石", "土", "金"],
    },
    {
        "field": "metal",
        "target_radical": 167,  # 金/钅
        "contexts": [
            "这把__很锋利。",          # This __ is sharp
            "工人用__敲钉子。",        # The worker hits nails with a __
            "古代用__做兵器。",        # In ancient times, weapons were made of __
            "她戴着__项链。",          # She wears a __ necklace
        ],
        "targets": ["铁", "铜", "钢", "锤", "银"],
        "distractors": ["木", "石", "布", "纸", "玉"],
    },
    {
        "field": "tree",
        "target_radical": 75,  # 木
        "contexts": [
            "院子里有一棵大__。",      # There's a big __ in the yard
            "秋天__叶变黄了。",         # In autumn the __ leaves turn yellow
            "他爬上了那棵__。",        # He climbed that __
            "用__做家具。",            # Make furniture from __
        ],
        "targets": ["树", "松", "柳", "桃", "梅"],
        "distractors": ["山", "河", "鸟", "云", "鱼"],
    },
    {
        "field": "bird",
        "target_radical": 196,  # 鸟
        "contexts": [
            "天上飞着一只__。",        # A __ flies in the sky
            "他养了一只__。",          # He keeps a pet __
            "树上停着__。",            # A __ rests on the tree
        ],
        "targets": ["鸽", "雀", "鹰", "鹅", "鸭"],
        "distractors": ["猫", "鱼", "羊", "马", "蛇"],
    },
    {
        "field": "fish",
        "target_radical": 195,  # 鱼
        "contexts": [
            "渔夫钓到了一条__。",      # The fisherman caught a __
            "海里有很多__。",          # There are many __ in the sea
            "我喜欢吃__。",            # I like to eat __
        ],
        "targets": ["鲤", "鲨", "鲸", "鳕", "鳗"],
        "distractors": ["猫", "鸟", "牛", "羊", "蛇"],
    },
    {
        "field": "insect",
        "target_radical": 142,  # 虫
        "contexts": [
            "草地上有许多__。",        # There are many __ in the grass
            "夏天最讨厌__。",          # In summer the most annoying are __
            "这只__在墙上。",          # This __ is on the wall
        ],
        "targets": ["蚊", "蝇", "蜂", "蚁", "蝶"],
        "distractors": ["猫", "狗", "鱼", "鸟", "马"],
    },
    {
        "field": "speech",
        "target_radical": 149,  # 言/讠
        "contexts": [
            "请你__清楚一点。",        # Please __ more clearly
            "他对我__了一句话。",      # He __ed a sentence to me
            "老师在__课文。",          # The teacher is __ing the text
        ],
        "targets": ["说", "讲", "谈", "诉", "讨"],
        "distractors": ["看", "走", "吃", "睡", "想"],
    },
]


# ── 2. encoder MLM scoring ──────────────────────────────────────────────────
def score_with_mlm(model_id: str, trust_remote_code: bool = False) -> List[Dict]:
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
        for trial in TRIALS:
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
                # Skip chars the tokenizer doesn't have
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

                # MRR over target ∪ distractor
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
def score_with_causal_lm(model_id: str, trust_remote_code: bool = False) -> List[Dict]:
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
        for trial in TRIALS:
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

    all_rows: List[Dict] = []

    for hf_id in mlm_models:
        print(f"\n=== {hf_id}  (MLM) ===")
        all_rows.extend(score_with_mlm(hf_id))

    for hf_id in causal_models:
        print(f"\n=== {hf_id}  (causal) ===")
        all_rows.extend(score_with_causal_lm(hf_id))

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
