"""
cloze_inference.py — paired permutation test + character-clustered bootstrap
on the cloze probe deltas.

Reviewer objection this answers (Claude, ChatGPT both flagged it):
    "The cloze probe reports point estimates with no significance testing
     and no confidence intervals. mean_delta = +0.36 vs -0.98 could easily
     be artifact of context writing, not radical geometry."

Method:
    For each (model, field) row in radical_cloze.csv we have
        delta_field = mean(target_logprob) - mean(distractor_logprob).
    We treat each (model, field) as an exchangeable observation under H0
    that radical identity has no effect on next-token preference. Within
    each model:
        - a paired-sign permutation test sign-flips the per-field delta
          B = 10000 times and reports p_two-sided.
        - a non-parametric bootstrap resamples the 8 fields with
          replacement B = 10000 times and reports a 95% CI on mean_delta.
    Both honour pair non-independence at the field level — the right
    cluster.

    For the *cross-model* comparison (does Qwen-3B systematically beat
    multilingual MLMs?), we run a model-pair bootstrap: per resample of
    fields, recompute each model's mean_delta from the same resample,
    then compute (mean_delta_qwen3b - mean_delta_mbert). The CI tells
    us whether the *gap* between models is robust.

Output:
    results/radical_cloze_inference.csv
        rows = (model, mean_delta, ci_lo, ci_hi, p_perm_two_sided, n_fields)
    results/radical_cloze_pairwise.csv
        rows = (model_a, model_b, gap_mean, gap_ci_lo, gap_ci_hi, gap_sig)

Runtime: ~10 sec on CPU. No embeddings needed; reads only radical_cloze.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import RESULTS_DIR, set_seed  # noqa: E402

set_seed()
B = 10000


def per_model_inference(deltas: np.ndarray, rng: np.random.Generator) -> dict:
    """Paired sign-flip permutation + non-parametric bootstrap for one model."""
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(deltas)
    obs_mean = float(deltas.mean())

    # 1. Paired sign-flip permutation: under H0, the sign of each field's
    #    delta is exchangeable. Two-sided p on |mean|.
    signs = rng.choice([-1.0, 1.0], size=(B, n))
    null_means = (signs * deltas[None, :]).mean(axis=1)
    p_two_sided = float(((np.abs(null_means) >= np.abs(obs_mean)).sum() + 1) / (B + 1))

    # 2. Non-parametric field bootstrap for 95% CI.
    boot_means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = deltas[idx].mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    return {
        "mean_delta": obs_mean,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_perm_two_sided": p_two_sided,
        "n_fields": n,
    }


def pairwise_gap_bootstrap(
    df_long: pd.DataFrame, model_a: str, model_b: str, rng: np.random.Generator
) -> dict:
    """Bootstrap the gap (mean_delta_a − mean_delta_b) across the same field
    resample. The shared resample is the right unit because both models are
    evaluated on the same field set."""
    fields_a = df_long[df_long["model"] == model_a].set_index("radical_field")["delta"]
    fields_b = df_long[df_long["model"] == model_b].set_index("radical_field")["delta"]
    common = sorted(set(fields_a.index) & set(fields_b.index))
    if len(common) < 3:
        return {
            "model_a": model_a, "model_b": model_b,
            "gap_mean": float("nan"), "gap_ci_lo": float("nan"),
            "gap_ci_hi": float("nan"), "gap_sig": False, "n_fields": len(common),
        }
    da = fields_a.loc[common].to_numpy()
    db = fields_b.loc[common].to_numpy()
    gap_obs = float(da.mean() - db.mean())
    n = len(common)
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = da[idx].mean() - db[idx].mean()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    sig = bool(ci_lo > 0 or ci_hi < 0)
    return {
        "model_a": model_a, "model_b": model_b,
        "gap_mean": gap_obs, "gap_ci_lo": float(ci_lo),
        "gap_ci_hi": float(ci_hi), "gap_sig": sig, "n_fields": n,
    }


def main():
    src = RESULTS_DIR / "radical_cloze.csv"
    if not src.exists():
        print(f"[fatal] {src} not found — run radical_cloze_probe.py first")
        sys.exit(1)

    df = pd.read_csv(src)
    if df.empty:
        print(f"[fatal] {src} is empty")
        sys.exit(1)

    rng = np.random.default_rng(42)

    # Per-model paired permutation + bootstrap
    rows = []
    for model, sub in df.groupby("model"):
        res = per_model_inference(sub["delta"].to_numpy(), rng)
        res["model"] = model
        rows.append(res)
    out = pd.DataFrame(rows).sort_values("mean_delta", ascending=False)
    out = out[["model", "mean_delta", "ci_lo", "ci_hi", "p_perm_two_sided", "n_fields"]]
    out_path = RESULTS_DIR / "radical_cloze_inference.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out.round(4).to_string(index=False))

    # Pairwise gap bootstrap: every model vs the lowest-mean-delta model.
    # The interesting gaps are the three positive-delta models (Qwen-3B,
    # Qwen-1.5B, Chinese-BERT) vs the three multilingual MLMs (mBERT,
    # XLM-R-base, XLM-R-large).
    targets = [
        ("Qwen/Qwen2.5-3B", "bert-base-multilingual-cased"),
        ("Qwen/Qwen2.5-3B", "xlm-roberta-large"),
        ("Qwen/Qwen2.5-1.5B", "bert-base-multilingual-cased"),
        ("Qwen/Qwen2.5-1.5B", "xlm-roberta-large"),
        ("hfl/chinese-bert-wwm-ext", "bert-base-multilingual-cased"),
        ("hfl/chinese-bert-wwm-ext", "xlm-roberta-large"),
        ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-1.5B"),
    ]
    gaps = [pairwise_gap_bootstrap(df, a, b, rng) for a, b in targets]
    gap_df = pd.DataFrame(gaps)
    gap_path = RESULTS_DIR / "radical_cloze_pairwise.csv"
    gap_df.to_csv(gap_path, index=False)
    print(f"\nWrote {gap_path}")
    print(gap_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
