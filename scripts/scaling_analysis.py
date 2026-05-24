"""
scaling_analysis.py — does radical signal scale (or wash out) with model size?

For each model in the configured "scaling subset", we already have the
last-layer Cohen's d in `results/layer_wise.csv`. This script:
    1. Pulls last-layer d (isotropy-corrected, char-pool) for each model
    2. Pulls each model's parameter count via HF model card / config
    3. Plots and saves the scaling trajectory

Output:
    results/scaling.csv
        rows = (model, params_M, n_layers, hidden_dim, layerwise_max_d,
                last_layer_d, last_layer_p_perm, last_layer_rsa_rho)
    figures/scaling_d_vs_params.png/pdf  produced in figures.py

Depends on: layer_wise_analysis.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import RESULTS_DIR, set_seed  # noqa: E402
from scripts.new.config import scaling_models, MODELS  # noqa: E402

set_seed()


def get_param_count(hf_id: str, trust_remote_code: bool = False) -> int:
    """Return the parameter count of a model. Loads the model on CPU; we
    only do this once per script run."""
    try:
        from transformers import AutoConfig, AutoModel
        # try config first (faster and avoids downloading weights if cached)
        try:
            cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
            # configs usually expose num_parameters via .num_parameters() on model;
            # we approximate from architecture if model isn't available.
        except Exception:
            cfg = None

        # Fallback: load just the model and count
        model = AutoModel.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
        n = sum(p.numel() for p in model.parameters())
        del model
        return int(n)
    except Exception as e:
        print(f"[warn] couldn't count params for {hf_id}: {e}")
        return -1


def main():
    layerwise = pd.read_csv(RESULTS_DIR / "layer_wise.csv")
    layerwise = layerwise[(layerwise["pool"] == "char") & (layerwise["iso"] == 1)]

    rows = []
    targets = scaling_models() if scaling_models() else MODELS
    for spec in targets:
        sub = layerwise[layerwise["model"] == spec.hf_id]
        if sub.empty:
            print(f"[skip] {spec.label}: no layer-wise rows")
            continue
        last_layer = sub["layer"].max()
        last_row = sub[sub["layer"] == last_layer].iloc[0]
        max_d = sub["cohens_d"].max()
        n_params = get_param_count(spec.hf_id, spec.trust_remote_code)

        rows.append({
            "model": spec.label,
            "model_id": spec.hf_id,
            "size_label": spec.size,
            "params_M": (n_params / 1e6) if n_params > 0 else np.nan,
            "n_layers": int(last_layer),
            "layerwise_max_d": float(max_d),
            "last_layer_d": float(last_row["cohens_d"]),
            "last_layer_delta": float(last_row["delta"]),
            "last_layer_p_perm": float(last_row["p_perm"]),
            "last_layer_rsa_rho": float(last_row["rsa_rho"]),
            "last_layer_intra_mean": float(last_row["intra_mean"]),
            "last_layer_inter_mean": float(last_row["inter_mean"]),
        })

    out = pd.DataFrame(rows).sort_values("params_M")
    out.to_csv(RESULTS_DIR / "scaling.csv", index=False)
    print(f"\nWrote {len(out)} rows to {RESULTS_DIR / 'scaling.csv'}")
    print(out[["model", "params_M", "last_layer_d", "layerwise_max_d"]].to_string(index=False))


if __name__ == "__main__":
    main()
