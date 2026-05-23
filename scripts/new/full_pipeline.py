"""
full_pipeline.py — master entry point.

Runs every script in dependency order. Cheap scripts always re-run (they
re-do work in seconds). Expensive scripts skip if their primary output
is already present, so it's safe to interrupt and rerun.

Usage:
    venv\\Scripts\\python.exe scripts/new/full_pipeline.py
    venv\\Scripts\\python.exe scripts/new/full_pipeline.py --skip extract,sentential
    venv\\Scripts\\python.exe scripts/new/full_pipeline.py --only figures

Exit codes:
    0  — every step that ran finished cleanly
    1  — at least one step raised
"""
from __future__ import annotations
import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radical_lib import RESULTS_DIR, CACHE_DIR  # noqa: E402


# Each tuple: (key, module path, name, output marker, always_run, prerequisites)
# `prerequisites` are output paths that must exist for this step to run sensibly.
STEPS = [
    ("dataset",            "scripts.new.dataset_builder",            "Build radical_dataset.csv",          "data/radical_dataset.csv",         False, []),
    ("liushu",             "scripts.new.classify_liushu",            "Classify liushu / radical_role",     "data/radical_dataset.csv",         True,  ["data/radical_dataset.csv"]),
    ("audit",              "scripts.new.tokenization_audit",         "Tokenization audit",                 "results/tokenization_audit_summary.csv", False, ["data/radical_dataset.csv"]),
    ("extract",            "scripts.new.extract_embeddings",         "Extract embeddings (heavy)",         "cache/embeddings",                 False, ["data/radical_dataset.csv"]),
    ("isotropy",           "scripts.new.isotropy_correction",        "Isotropy correction",                "cache/embeddings_iso",             False, ["cache/embeddings"]),
    ("glyph_baseline",     "scripts.new.glyph_only_baseline",        "Pure-vision ResNet baseline",        "cache/embeddings/glyph_only__resnet18", False, ["data/radical_dataset.csv"]),
    ("layerwise",          "scripts.new.layer_wise_analysis",        "Layer-wise cohesion analysis",       "results/layer_wise.csv",           False, ["cache/embeddings_iso"]),
    ("pseudoradical",      "scripts.new.pseudoradical_control",      "Pseudoradical null partition",       "results/pseudoradical_control.csv", False, ["cache/embeddings_iso"]),
    ("freq_matched",       "scripts.new.frequency_matched_pairs",    "Frequency-matched pair control",     "results/frequency_matched.csv",    False, ["cache/embeddings_iso"]),
    ("random_init",        "scripts.new.random_init_baseline",       "Random-init noise floor",            "results/random_init_baseline.csv", False, ["data/radical_dataset.csv"]),
    ("semantic",           "scripts.new.expanded_semantic_control",  "Expanded semantic control",          "results/expanded_semantic_control.csv", False, ["cache/embeddings_iso"]),
    ("probing",            "scripts.new.probing_classifier",         "Linear probing classifier",          "results/probing.csv",              False, ["cache/embeddings_iso"]),
    ("phon_vs_sem",        "scripts.new.phonetic_vs_semantic_radicals", "Phonetic vs semantic radicals",   "results/phonetic_vs_semantic_radicals.csv", False, ["cache/embeddings_iso", "data/radical_dataset.csv"]),
    ("cross_japanese",     "scripts.new.cross_script_japanese",      "Cross-script Japanese kanji",        "results/cross_script_japanese.csv", False, ["cache/embeddings_iso"]),
    ("glyph_comp",         "scripts.new.glyph_comparison",           "Glyph-aware vs standard",            "results/glyph_comparison.csv",     True,  ["results/layer_wise.csv", "results/expanded_semantic_control_pooled.csv"]),
    ("scaling",            "scripts.new.scaling_analysis",           "Scaling analysis",                   "results/scaling.csv",              True,  ["results/layer_wise.csv"]),
    ("cooccurrence",       "scripts.new.cooccurrence_baseline",      "PMI variance decomposition",         "results/variance_decomposition.csv", False, ["cache/embeddings_iso"]),
    ("orth_arith",         "scripts.new.orthographic_arithmetic",    "Orthographic arithmetic",            "results/orthographic_arithmetic.csv", False, ["cache/embeddings_iso"]),
    ("activation_patch",   "scripts.new.activation_patching",        "Activation patching (geometric)",    "results/activation_patching.csv",  False, ["cache/embeddings_iso"]),
    ("sentential",         "scripts.new.sentential_context",         "Sentential context (heavy)",         "results/sentential_cohesion.csv",  False, ["cache/embeddings"]),
    ("downstream",         "scripts.new.downstream_validation",      "Downstream validation",              "results/downstream_validation.csv", False, ["cache/embeddings_iso"]),
    ("cloze",              "scripts.new.radical_cloze_probe",         "Radical cloze probe",                "results/radical_cloze_summary.csv", False, ["data/radical_dataset.csv"]),
    ("figures",            "scripts.new.figures",                    "All paper figures",                  "figures/fig_layer_wise_d.png",     True,  ["results/layer_wise.csv"]),
    ("report",             "scripts.new.results_report",             "Auto-generated analysis report",     "results/_REPORT.md",               True,  ["results/layer_wise.csv"]),
]


def output_exists(out_path: str) -> bool:
    p = Path(out_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / out_path
    if p.is_dir():
        return any(p.iterdir())
    return p.exists()


def run_step(key: str, module_path: str, name: str, out_marker: str, always_run: bool,
             prerequisites: list, force: bool) -> bool:
    # check prerequisites
    for prereq in prerequisites:
        if not output_exists(prereq):
            print(f"\n[skip] {key:18s}  prerequisite missing: {prereq}")
            return True
    if (not always_run) and (not force) and output_exists(out_marker):
        print(f"\n[skip] {key:18s}  {name}  (output {out_marker} exists)")
        return True
    print(f"\n{'=' * 70}\n[run]  {key:18s}  {name}\n{'=' * 70}")
    t0 = time.time()
    try:
        mod = importlib.import_module(module_path)
        if hasattr(mod, "main"):
            mod.main()
        else:
            print(f"[warn] {module_path} has no main()")
    except SystemExit as e:
        if e.code:
            print(f"[error] {key} called sys.exit({e.code})")
            traceback.print_exc()
            return False
    except Exception:
        print(f"[error] {key} raised:")
        traceback.print_exc()
        return False
    dt = time.time() - t0
    print(f"[done] {key} in {dt:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", default="",
                        help="comma-separated step keys to skip")
    parser.add_argument("--only", default="",
                        help="comma-separated step keys to run (skip everything else)")
    parser.add_argument("--force", action="store_true",
                        help="re-run all steps even if their output exists")
    parser.add_argument("--fast", action="store_true",
                        help="enable RADICAL_FAST=1 + skip the slowest optional steps "
                             "(cooccurrence, sentential). Use this for quick iteration "
                             "on free GPUs (Kaggle / Colab T4).")
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    if args.fast:
        os.environ["RADICAL_FAST"] = "1"
        # Wikipedia-streaming steps are the bottleneck after extract.
        # Skip them in fast mode unless the user explicitly asks for them
        # via --only.
        if not only:
            skip = skip | {"cooccurrence", "sentential"}
        print("[fast] using 5-model subset, ~5 layers per model, bf16 inference")
        print("[fast] skipping cooccurrence + sentential (override with --only)")

    fails = []
    for key, mod_path, name, marker, always_run, prereqs in STEPS:
        if only and key not in only:
            continue
        if key in skip:
            print(f"\n[skip] {key:18s} (user-requested)")
            continue
        ok = run_step(key, mod_path, name, marker, always_run, prereqs, force=args.force)
        if not ok:
            fails.append(key)

    print(f"\n{'=' * 70}\nSummary:")
    if fails:
        print(f"  {len(fails)} step(s) failed: {fails}")
        sys.exit(1)
    print("  all done ✔")


if __name__ == "__main__":
    main()
