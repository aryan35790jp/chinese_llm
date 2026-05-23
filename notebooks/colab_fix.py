"""
colab_fix.py — quick recovery cell for the Colab notebook.

Drop this content into ONE cell at the bottom of the existing colab_run.ipynb
(below the existing cell 14) and run only that one cell. It does three things:

1. Deletes the smoketest-leftover pseudoradical_control.csv and
   frequency_matched.csv so the next run rebuilds them on the real
   8-model embedding cache that's already on disk.
2. Deletes layer_wise.csv so it gets rebuilt with full 13-layer
   resolution for mBERT and Chinese-BERT (the rest of the models stay
   at 5-layer sampling).
3. Re-runs only those three scripts (~5 min total), then regenerates
   the report and re-zips for download.
"""
import os, pathlib

# 1. Wipe the smoketest leftovers
for fname in ("pseudoradical_control.csv", "frequency_matched.csv",
               "layer_wise.csv", "_REPORT.md"):
    p = pathlib.Path("results") / fname
    if p.exists():
        p.unlink()
        print(f"removed {p}")

os.environ["RADICAL_FAST"] = "1"
print("running with RADICAL_FAST=1, embeddings already cached on disk")
