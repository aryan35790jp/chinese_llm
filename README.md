# Radical-Aligned Structure in Multilingual Transformer Representations of Chinese Characters

Probing whether transformer-based language models encode radical (部首)
compositionality in their learned representations of Chinese characters.

This is the extended pipeline for the ACL ARR 2026 submission. It supersedes
the original 2-model / 4-semantic-field analysis with:

- **11 models** spanning multilingual / Chinese / Japanese / glyph-aware /
  pure-vision baselines (`mBERT`, `Chinese-BERT`, `MacBERT`, `XLM-R base/large`,
  `ERNIE 3.0`, `UER-tiny/small`, `ChineseBERT-glyph`, `JP-BERT char/subword`,
  rendered-PNG → frozen ResNet-18).
- **All hidden layers** for every model, three pool types each (`char`,
  `mean`, `cls`).
- **Anisotropy correction** (Mu & Viswanath all-but-the-top) before every
  cosine measurement.
- **20+ semantic fields** for the controlled comparison, generated from a
  hand-curated mixed-radical taxonomy with a fallback path that doesn't
  require OpenHowNet.
- **Linear probes** for radical category and semantic field at every layer.
- **Phonetic vs semantic radical role** split via CHISE IDS.
- **Cross-script replication** on Japanese kanji.
- **Co-occurrence / PMI variance decomposition** that quantifies how much
  of the radical effect is form, semantics, distributional context, or
  frequency.
- **Mikolov-style orthographic arithmetic** and **geometric activation
  patching** as causal-flavored interventions.
- **Sentential-context analysis** comparing isolated vs in-sentence
  embeddings.
- **Downstream validation** against PKU-500 word similarity.

## Layout

```
chinese_llm_composition/
├── data/
│   ├── unihan/                       # Unihan database files
│   ├── radical_dataset.csv           # canonical char→radical (+ liushu_class, radical_role, stroke_count)
│   ├── radical_summary.csv
│   └── tokenization_coverage.csv     # per-model single-token coverage
│
├── radical_lib/                      # shared library — every script imports from here
│   ├── core.py                       # paths, seeds, device
│   ├── data.py                       # dataset accessors
│   ├── embeddings.py                 # per-(model, layer, pool) cache layout
│   ├── isotropy.py                   # Mu & Viswanath all-but-the-top
│   ├── stats.py                      # cohens_d, welch, bootstrap, permutation, holm, RSA
│   └── plotting.py                   # publication style + save_fig
│
├── scripts/
│   ├── (legacy scripts)              # original 2-model pipeline, kept for reference
│   └── new/
│       ├── config.py                 # MODELS, SENTENCE_CORPUS, knobs
│       ├── full_pipeline.py          # master entry point with checkpointing
│       │
│       ├── dataset_builder.py        # rebuild radical_dataset.csv
│       ├── tokenization_audit.py     # appendix table
│       ├── classify_liushu.py        # CHISE IDS → liushu_class + radical_role
│       │
│       ├── extract_embeddings.py     # all models × all layers × {char, mean, cls}
│       ├── isotropy_correction.py    # Mu & Viswanath on every cached layer
│       │
│       ├── glyph_only_baseline.py    # rendered-PNG → frozen ResNet-18
│       │
│       ├── layer_wise_analysis.py    # centerpiece: layer-wise Cohen's d
│       ├── expanded_semantic_control.py
│       ├── probing_classifier.py
│       ├── phonetic_vs_semantic_radicals.py
│       ├── cross_script_japanese.py
│       ├── glyph_comparison.py
│       ├── scaling_analysis.py
│       │
│       ├── cooccurrence_baseline.py  # PMI-driven variance decomposition (the money figure)
│       ├── orthographic_arithmetic.py # Mikolov-style E(c) − E(R₁) + E(R₂)
│       ├── activation_patching.py    # geometric ablation
│       ├── sentential_context.py     # in-sentence embeddings
│       ├── downstream_validation.py  # PKU-500
│       │
│       ├── figures.py                # every paper figure
│       │
│       ├── _check_syntax.py          # compile-check every file
│       ├── _smoke_test.py            # batch A unit tests
│       ├── _smoke_test_b.py          # batch B unit tests
│       └── _integration_test_c.py    # synthetic-data integration tests for batch C
│
├── cache/
│   ├── embeddings/{model_tag}/layer{NN}_{pool}.npy   # extract_embeddings output
│   ├── embeddings_iso/{model_tag}/layer{NN}_{pool}.npy  # isotropy_correction output
│   ├── char_ppmi.npz                 # cooccurrence_baseline (cached)
│   ├── char_counts.npy
│   ├── sentences.json                # sentential_context (cached)
│   └── sentential/{model_tag}/layer{L}.npy
│
├── results/                          # canonical CSV / NPY outputs
│   ├── layer_wise.csv                # the centerpiece table
│   ├── tokenization_audit_summary.csv
│   ├── expanded_semantic_control.csv
│   ├── expanded_semantic_control_pooled.csv
│   ├── probing.csv
│   ├── phonetic_vs_semantic_radicals.csv
│   ├── cross_script_japanese.csv
│   ├── glyph_comparison.csv
│   ├── scaling.csv
│   ├── variance_decomposition.csv    # the money figure
│   ├── orthographic_arithmetic.csv
│   ├── orthographic_arithmetic_summary.csv
│   ├── activation_patching.csv
│   ├── sentential_cohesion.csv
│   ├── downstream_validation.csv
│   ├── downstream_per_radical.csv
│   └── (legacy) main_results.csv, semantic_control_results.csv, *.npy
│
├── figures/                          # 300 DPI PNG + PDF for every figure
└── paper/
```

## Reproduce

```bash
python -m venv venv
venv\Scripts\activate                  # Windows
# source venv/bin/activate              # Mac/Linux

pip install -r requirements.txt

# 0. Download Unihan
mkdir data\unihan
curl -L -o data\unihan\Unihan.zip https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip
cd data\unihan && powershell Expand-Archive Unihan.zip . && cd ..\..

# 1. Verify environment (under a minute)
python scripts/new/_check_syntax.py
python scripts/new/_smoke_test.py
python scripts/new/_smoke_test_b.py

# 2. Run the full pipeline. Heavy steps (extract_embeddings,
#    cooccurrence_baseline, sentential_context) skip themselves once
#    their output is cached. Safe to interrupt and rerun.
python scripts/new/full_pipeline.py
```

Selective re-runs:

```bash
python scripts/new/full_pipeline.py --only figures           # just rebuild figures
python scripts/new/full_pipeline.py --skip extract,sentential  # skip the heaviest
python scripts/new/full_pipeline.py --force                  # re-run every step
```

## Heaviest steps and where they should live

| Step                     | Compute            | RAM    | Runtime          |
|--------------------------|--------------------|--------|------------------|
| `extract_embeddings.py`  | GPU (A100 ideal)   | 16 GB  | 3 h A100 / 15 h T4 |
| `cooccurrence_baseline.py` (first run) | CPU, network | 8 GB   | 30 min for PMI build |
| `sentential_context.py`  | GPU                | 16 GB  | 90 min total     |
| `layer_wise_analysis.py` | CPU                | 8 GB   | 1 hr             |
| everything else          | CPU                | <8 GB  | minutes          |

## Statistical methods

- **Anisotropy correction**: mean-centering, per-coordinate standardization,
  removal of top-k principal components (Mu & Viswanath 2018,
  "All-But-the-Top"). Default k=2.
- **Cohen's d** with pooled unbiased standard deviation.
- **Welch's unequal-variance t-test** for the primary comparison.
- **Permutation test** (1,000 shuffles for corpus-scale, 5,000 for the
  semantic control). Continuity-corrected one-sided p.
- **Bootstrap 95% CI** (1,000 resamples) for the mean difference.
- **Holm–Bonferroni** correction across all primary comparisons.
- **Representational Similarity Analysis (RSA)**: Spearman correlation
  between the embedding RDM and the binary same-radical RDM.
- **Variance decomposition**: OLS of `bert_cosine ~ same_radical + ppmi +
  freq_diff + stroke_diff` on a 200k-pair sample, partial R² for each
  predictor.

## License

Research use. Dataset derived from Unicode Unihan (see Unicode terms of use).
