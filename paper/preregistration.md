# Preregistration

**Paper:** Radical-Aligned Structure in Multilingual Transformer Representations of Chinese Characters

**Status:** Locked before observing results from the expanded pipeline (mBERT and Chinese-BERT 2-model results from the original paper are *seen*; everything new is *unseen*).

**Date locked:** 2026 (this turn).

This document specifies, in advance, what we predict and how we will decide between competing hypotheses. Any deviation between this document and the final paper will be disclosed as a deviation, with reason.

---

## 1. Primary research question

Do transformer language models encode Kangxi radical category as a representational dimension that is *separable* from semantic content, distributional context, character frequency, and visual stroke complexity?

## 2. Decomposable hypotheses

We decompose the radical signal into four candidate sources:

- **(S) Semantic.** Radicals correlate with meaning categories; clustering reflects semantics.
- **(D) Distributional.** Same-radical chars share contexts; clustering reflects co-occurrence.
- **(F) Frequency.** Same-radical chars share frequency strata; clustering reflects frequency-band geometry.
- **(O) Orthographic.** Models encode visual/structural form independently of S, D, F.

We will fit a single regression on a 200k-pair sample:

```
bert_cosine ~ same_radical + ppmi + freq_diff + stroke_diff
```

with all predictors standardized, and report partial R² for each.

## 3. Pre-specified predictions

### H1. For *standard distributional models* (mBERT, Chinese-BERT, MacBERT, XLM-R, ERNIE, UER):

- **H1a.** Pooled semantic-control Cohen's d will be statistically indistinguishable from zero ($|d|$ < 0.10, $p_{\text{perm}}$ > 0.05).
- **H1b.** In the variance decomposition, $\text{partial } R^2(\text{ppmi}) > \text{partial } R^2(\text{same\_radical})$.
- **H1c.** Frequency-matched effect $d_{\text{matched}}$ will be smaller than unmatched $d_{\text{unmatched}}$ by at least 0.02.
- **H1d.** Pseudoradical $p_{\text{pseudo}}$ will be < 0.05 (the effect is specific to Kangxi, not any 68-group partition), but the *magnitude* of $d_{\text{real}} - d_{\text{random\_partition\_mean}}$ will be small.

### H2. For *glyph-aware* (ChineseBERT) and *vision-only* (rendered-PNG → ResNet-18):

- **H2a.** Pooled semantic-control Cohen's d will be substantially larger than zero ($d \geq 0.20$, $p_{\text{perm}}$ < 0.01).
- **H2b.** $\text{partial } R^2(\text{stroke\_diff})$ and/or $\text{partial } R^2(\text{same\_radical})$ will exceed $\text{partial } R^2(\text{ppmi})$.
- **H2c.** $d_{\text{form\_specific}} = d_{\text{corpus}} - d_{\text{semantic\_ctrl}}$ will be at least 0.15 larger for the glyph/vision class than for the standard class.

### H3. Random-init noise floor:

- **H3.** $d_{\text{random\_init}} < 0.05$ for both architectures we test (Chinese-BERT random, XLM-R random). If this is violated, the architecture itself encodes radical-aligned structure even before training, which we will report as a separate finding.

### H4. Cross-script generalization:

- **H4a.** Japanese BERT on the Joyo subset will show $d > 0$ (semantic clustering generalizes) but smaller than Chinese-BERT on the same subset (Chinese has stronger distributional reinforcement).
- **H4b.** If Japanese $d \approx$ Chinese $d$ on the same subset, this argues against the distributional account and weakly for an orthographic one.

### H5. Layer-wise emergence:

- **H5a.** Cohen's d will *peak in the middle layers* (layers 4–8 of 12, or 8–16 of 24), consistent with the established "middle layers carry semantic content" finding (Tenney et al. 2019). Last-layer d will be slightly lower than peak.
- **H5b.** RSA Spearman ρ will track Cohen's d closely (Pearson correlation > 0.9 across layers within a model).

### H6. Probing:

- **H6a.** Linear probe macro-F1 for the 68-radical task will be strictly lower than for the 20-semantic-field task at every layer of every standard distributional model.
- **H6b.** For the glyph-aware model, radical macro-F1 will exceed semantic-field macro-F1 at *some* layer.

### H7. Orthographic arithmetic:

- **H7a.** Mean retrieval lift will be < 2 for standard distributional models (no meaningful linear compositionality).
- **H7b.** Mean retrieval lift will be > 3 for glyph-aware/vision-only models if H2 holds.

## 4. Stopping rules and sensitivity analysis

We will not iteratively look at the data and adjust the hypotheses. The hypotheses above are locked. After observing results we will:

1. Mark each prediction as **supported / not supported / partial**.
2. For any non-supported prediction, run **one** post-hoc analysis to characterize the deviation, clearly labeled as exploratory in the paper.
3. Report effect-size confidence intervals (bootstrap, 1,000 resamples) alongside every point estimate.

## 5. What would falsify the paper's core claim?

The paper's core claim is "radical-aligned structure in standard distributional models is fully reducible to semantics + distributional context + frequency, while glyph-aware models retain a form-specific residual."

This claim is **falsified** if any of the following hold:

- $d_{\text{semantic\_ctrl}} \geq 0.20$ in any standard distributional model with $p_{\text{perm}} < 0.01$. (Standard models have non-reducible radical signal.)
- $d_{\text{form\_specific}} \leq 0.05$ for the vision-only baseline. (Even pure form provides no distinct signal — implausible but possible.)
- $\text{partial } R^2(\text{same\_radical}) > \text{partial } R^2(\text{ppmi})$ in the regression for any standard model. (Radical identity dominates over distributional context — implausible but worth checking.)

Any falsification will lead to a substantive rewrite of the contribution, not a quiet redefinition.

## 6. Multiple-comparison plan

Across all primary tests (per model × per metric × per layer for the centerpiece, plus the 5 secondary controls), we apply Holm–Bonferroni within each "family" (centerpiece corpus-scale, centerpiece semantic-control, controls). We do not apply a single global correction — that would be over-conservative across families that test independent claims.

## 7. Reproducibility commitment

The complete analysis pipeline (every script in `scripts/new/`), the dataset (`data/radical_dataset.csv`), the cached embeddings (after permission for redistribution from each model's author), the figures (`figures/`), and this preregistration document are released together. A `results/_REPORT.md` is auto-generated by `scripts/new/results_report.py` from the CSVs and is included verbatim in the supplementary materials.

## 8. Authorship and contribution statement

Aryan Maity (sole researcher) designed the study, implemented all experiments, performed the analyses, and wrote the paper. AI tools were used as a programming and writing aide; all analytic decisions and final conclusions were made and verified by the author.
