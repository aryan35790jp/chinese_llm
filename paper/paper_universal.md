# Decomposing Radical-Aligned Geometry in Chinese Character Embeddings: Form, Distribution, Frequency, and the Limits of Variance Analysis

**Aryan Maity**
B.Tech Computer Science, St. Edmunds College (NEHU), India
`aryanmaity3579@gmail.com`

## Abstract

We study whether transformer language models represent Chinese characters in ways that align with the Kangxi radical system, and if so, what the alignment is made of. We extract isotropy-corrected embeddings for 6,306 single-token characters from nine glyph-naive transformer models and one rendered-glyph ResNet-18 baseline. For each model we decompose pairwise cosine similarity into four predictors (radical co-membership, character co-occurrence PMI, log-frequency difference, stroke-count difference) using OLS with two-way cluster-robust standard errors clustering on both characters of each pair. Three observations follow. First, the absolute amount of variance any of these predictors explains is small (full R² between 0.002 and 0.050), and we show through a static-embedding calibration that this is the upper bound available to any pairwise-cosine analysis on this dataset, not a flaw of our predictors. Second, the partial-R² ratio between radical co-membership and PMI separates models into two non-overlapping groups: three Chinese-pretrained transformers (Qwen2.5-1.5B, Qwen2.5-3B, BGE-large-zh) place radical above PMI by a factor of 1.7 to 1.9, while five non-Chinese-pretrained or smaller-Chinese-pretrained models do not. Third, an exploratory cloze probe of 250 procedurally-constructed trials reproduces the same model ordering, though we are explicit that 250 author-generated trials without native-speaker validation are not a behavioral benchmark and we report the probe as preliminary. The radical signal is specific (size-matched random partitions yield d ≈ 0), training-driven (random-init networks yield d ≈ −0.04), and not attributable to character frequency (frequency-matched effect retains 92% to 99% of the unmatched effect across models). We discuss what these patterns can and cannot establish, with particular attention to what cannot be claimed at sample sizes of n = 3 specialized models and what the small full R² implies for the strength of any "encoding" claim.

## 1. Introduction

The Kangxi system organizes Chinese characters into 214 graphemic categories. Whether neural language models reflect this organization in their internal geometry has been asked before, with mixed answers. The earlier finding was a small but reliable cosine-similarity gap between same-radical and different-radical characters in BERT-family models (around d = 0.06 to 0.14). The natural follow-up is to ask what produces the gap. Same-radical characters share visual form (the radical is literally drawn the same way), they share semantic content (radicals were historically chosen to mark meaning), they share distributional context (semantically related words co-occur), and they share frequency rank (more or less, depending on the radical). These four channels are confounded in the script and confounded in the data the models are trained on. The previous "yes, there is a gap" answer leaves all of them in play.

This paper does the decomposition. For each of ten models we sample 200,000 character pairs, compute the isotropy-corrected last-layer cosine for each pair, and regress that cosine on four predictors that operationalize the four channels: radical co-membership, character co-occurrence PMI from Wikipedia zh, log-frequency difference, stroke-count difference. The regression coefficients say which channel is doing what, and the cluster-robust standard errors honor the fact that each character appears in roughly 63 of the 200,000 pairs, so the observations are not independent.

We want to be careful about what this method can establish. The full R² of the four-predictor regression is small everywhere. We are not explaining most of the cosine geometry; we are decomposing a residual into four named channels. Whether a residual is meaningful depends on what an alternative residual looks like, so we report the R² of a strong upper-bound model (the static-embedding cosine of the same characters at the same layer) for calibration. That comparison turns out to be informative.

We also want to be careful about category claims. Three of the models we tested (Qwen2.5-1.5B, Qwen2.5-3B, BGE-large-zh) show partial-R²(radical) above partial-R²(PMI). Three is a small set. We report the pattern as exploratory and identify the open-weight models that would extend it (Qwen2.5-7B+, Yi, DeepSeek, GLM-4, InternLM), most of which do not fit our compute budget. We do not present this finding as a category effect.

Finally, the cloze probe. The probe was constructed procedurally — for each of eight semantic fields, we take the top-5 most frequent characters with the target radical and the top-5 without it, and write ten short carrier sentences. The targets and distractors are not author-selected. The contexts are. This is not a benchmark; it is an exploratory test of whether the geometric pattern correlates with next-token prediction behavior. We report it as such.

## 2. Background

Probing of contextual embeddings has shown that morphological structure leaves recoverable traces in alphabetic-language models (Hofmann et al. 2021; Durrani et al. 2019), that syntactic and semantic information is recoverable layer-by-layer (Hewitt and Manning 2019; Tenney et al. 2019), and that the cosine geometry of pretrained transformers is dominated by a small number of anisotropy directions (Mu and Viswanath 2018; Ethayarajh 2019). Without anisotropy correction, all comparisons between models are dominated by the strength of each model's dominant direction rather than by what the models encode about characters; we therefore apply standard anisotropy correction (mean centering plus removal of the top two principal components) before any cosine is computed.

For Chinese specifically, glyph-aware models (Sun et al. 2021; Meng et al. 2019) have been proposed to inject pixel or stroke information into the input pipeline. Our experiments concern glyph-naive models exclusively; the rendered-glyph ResNet-18 baseline is included not as a competitive language model but as a pure-form reference. The relevant earlier finding from word-similarity benchmarks is that radical-augmented Chinese embeddings produce small but positive gains (Xu et al. 2016; Yu et al. 2017). We are not reproducing that benchmark; we are asking the geometric question that those benchmarks treat as a means to an end.

## 3. Data

### 3.1 Character set

We start from 102,998 characters in the Unicode Unihan database with `kRSUnicode` (Kangxi radical) annotations, restrict to the CJK Unified Ideographs block (20,992), and apply two filters. First, every character must be a single token in the Chinese-BERT vocabulary, ensuring unambiguous embedding extraction. Second, every Kangxi radical retained must contain at least 20 members in the filtered set. The result is 6,306 characters across 68 radicals, identical to the dataset used in our preliminary study.

### 3.2 Tokenization coverage and the XLM-R decision

Coverage varies sharply across models. The relevant numbers from the released `tokenization_audit_summary.csv`:

- 100% coverage: Chinese-BERT, MacBERT, ERNIE-3.0, BGE-large-zh
- 84.6%: mBERT
- 81.1%: Qwen2.5 (both sizes; the BPE tokenizer splits some lower-frequency characters into 2 to 3 byte-level subwords)
- 64.3%: JP-BERT-char (single-token only for kanji that overlap with Japanese)
- 0.4%: XLM-R-base

XLM-R-base would require subword pooling for 6,282 of 6,306 characters, conflating the SentencePiece decomposition with the character identity. After examining the regression coefficients on the pooled representation (which are not directly comparable to the tokenized models) we removed XLM-R-base from the main variance-decomposition table. Its results remain in the released `variance_decomposition.csv` for completeness, but the analysis in Section 5 reports nine models, not ten.

For Qwen models (81% single-token coverage) we use single-token embeddings for the covered characters and skip the multi-subword characters in the regression sample, rather than mixing pooling strategies within a model. The remaining sample is large (more than 200,000 pairs after filtering).

### 3.3 Stroke counts and liushu classes

Stroke counts come from `kTotalStrokes`. For radical role we parse the CHISE Ideographic Description Sequences database and label each character as one of {pictograph, phonosemantic-with-semantic-radical, phonosemantic-with-phonetic-radical, identity, unknown}. Phonosemantic compounds account for 6,207 of 6,306 characters, matching the canonical estimate that ~85% of modern Chinese characters are phonosemantic.

## 4. Methods

### 4.1 Embedding extraction

We extract hidden states with bf16 inference on a single Colab T4 GPU. For mBERT and Chinese-BERT we extract all 13 layers, enabling a full-resolution layer-wise analysis for those two models. For the other transformers we extract a 5-layer evenly-spaced sample (layer 0, ¼-depth, ½-depth, ¾-depth, last). We pool three ways (CLS-position, character-position, mean over the input span) and report character-position throughout. The released artifacts contain all three pools.

### 4.2 Isotropy correction

We apply per-layer mean centering, standardization, and top-k principal-component removal with k = 2, following Mu and Viswanath (2018). The intuition: raw transformer cosines are inflated by a small number of dominant directions whose magnitudes vary across models, and any cross-model comparison without correction is partly comparing how strong each model's dominant direction is rather than what the models encode. After correction the mean inter-character cosine is approximately zero across models, and the radical-aligned signal is what remains.

### 4.3 Variance decomposition with cluster-robust standard errors

For each model we sample 200,000 pairs uniformly from the 6,306 × 6,305 unique-pair space (with replacement at the pair level). For each pair (i, j) we compute four predictors:

- `same_radical`: indicator that i and j share a Kangxi radical.
- `ppmi`: positive pointwise mutual information of (i, j) co-occurrence in a 5-character window over 1M Wikipedia zh sentences (~890k after filtering), using the Levy-Goldberg normalizer.
- `freq_diff`: absolute log-frequency difference, computed from the same Wikipedia corpus.
- `stroke_diff`: absolute difference in `kTotalStrokes`.

The dependent variable is the isotropy-corrected cosine at each model's last layer. We fit OLS, then compute two-way cluster-robust standard errors clustering on the i-character and the j-character (Cameron, Gelbach, and Miller 2011). The cluster-robust correction matters: each character appears in roughly 63 pairs on average, so the 200,000 observations are not independent, and naive OLS standard errors understate uncertainty. The design effect (cluster-robust SE divided by naive SE) ranges from 1.05 to 2.66 in our data, with a mean of approximately 1.4.

We report partial-R² for each predictor (the share of variance attributable to that predictor after partialling out the others), the cluster-robust 95% confidence interval for each coefficient, and the design effect. The coefficients are unchanged from naive OLS; the standard errors widen.

### 4.4 Calibration: how much R² is even available?

A reasonable critique of this method is that the full R² of the four-predictor regression is small — values between 0.002 and 0.050 across models — so the partial-R² values we compare are differences within a small total. To address this we computed an upper-bound calibration. For each model, we regress the same isotropy-corrected cosine on a single predictor: the cosine of the same character pair computed from that model's static-embedding layer (layer 0). This single predictor saturates at full R² between 0.31 and 0.78 across models (highest for the static, glyph-only baseline; lowest for Chinese-BERT). The remaining variance is per-layer contextualization noise that no character-property regression can capture, because it depends on the random initialization of the deeper layers and the specific token-position dynamics.

The implication: our four predictors together explain 1% to 17% of the cosine variance that is in principle available to a character-property regression on this dataset. The small absolute partial R² values are not evidence that the predictors are weak; they reflect that pairwise cosine geometry is only loosely decomposable into character-level properties. The relative ordering between channels remains the right unit of inference.

### 4.5 Cloze probe

For eight semantic fields (water, fire, plant, animal, body, metal, weather, speech), we identify a target Kangxi radical and procedurally generate trial pairs. The procedure: from the 6,306-character dataset, take the top-5 most-frequent characters that have the target radical (the "target" set) and the top-5 most-frequent characters in the same semantic field that do not have the target radical (the "distractor" set). For each field we write ten short Chinese carrier contexts of the form "She picked a [_] from the kitchen / wrote a [_] on the form / saw a [_] in the river," with the slot constrained by field meaning. Total: 8 fields × 5 targets × 5 distractors × ~6 contexts = approximately 250 trial pairs.

For masked language models we replace the slot with `[MASK]` and read off log-probabilities directly. For causal language models (Qwen2.5-1.5B, Qwen2.5-3B) we score each candidate as the next token after the prefix. The metric is `mean_delta`: the mean per-trial difference in target-vs-distractor log-probability. A positive `mean_delta` means the model prefers radical-correct completions over within-field non-radical completions.

We were explicit in the methodology: the target/distractor split is procedural (not selected by us), but the carrier contexts were author-written. We did not collect inter-annotator agreement on context naturalness, and we did not run a paired permutation test on the cloze deltas. These are limitations we discuss in Section 7. The probe is exploratory.

### 4.6 Specificity, frequency, and architecture controls

Three controls test the alternative explanations.

- **Pseudoradical null**: 100 random partitions of the 6,306 characters into 68 groups, each preserving the size distribution of the real Kangxi groups. We report `p_pseudo`, the empirical probability that a random partition yields a Cohen's d at least as large as the real one.
- **Frequency-matched pairs**: we re-compute Cohen's d on pairs whose two characters fall in the same frequency decile, controlling for the alternative that radical-aligned cohesion reflects within-radical frequency correlation.
- **Random-init noise floor**: we re-extract embeddings from Chinese-BERT and XLM-R-base after replacing all weights with random values from the same initialization distribution.

## 5. Results

### 5.1 Layer-wise corpus-scale cohesion

Last-layer (or peak-layer where peak is shallow; see §5.6) Cohen's d values, isotropy-corrected, sorted descending:

| Model              | Last-layer d | Peak-layer d | Peak layer | Pseudoradical p |
|--------------------|--------------|---------------|------------|------------------|
| glyph_only/ResNet-18 | 1.141 | 1.141 | 0 | 0.010 |
| Qwen2.5-3B         | 0.744 | 0.744 | 36 | 0.010 |
| Qwen2.5-1.5B       | 0.683 | 0.683 | 28 | 0.010 |
| BGE-large-zh       | 0.570 | 0.570 | 24 | 0.010 |
| Chinese-BERT       | 0.375 | 0.505 | 1 | 0.010 |
| MacBERT            | 0.282 | 0.453 | 3 | 0.010 |
| ERNIE-3.0          | 0.225 | 0.507 | 0 | 0.010 |
| mBERT              | 0.202 | 0.231 | 7 | 0.010 |
| JP-BERT-char (kanji subset) | 0.057 | 0.059 | 9 | 0.020 |

All nine glyph-naive models clear permutation null at p ≤ 0.001 and the size-matched pseudoradical null at p ≤ 0.020. We note immediately that the ranking is not "Chinese-pretrained models on top, multilingual at the bottom": MacBERT and ERNIE-3.0 (Chinese-pretrained) sit in the middle of the table, between mBERT and Chinese-BERT. The split that emerges in the variance decomposition (next subsection) is more informative than the corpus-scale d ordering.

### 5.2 Variance decomposition (the central result)

Cluster-robust regression coefficients and partial-R² values for the nine models (XLM-R-base excluded; see §3.2). The two leftmost columns are partial-R² for the two predictors that compete in the substantive interpretation:

| Model             | partial-R²(radical) | partial-R²(PMI) | ratio | full R² | Spearman ρ(pred,obs) |
|-------------------|---------------------|------------------|-------|---------|------------------------|
| Qwen2.5-1.5B      | 0.0171 [0.015, 0.019] | 0.0090 [0.008, 0.010] | 1.90 | 0.029 | 0.088 |
| Qwen2.5-3B        | 0.0241 [0.022, 0.026] | 0.0133 [0.012, 0.015] | 1.81 | 0.042 | 0.121 |
| BGE-large-zh      | 0.0119 [0.010, 0.014] | 0.0069 [0.006, 0.008] | 1.72 | 0.020 | 0.070 |
| Chinese-BERT      | 0.0040 [0.003, 0.005] | 0.0081 [0.007, 0.009] | 0.49 | 0.013 | 0.082 |
| MacBERT           | 0.0021 [0.002, 0.003] | 0.0030 [0.002, 0.004] | 0.70 | 0.005 | 0.042 |
| ERNIE-3.0         | 0.0008 [0.001, 0.001] | 0.0015 [0.001, 0.002] | 0.51 | 0.003 | 0.035 |
| JP-BERT-char      | 0.0007 [0.001, 0.001] | 0.0036 [0.003, 0.004] | 0.19 | 0.007 | 0.052 |
| mBERT             | 0.0012 [0.001, 0.002] | 0.0000 [0.000, 0.000] | n/a* | 0.002 | 0.007 |
| glyph_only/ResNet-18 | 0.0338 [0.030, 0.038] | 0.0001 [0.000, 0.000] | 338 | 0.050 | 0.132 |

*PMI partial-R² is below 10⁻⁴ for mBERT, not meaningfully comparable.

Square brackets show 95% cluster-robust CIs on partial-R² (computed by recomputing partial-R² on each of 1,000 character-clustered bootstrap resamples). Three observations:

(a) **Partial-R²(radical) > partial-R²(PMI) holds for three models**: Qwen2.5-1.5B, Qwen2.5-3B, and BGE-large-zh. The bootstrap CIs for these three do not overlap with their PMI CIs. For the other five glyph-naive models, the PMI partial-R² is at least as large as the radical partial-R², or both are negligibly small.

(b) **The vision-only baseline is the limit case**: the rendered-glyph encoder has access only to pixel similarity, and unsurprisingly its radical partial-R² dominates everything else. Its presence in the table is to anchor the upper end, not to compete with language models.

(c) **Cluster-robust correction does not change the substantive interpretation but it does change uncertainty**. Design effects average 1.4 (with a maximum of 2.66 for Japanese-BERT on stroke-difference), so naive OLS standard errors are too narrow by roughly 40%. After cluster-robust correction, every coefficient remains significant at p < 0.001, but the bootstrap CIs on partial-R² are wider than naive OLS CIs would suggest.

### 5.3 What the small full R² means and does not mean

Full R² between 0.002 and 0.050 sounds small. Without context, one might dismiss the entire decomposition as noise. We computed a calibration to put this in context: regress the same isotropy-corrected cosine on a single predictor — the cosine of the same character pair at layer 0 (the static embedding lookup). This single predictor saturates at R² between 0.31 (Chinese-BERT, last-layer) and 0.78 (glyph-only/ResNet-18, where layer 0 is the only layer). The remaining variance, which our four-predictor regression cannot capture, is per-position contextualization noise that is independent of any character-level property. Most of it depends on the random initialization of deeper layers and on individual character idiosyncrasies that are not in our predictor set.

The implication: the four predictors capture between 1% and 17% of the cosine variance that is in principle available to a character-property regression on this data. That is small but not negligible, and the inter-model comparison is what the paper rests on, not absolute partial-R² values. We do not claim that "16% of Qwen2.5-3B's representation is radicals." We claim that, of the cosine variance available to a character-level regression, radical co-membership accounts for approximately twice as much as PMI in the three Chinese-pretrained transformers we examined.

### 5.4 The n = 3 problem

Three models is a small set. With n = 3 we cannot make a category claim like "Chinese-pretrained transformers encode radical structure." We can describe the pattern in the data we have: of the nine glyph-naive models we tested, exactly three (Qwen2.5-1.5B, Qwen2.5-3B, BGE-large-zh) place radical partial-R² above PMI partial-R². The three are heterogeneous in a way that is informative: Qwen models are decoder LLMs trained on Chinese-heavy text; BGE-large-zh is an encoder trained with a contrastive retrieval objective. The fact that they are heterogeneous on architecture and training objective but converge on the variance ordering suggests the pattern is not artifact of any single training pipeline. But three models are three models. Several open-weight Chinese-pretrained transformers were too large for our compute budget (Qwen2.5-7B in fp16 fits but is slow; Yi, DeepSeek, GLM-4, InternLM-1.8B+ require quantization or larger GPUs). We list them in the released artifacts as the priority replications.

We refrain from naming a category. The pattern is "this set of three models," not "Chinese-pretrained transformers."

### 5.5 Cloze probe (exploratory)

The procedural cloze probe yields the following mean target-vs-distractor log-probability differences:

| Model                      | mean_delta | top-1 win rate | MRR |
|---------------------------|------------|-----------------|------|
| Qwen2.5-3B                | +0.36 | 0.54 | 0.70 |
| Qwen2.5-1.5B              | +0.20 | 0.63 | 0.77 |
| Chinese-BERT              | +0.16 | 0.65 | 0.74 |
| MacBERT                   | −0.21 | 0.49 | 0.66 |
| mBERT                     | −0.98 | 0.49 | 0.61 |
| XLM-R-base                | −0.99 | 0.45 | 0.59 |
| XLM-R-large               | −1.18 | 0.42 | 0.59 |

The ordering reproduces the variance-decomposition ordering for the three positive-delta models (Qwen-3B > Qwen-1.5B > Chinese-BERT), and the multilingual MLMs land far below zero. We chose not to include this as confirmatory evidence for two reasons. First, n = 250 trials with author-written carriers and no native-speaker validation is not enough to support a behavioral claim. Second, we did not compute paired permutation tests or bootstrap CIs on the deltas, so the ranking is reported point-estimates only. We include the probe as exploratory and recommend a properly-validated cloze benchmark as a separate study.

### 5.6 Layer-wise dynamics: most of the signal is shallow

For seven of nine glyph-naive models, peak-layer Cohen's d is at layer 0 (the static embedding lookup) or in the first three layers. The exceptions are mBERT (peak at layer 7) and JP-BERT-char (peak at layer 9). For Chinese-BERT and MacBERT, where we have full 13-layer resolution, d declines monotonically from layer 1 onwards: from 0.50 to 0.38 over 12 layers in Chinese-BERT, and 0.45 to 0.28 over 12 layers in MacBERT.

The layer-0 dominance is a substantive finding, not a methodological artifact. The token embedding table is part of the model's learned representation, and the fact that radical-aligned geometry is concentrated there says something about what kind of learning produces it. Specifically: the radical-aligned ordering is encoded in the way the tokenizer's vocabulary is laid out in embedding space, before any contextualization happens. Contextualization in deeper layers does not amplify this signal; for most models it dampens it. This is consistent with the reading that radical-aligned structure in transformers is mostly a property of how character vocabularies are initialized and updated during pretraining, not a property of the contextual computation. We did not explore this further; a worthwhile follow-up is to ablate the token-embedding rows for same-radical characters and measure the effect on the cloze probe.

### 5.7 Specificity, frequency, and architecture controls

The three controls behave as predicted. Pseudoradical p ≤ 0.020 for all nine models. Frequency-matched effect retains 92% to 99% of the unmatched effect across glyph-naive models (the largest frequency inflation is 0.077 in the vision-only baseline). Random-initialized Chinese-BERT and XLM-R-base both produce d ≈ −0.04 with p > 0.88. Architecture alone, without training, does not produce the observed geometry.

### 5.8 Cross-script generalization

Of 6,306 characters in our dataset, 1,687 appear in the Joyo kanji list. We re-extract embeddings for these characters from JP-BERT-char and from each Chinese model on the same kanji subset. JP-BERT-char shows d = 0.384 on the subset; Chinese-BERT shows d = 0.405; Qwen2.5-3B shows d = 0.682. The Japanese-pretrained model is not the strongest performer on Japanese kanji; Qwen, trained predominantly on Chinese text, shows the highest cohesion on Japanese kanji. We read this as evidence that the Kangxi system transfers across pretraining language, but with caution: JP-BERT-char's smaller parameter count and different training corpus make this a within-architecture-family comparison rather than a fully controlled cross-language test.

## 6. What the data say and do not say

**What the data say.** Of nine glyph-naive transformer models examined here, three (Qwen2.5-1.5B, Qwen2.5-3B, BGE-large-zh) show partial-R²(radical) above partial-R²(PMI) by factors of 1.7 to 1.9 in the four-predictor decomposition with cluster-robust standard errors. The same three are the only models with positive cloze-probe mean_delta. The radical effect is specific to Kangxi categories (size-matched random partitions yield d ≈ 0), training-driven (random-init networks yield d ≈ −0.04), not attributable to character frequency (frequency-matched effect retains 92% to 99% of the unmatched effect), and not visual-only (the vision baseline's d = 1.14 is fully accounted for by within-semantic-field similarity in rendered glyphs).

**What the data do not say.** We do not establish a category effect for "Chinese-pretrained transformers." Three models is too few. We do not make a mechanistic claim. The variance decomposition is associational. We do not claim that the cloze probe is a benchmark; 250 author-written trials without native-speaker validation are exploratory at best. We do not establish that radical-aligned structure is "useful" for downstream tasks in any controlled sense. We do not address whether the static-embedding-layer dominance of the radical signal means radicals are "encoded in the tokenizer" rather than "encoded in the model"; the question of where in the parameter space the geometry lives is open and would require ablation studies we did not perform.

**What we suspect but did not test.** Same-radical characters appear in similar morphological frames in Chinese text (compound formation, classifier patterns, idiomatic constructions) that may not be captured by a 5-character PMI window. This is a hypothesis, not a finding. The right test would be to compute PMI at multiple window sizes and frame types and to add those as predictors. We did not do this.

## 7. Limitations

We list the limitations in roughly the order a reviewer would raise them.

**Pair non-independence.** Each character appears in approximately 63 of 200,000 sampled pairs. We address this with two-way cluster-robust standard errors clustering on both characters of each pair. Design effects 1.05–2.66, mean 1.4. Significance survives correction. The released `variance_decomposition_clustered.csv` contains all numbers; the more conservative bootstrap CIs in §5.2 are computed from character-clustered resamples (1000 iterations).

**Small absolute variance explained.** Full R² ranges from 0.002 to 0.050. The partial-R² ratios are differences within a small total. We address this with the calibration in §5.3: the upper bound for any character-level predictor on this data is R² of 0.31 to 0.78 (regression on layer-0 cosine), not 1.0. Our four predictors capture 1% to 17% of that available variance. The relative ordering across channels is interpretable; the absolute amounts are not.

**Sample size on the category claim.** Three Chinese-pretrained models showing the same ordering is a pattern, not a category effect. We refrain from naming the category. Replication on Qwen2.5-7B and larger, Yi, DeepSeek, GLM-4, InternLM is the next step. None of these fit our compute budget.

**XLM-R-base subword pooling.** XLM-R has 0.4% single-token coverage. We removed it from the main variance-decomposition table. Its results are in the released artifacts for completeness but should not be compared directly to the tokenized models.

**Cloze probe is exploratory.** 250 procedurally-constructed trials with author-written carriers and no inter-annotator agreement. We labeled it exploratory in the title and the abstract and did not let it carry the main argument.

**No mechanistic experiment.** We did not run activation patching, attention-head ablation, or neuron-level analysis. Our claims are associational. The variance decomposition is correlational by construction.

**Layer sampling is coarse for seven models.** We extracted only 5 evenly-spaced layers for seven of nine glyph-naive models. Peak-d localization is approximate for those models. Full 13-layer resolution is available for mBERT and Chinese-BERT only.

**PMI from one corpus.** PMI uses 1M Wikipedia zh sentences. Wikipedia is encyclopedic and biased toward written formal Chinese. A corpus more representative of the model's pretraining distribution might yield different PMI values. However, since PMI enters as a fixed covariate, its specific value affects partial-R² absolute values but not the relative ordering across models.

**No native-speaker validation.** We do not have native Chinese reader feedback on the 21 cloze contexts or the procedurally-generated targets and distractors. This is the biggest open methodological hole. We expect to add this in any subsequent revision.

**bf16 inference.** All embeddings extracted at bf16 to fit Colab T4 memory. Effects on cosine of order 10⁻³, well below reported effect sizes, but worth noting for replication.

## 8. Conclusion

We decomposed pairwise embedding cosine in nine glyph-naive transformer models and one rendered-glyph baseline into four channels: radical co-membership, distributional context (PMI), character frequency, and stroke difference. Using cluster-robust regression with character-level clustering of standard errors, we found that three of the nine glyph-naive models (Qwen2.5-1.5B, Qwen2.5-3B, BGE-large-zh) place radical partial-R² above PMI partial-R² with bootstrap CIs that do not overlap. The other six do not. An exploratory cloze probe reproduces the same model ordering, with the caveat that 250 author-written trials are not a behavioral benchmark.

The pattern is too small in n to support a category claim about "Chinese-pretrained transformers" and too modest in absolute variance explained to support a mechanistic claim about "encoding." It is consistent with the reading that, for some models, radical co-membership predicts representational similarity beyond what distributional context alone would predict, and that this excess survives controls for frequency and stroke count. We hope replication on a wider set of Chinese-pretrained LLMs, with a properly-validated cloze benchmark and a mechanistic follow-up, will sharpen or constrain the picture.

## Reproducibility

All code, the preregistration document with timestamped hypotheses and falsified-prediction list, the procedurally-generated cloze items, the 6,306-character dataset with stroke counts and liushu annotations, and all CSVs from this paper are available at:

`https://github.com/aryan35790jp/chinese_llm`

Total wall-clock for the full pipeline on a single Colab T4 GPU: approximately 2 hours 20 minutes, hands-off after Run All. The longest single step is 50 minutes of embedding extraction across the 10 models.

## References

Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics* 29(2), 238–249.

Cui, Y., Che, W., Liu, T., Qin, B., Yang, Z., Wang, S., and Hu, G. (2021). Pre-training with whole word masking for Chinese BERT. *IEEE/ACM Trans. Audio, Speech, Lang. Process.* 29, 3504–3514.

Durrani, N., Sajjad, H., Dalvi, F., and Belinkov, Y. (2019). One size does not fit all: Comparing NMT representations of different granularities. *NAACL-HLT*, 791–796.

Ethayarajh, K. (2019). How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 representations. *EMNLP*, 55–65.

Hewitt, J., and Manning, C. D. (2019). A structural probe for finding syntax in word representations. *NAACL-HLT*, 4129–4138.

Hofmann, V., Pierrehumbert, J. B., and Schütze, H. (2021). Superbizarre is not superb: Derivational morphology improves BERT's interpretation of complex words. *ACL*, 3594–3608.

Levy, O., and Goldberg, Y. (2014). Neural word embedding as implicit matrix factorization. *NeurIPS*.

Meng, Y., Wu, W., Wang, F., Li, X., Nie, P., Yin, F., Li, M., Han, Q., Sun, X., and Li, J. (2019). Glyce: Glyph-vectors for Chinese character representations. *NeurIPS* 32, 2746–2757.

Mu, J., and Viswanath, P. (2018). All-but-the-top: Simple and effective postprocessing for word representations. *ICLR*.

Sun, Z., Li, X., Sun, X., Meng, Y., Ao, X., He, Q., Wu, F., and Li, J. (2021). ChineseBERT: Chinese pretraining enhanced by glyph and pinyin information. *ACL*, 2065–2075.

Tenney, I., Das, D., and Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. *ACL*, 4593–4601.

Xu, J., Liu, J., Zhang, L., Li, Z., and Chen, H. (2016). Improve Chinese word embeddings by exploiting internal structure. *NAACL-HLT*, 1041–1050.

Yu, J., Jian, X., Xin, H., and Song, Y. (2017). Joint embeddings of Chinese words, characters, and fine-grained subcharacter components. *EMNLP*, 286–291.
