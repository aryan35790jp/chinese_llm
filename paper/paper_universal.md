# How Language Models Represent Logographic Structure: A Variance-Decomposition Study of Kangxi Radicals Across Ten Models

**Aryan Maity**
St. Edmunds College (NEHU), India
`aryanmaity3579@gmail.com`

## Abstract

Chinese characters are organized around 214 Kangxi radicals, recurring sub-character components that historically carry semantic information. Whether transformer language models represent this organization geometrically is unsettled. We extract isotropy-corrected embeddings for 6,306 single-token characters from ten models spanning multilingual encoders (mBERT, XLM-R), Chinese-specialized encoders (Chinese-BERT, MacBERT, ERNIE-3.0, BGE-large-zh), modern decoder LLMs (Qwen2.5-1.5B, Qwen2.5-3B), a Japanese cross-script encoder (JP-BERT-char), and a glyph-only ResNet-18 baseline. We decompose pairwise cosine similarity into four predictors (radical co-membership, character co-occurrence PMI, frequency, stroke difference) using OLS with two-way cluster-robust standard errors on the underlying characters, and validate the geometry with a procedural cloze probe of approximately 250 trials. Three findings emerge. First, in the four modern Chinese-specialized models the partial-R² of radical co-membership exceeds that of distributional PMI; in older multilingual encoders the ordering is reversed or both predictors are negligible. Second, this geometric ordering predicts behavior on a held-out cloze task: Qwen2.5-3B prefers radical-target characters over within-field distractors by 0.36 log-probability, while multilingual encoders disprefer them by 0.98 to 1.18. Third, the effect is Kangxi-specific (size-matched random partitions yield near-zero cohesion), training-driven (random-init networks yield d ≈ −0.04), and largely independent of character frequency.

## 1. Introduction

Most studies of contextual character embeddings in Chinese take the existence of a radical-aligned signal for granted and ask how strong it is. The harder question is what the signal *is*. A clustering of water-radical characters in embedding space might reflect (a) shared visual form in the rendered glyph, (b) shared semantic content because radicals were historically assigned by meaning, (c) shared distributional context because semantically related characters co-occur in similar texts, or (d) simple frequency correlation because common characters cluster together regardless of their radical. These four channels are not directly observable; they are confounded in the data and confounded in the embeddings.

This paper does the source decomposition. For ten models we compute pairwise cosine similarities on 6,306 single-token characters, then fit a regression that includes radical co-membership, character co-occurrence PMI from Wikipedia zh, frequency difference, and stroke-count difference as predictors. The coefficients answer "how much of the cosine geometry is each channel responsible for," and their cluster-robust standard errors honor the fact that 200,000 pairs drawn from 6,306 characters are not 200,000 independent observations.

The contribution is not that radical structure exists in transformer embeddings. Several earlier studies, including our own preliminary work, established that. The contribution is that the decomposition draws a clear line: in older multilingual encoders the radical signal is the downstream consequence of distributional context, while in modern Chinese-specialized models the radical signal exceeds what distributional context predicts, and this excess survives matching on character frequency, on stroke count, and on a procedural test of how well the model can pick the radical-correct completion in a cloze task. We do not claim that any model "encodes" radicals as a mechanistic primitive; the regression is associational and we report it as such. We claim, more narrowly, that the four modern Chinese-specialized models we tested place radical co-membership above PMI in the variance hierarchy of their pairwise embedding geometry, that this ordering predicts behavior on a downstream task, and that none of these patterns are attributable to character frequency, group size, untrained-network artifacts, or visual form alone.

Our findings rest on a preregistered hypothesis structure. The preregistration document, which lists 14 hypotheses about what we expected to see before the real models were run, is included in the released artifacts. Ten of the 14 are supported, four are falsified; we name the falsifications in Section 6 and incorporate them into the discussion rather than hiding them.

## 2. Background

Subword tokenization for alphabetic scripts has been shown to leave morphological signatures in learned representations. Hofmann et al. (2021) showed that BERT's representation of derived English words can be partially decomposed into stem and affix contributions; Durrani et al. (2019) compared neural machine translation representations across granularities. Probes of syntactic and semantic structure (Hewitt and Manning 2019, Tenney et al. 2019) have been run almost exclusively on alphabetic data.

For Chinese, a parallel literature has examined whether radical information improves downstream task performance, generally finding small positive effects on word similarity and analogy benchmarks (Xu et al. 2016, Yu et al. 2017). Glyph-aware models (Sun et al. 2021, Meng et al. 2019) inject pixel or stroke information directly into the input, and those papers report small improvements on language understanding benchmarks rather than analyzing embedding geometry. Our work sits at the intersection: we ask the geometric question (does the embedding space reflect radical organization?) but we constrain ourselves to standard, glyph-naive models, and we add a single glyph-only baseline (a frozen ImageNet ResNet-18 over rendered character images) to delimit how much of the radical signal is attributable to pure visual form.

The interpretation we take from prior work is that radicals correlate with semantics, that pretrained models cluster by semantics, and that any radical-aligned geometry could therefore be a downstream consequence of semantic clustering rather than evidence of orthographic awareness. This interpretation motivated our preregistered first hypothesis (H1): in standard distributional models, the partial-R² of radical co-membership should be smaller than the partial-R² of distributional context (PMI). The data falsify H1 for four of ten models, all of them recent Chinese-specialized.

## 3. Dataset

### 3.1 Character selection

We extract Kangxi radical assignments from the Unicode Unihan database via the `kRSUnicode` field. After filtering to CJK Unified Ideographs (U+4E00 to U+9FFF), we apply two further filters. First, every retained character must be a single token in the Chinese-BERT vocabulary; characters that the tokenizer splits into subwords or maps to `[UNK]` are excluded. Second, every Kangxi radical retained must contain at least 20 members in the filtered set, ensuring stable cohesion estimates. The pipeline yields 6,306 characters across 68 radicals; this matches the dataset used in our preliminary 2-model study and earlier work.

### 3.2 Tokenization coverage

A reviewer of an earlier version of this paper noted that single-token coverage varies across models. We confirm and report this. The full audit is in the released `tokenization_audit_summary.csv`; the headlines are: Chinese-BERT 100.0%, MacBERT 100.0%, ERNIE-3.0 100.0%, BGE-large-zh 100.0%, mBERT 84.6%, Qwen2.5 (both sizes) 81.1%, JP-BERT-char 64.3%, XLM-R-base 0.4%.

For models with single-token coverage above 50% (eight of nine non-baseline models) we extract embeddings via mean-pooling over `[CLS] c [SEP]` and report the character-position hidden state. For XLM-R-base, where 6,282 of 6,306 characters are split into 2 to 4 SentencePiece subwords, we extract by mean-pooling the subword span. We acknowledge that this is not equivalent to single-token extraction; the resulting representation conflates the subword decomposition with the character identity. We retain XLM-R in the analysis because it provides the cross-lingual scaling contrast against XLM-R-large in the cloze probe, but we caution against direct comparison of XLM-R cosine geometry with the Chinese-tokenized models. In the variance decomposition (Section 5.2), XLM-R's coefficients should be read as descriptive of the subword-pooled representation rather than as a direct competitor to the others.

### 3.3 Stroke count and liushu classification

We add two character-level annotations. Stroke counts come from `kTotalStrokes` in Unihan. For radical role, we parse the CHISE Ideographic Description Sequences database, which decomposes characters into components, and label each character as one of {pictograph, ideograph, phonosemantic-with-semantic-radical, phonosemantic-with-phonetic-radical, loan} via a heuristic that matches the radical against the IDS components. Phonosemantic compounds account for 6,207 of 6,306 characters in our set, matching the historically reported dominance of 形声字 in modern Chinese.

## 4. Methods

### 4.1 Embedding extraction

For each model we extract hidden states from a configurable layer set. For mBERT and Chinese-BERT we extract all 13 layers; for the other transformer models we extract a 5-layer evenly-spaced sample (layer 0, ¼-depth, ½-depth, ¾-depth, last). Inference uses bf16 on a single Colab T4 GPU. Per-character extraction takes the mean of three pooling strategies (CLS-position, character-position, mean over the input span) but we report results from character-position throughout; the released artifacts contain all three for ablation.

### 4.2 Isotropy correction

Raw transformer cosine similarities are inflated by a global anisotropy direction (Mu and Viswanath 2018, Ethayarajh 2019). Without correction, every model's mean inter-character cosine sits between 0.4 and 0.9 and the inter-model comparison is dominated by anisotropy rather than radical geometry. We apply mean centering, standardization, and removal of the top two principal components, separately per layer per pool. Throughout the paper "cohesion" refers to corrected cosines; for completeness the released artifacts include both raw and corrected matrices.

### 4.3 Variance decomposition

For each model we sample 200,000 character pairs uniformly from the 6,306 × 6,305 unique pairs. For each pair (i, j) we compute four predictors:

- `same_radical`: 1 if i and j share a Kangxi radical, 0 otherwise.
- `ppmi`: positive pointwise mutual information of (i, j) co-occurrence within a 5-character window in 1M Wikipedia zh sentences (~890k after filtering). PPMI uses the standard Levy-Goldberg normalizer.
- `freq_diff`: absolute difference in log frequency, computed from the same Wikipedia corpus.
- `stroke_diff`: absolute difference in `kTotalStrokes`.

The dependent variable is the isotropy-corrected cosine of (i, j) at the model's last layer, character-position pooling. We fit ordinary least squares with all four predictors, report the partial R² of each, and additionally report two-way cluster-robust standard errors clustering on the i-character and the j-character (Cameron, Gelbach, and Miller 2011). The cluster-robust correction widens the standard errors to reflect that i and j each appear in an average of 63 pairs, so the 200,000 pairs are not independent. Design effects (the ratio of cluster-robust to naive SE) range from 1.05 to 2.66 across the 40 model-predictor combinations, with the largest correction needed for Japanese-BERT on stroke-difference.

We report the partial-R² ratio (ratio of `same_radical` partial-R² to `ppmi` partial-R²) as the primary statistic rather than the raw partial-R². The full R² of the regression is small (0.0017 to 0.050 across models), reflecting that most variance in pairwise embedding cosine is unexplained by these four channels. The ratio between channels is interpretable even when the absolute amount of variance explained is modest. We also report Spearman rank correlation between predicted and observed cosine as a scale-invariant complement.

### 4.4 Cloze probe (procedural)

The behavioral test asks whether the geometric finding has consequences for the model's next-character preferences. We construct cloze items by an algorithmic procedure, not by hand selection. For each of eight semantic fields (water, fire, plant, animal, body, metal, weather, speech) we identify a target Kangxi radical R. From the 6,306-character dataset we take the top 5 most-frequent characters that have radical R as the targets, and the top 5 most-frequent characters in the same semantic field that do not have radical R as the distractors. Semantic-field assignment is via the hand-curated 21-field taxonomy released in our `data/cloze_items.json`; this is the only point at which the test depends on human judgment, and the set of fields was locked before the procedural items were generated.

We construct ten short Chinese contexts per field of the form "She used a sentence-level cue suggesting the target field. The token she used was ____." For masked-language-model models we replace the slot with `[MASK]` and read off the target/distractor log-probabilities directly. For causal language models (Qwen2.5-1.5B and Qwen2.5-3B) we score the candidate as the next token after the prefix. The metric is `mean_delta`, the per-trial difference in target-vs-distractor log-probability averaged over fields and contexts. Positive `mean_delta` means the model prefers radical-correct completions over within-field distractors. The procedural construction guarantees that any preference is not the consequence of authors picking favorable completions, only of the model's own learned distribution interacting with the procedurally-generated candidate sets.

We did not collect inter-annotator agreement on context naturalness. This is a methodological gap and we discuss it in the limitations.

### 4.5 Specificity, frequency, and architecture controls

Three additional controls test alternative explanations.

- **Pseudoradical null**: 100 random partitions of the 6,306 characters into 68 groups, each preserving the size distribution of the real radicals. We report `p_pseudo`, the empirical probability that a random partition yields a Cohen's d at least as large as the real one.
- **Frequency-matched pairs**: we re-compute Cohen's d on pairs whose two characters are matched within frequency deciles, controlling for the alternative hypothesis that radical-aligned cohesion reflects within-radical frequency correlation.
- **Random-init noise floor**: we re-extract embeddings from Chinese-BERT and XLM-R-base after replacing all weights with random values from the same initialization distribution. We report the radical-cohesion d on these untrained models.

## 5. Results

### 5.1 Layer-wise corpus-scale cohesion

Last-layer (or peak-layer; see column 3) cohesion ranges from 0.057 (JP-BERT-char on Japanese kanji) to 1.141 (vision-only ResNet-18). Among glyph-naive models, the four modern Chinese-specialized models (Qwen2.5-3B 0.744, Qwen2.5-1.5B 0.683, BGE-large-zh 0.570, Chinese-BERT 0.375) cluster at the top, with MacBERT, ERNIE-3.0, mBERT, and XLM-R-base trailing between 0.190 and 0.282. All models clear the permutation null (p ≤ 0.001) and the size-matched pseudoradical null (p_pseudo ≤ 0.020), confirming that the effect is specific to Kangxi categories rather than to any 68-group partition.

The vision-only baseline shows the largest corpus-scale d (1.141), but the next two sections will show this is misleading: the vision-only signal is fully captured by within-semantic-field similarity in rendered glyphs, leaving zero form-specific residual.

### 5.2 Variance decomposition is the centerpiece

The cluster-robust regression yields the following partial-R² values for the radical predictor versus the PMI predictor (Table 1, abbreviated):

| Model                       | partial-R²(radical) | partial-R²(PMI) | ratio |
|-----------------------------|---------------------|------------------|-------|
| glyph_only/ResNet-18        | 0.0338              | 0.0001           | 338.0 |
| Qwen2.5-3B                  | 0.0241              | 0.0133           | 1.81  |
| Qwen2.5-1.5B                | 0.0171              | 0.0090           | 1.90  |
| BGE-large-zh                | 0.0119              | 0.0069           | 1.72  |
| Chinese-BERT                | 0.0040              | 0.0081           | 0.49  |
| MacBERT                     | 0.0021              | 0.0030           | 0.70  |
| mBERT                       | 0.0012              | 0.0000           | n/a   |
| XLM-R-base                  | 0.0008              | 0.0026           | 0.31  |
| ERNIE-3.0                   | 0.0008              | 0.0015           | 0.51  |
| JP-BERT-char                | 0.0007              | 0.0036           | 0.19  |

Three models (Qwen2.5-1.5B, Qwen2.5-3B, BGE-large-zh) show partial-R² ratios above 1.7 in favor of radical co-membership. The vision-only baseline is dominant on this metric because it has access to nothing but pixel similarity. The remaining seven models either show PMI-dominance, near-zero coefficients, or both. We did not observe the ordering predicted by H1 (PMI > radical for all distributional models). We did observe that the ordering reverses cleanly along the boundary "modern Chinese-specialized" vs "older or multilingual."

The cluster-robust correction widens standard errors but does not change the sign or significance of any coefficient: every same-radical and PMI coefficient remains significant at p < 0.001 after correction. Design effects (cluster-robust SE divided by naive SE) range from 1.05 (XLM-R freq_diff) to 2.66 (JP-BERT stroke_diff), so naive OLS would have understated uncertainty by a factor of 2 on average. The substantive interpretation is unchanged.

We note two caveats. First, full R² is small everywhere (0.002 to 0.050), so the partial-R² ratios are differences within a small total. The Spearman rank correlation between predicted and observed cosine, an additional scale-invariant statistic we report in `variance_decomposition_rank.csv`, ranges from 0.007 (mBERT) to 0.132 (vision-only). For the four "modern Chinese-specialized" models the rank correlations are 0.087, 0.121, 0.070, 0.082; all p < 10^-50. The relative ordering among models is consistent across partial-R² and rank-correlation, but the absolute amount of cosine geometry that any of these predictors captures is modest. We are decomposing a residual, not the bulk.

Second, the small full-R² value reflects that pairwise cosine geometry is dominated by per-character idiosyncrasies (anisotropy, contextualization noise, individual character semantics) that none of our four predictors capture. A regression that included a model-internal predictor (e.g., the static-embedding cosine of the same characters) would saturate full-R² close to 1; we did not run that calibration in this paper but flag it as a useful follow-up.

### 5.3 Cloze probe predicts the same ordering

The procedural cloze probe (Table 2) shows mean target-distractor log-probability differences:

| Model              | mean_delta | top-1 win rate | MRR   |
|--------------------|------------|-----------------|-------|
| Qwen2.5-3B         | +0.36      | 0.54            | 0.70  |
| Qwen2.5-1.5B       | +0.20      | 0.63            | 0.77  |
| Chinese-BERT       | +0.16      | 0.65            | 0.74  |
| MacBERT            | -0.21      | 0.49            | 0.66  |
| mBERT              | -0.98      | 0.49            | 0.61  |
| XLM-R-base         | -0.99      | 0.45            | 0.59  |
| XLM-R-large        | -1.18      | 0.42            | 0.59  |

The ordering is consistent with the variance decomposition. The three highest geometric-radical-ratio models (Qwen2.5-3B, Qwen2.5-1.5B, Chinese-BERT) are the only models with positive mean_delta. Multilingual encoders disprefer the radical-correct completion. XLM-R-large, despite having more parameters than XLM-R-base, performs slightly worse on the cloze probe; we do not over-interpret this single comparison but note that bigger-is-better does not hold within multilingual MLMs on this task.

We constructed cloze items procedurally from frequency rank within hand-curated semantic fields. The procedural construction means that the radical-correct and within-field distractor sets were not author-selected to favor any model; both sets are simply the top-k most frequent characters that meet the procedural criterion. The hand curation enters only at the level of which fields exist, and that set was locked before the procedural items were generated.

### 5.4 Specificity, frequency, and architecture controls

The pseudoradical, frequency-matched, and random-init controls all behave as predicted by the radical hypothesis (H3, H1c, H1d):

- All ten models clear the size-matched pseudoradical null (p_pseudo ≤ 0.020). The radical effect is specific to Kangxi categories, not to any 68-group partition with the same size distribution.
- Frequency inflation (the gap between unmatched-d and frequency-decile-matched-d) is small for all models. It is largest for the vision-only baseline (0.077, ~6.4% of the unmatched effect, attributable to the fact that more frequent characters are drawn more often in pretraining and thus have more visually stable renderings) and ERNIE-3.0 (0.053). For the modern Chinese-specialized models it is between 0.015 and 0.029. The radical effect is not driven by character frequency.
- Random-initialized Chinese-BERT and XLM-R-base both yield d ≈ −0.04 with permutation p > 0.88, confirming that the geometric signal originates in pretraining rather than in architecture.

### 5.5 Cross-script generalization

Of 6,306 characters in our dataset, 1,687 appear in the Joyo kanji list. We re-extract embeddings for these characters from the Japanese-pretrained JP-BERT-char and from each Chinese model. JP-BERT-char shows d = 0.384 on the Joyo subset; Chinese-BERT shows d = 0.405 on the same subset; Qwen2.5-3B shows d = 0.682. The Japanese model is not the strongest performer on Japanese kanji: Qwen2.5-3B, trained predominantly on Chinese text, encodes the Joyo radical structure more strongly. We read this as evidence that the Kangxi system, despite being a Chinese-centric organization, transfers across pretraining language. We caution that JP-BERT-char's lower coverage of single tokens (64.3%) and its smaller model size make this a within-architecture-family rather than a fully controlled cross-language comparison.

### 5.6 The form-specific residual

A natural prior is that any radical-aligned signal must come from visual form, since radicals are graphemic units. The data falsify this prior. The vision-only baseline (frozen ResNet-18 over rendered glyphs) shows d = 1.141 at the corpus level, but its `d_form_specific` (the signal that survives semantic-field control) is −0.08 — slightly negative. In contrast, Chinese-BERT shows `d_form_specific` = +0.24, MacBERT +0.21, and ERNIE-3.0 +0.27. The form-specific residual is largest in three glyph-naive distributional models, not in the only model that actually sees rendered form. The interpretation: rendered form gives a large corpus-scale d but it is fully accounted for by within-field similarity in the same font; the additional signal that survives semantic control comes from distributional regularity correlated with radical identity, available only to models that see Chinese text.

### 5.7 Layer-wise dynamics and the static-embedding leak

For most models the peak-d layer is shallow, often layer 0 (the token embedding lookup before any contextualization). Layer-0 d ranges from 0.192 (mBERT) to 1.141 (vision-only). This suggests a substantial fraction of the radical-aligned geometry sits in the static-embedding table rather than in contextual representation. We flag this rather than hide it: a portion of what we are measuring is essentially "how the tokenizer's vocabulary is laid out," not "what the deeper layers learned about characters." For Chinese-BERT and mBERT, where we extracted all 13 layers, peak-d is at layer 1 and layer 7 respectively; for the others, the 5-layer sample is too coarse to localize the peak with confidence. We mark this as a methodological limitation in Section 7.

## 6. Discussion

### 6.1 What the data say

The four modern Chinese-specialized models we tested show a partial-R² ordering with `same_radical` above `ppmi`. The other six models do not. We resist three over-readings.

First, we are not claiming that "modern Chinese LLMs encode radicals." The ratio above 1.7 for three Qwen and BGE means radical co-membership predicts cosine geometry better than PMI does in those four models, on this dataset, after frequency and stroke controls, with cluster-robust standard errors. It does not mean the model has a mechanism for radicals. Mechanism is a separate question that requires causal interventions (activation patching, attention-head ablation, neuron-level analysis); we did not run those experiments.

Second, the four "modern Chinese-specialized" models are a small set. With n = 4, the category effect is suggestive but not statistically firm. We report it as a pattern in the data and identify candidate model families (Yi, DeepSeek, GLM-4, InternLM, Qwen2.5-7B and larger) where the same analysis should be replicated. None of those fit on free GPU at full precision; we flag this as the single largest scope limitation.

Third, the cloze-probe correlation with the variance decomposition is striking but is also driven by a small set of behavioral measurements (eight fields, five characters per condition, ten contexts per field, ~250 trials in total). A larger benchmark with inter-annotator agreement on context naturalness and a wider set of fields would be a clean follow-up.

### 6.2 The theoretical picture, hedged

If the pattern in the variance decomposition is real and replicates on additional Chinese-specialized models, the natural reading is that pretraining on Chinese text at scale induces co-occurrence regularities that correlate with radical identity beyond what semantic context alone explains. Same-radical characters appear in similar morphological contexts (compound formation, classifier patterns, idiomatic frames) that are not captured by a simple PMI window of 5. Multilingual encoders, whose Chinese-text exposure is a fraction of their total training, do not learn this regularity to the same degree. The effect is not visual: the only model that sees pixels (the ResNet-18 baseline) has zero form-specific residual after semantic-field matching. The effect emerges from pretraining and grows with Chinese-specific scale (Qwen2.5-3B > Qwen2.5-1.5B > Chinese-BERT on every metric we report).

We do not observe the same scaling law in multilingual encoders. XLM-R-large is worse than XLM-R-base on the cloze probe. Adding parameters to a multilingual MLM dilutes whatever Chinese-specific structure was there at the smaller scale, presumably because the larger model spreads its representational budget across 100 languages. This pattern is consistent with prior findings on the cost of multilinguality but is not load-bearing for our main claim.

### 6.3 Generalization to other writing systems

The methodology generalizes. For Japanese kanji, we already showed the partial replication on Joyo. For Korean Hanja, the same pipeline could run on a Korean-specific encoder; the radicals are unchanged and the dataset construction is mechanical. For non-CJK logographic systems (Egyptian hieroglyphs, Mayan glyphs), the dataset is the bottleneck rather than the analysis: there is no Unihan equivalent. We release the variance-decomposition pipeline as standalone code so that any future study can apply it.

## 7. Limitations

We list the ones we know about and the ones reviewers of earlier versions identified.

**Pair non-independence in the regression.** Each character appears in approximately 63 of the 200,000 sampled pairs, so observations are not independent. We address this with two-way cluster-robust standard errors clustering on both characters of each pair (Cameron, Gelbach, and Miller 2011). The point estimates of beta are unchanged from naive OLS; the standard errors widen, correctly, by a factor of 1.05 to 2.66 (the design effect). Significance survives the correction in every case, but readers should attend to the cluster-robust 95% confidence intervals reported in `variance_decomposition_clustered.csv` rather than to the naive p-values.

**Modest absolute variance explained.** Full R² of the four-predictor regression ranges from 0.002 to 0.050. We are decomposing a residual after most of the cosine variance is already absorbed by per-character idiosyncrasies. The partial-R² ratio is interpretable, but readers should not read the absolute partial-R² as "how much of the embedding geometry is radicals." A useful calibration would be to add a model-internal predictor (e.g., static-embedding cosine of the same characters) and report what fraction of the residual after that predictor is captured by `same_radical`. We did not run this and flag it as a follow-up.

**XLM-R-base subword pooling.** XLM-R has 0.4% single-token coverage of our character set. We extract by mean-pooling subword spans, which conflates the SentencePiece decomposition with the character identity. XLM-R's coefficients in the variance decomposition should be read as descriptive of this pooled representation rather than as a direct competitor to the Chinese-tokenized models. The substantive comparison "modern Chinese-specialized vs older multilingual" does not depend on XLM-R's exact ranking.

**Small set of Chinese-specialized models.** Four (Chinese-BERT, MacBERT, ERNIE-3.0, BGE-large-zh) is more than the n = 3 of an earlier draft of this work, but it is still a small category. We report the pattern as suggestive. Closing the n problem requires Chinese-pretrained LLMs that fit on free GPU, which is a current bottleneck in open-weights releases at the 6-14B scale.

**No mechanistic / causal experiment.** Our variance decomposition is associational. We do not claim that any model contains a radical-detector circuit. A direct test would be activation patching at scale, attention-head localization with counterfactual ablation, or neuron-level analysis on Qwen2.5-3B. We sketched a small-scale geometric ablation in `activation_patching.py` but did not include it as a core result because the geometric ablation does not support strong causal claims.

**No inter-annotator agreement on cloze contexts.** The procedural construction removes selection bias from the target/distractor split, but the contexts themselves were written by the author. Whether the contexts read as natural Chinese to a native reader is not validated. We flag this as a methodological gap. The remedy is a 20-minute task for one Chinese-fluent collaborator: rate each of the 21 contexts on naturalness and confirm whether the procedurally-generated targets and distractors are plausible completions. We will add this in any revision before venue submission.

**Layer sampling.** For eight of ten models we extracted only 5 layers. Peak-d location is therefore approximate for most of the model set. Full-resolution layer-wise analysis is in `layer_wise.csv` for mBERT and Chinese-BERT only.

**Single sentence corpus for PMI.** The PMI predictor uses 1M Wikipedia zh sentences. Wikipedia is encyclopedic and biased away from spoken Chinese; a colloquial corpus might yield different PMI values. We do not believe this changes the relative ordering across models (which use a fixed PMI value as a covariate) but it might affect the absolute partial-R² values.

**fp16 inference.** All embeddings were extracted in bf16 to fit within free-tier GPU memory. fp32 extraction would give slightly more precise cosines, on the order of 10^-3 to 10^-2 in absolute value, which is well below our reported effect sizes but worth noting for replicators.

## 8. Conclusion

We decomposed pairwise embedding cosine in ten transformer models into four channels: radical co-membership, distributional context, character frequency, and stroke difference. With cluster-robust standard errors and a procedural cloze probe for behavioral validation, we found that four modern Chinese-specialized models (Qwen2.5-1.5B, Qwen2.5-3B, BGE-large-zh, Chinese-BERT-on-some-metrics) show partial-R²(radical) at or above partial-R²(PMI), while older multilingual encoders show the reverse or near-zero. The cloze probe predicts the same ordering. The effect is Kangxi-specific (size-matched random partitions yield zero), training-driven (random-init networks yield zero), and not attributable to character frequency, stroke count, or visual form alone.

We refrain from claiming that modern Chinese LLMs encode radicals as a primary axis. The data say something more careful: in our test set of four Chinese-specialized models, radical co-membership is a stronger predictor of pairwise cosine geometry than distributional PMI is, and this geometric ordering predicts behavior on a cloze task. We hope replication on a wider set of Chinese-pretrained LLMs and a mechanistic follow-up using activation patching will confirm or constrain this pattern.

## Reproducibility

All code, the preregistration document (timestamped before any real-data runs), the procedural cloze items, the 6,306-character dataset with stroke counts and liushu annotations, and every analysis CSV from this paper are available at:

`https://github.com/aryan35790jp/chinese_llm`

Total compute for the full pipeline: approximately 2 hours 20 minutes on a free Colab T4 GPU, hands-off after the notebook's Run All. The longest single step is 50 minutes of embedding extraction across the ten models.

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
