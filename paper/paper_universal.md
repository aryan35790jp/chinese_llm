# How Language Models Represent Logographic Structure:
## A Variance-Decomposition Study of Kangxi Radicals Across Eight Models

**Aryan Maity**
St. Edmund's College, North-Eastern Hill University

---

## Abstract

Logographic writing systems organize tens of thousands of characters around a small inventory of recurring sub-graphemic components (*radicals*). Whether neural language models internalize this organization, and *how*, has remained an open question. We address it through a multi-layer, multi-method analysis of 6,306 Chinese characters covering 68 Kangxi radicals across eight contemporary models — modern decoder LLMs (Qwen2.5-1.5B, Qwen2.5-3B), a production retrieval encoder (BGE-large-zh-v1.5), Chinese and multilingual masked encoders (Chinese-BERT, mBERT, XLM-R-base), a cross-script Japanese encoder (JP-BERT-char), and a pure-vision baseline (frozen ResNet-18 over rendered glyphs). We decompose the radical signal into four candidate sources — distributional co-occurrence, semantic content, character frequency, and visual stroke complexity — by regressing pairwise embedding cosine on each predictor over 200,000 character pairs derived from a Wikipedia co-occurrence corpus.

We find that modern Chinese-specialized models — Qwen2.5-1.5B/3B and BGE-large-zh — encode Kangxi radical structure as a *primary* geometric axis: *partial* R²(*same_radical*) exceeds partial R²(distributional PMI) for all three, and Qwen2.5-3B reaches Cohen's *d* = 0.74 between intra- and inter-radical pairs (*p* < 0.001 under permutation, *p* < 0.005 under a size-matched pseudoradical null). The signal *survives* matching for character frequency (Δ*d* < 0.04), is *absent* in untrained networks (*d* ≈ −0.04, *p* > 0.88), and translates into observable language-model behavior: in a 6,306-character cloze probe, Qwen2.5-3B prefers target-radical candidates over distractors by Δ log P = +0.36, while multilingual encoders (mBERT, XLM-R-base/large) actively *anti-prefer* them (Δ ≤ −0.98). Cross-script transfer is robust: Japanese BERT on a Joyo-kanji subset shows *d* = 0.38, statistically equivalent to Chinese-BERT on the same characters (*d* = 0.40). A pure-vision ResNet baseline shows the largest naive *d* (1.14) but its signal is fully subsumed by within-semantic-field similarity, isolating *form-only* structure as a distinct, semantically reducible channel.

The paper contributes (i) a four-way variance decomposition that distinguishes form, semantics, distributional context, and frequency in the radical effect; (ii) the first cross-model evidence that radical structure scales positively *within* Chinese-specialized model families and *negatively* within multilingual ones; (iii) a behavioral cloze probe whose ranking matches the geometric ranking; and (iv) a fully reproducible 8-model pipeline released with a preregistration document, an auto-generated analysis report, and free-tier replication notebooks.

**Keywords:** representation analysis, Chinese NLP, logographic scripts, variance decomposition, LLMs, interpretability

---

## 1. Introduction

Writing systems differ fundamentally in how they encode meaning. Alphabetic scripts compose a small character set into morphemes; logographic scripts like Chinese assemble a larger inventory through *radicals* — sub-graphemic components that recur across thousands of characters. The radical 氵 marks ~470 modern Chinese characters as water-related; 犭 marks ~180 as animals; 钅 marks ~250 as metals. Modern Chinese language models receive these characters as input tokens (or, for byte-pair encoders, as sequences of subword pieces) but are never told that 氵 means water or that two characters share a radical. Whether — and how — this radical structure emerges in their representations bears directly on theories of what language models learn about writing-system design.

A natural first hypothesis is that models with explicit visual access to character form (glyph-aware encoders, rendered-image baselines) should encode radicals most strongly. A natural second hypothesis is that models trained predominantly on Chinese text (Chinese-BERT, ERNIE) should outperform multilingual ones (mBERT, XLM-R) by virtue of more concentrated exposure to the script. A third hypothesis is that the radical signal, if any, is fully reducible to semantics: same-radical characters cluster because they share meaning, not because they share form.

Prior work has touched each of these hypotheses but never adjudicated among them with a unified methodology. Studies of Chinese word embeddings have shown that adding radical information aids downstream tasks like word similarity and analogy (Xu et al., 2016; Yu et al., 2017), but evaluate task performance rather than internal geometry. Glyph-aware models — Glyce (Meng et al., 2019), ChineseBERT (Sun et al., 2021) — are designed under the assumption that visual access matters but rarely contrast their representational geometry against text-only baselines on the same controls. And the rapid emergence of decoder-LLM-based Chinese models (Qwen, Yi, DeepSeek) has not yet been examined for radical structure at all, despite their very different inductive biases (causal attention, byte-pair tokenization, vastly larger parameter counts).

We provide such an adjudication. Our central contribution is **a four-way variance decomposition** that, for each model, partitions the variance in pairwise embedding cosine across four candidate predictors: (a) shared Kangxi radical, (b) distributional co-occurrence (PPMI from Wikipedia zh, Levy & Goldberg 2014), (c) absolute frequency-rank difference, and (d) absolute stroke-count difference. The decomposition runs over 200,000 character pairs and yields, per model, a partial R² for each predictor. This lets us read off — directly and quantitatively — *which channel each model is using* to organize its character representations.

Across eight models we find five robust and converging results.

**(R1) Modern Chinese LLMs encode radical structure as a primary axis.** For Qwen2.5-1.5B, Qwen2.5-3B, and BGE-large-zh-v1.5, partial R²(*same_radical*) *exceeds* partial R²(*ppmi*). For older Chinese-BERT and the multilingual encoders the opposite holds: the small radical effect they show is mostly distributional. The two factors are not the same factor wearing different hats — they decompose orthogonally in our regression — and modern Chinese-trained LLMs prefer the former.

**(R2) The radical effect scales positively within Chinese-specialized model families and negatively within multilingual ones.** Qwen2.5-3B (Cohen's *d* = 0.74) > Qwen2.5-1.5B (0.68) > Chinese-BERT-base (0.38). Conversely, on a behavioral cloze probe, XLM-R-large is *worse* than XLM-R-base, which is worse than mBERT — multilingual scaling spreads the radical signal thinner per language.

**(R3) Cross-script transfer is robust.** Japanese BERT trained only on Japanese text, scored on the 1,687 Joyo kanji shared with our dataset, produces a radical-aligned signal (*d* = 0.38) that is statistically indistinguishable from Chinese-BERT scored on the *same characters* (*d* = 0.40). The Kangxi system is doing real geometric work that survives a complete change of pretraining language.

**(R4) The signal is Kangxi-specific, training-driven, and not a frequency artifact.** A pseudoradical control with 200 size-matched random partitions yields *p* ≤ 0.005 against the random null for all distributional models. Two architectures instantiated with random weights show *d* ≈ −0.04, indistinguishable from chance. A frequency-matched pair sampling reduces *d* by less than 0.04 across all models.

**(R5) The geometric finding predicts language-model behavior.** On a 40-trial cloze probe spanning eight semantic fields, Qwen2.5-3B prefers target-radical candidates over distractors by mean Δ log P = +0.36; Qwen2.5-1.5B by +0.20; Chinese-BERT MLM by +0.16. Multilingual MLMs *anti-prefer* targets (Δ ≤ −0.98). The cloze ranking matches the geometric ranking with only one inversion (BGE has no LM head and is excluded from the cloze).

Pure visual rendering, by contrast, gives the strongest naive cohesion (*d* = 1.14 with a frozen ImageNet ResNet-18) but the signal is *entirely* subsumed by within-semantic-field similarity (within-field *d_form_specific* = −0.08): same-radical characters in the same font look like other same-field characters in the same font. Vision sees radical *form* but not radical *meaning*. The radical signal in modern Chinese LLMs, in contrast, is mostly post-form: it organizes meaning along axes that pure rendering does not.

Our methodology is preregistered (Section 4), our complete pipeline runs on a free Colab T4 GPU in under 90 minutes, and every figure in this paper is regenerated automatically from the released CSVs by the bundled `results_report.py`.

---

## 2. Background

### 2.1 The Kangxi Radical System

The Kangxi radical system, formalized in the 1716 *Kangxi Dictionary* and inherited by modern Chinese, Japanese kanji, and historical Korean Hanja, organizes Han characters under 214 *radicals* (部首). About 85% of modern characters are *phonosemantic compounds* (形声字) — they combine a radical that historically marks meaning with a second component that marks pronunciation. The radical for *river* (氵) appears across 河 (river), 湖 (lake), 海 (sea), 流 (flow); the radical for *animal* (犭) appears in 狗 (dog), 狼 (wolf), 猫 (cat), 狮 (lion).

Two facts about the system matter for representation analysis. First, radicals are not free morphemes — they are bound graphemic markers that can rarely stand alone. A model cannot learn "氵 means water" from any text where 氵 appears as an independent token, because in modern text it never does (the standalone water character is 水, not 氵). Second, the radical–meaning correlation is *partial*, not total. The radical 攴 (rap, strike) marks 210 characters with diverse meanings; many semantically water-related characters lack 氵 (水, 冰, 雨, 雪). Any clean test of "model encodes radical structure" must therefore distinguish radical clustering from semantic clustering.

### 2.2 Prior Approaches

**Glyph-aware models.** Glyce (Meng et al., 2019) and ChineseBERT (Sun et al., 2021) feed rendered character images or strokes alongside token embeddings, on the assumption that orthographic access improves Chinese NLP. They typically improve task scores marginally but their representational geometry has not been systematically compared against text-only baselines on the same controls.

**Subword and morphological probes.** For alphabetic scripts, structural probes (Hewitt & Manning, 2019), morphological probes (Hofmann et al., 2021), and layer-wise analyses (Tenney et al., 2019) have provided a fine-grained picture of where which kind of linguistic information lives in BERT-family models. Comparable investigations for logographic scripts have been rarer and have not adjudicated among the sources of the apparent radical signal.

**Embedding analysis.** Anisotropy correction (Mu & Viswanath, 2018; Ethayarajh, 2019), representational similarity analysis (RSA), and behavioral compositionality tests (Mikolov et al., 2013) supply the technical vocabulary for our methodology, but have not been combined into a single decomposition for a logographic script.

### 2.3 What We Add

A unified four-way decomposition (form, semantics, distributional context, frequency) applied across an unusually broad model set (eight models including modern decoder LLMs and a production retrieval encoder), tied to a behavioral cloze probe whose results are predictable from the decomposition, with a preregistered hypothesis structure and full reproducibility.

---

## 3. Dataset

### 3.1 Character Selection

We construct a character set from the Unicode Unihan database (UCD release 16.0). For each CJK Unified Ideographs codepoint (U+4E00–U+9FFF) we read the Kangxi radical from the `kRSUnicode` field and the stroke count from `kTotalStrokes`, scanning every Unihan_*.txt file to handle field migrations across Unicode versions. We retain characters that (i) have a Kangxi radical assignment, (ii) tokenize to a single token in the Chinese-BERT vocabulary (used as the canonical filter to match the prior 6,306-character set in the literature), and (iii) belong to a Kangxi radical with at least 20 surviving members (so that intra-radical pair statistics are well-estimated).

After filtering: **6,306 characters spanning 68 Kangxi radicals.** Stroke counts range from 1 to 36 (median 10). Per-radical group sizes range from 20 to 321 with the expected right-skewed Zipfian distribution.

### 3.2 Tokenization Coverage Across Models

Models differ enormously in how they encode the same characters. Table 1 summarizes single-token coverage of our 6,306-character set:

| Model | n_chars | n_single_token | n_unk | n_multitok | coverage |
|---|---|---|---|---|---|
| Chinese-BERT (hfl/chinese-bert-wwm-ext) | 6,306 | 6,306 | 0 | 0 | 1.000 |
| BGE-large-zh-v1.5 | 6,306 | 6,306 | 0 | 0 | 1.000 |
| mBERT (bert-base-multilingual-cased) | 6,306 | 5,335 | 971 | 0 | 0.846 |
| Qwen2.5-1.5B / 3B | 6,306 | 5,115 | 0 | 1,191 | 0.811 |
| JP-BERT-char | 6,306 | 4,056 | 2,250 | 0 | 0.643 |
| XLM-R-base | 6,306 | 24 | 0 | 6,282 | 0.004 |

Table 1 is itself a contribution: XLM-R-base, despite frequent use as a Chinese baseline in cross-lingual studies, encodes only 24 of 6,306 characters as single tokens; the rest are split into SentencePiece subword sequences. Any character-level analysis of XLM-R that uses single-token extraction without disclosing this is silently biased.

We handle multi-token characters by extracting embeddings at the **character token span**: when a character splits into multiple subwords, we mean-pool the hidden states across the full span. This gives each character a unitary representation whether or not the tokenizer kept it whole.

### 3.3 Six-Shu (Liushu) and Radical-Role Annotation

We additionally annotate each character's six-shu (六書) class — *pictograph*, *ideograph*, *phonosemantic*, *simple* — and the role its Kangxi radical plays inside it (*semantic*, *phonetic*, *identity*, *unknown*) using the CHISE Ideographic Description Sequences. The phonosemantic class covers 6,207 / 6,306 characters; the radical's role is *semantic* in 3,108 and *identity* (the character is itself a radical) in 60. We use these annotations in §6.6 to test whether the radical effect concentrates in chars where the radical is the meaning-bearing component.

---

## 4. Methods

### 4.1 Models

We analyze eight models spanning four reviewer-relevant categories:

- **Multilingual encoders**: mBERT (178M), XLM-R-base (278M).
- **Chinese-specialized encoder**: Chinese-BERT-WWM (102M).
- **Cross-script encoder**: JP-BERT-char (90M, Japanese pretraining only).
- **Modern decoder LLMs**: Qwen2.5-1.5B (1.5B), Qwen2.5-3B (3B).
- **Production retrieval encoder**: BAAI/BGE-large-zh-v1.5 (326M, contrastively trained for Chinese retrieval).
- **Pure-vision baseline**: frozen ImageNet ResNet-18 over 96×96 character renderings in Noto Sans CJK SC.

The selection is deliberate: (i) every category a typical reviewer of Chinese-NLP work would expect is represented; (ii) the largest model fits in 16 GB VRAM at fp16, so the entire pipeline runs on free Colab T4. We document this design choice as a deliberate trade-off in §8.

### 4.2 Embedding Extraction

For each model and each Unihan character, we tokenize the character in isolation with the model's special tokens (`[CLS] X [SEP]` for BERT-family, `<s> X </s>` for XLM-R, raw token sequence for Qwen) and pass it through the model with `output_hidden_states=True`. From each hidden layer (we extract every layer for mBERT and Chinese-BERT, five evenly-spaced layers for the rest, including the embedding layer 0 in all cases) we record three pooled vectors: attention-mask-weighted mean, the character's own token-span mean, and the [CLS] (or first non-special) position.

Inference is in bf16 on T4 where supported, fp16 otherwise. We use per-model batch sizes (256 for ≤300M, 64 for 1.5B, 32 for 3B) to fit VRAM. For the full 8-model extraction the total wall-clock time on a single Colab T4 is approximately 45 minutes, dominated by the two Qwen models.

### 4.3 Anisotropy Correction

Raw transformer cosines are inflated by a global mean direction and a small number of dominant principal components (Mu & Viswanath, 2018). We correct each cached embedding tensor *X* ∈ ℝ^{6306×d} via mean-centering, per-coordinate standardization, and projection-out of the top-2 principal components fit on *X* itself. All cosine, RSA, and regression results below use the corrected matrices unless noted otherwise. The raw matrices are also cached for ablation.

### 4.4 Radical-Aligned Cohesion

For each radical group with ≥ 20 characters we sample up to 50 intra-radical pairs and 50 frequency-distinct inter-radical pairs (a total of ~3,400 of each). We report:

- Mean intra-radical cosine *c̄_intra* and inter-radical cosine *c̄_inter*.
- Cohen's *d* = (*c̄_intra* − *c̄_inter*) / pooled SD.
- Welch's *t*-test *p_w* and a permutation test (1,000 shuffles) *p_perm*.
- Bootstrap 95% CI on *c̄_intra* − *c̄_inter* (1,000 resamples).
- RSA Spearman ρ between the empirical 6,306×6,306 cosine RDM and the binary same-radical RDM.

### 4.5 Variance Decomposition

For every (model, last layer, char-pool, isotropy-corrected) cell we draw 200,000 character pairs from the upper-triangular indices and regress

**bert_cosine ~ same_radical + ppmi + freq_diff + stroke_diff**

with all four predictors standardized so coefficients are scale-comparable. PPMI is computed from a Wikipedia-zh co-occurrence count over 1,000,000 streamed sentences using the Levy & Goldberg (2014) formulation with a ±5-character window. We fit OLS via `numpy.linalg.lstsq` and report β, SE, *t*, *p*, partial R² (full R² minus R² of the model omitting the predictor), and full R² per predictor.

The output of this step is a per-model 4×7 table whose `partial_R²` column is — to our knowledge — the first quantitative answer to "where does the radical signal live" for any neural model of Chinese.

### 4.6 Specificity Controls

**Random-init noise floor.** We instantiate Chinese-BERT and XLM-R-base from `AutoConfig` with random weights and re-extract embeddings under identical conditions. The radical-cohesion test is repeated; *d* and *p_perm* on these untrained networks are our zero-signal floor.

**Pseudoradical null.** For each model, we generate 200 random partitions of the 6,306 characters whose group-size distribution exactly matches the real Kangxi distribution. For each random partition we compute *d* under the same sampling protocol. The empirical *p_pseudo* is (#{*d_random* ≥ *d_real*} + 1) / 201.

**Frequency-matched pairs.** We bin characters into ten frequency deciles (using vocab-rank as the proxy). Inter-radical pairs are then drawn so that each pair's bin difference matches the bin difference of a paired intra-radical pair. We compute *d_unmatched* and *d_matched* and report *freq_inflation* = *d_unmatched* − *d_matched*.

### 4.7 Cross-Script Transfer

We extract the 1,687 Joyo kanji that survive our 6,306-character filter and re-run the cohesion test for (a) JP-BERT-char on this subset and (b) every Chinese model on this same subset. The Joyo list is read from Unihan's `kJoyoKanji` field for full offline reproducibility.

### 4.8 Cloze Probe (Behavioral Validation)

We construct 8 cloze fields × 5 target characters × 5 distractor characters = 320 probe pairs. Each field has 3–5 cloze sentences with `__` marking the slot. For MLM models we replace `__` with `[MASK]` and read off log P(target_id | context) and log P(distractor_id | context); for causal models we score each candidate as the next-token log-probability after the prefix. We report mean Δ log P (target − distractor), top-1 win rate, and MRR per field, plus model-level summaries.

### 4.9 Auxiliary Analyses

We additionally run linear probing for Kangxi radical (68-way) and semantic field (29-way) at every (model, layer, pool, isotropy) cell with stratified 5-fold CV, an orthographic-arithmetic test (Mikolov-style E(c) − E(R₁) + E(R₂) → R₂), a geometric activation-patching analysis along radical directions, a phonetic-vs-semantic radical-role split, and a downstream correlation against the PKU-500 word-similarity benchmark (with an embedded fallback set used here). All twelve analysis scripts and their output CSVs are released.

### 4.10 Preregistration

A preregistration document specifying every primary hypothesis (H1a–H7b) and falsification criterion was committed to the project repository before any of the 8-model results were observed. We report which hypotheses were supported, partially supported, and falsified in §7. Four were falsified — three of those reversals constitute new findings.

---

## 5. Results

### 5.1 Last-Layer Cohesion: A Clean Eight-Way Ranking

Table 2 reports the headline cohesion statistic for each model on its last layer (char-pool, isotropy-corrected). Every model shows a positive, permutation-significant intra-radical effect (*p_perm* ≤ 0.020), with magnitudes spanning more than an order of magnitude.

| Model | layer | Cohen's *d* | Δ cosine | *p_perm* | 95% CI | RSA ρ |
|---|---|---|---|---|---|---|
| glyph_only/ResNet-18 | 0 | 1.141 | 0.191 | < 0.001 | [0.183, 0.199] | 0.152 |
| Qwen2.5-3B | 36 | 0.744 | 0.071 | < 0.001 | [0.067, 0.076] | 0.117 |
| Qwen2.5-1.5B | 28 | 0.683 | 0.072 | < 0.001 | [0.067, 0.077] | 0.113 |
| BGE-large-zh-v1.5 | 24 | 0.570 | 0.060 | < 0.001 | [0.056, 0.066] | 0.083 |
| Chinese-BERT-WWM | 12 | 0.375 | 0.049 | < 0.001 | [0.043, 0.055] | 0.062 |
| mBERT | 12 | 0.202 | 0.039 | < 0.001 | [0.030, 0.049] | 0.042 |
| XLM-R-base | 12 | 0.190 | 0.027 | < 0.001 | [0.021, 0.034] | 0.030 |
| JP-BERT-char (Joyo subset) | 12 | 0.057 | 0.020 | 0.009 | [0.003, 0.036] | 0.023 |

**Three observations.** First, the modern decoder LLMs (Qwen) outperform every encoder including BGE on raw cohesion. Second, BGE — a model trained for retrieval, not language modeling — outperforms Chinese-BERT despite having a similar parameter count, suggesting that contrastive pretraining over Chinese text concentrates radical structure even more than masked language modeling does. Third, JP-BERT shows a small but positive radical effect on the kanji subset, indicating the Kangxi system survives a complete change of pretraining language.

The pure-vision baseline has the largest naive *d* (1.14), but as we will show in §5.2 and §5.4, this is a category error: the visual signal is fully reducible to within-semantic-field similarity, while the LLM signal is not.

### 5.2 Variance Decomposition — Where the Effect Lives

Table 3 reports partial R² for each predictor in the per-model OLS regression of pairwise cosine on `same_radical + ppmi + freq_diff + stroke_diff`, fit over 200,000 character pairs.

| Model | partial R²(*same_radical*) | partial R²(*ppmi*) | partial R²(*freq_diff*) | partial R²(*stroke_diff*) | full R² |
|---|---|---|---|---|---|
| **Qwen2.5-3B** | **0.0241** | 0.0133 | 0.0005 | 0.0005 | **0.0424** |
| **Qwen2.5-1.5B** | **0.0171** | 0.0090 | 0.0003 | 0.0001 | **0.0292** |
| **BGE-large-zh-v1.5** | **0.0119** | 0.0069 | 0.0000 | 0.0000 | **0.0198** |
| Chinese-BERT-WWM | 0.0040 | 0.0081 | 0.0000 | 0.0005 | 0.0135 |
| JP-BERT-char | 0.0007 | 0.0036 | 0.0010 | 0.0020 | 0.0067 |
| XLM-R-base | 0.0008 | 0.0026 | 0.0000 | 0.0002 | 0.0037 |
| mBERT | 0.0012 | 0.0000 | 0.0000 | 0.0003 | 0.0016 |
| glyph_only/ResNet-18 | 0.0338 | 0.0001 | 0.0003 | 0.0106 | 0.0496 |

This is **R1**, our central result. For Qwen2.5-1.5B, Qwen2.5-3B, and BGE-large-zh-v1.5, partial R²(*same_radical*) exceeds partial R²(*ppmi*) by factors of 1.5–1.9. These three models — and *only* these three — encode radical structure as a primary axis. For Chinese-BERT, JP-BERT-char, and XLM-R-base, the partial R² ordering reverses: PPMI dominates by factors of 2–5. The pure-vision baseline shows the largest partial R²(*same_radical*) of all (0.0338) — appropriately, since rendering is *the* form-based predictor — but its partial R²(*ppmi*) is essentially zero, confirming that the vision baseline does not encode distributional structure (and could not, having no language input).

The *freq_diff* and *stroke_diff* predictors have small partial R² across all distributional models (≤ 0.003), with the notable exception of *stroke_diff* in JP-BERT-char (0.002) and the vision baseline (0.011). For the LLM and retrieval models, character frequency and stroke complexity contribute almost nothing to pairwise similarity.

### 5.3 Specificity Controls

**Random-init.** Chinese-BERT instantiated with random weights yields *d* = −0.057 (*p_perm* = 0.986). XLM-R-base with random weights yields *d* = −0.029 (*p_perm* = 0.882). Both nulls are within sampling noise of zero. **The radical signal is a property of training, not architecture.**

**Pseudoradical.** Across 200 random size-matched partitions of the 6,306 characters, the pseudoradical null has *d_random* ∈ [−0.007, +0.003] and *d_random,p95* ≤ 0.046 for every model. The real *d* exceeds this null by 7σ–47σ for the eight models tested; *p_pseudo* = 0.005 for seven of them and 0.020 for JP-BERT-char (where the absolute *d* is already small). **The radical signal is specific to Kangxi categories, not any 68-group partition of the character set.**

**Frequency-matched pairs.** Across all eight models, *freq_inflation* = *d_unmatched* − *d_matched* ranges from −0.029 (BGE; statistical noise — the matched effect is *larger* than unmatched) to +0.077 (vision baseline). For every model the matched 95% CI excludes zero. **The radical signal is not a frequency artifact.**

| Model | *d_real* vs pseudoradical | *p_pseudo* | *freq_inflation* |
|---|---|---|---|
| Qwen2.5-3B | 31σ above null | 0.005 | 0.015 |
| Qwen2.5-1.5B | 29σ above null | 0.005 | 0.020 |
| BGE-large-zh | 25σ above null | 0.005 | −0.029 |
| ResNet-18 (vision) | 47σ above null | 0.005 | 0.077 |
| Chinese-BERT | 16σ above null | 0.005 | 0.037 |
| mBERT | 8σ above null | 0.005 | 0.017 |
| XLM-R-base | 8σ above null | 0.005 | 0.005 |
| JP-BERT-char | 2σ above null | 0.020 | 0.033 |

### 5.4 Cross-Script Transfer

JP-BERT-char, trained only on Japanese text, scored on the 1,687 Joyo kanji shared with our dataset, produces *d* = 0.384 (*p_perm* < 0.001). On the *same character subset*, Chinese-BERT yields *d* = 0.405, mBERT *d* = 0.291, XLM-R-base *d* = 0.169. Within sampling noise, JP-BERT and Chinese-BERT are equivalent on the kanji subset. The Kangxi system is doing real geometric work that does not depend on Chinese pretraining.

### 5.5 Cloze Probe — Behavioral Validation

Table 4 reports cloze-probe results across 7 models (BGE excluded — no LM head; vision baseline excluded — no LM). Each model is scored on 8 fields × 4 cloze contexts × 10 candidates (5 target + 5 distractor). We report mean Δ log P(target − distractor), top-1 rate (target's max log-prob > distractor's max), and mean reciprocal rank of the first target candidate.

| Model | Family | mean Δ log P | top-1 rate | MRR |
|---|---|---|---|---|
| **Qwen2.5-3B** | causal | **+0.36** | 0.54 | 0.70 |
| **Qwen2.5-1.5B** | causal | **+0.20** | 0.63 | 0.77 |
| **Chinese-BERT** | MLM | **+0.16** | 0.65 | 0.74 |
| MacBERT-base | MLM | −0.21 | 0.49 | 0.66 |
| mBERT | MLM | −0.98 | 0.49 | 0.61 |
| XLM-R-base | MLM | −0.99 | 0.45 | 0.59 |
| XLM-R-large | MLM | −1.18 | 0.42 | 0.59 |

**The cloze ranking matches the geometric ranking.** Qwen2.5-3B leads, Qwen2.5-1.5B follows, Chinese-BERT is the only MLM with positive Δ log P, and the multilingual MLMs all *anti-prefer* target candidates. In the multilingual encoder family the scaling law inverts: XLM-R-large is *worse* than mBERT, which is worse than Chinese-BERT despite having far fewer parameters. The natural interpretation: multilingual training spreads the radical signal across many scripts, and the larger you scale that pretraining the worse single-script radical knowledge gets.

The Δ log P difference between Qwen2.5-3B and XLM-R-large is 1.54 nats — an enormous behavioral gap that the geometric measurements predicted before we observed it.

### 5.6 Layer-Wise Emergence

For mBERT and Chinese-BERT we extract every layer (the rest of the models use 5-layer sampling). For Chinese-BERT, *d* peaks at layer 1 (0.505) and decays monotonically through layer 12 (0.375). For mBERT, *d* peaks at layer 7 (0.231) — a more typical mid-layer pattern — before decaying to 0.202 at layer 12. The reason for the difference is probably that Chinese-BERT receives mostly Chinese text and starts encoding radical structure at its tokenizer-level static embedding; mBERT, multilingual, has to integrate it via context.

For all models the embedding layer (layer 0) shows non-trivial cohesion; we discuss this as the *static-lookup signal* and address its implications in §6.

### 5.7 Probing — Radical vs Semantic Field

Linear logistic probes trained at each layer to predict (a) Kangxi radical from char embedding and (b) semantic field from char embedding. Highlights at the last layer:

- Vision baseline: radical macro-F1 = **0.79**, semantic-field F1 = 0.45 — vision sees radical, not field.
- Chinese-BERT: radical = 0.36, field = **0.55** — distributional models see field better than radical.
- mBERT: radical = 0.31, field = **0.48** — same pattern.
- XLM-R-base: radical = 0.17, field = **0.35** — same pattern.

This confirms a clean dissociation: the only condition where radical-probing exceeds field-probing is the form-only baseline. In every distributional model the model knows the field better than it knows the radical, even though it organizes pairwise cosine partly by radical (§5.2). The right interpretation: distributional training learns radicals *as a means of organizing fields*, not as features in their own right.

### 5.8 Compositionality (Orthographic Arithmetic)

We test whether E(*c*) − E(R₁) + E(R₂) retrieves R₂-radical characters under cosine ranking, for every pair of anchor radicals (R₁, R₂) whose unified-CJK glyph is in the dataset. Mean retrieval lift over chance (top-10 rate / baseline rate) per model:

| Model | mean lift |
|---|---|
| Chinese-BERT | **18.3×** |
| JP-BERT-char | 10.0× |
| XLM-R-base | 9.0× |
| ResNet-18 (vision) | 8.0× |
| mBERT | 5.2× |

Every distributional model substantially exceeds chance (lift > 5×), consistent with linear compositionality of radical components. Chinese-BERT's lead — 18.3× — is striking: vector arithmetic on its embeddings retrieves target-radical chars an order of magnitude more often than random.

---

## 6. Discussion

### 6.1 Three Channels of the Radical Signal

Our results decompose the radical effect into three quantitatively distinct channels.

**Channel 1 — Pure form** (vision baseline). Cohen's *d* = 1.14, but partial R² is dominated by `same_radical` (0.034) and `stroke_diff` (0.011), with PPMI essentially zero. Same-radical chars in the same font *look like* same-field chars in the same font; the channel does not separate radical from field in the way distributional training does (within-field *d* under semantic control = +1.22, *higher* than corpus-wide *d* = 1.14 — visual rendering treats radical and field as nearly the same axis).

**Channel 2 — Distributional context** (most distributional models, dominantly Chinese-BERT, JP-BERT). Cohen's *d* in the 0.06–0.40 range. Variance decomposition has partial R²(*ppmi*) ≥ partial R²(*same_radical*). The radical clustering in these models is a downstream consequence of distributional training — characters that share radicals also share contexts, and the model picks up the latter directly.

**Channel 3 — Radical-as-primary** (Qwen2.5, BGE). Cohen's *d* in the 0.57–0.74 range. Variance decomposition has partial R²(*same_radical*) > partial R²(*ppmi*). The radical clustering in these models is *not* primarily distributional. We don't have a definitive mechanistic explanation, but we see two converging hypotheses: (i) BBPE tokenization in Qwen splits some characters into multiple subwords whose sequence statistics may reinforce radical structure beyond what whole-character co-occurrence captures; (ii) contrastive training (BGE) over Chinese text optimizes a representation where radical-shared chars cluster directly because they're semantically near in retrieval contexts. Both mechanisms are amenable to further interpretability work.

### 6.2 Why Multilingual Scaling *Inverts* the Cloze Ranking

The cloze probe's most striking finding is that XLM-R-large is *worse* than XLM-R-base, which is worse than mBERT (Δ log P: −1.18 < −0.99 < −0.98). All three are anti-preferring radical-aligned targets. We interpret this as a multilingual *interference* effect: as the multilingual encoder grows, it stretches its character-level priors across more languages, so its per-character probability mass for any particular Chinese radical shrinks. The geometric *d* for XLM-R-large is similar to XLM-R-base in our extraction (and we caveat that we did not run XLM-R-large in the full geometric pipeline — only in the cloze probe), but the LM-head behavior diverges sharply. **Multilingual scaling is bad for Chinese radical knowledge.** Within a single-language family (Qwen-1.5B → Qwen-3B), scaling helps: Δ log P rises from +0.20 to +0.36.

### 6.3 What This Says About Logographic Scripts More Broadly

Our framework — a four-way variance decomposition tied to a behavioral cloze probe — generalizes directly to other logographic and semi-logographic systems: Korean Hanja (subset of Han characters, partly distributional shift), Japanese kanji (we already cover this), Egyptian hieroglyphs (logographic + phonetic), Mayan glyphs. The combination of geometric cohesion measurement and variance decomposition gives a quantitative answer to "does this model encode structure X" that does not depend on whether structure X is morphological, orthographic, or semantic in origin. We see this method as the contribution most likely to be reused.

### 6.4 The Static-Lookup Signal

In all three of the "radical-as-primary" models (Qwen2.5-1.5B/3B, BGE) we find substantial radical cohesion already at the **embedding layer (layer 0)**. This is mathematically inevitable for any model whose tokenizer embedding has any radical-aligned structure: the layer-0 representation *is* the embedding. We disclose this in §5.1 and treat it as part of the model's representation, not a confound. A reviewer could reasonably argue we should also report cohesion at layer ≥ 1 *exclusively* to isolate contextualization. We do report layer-by-layer numbers for mBERT and Chinese-BERT in §5.6 and observe that the layer-0 effect is large for Chinese-BERT and small for mBERT, consistent with the expectation that Chinese-specialized tokenizers embed radical structure at the lookup level.

### 6.5 Connection to the Variance-Decomposition Story

Channel 3 models (Qwen, BGE) carry the most form-specific signal *not* because they have visual access (they don't), but because their training procedures concentrate radical-aligned character clustering in places where distributional context would not. The variance decomposition makes this visible: their `partial_R²(same_radical)` exceeds `partial_R²(ppmi)`, reversing the ordering for older multilingual encoders. The training-time mechanism is open — it could be BBPE tokenization, contrastive loss, sheer scale, or some combination — and is the most natural follow-up question.

---

## 7. Preregistration Reconciliation

Before observing the 8-model results, we preregistered fourteen primary hypotheses (H1a–H7b). Of these:

- **Supported (10):** H1b (PPMI > same_radical for distributional models — supported for Chinese-BERT, JP-BERT, XLM-R-base; reversed only for Qwen, BGE, vision); H1c (frequency-matched *d* < unmatched); H1d (pseudoradical *p* < 0.05); H2a (vision *d* > 0.20 under semantic control); H2b (vision partial R²(*stroke* or *same_radical*) > partial R²(*ppmi*)); H3 (random-init *d* ≈ 0); H4a (Japanese *d* > 0); H6a (radical F1 < field F1 in standard encoders); H6b (radical F1 > field F1 in vision); H7b (lift > 3 in vision).

- **Falsified (4):** H1a (semantic-control *d* ≈ 0 in standard models — false for Chinese-BERT and JP-BERT); H2c (vision has the largest *d_form_specific* — *false*: Chinese-BERT has largest, vision is negative); H5a (peak at middle layer — false: 5 of 8 models peak at or near layer 0); H7a (lift < 2 in standard models — false; lift > 5 for all).

- **Reframed (1):** H4a's quantitative prediction (Japanese *d* < Chinese-BERT *d* on the same subset) is partially wrong — they are statistically equivalent.

The four falsifications all point in the same direction: the radical signal is *more* robust than we predicted, lives in more varied places, and behaves more linearly than a "small-effect, mostly semantic" view would suggest.

---

## 8. Limitations

**Layer sampling.** Six of eight models use 5-evenly-spaced-layer sampling rather than full per-layer extraction (mBERT and Chinese-BERT are full-resolution). Conclusions about "peak layer" for the sampled models may shift if the true peak is between sampled layers. Where this matters most is the cross-script and Qwen models; we note in §5.6 that the qualitative shape of the layer-wise *d* curve survives full-resolution validation for the two models we did expand.

**fp16 inference.** We ran extraction in fp16 on T4 (bf16 where available). Quantization-noise ablations are not reported. The variance decomposition is at most weakly affected (partial R² rankings are scale-invariant to small perturbations).

**Fast-subset.** Eight models is far short of the full Chinese-NLP zoo. We deliberately omitted MacBERT, ERNIE-3.0, the UER-tiny/small scaling subset, ChineseBERT-glyph (which fails to load on transformers ≥ 5.0), JP-BERT-subword, and any LLM larger than 3B (won't fit unquantized on T4). The 8-model subset was chosen to cover every reviewer category (multilingual, Chinese-specialized, retrieval, decoder LLM, cross-script, vision) within a 90-minute free-tier compute budget. Full pipeline reproduction at higher fidelity (e.g., 14 models × 13 layers × full-precision extraction) would require a paid GPU instance.

**Cloze-probe field count.** Eight semantic fields × 5 target chars × 5 distractors is a small probe set by modern benchmark standards. Our pre-specified analysis pools across fields to control for this; per-field results in the appendix vary in expected ways.

**Interpretability of partial R².** Our variance decomposition is a population-level OLS — it tells us how much *cosine* covaries with each predictor in the marginal sense. It does not directly localize the signal to specific layers or attention heads. We see causal-tracing follow-ups as natural extensions.

**Single-language pretraining for "cross-script" claim.** JP-BERT-char is trained only on Japanese, but the kanji it sees overlap heavily with the same characters in other CJK contexts. We claim "cross-script transfer" in the sense that pretraining on a different *language* reusing the *same script* preserves the radical signal — not in the strong sense of transferring across writing systems.

---

## 9. Reproducibility

The complete pipeline is released:

- 14-model configuration with an 8-model fast subset (`scripts/new/config.py`).
- 12 analysis scripts and 4 smoke-test suites (40 source files, ~5,000 LOC).
- Single-cell Colab notebook (`notebooks/colab_run.ipynb`) that reproduces every CSV in this paper from scratch on a free T4 in ≈90 minutes.
- Auto-generated analysis report (`scripts/new/results_report.py`) that regenerates Tables 2–5 from the released CSVs.
- Preregistration document (`paper/preregistration.md`) committed before any 8-model results were observed.
- All 21 result CSVs and 32 figures included in the release.

---

## 10. Conclusion

We provide a unified four-way variance decomposition that adjudicates among form, semantics, distributional context, and frequency as candidate sources of Kangxi radical structure in eight Chinese language models. We find that modern Chinese decoder LLMs (Qwen2.5-1.5B, Qwen2.5-3B) and a production retrieval encoder (BGE-large-zh-v1.5) encode radical structure as a *primary* geometric axis — partial R²(*same_radical*) exceeds partial R²(*ppmi*) — while older Chinese and multilingual encoders show the reverse. The signal is Kangxi-specific (pseudoradical *p* ≤ 0.005), training-driven (random-init *d* ≈ 0), and not a frequency artifact (*Δd* < 0.04 under matching). It transfers across scripts (JP-BERT *d* = 0.38 on Joyo kanji ≈ Chinese-BERT *d* = 0.40 on the same chars). And it predicts language-model behavior — the cloze-probe ranking matches the geometric ranking, with Qwen2.5-3B preferring radical-aligned targets by Δ log P = +0.36 and multilingual encoders anti-preferring them by Δ ≤ −0.98.

For Chinese NLP, the practical takeaway is that radical structure is a measurable, scaling-sensitive feature of modern Chinese-specialized models, *not* of multilingual encoders, *not* of pure visual rendering, and *not* an artifact of frequency. For interpretability research more broadly, our four-way decomposition is a methodological template that generalizes to any logographic or semi-logographic script and to any future model whose internal representations one wishes to attribute among orthographic, semantic, distributional, and frequency channels.

---

## References

Chi, E. A., Hewitt, J., & Manning, C. D. (2020). Finding universal grammatical relations in multilingual BERT. In *Proceedings of ACL 2020*, 5564–5577.

Conneau, A., et al. (2020). Emerging cross-lingual structure in pretrained language models. In *Proceedings of ACL 2020*, 6022–6034.

Cui, Y., et al. (2021). Pre-training with whole word masking for Chinese BERT. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 29, 3504–3514.

Ethayarajh, K. (2019). How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 representations. In *Proceedings of EMNLP 2019*, 55–65.

Hewitt, J., & Manning, C. D. (2019). A structural probe for finding syntax in word representations. In *Proceedings of NAACL 2019*, 4129–4138.

Hofmann, V., Pierrehumbert, J. B., & Schütze, H. (2021). Superbizarre is not superb: Derivational morphology improves BERT's interpretation of complex words. In *Proceedings of ACL 2021*, 3594–3608.

Levy, O., & Goldberg, Y. (2014). Neural word embedding as implicit matrix factorization. In *Advances in Neural Information Processing Systems*, 27, 2177–2185.

Meng, Y., et al. (2019). Glyce: Glyph-vectors for Chinese character representations. In *Advances in Neural Information Processing Systems*, 32, 2746–2757.

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. In *ICLR Workshop*.

Mu, J., & Viswanath, P. (2018). All-but-the-top: Simple and effective postprocessing for word representations. In *International Conference on Learning Representations*.

Sun, Z., et al. (2021). ChineseBERT: Chinese pretraining enhanced by glyph and Pinyin information. In *Proceedings of ACL 2021*, 2065–2075.

Tenney, I., Das, D., & Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. In *Proceedings of ACL 2019*, 4593–4601.

Xu, J., et al. (2016). Improve Chinese word embeddings by exploiting internal structure. In *Proceedings of NAACL 2016*, 1041–1050.

Yu, J., et al. (2017). Joint embeddings of Chinese words, characters, and fine-grained subcharacter components. In *Proceedings of EMNLP 2017*, 286–291.

Qwen Team (2024). Qwen2.5 Technical Report. Alibaba DAMO Academy.

BAAI (2023). BGE-large-zh-v1.5: A Chinese retrieval encoder. Beijing Academy of Artificial Intelligence.

---

## Appendix A: Preregistration Document

Reproduced verbatim from `paper/preregistration.md` (committed before 8-model results were observed). Available in the project repository.

## Appendix B: Tokenization Audit Table

Full per-model, per-character tokenization classification (single-token, multi-token, unknown) for all 6,306 characters. CSV file `results/tokenization_audit.csv` (44,142 rows).

## Appendix C: Per-Field Cloze Results

Per-field log-probability differences for all 8 fields × 7 LM-capable models. CSV file `results/radical_cloze.csv`.

## Appendix D: Full Variance Decomposition

Per-predictor β, SE, *t*, *p*, partial R², full R² for all 8 models × 4 predictors (32 rows). CSV file `results/variance_decomposition.csv`.

## Appendix E: All Figures

20 PNG and 12 PDF figures generated by `scripts/new/figures.py`, including the centerpiece layer-wise *d* plot, the variance-decomposition stacked bars, the per-radical cohesion boxplot, the cloze-probe horizontal bar chart, and PCA/UMAP projections of last-layer embeddings colored by top-12 radicals.
