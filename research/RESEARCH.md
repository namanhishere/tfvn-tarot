# Research Report: Vietnamese-First Tarot LLM

> Consolidated research findings across data collection, model selection, CPU inference,
> training methodology, environment, and toolchain compatibility. Prepared for the
> `tfvn-tarot` project. Findings marked **[measured]** are first-party measurements on
> this project's own corpus or hardware; everything else is sourced from primary
> documentation, papers, or verified external sources with citations inline.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Corpus research](#2-corpus-research)
3. [Model selection research](#3-model-selection-research)
4. [CPU inference research](#4-cpu-inference-research)
5. [Training methodology research](#5-training-methodology-research)
6. [Evaluation methodology research](#6-evaluation-methodology-research)
7. [Environment findings](#7-environment-findings)
8. [Toolchain compatibility findings](#8-toolchain-compatibility-findings)
9. [Decision summary](#9-decision-summary)
10. [Evidence ledger](#10-evidence-ledger)

---

## 1. Executive summary

Five parallel research lanes (Vietnamese corpora, HuggingFace datasets, GitHub datasets,
public-domain texts, Kaggle + reading transcripts) produced the 490 MB corpus at
`data/`. The design phase then verified: the Qwen3.5 model family exists but carries
deployment blockers; Qwen3-1.7B is the best CPU-deployable base for this product; the
Vietnamese tokenizer landscape is dominated by one byte-identical Qwen tokenizer;
Vietnamese costs ~1.31–1.33× more tokens than English; quantization damage to non-English
generation is understated by automatic metrics by ~10×; and the corpus has structural
defects that require specific handling (orientation collapse risk is the top quality
risk).

The environment probe revealed the local box is a QEMU VM (SSE4.2 only, no AVX) with a
6 GB Pascal GPU (compute capability 6.1, no bf16), forcing cloud GPU rental for training
and projections-only latency claims. Toolchain research pinned the exact compatible
software versions, including the `torch==2.13.0+cu126` Pascal requirement and the
transformers v5 breaking-change surface.

---

## 2. Corpus research

### 2.1 What was collected (490 MB total)

| Layer | Contents | License status |
|---|---|---|
| English card meanings | 9 JSON sets; best = `tarotoo_cards.json` (78 cards × 22 fields, 7 `_reversed` fields) | MIT |
| Waite authority | `StarTarotOnline/tarot-rws-historical-meanings` — 78 rows, verbatim Waite 1910 with per-field citations | CC BY 4.0 |
| Spreads + decks | `tarotschema_spreads.json` (21 spreads with positional meanings), `tarotschema_decks.json` (7 decks incl. 80- and 86-card) | MIT |
| Vietnamese | `Tarot-Vietnamese-API/` — 156 records, 6 domain fields, 156 images (78 pre-rotated reversed) | MIT |
| Vietnamese fluency | `jakeveo05/chinese-traditional-knowledge` — 33.6M chars Vietnamese esoteric books | CC BY 4.0 |
| Public domain texts | Waite PKT 1911, Mathers 1888, de Laurence 1918, 4 cartomancy works, Grand Etteilla (French) | PD (pre-1931) |
| RWS deck images | 78 PNGs, Pamela Colman Smith 1909 | MIT / PD |
| Reading transcripts | Dendory 5,769; yeji-processed 27,735; tarot-oracle-instruct 115 | MIT / mixed |

### 2.2 Cross-source join analysis [measured]

**78 canonical RWS cards join cleanly on a numeric spine with 0 conflicts** across
tarotoo, StarTarotOnline, Blacik, smallcat419, and the multimodalart image index
(0–77). All other sources map via an alias table:

- Magician = *The Juggler* (Mathers); Fool = *The Foolish Man*; Strength = *Fortitude*;
  Judgement = *The Last Judgment* / *Judgment*; World = *The Universe*; Wands =
  *Sceptres*/*Batons*; Pentacles = *Coins*; Page = *Knave*; Two = *Deuce*.
- Vietnamese ids are `canonical_id + 1` (holds for 76/78).
- **Mathers 1888 must never be joined numerically**: minors run King→Ace *descending*
  from 22, so Ace of Pentacles is 77, not 64.
- Etteilla is its own deck — no RWS correspondence.

### 2.3 Data quality defects found [measured, file-by-file]

| Defect | Severity | Handling |
|---|---|---|
| Vietnamese `Page of Pentacles` **missing**; `Knight of Pentacles` duplicated at ids 75/76 with byte-identical prose | **Blocking** | Key by name, not id; assert-then-collapse; Page marked `synthetic_no_anchor` |
| **5 of 6 Vietnamese prose fields byte-identical between upright and reversed** — only `title_secondary` differs → authentic Vietnamese reversed meanings barely exist | **Blocking** | W2.1 orientation-attribution test decides whether upright text is genuinely upright or orientation-agnostic |
| `tellang` tarot rows = **100% verbatim duplicates** of Dendory (5,769/5,769) | High | Drop entirely; zero unique content |
| Dendory has **zero orientation encoding** and is ChatGPT English | High | Structural mining only, never SFT targets |
| `data.txt` has **no trailing newline** (155 lines / 156 records) | High | Never size by `wc -l` |
| id 48 `name` contains embedded newline | Medium | Normalize before `endswith('ngược')` tests |
| id 13 work/money headers **swapped** | Medium | Swap back, content included |
| `title_heath` misspelled 156/156 | Medium | Rename to `title_health` |
| `lookfate` `The World 世界` duplicated at idx 21/22, idx 22 has final 79 chars doubled | Medium | Keep idx 21; whole-record dedupe misses this |
| Waite `Four of Pentacles` uses **`Reversed;` (semicolon)** — strict regex drops it | Medium | Handle; provenance assertion catches regression |
| Waite `Two of Cups` genuinely has **no reversed text** | Medium | Fall back to Mathers (provenance = `mathers`) |
| Mathers HTML: malformed nesting, Google Analytics scripts, shop banner | Low | Lenient regex strip, not XML parser |
| reddit card-name chaos: 102 raw strings → 76 canonical cards | Low | Gazetteer + alias normalization |
| reddit hard truncation at 1500/2500 chars | Low | Known bound; strip mid-word cuts |
| jakeveo05: 27,164 `--- Page N ---` markers, duplicate feng-shui doc pair | Low | Strip markers, dedupe |
| `sức khoẻ`/`sức khỏe` content variant | Low | Normalize to `sức khỏe` (affects keyword back-check) |
| TarotSchema truncated names (`L'Impératri`, `</st`) | Low | Deck records only; don't affect spread extraction |

### 2.4 Reading transcript reality

- **5,963 unique readings** after dedupe: Dendory 5,769 + sunkencity 115 + reddit 76 + agent traces 3.
- **Only 79 records are human-written readings with explicit orientation** (76 reddit + 3 traces), and 76 of those are **English**.
- Reddit readings: mean 226 tokens, p95 504; 10.3% of drawn cards reversed.
- Dendory questions: mean 204 tokens — real questions worth adapting as Vietnamese input seeds.

### 2.5 Public-domain text parsing [measured]

| Text | Words | Parse result |
|---|---|---|
| Waite PKT 1911 (Wikisource, proofread) | 35,731 | 78/78 upright, 77/78 reversed; 3 blocks; em-dash majors pattern; inline minors pattern |
| Mathers 1888 mtar03 | 2,085 | **78/78 in one pass**; `N. Name.-- upright; R. reversed`; 2 regex-exception cards; 19 duplicate-numbered addenda to segregate |
| de Laurence 1918 | 39,420 | **Zero `Divinatory Meanings:` / `Reversed:`** — hard-wrapped, colon-less, `_italic_` markers; needs unwrap + strip |
| Gutenberg boilerplate | — | delaurence 1–32/4341–4691; mohammed_ali 1–33/4371–4725 (**footer not `*** END ***`**); cielo 1–31/5937–6287 |

### 2.6 Vietnamese output-style finding (highest-leverage single decision)

Verified across every credible Vietnamese source: **Vietnamese readers write card names in
English inside Vietnamese prose** ("khi **The Lovers** xuất hiện ở trạng thái ngược",
"**Queen of Cups** xuôi là người phụ nữ nhân hậu"). Vietnamese glosses (Kẻ Khờ, Nữ Tư Tế)
appear **only in lookup tables, never as working names**. The corpus contains zero
occurrences of `Tiểu Đồng`/`Hiệp Sĩ`/`Kị Sĩ` in running prose. This must be learned in
the weights — a per-token decision made hundreds of times per generation.

Also: Vietnamese distinguishes **bài xuôi/ngược** (physical orientation) from
**nghĩa xuôi/ngược** (interpretive register) — English has no clean equivalent; getting
this right is a strong fluency signal.

---

## 3. Model selection research

### 3.1 Qwen3.5 family verification (Qwen-team, released 2026-02-28)

**Both requested models exist**: `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B`, Apache-2.0.
But two architectural facts reshape any plan:

1. **Every Qwen3.5 model is natively multimodal** (`Qwen3_5ForConditionalGeneration`)
   with a vision tower — ~310M of the 2B's 2.27B params are vision (text-only app: dead weight).
2. **Hybrid attention**: 24 layers = 18 GatedDeltaNet (linear) + 6 Gated Attention.
   Only the 6 full-attn layers expose `q/k/v/o_proj`; the 18 linear layers expose
   `linear_attn.{in_proj_qkv,in_proj_a,in_proj_b,in_proj_z,out_proj}`. `conv1d`/
   `A_log`/`dt_bias` are not LoRA-compatible shapes.

Key deployment facts:

| Fact | Value |
|---|---|
| Thinking mode | OFF by default (good); `/think` `/nothink` soft switches **not supported** |
| Benchmarks (non-thinking) | 2B: MMMLU 56.9, IFEval 61.2. 0.8B: MMMLU 34.1, IFEval 52.1, **MAXIFE 39.2** |
| Official GGUF | **None exists for any Qwen3.5 model** (community only: unsloth, bartowski) |
| llama.cpp support | Merged 2026-02-10 after one revert; fix tail through June 2026 |
| **Open bug #24714** | **Forced full prompt reprocessing (no prompt-cache reuse) on Qwen3.5-2B-MTP-GGUF:Q4** — multi-turn CPU latency killer |
| Open bugs | #24737 (4B block count), #24812 (Vulkan garbage) |
| Qwen3-1.7B comparison | IFEval **68.2** (better than Qwen3.5-2B's 61.2); MMMLU 46.7; thinking ON by default (must pass `enable_thinking=False`) |

### 3.2 Tokenizer fertility measurements [measured on 496,062 chars of real corpus]

| Tokenizer | VI chars/tok | VI bytes/tok | Verdict |
|---|---|---|---|
| Qwen3.5 (248,320 vocab) | ~3.95 | ~5.10 | Best; ~7.1–9.2% fewer VI tokens |
| **Qwen3 / Qwen2.5 / SeaLLMs-v3 / Sailor2 (151,6xx)** | **3.663** | **4.912** | Baseline — **all byte-identical** |
| Phi-4-mini (200,029) | 3.594 | 4.819 | Slightly worse |
| Granite 4 (100,352) | 2.040 | 2.736 | **Disqualified** |
| Mistral (32,768) | 1.545 | 2.072 | **Disqualified — 3.19× penalty** |

Key findings:

- **Every SEA/Vietnamese-specialized model <4B reuses the stock Qwen tokenizer with zero
  Vietnamese vocabulary expansion** — specialization is weights-only, so there is no
  tokenizer argument for any of them.
- Vietnamese costs **~1.31–1.33× more tokens than English** for equivalent content.
- Diacritics are **not** a byte-fallback problem: zero byte-fallback tokens across 110
  stress-test VI syllables in Qwen/Llama/Gemma. Consequence: a quantized model cannot
  emit mojibake; it can only pick the **wrong whole syllable** (hỏi vs hồi) — invisible
  to UTF-8 validation, requires native-speaker or tone-checker eval.
- Vietnamese-native tokenizers barely win anymore: VinaLLaMA's 46K vocab beats Qwen3.5
  by only 2.8% on a dead Llama-2 architecture. PhoGPT's 20K vocab is a **trap**: English
  costs +48% tokens and card names shatter (`Knight of Wands` → 7 tokens).

### 3.3 Vietnamese-specialized model landscape

Every candidate <4B is **abandoned**: PhoGPT (last mod 2024-11), VinaLLaMA (2023-12),
Vistral (2024-02), SeaLLMs-v3 (2024-07, custom "seallms" license), Sailor2 (2025-02).
VMLU evidence: a Vietnamese-tuned Qwen2.5-**0.5B** (BloomVN) scores 29.43 vs 25% chance;
a tuned Qwen2.5-**3B** (Vintern-3B-beta) scores 54.81, beating Vistral-7B (50.07).
SEA-HELM's Vietnamese leaderboard contains **no model below 3B at all**.

### 3.4 Instruction-following at tiny scale

The 0.8B→2B discontinuity is the largest in the Qwen3.5 family (MMMLU +22.8, MAXIFE
+21.4). MAXIFE 39.2 at 0.8B means format compliance below 50% before persona and
code-mixing are layered on. Corroborating: "One Token Away from Collapse" (arXiv
2604.13006) shows surface constraints cost Qwen-2.5-7B 40.4% comprehensiveness and
74.7% word count; "When Thinking Fails" (2505.11423) shows self-reflection *degrades*
1–1.5B models. **~2B is the practical floor.**

### 3.5 Sequence budget [measured with Qwen tokenizer]

| Component | Tokens |
|---|---|
| Full VI card record (6 fields) | 868 mean / 1018 p95 |
| Compact card context | 58/card → 3-card 173, 10-card 578 |
| Spread record (full) | 2,396 mean — too large, compress to ~15/position |
| Reddit question | 204 mean |
| Reddit reading | 226 mean / 504 p95 |
| **p95 assembled training example** | **~1,700** |

---

## 4. CPU inference research

### 4.1 Runtime comparison verdict

**llama.cpp `llama-server`** — first-class CPU origin target, GGUF with K/I-quants,
imatrix tooling, best single-stream decode. vLLM CPU backend is explicitly secondary
(no GGUF, AMX-oriented); OpenVINO is competitive only on Intel and its own docs warn
small models (<1B) suffer more from low-bit compression; Ollama adds 10–30% overhead
over raw llama.cpp.

### 4.2 Quantization evidence

**The Q4_K_M penalty triples as models shrink** (WikiText2 perplexity Δ vs FP16/Q8):

| Model | Q4_K_M | Q5_K_M | Q6_K |
|---|---|---|---|
| 7B | +1.28% | +0.40% | +0.13% |
| 3B | +1.75% | +1.10% | +0.22% |
| 1.5B | **+3.62%** | **+0.41%** | +0.21% |
| 0.5B | +2.49% | +2.13% | +0.44% |

**Marchisio et al. (EMNLP 2024 Findings)** on multilingual quantization: automatic
metrics severely understate damage — Japanese −1.7% automatic vs **−16.0% human**;
French −0.3% automatic vs **−16.6% human**; smaller models degrade more; long-form
generation is where humans notice; and low precision may harm the ability to *maintain*
the output language once decoding begins (mid-generation language drift).

**I-quants (IQ4_XS) rejected**: non-linear codebook lookups are CPU-unfriendly in the
compute-starved 4–8 core regime (ikawrakow's own analysis, issue #5290).

**Recommended: Q5_K_M body + Q8_0 output tensor + Q6_K token embeddings + f16 KV.**
The output-tensor precision is the single most Vietnamese-specific decision: wrong-
whole-syllable selection is a final-layer logit-discrimination failure among tonal
near-neighbors.

### 4.3 Measured CPU throughput (published + roofline-calibrated)

Single-stream CPU decode is **memory-bandwidth bound**; achieved 54–72% of theoretical
DRAM bandwidth (calibrated to i7-8700/i7-6700 measurements). Projected Q5_K_M:

| Hardware | 600-token VI reading |
|---|---|
| 6-core DDR4-3200 | ~32 s |
| 4-core DDR4-2400 (floor) | ~64 s |
| 8-core DDR5-5600 | ~16 s |

**Thread count is the largest free win**: SMT/hyperthreads measurably hurt (Ryzen 7 260:
8 physical threads 23.6 t/s vs 16 logical **6.2 t/s** — 3.8× regression). Ship
`--threads` = physical cores, never `nproc`. Prefill and decode prefer different thread
counts (`-t` vs `-tb`).

### 4.4 KV cache architecture

Per-token KV @4k f16: Qwen2.5-1.5B (2 KV heads) = 112 MiB; Qwen2.5-3B = 144 MiB;
**Qwen3-1.7B (8 KV heads) = 448 MiB**; Qwen3-4B = 576 MiB. Computed for Qwen3-1.7B:
28 layers × 2 (K+V) × 8 heads × 128 dim × 4096 ctx × 2 B = 448 MiB — fits in 4 GB
free RAM alongside ~1.25 GiB Q5_K_M weights with ~2.3 GiB headroom.

### 4.5 Speculative decoding

**Skip it for this workload.** Measured: 1.72× overall on a 3B CPU target but **1.01×
on open-ended chat** (vs 2.03× math) — acceptance rate tracks output predictability,
and a Vietnamese tarot reading is free-form prose. ggerganov: "If you try to generate
free-form text, the acceptance rate drops significantly." No viable distribution-
matched ~0.3B draft for a Vietnamese LoRA fine-tune exists. One cheap exception worth
testing: draft-free n-gram self-speculation (`--spec-type ngram-mod`, ~16 MB) — keep
only if acceptance >30%.

### 4.6 imatrix with Vietnamese calibration

Target-language-only imatrix beat English-only by 3.8% on the equivalent French
experiment (ikawrakow, discussion #5263); **English-only was worse than random tokens**.
Caveat: the effect shrinks at 4+ bits. Build a ~70% Vietnamese (register-matched),
~20% English (card names inline), ~10% structural corpus; 200–400 chunks at `-c 4096`.
The imatrix claim is only falsifiable against a no-imatrix quant from the same F16.

---

## 5. Training methodology research

### 5.1 LoRA configuration (converged evidence)

| Setting | Value | Evidence |
|---|---|---|
| Target modules | **all-linear, including MLP** | Thinking Machines "LoRA Without Regret": parameter-matched attention-only *underperforms* MLP-only |
| Rank | 32 | ~35M params ≈ 70 Mbit capacity vs ~10.5 Mbit demand (15k × 700 tok) ≈ 6.7× headroom |
| Alpha | 32 fixed | α=2r and α=32 coincide at r=16 (why both circulate); changing α requires changing LR |
| Dropout | 0.0 | ALLoRA: dropout "fails to converge as a reliable regularizer for short training episodes" |
| LR | 1e-4 | ≈10× full-FT LR, fit across 14 Llama/Qwen models by Thinking Machines |
| Schedule | cosine → 1%, 3% warmup | `warmup_ratio` **removed** in transformers v5 — express as warmup steps |
| Epochs | 2 as a stopping rule | eval every ¼ epoch |
| Batch | 32 effective | LoRA is measurably less tolerant of large batches, independent of rank |
| Embeddings | frozen | tied on Qwen3 → ~311M params, ~9× the adapter; no fertility problem to fix |

**Variants verdict**: rsLoRA only pays above ~14B; LoRA+ is provably a reparametrization
of `init_A/LR_A`; PiSSA shows spectral collapse; DoRA's gains are RL/VLA-only. **Vanilla
LoRA wins at 1–4B text SFT.**

**Packing**: skip. Token-level averaging across a packed batch upweights long sequences,
systematically downweighting short safety refusals (the highest-stakes slice).
`group_by_length` reproduces the same distortion via refusal-clustering → also dropped.

**Loss**: completions-only is correct here *for a reason* — instruction-modelling
loss-over-instructions wins only for long-instruction/short-output or tiny datasets,
the mirror image of this case. Train on ALL assistant turns in multi-turn, not just the last.

### 5.2 Synthetic data generation

- **Self-Instruct: no** (built to discover an unknown task distribution); **Evol-Instruct:
  no as-is** (drifts off-domain, inflates length); **Magpie: wrong tool, right trick**
  (steal the hot-prompts/cool-readings temperature split).
- **Schema-driven enumeration wins for a closed domain**: card × orientation × position ×
  spread × context × register × length — coverage becomes auditable.
- **Anti-collapse stack** (Dynamic Context Evolution, arXiv 2604.07147): naive prompting
  collapses 5.6±2.0% with 2–17 unstable clusters; with semantic memory + adaptive prompt
  evolution + verbalized tail sampling: 0.0±0.0%, stable 17–18 clusters, ~$0.50/1,000
  candidates. **Deduplication and prompt evolution are individually insufficient but
  jointly effective.**
- Multi-source synthetic data (≥2 teachers or personas) mitigates distribution collapse
  (ACL Findings 2026); ~30% real human anchor data changes the character of the mix
  (EMNLP 2025, >1000 LLMs).

### 5.3 Quality filtering — cheapest first

1. **Programmatic**: schema, language ID, diacritics, card-containment, orientation
   consistency, keyword collision, MinHash+embedding dedup.
2. **IFD** (NAACL 2024): 10% of data matched full-data performance — but the scoring
   model matters (base Qwen3-1.7B has weak VI; verify the kept/rejected sets differ).
3. **Deita 3-axis** (ICLR 2024): 6k samples matched SOTA baselines with 10× less data.
4. **Judge rubric gate** (AlpaGasus: 9k from 52k beat the full set, 5.7× faster).

Skip perplexity filtering as a primary signal (selects for typicality) and reward models
(no Vietnamese one exists).

### 5.4 Dataset size evidence

LIMA's 1k-curated regime applies to pure style transfer on an existing capability. Here
the capability doesn't exist (closed-set factual coverage 78×2×~6 ≈ 936 combos +
multi-turn + safety), so **tiering wins**: verified core (~3–5k, all gates incl.
calibrated judge) + bulk (programmatic gates only), size as an ablation outcome.

### 5.5 Preference optimization

**Skip DPO/ORPO/KTO for v1.** Preference collapse rates: ORPO 20.6% vs 11.0% for a
diversity-preserving alternative (p<0.001); DPO/ORPO degrade representational
separability while KTO/GRPO improve it. If a nameable defect survives SFT eval, prefer
KTO (binary labels are free from the filter pipeline) or DPO (pre/post-filter pairs);
scope 3–5k pairs; gate on diversity metrics.

---

## 6. Evaluation methodology research

### 6.1 LLM-as-judge discipline

- **Report Cohen's κ / Krippendorff's α, never raw agreement**: kappa deflation 33–41 pp
  on MT-Bench (Reliability without Validity, 21 judges, 541k judgments). "GPT-4 agrees
  with humans ~80%" is raw, uncorrected for chance.
- **Consistency–bias paradox**: >0.95 test-retest reliability coexists with >0.10
  position bias in production judges. Reproducibility ≠ validity.
- **Teacher ≠ judge — structural**: self-recognition correlates linearly with
  self-preference (Panickssery et al.). DeepSeek V4 Flash must not judge its own output.
- Position bias: pairwise both orders, order-consistent verdicts only; rubric scoring
  carries its own position bias in both option order and criterion order — 3 random
  permutations averaged.
- Verbosity bias is largely an artifact of pointwise scoring (<0.011 under a single
  pairwise rubric); use pairwise + length-controlled aggregates.

### 6.2 Vietnamese fluency — humans own it

GPT-4-class evaluators are biased toward higher scores; calibration against native
speakers is "necessary, especially in low-resource and non-Latin script languages"
(Hada et al., EACL 2024). BabelJudge (Jun 2026): composite reliability fell 0.714
(Hindi) → 0.550 (Swahili), order consistency collapsing to 0.480 (near-random under
swaps) — invisible to accuracy. **BabelJudge's gold-labelling-by-degradation audits a
judge with zero human annotation — run it on Vietnamese first.**

### 6.3 Human eval sizing (Card et al. power analysis)

Paired pairwise A/B, α=0.05, 80% power: 65/35 → ~85 judgments; **60/40 → ~194**;
55/45 → ~783. Below ~100 items you cannot distinguish a real 10-point win-rate move
from noise. Guidelines: 5 criteria, 1–5 scale, Vietnamese exemplar at every level;
give annotators the KB row (converts knowledge task → verification task, where high κ
lives); require justification for scores ≤2; targets κ ≥ 0.7 objective, α ≥ 0.5–0.6
subjective.

### 6.4 Faithfulness for a closed fact set

Three programmatic checks: card-name containment, orientation consistency, keyword
collision. MiniCheck (770M, GPT-4-level at 400× lower cost) is English-trained —
validate Vietnamese detection rate on mechanically perturbed KB-licensed prose before
gating; fall back to the deterministic KB-licensing check. FactScore-style precision
as the headline number. Self-consistency sampling (k=5, T=0.8) as a free unsupervised
tripwire. **Longer, more elaborate output correlates with more unsupported claims**
(IndustryBench) — keep faithfulness a separate axis from quality.

### 6.5 Automatic metrics for Vietnamese

Use: chrF++/COMET/BERTScore-PhoBERT (translation QA only), pairwise judge win-rate,
faithfulness precision, distinct-2/3 + self-BLEU (diversity gate), independent VI-LM
perplexity (anomaly detector). **Never BLEU/ROUGE for readings** (no reference; penalizes
variety; syllable-level whitespace breaks word scoring).

### 6.6 Safety eval

**Over-refusal is the more likely failure**: Death/The Tower/Ten of Swords denote harm
by name while queries are benign. Report harmful AND benign refusal rates together,
always. Build a Vietnamese tarot XSTest with **matched pairs** (same topic, vocabulary,
different intent). Test natively in Vietnamese and code-switched: low-resource languages
show ~3× harmful content likelihood; cross-language safety agreement as low as 12.8%.
Existing coverage: SEA-HELM (VI safety pillar), MultiJail-vi. **No Vietnamese XSTest or
OR-Bench exists — build 300–500 prompts.**

---

## 7. Environment findings [measured]

| Fact | Value | Consequence |
|---|---|---|
| GPU | NVIDIA **P106-100, 6 GB, compute 6.1** (Pascal), driver 580.173.02 | No bf16, no Tensor Cores, **fp16 ALU at 1/64 of fp32** (CUDA guide CC 6.1 table) |
| CPU | `QEMU Virtual CPU v2.5+`, 6 cores, **SSE4.2 only** — no AVX/AVX2/FMA/F16C | llama.cpp scalar fallback only; local tok/s does NOT transfer |
| RAM | 12 GiB total, 7.9 available | adequate |
| Disk | **28 GB free**, 92% full; `/tmp` is the SAME filesystem | no independent scratch budget |
| Python | `.venv` py3.12.3, `transformers 5.14.1` | no training stack installed |
| Toolchain | git, make, gcc 13.3, node 22, bun, uv, docker, nvcc 12.0 | **cmake ABSENT → blocks llama.cpp build** |
| Git | **not a repo**; nested `.git` at `data/vietnamese/Tarot-Vietnamese-API/` | 975 MB unversioned; gitlink trap |
| Network | huggingface.co HTTP/2 200 | model download OK |

---

## 8. Toolchain compatibility findings

### 8.1 QLoRA on Pascal — works, with strict constraints

- bitsandbytes NF4/FP4 and 8-bit optimizers: officially supported at **CC 6.0+**;
  LLM.int8() needs 7.5+. Binary-verified: the 0.50.0 PyPI wheel's
  `libbitsandbytes_cuda126.so` contains `sm_60` cubins with `gemm_4bit_simt`,
  `kQuantizeBlockwise` (NF4), and `kOptimizerStatic8bit*`. No source build, no flags —
  **conditional on a cu126-or-older torch**.
- **torch sm_61 ships only in `2.13.0+cu126` from the cu126 index.** Default PyPI wheels
  since 2.8.0 are cu128+ with no sm_61 SASS; release wheels are SASS-only
  (`_PTX_ARCHES = {120}`) — **no JIT fallback**; a wrong wheel hard-fails. torch 2.7.0
  was the last plain `pip install torch` that worked on Pascal.
- **Trap: `torch.cuda.is_bf16_supported()` returns True on Pascal** via emulation
  (`including_emulation=True` default). Gate on `get_device_capability()[0] >= 8`.
- **Triton floor is CC 8.0+** → no Unsloth, no Liger fused cross-entropy, no
  `torch.compile` locally.
- **Unsloth out twice** (transformers ≤5.5.0 conflict + Triton floor); **Axolotl out**
  (pins torch ≤2.12.1 + Triton paths). **Plain PEFT + TRL is the only stack with no
  architecture floor.**
- Memory reality: Qwen3's 151,936 vocab makes logits + fp32 CE + logit grads cost
  2.5–3.1 GB at seq 2048 — more than the 0.73 GB of NF4 weights. Measured comparable:
  Qwen2.5-1.5B (same vocab) peaked **6.2 GB at bs1/seq512 on an 8 GB RTX 4060** — above
  the 6 GB card's total. Practical local ceiling: **seq 1024, bs 1, grad-accum 8–16,
  r=8–16, paged_adamw_8bit**.

### 8.2 transformers v5 breaking surface (5.14.1 installed)

- `load_in_4bit=` / `load_in_8bit=` **removed** → `quantization_config=BitsAndBytesConfig(...)`.
- `Trainer(tokenizer=)` → `processing_class=`; `Trainer.train(model_path=)` →
  `resume_from_checkpoint`.
- `warmup_ratio` **removed** (→ `warmup_steps` accepts floats); `overwrite_output_dir`,
  `logging_dir`, `jit_mode_eval` removed.
- `apply_chat_template` now returns a **BatchEncoding** (silently breaks hand-rolled
  collators); `encode_plus` → `__call__`; `batch_decode` → `decode`.
- `config.rope_theta` → `config.rope_parameters['rope_theta']`; TensorFlow/Jax removed.
- peft/trl/datasets/accelerate are all **v5-clean, unpinned** — no `transformers<5` in
  the core stack (bitsandbytes' `<5` is a test-extra constraint only).

### 8.3 Recommended pinned set (local, Pascal)

```
--extra-index-url https://download.pytorch.org/whl/cu126
torch==2.13.0+cu126
transformers==5.14.1
peft==0.20.0
trl==1.9.2
accelerate==1.14.0
datasets==5.0.1
bitsandbytes==0.50.0
```

Cloud GPU: same versions, standard PyPI torch (no cu126 needed on CC ≥ 8).

---

## 9. Decision summary

| Decision | Choice | Decisive reason |
|---|---|---|
| Base model | **Qwen3-1.7B** | Prompt-cache reuse (10–20× multi-turn factor) beats tokenizer efficiency (~7%); best IFEval; official GGUF; ~22 t/s on 6-core |
| Fallback | Qwen2.5-3B (8-core installs) | 2 KV heads → 144 MiB KV; VMLU 54.81 tuned evidence |
| Watch | Qwen3.5-2B | Re-evaluate when #24714/#24737/#24812 close and GGUF is perplexity-validated |
| Training venue | **Rented cloud GPU (24 GB+, CC ≥ 7.5)** | 6 GB Pascal caps seq at 1024, truncating the p95 1,700-token examples; no bf16; ~10–30× slower |
| Quantization | **Q5_K_M + Q8_0 output + Q6_K embd + f16 KV** | Q4 penalty triples at small scale; output tensor protects tonal discrimination |
| Latency claims | **Projections only, caveated** | Local box has no AVX; no transferable first-party numbers |
| Vietnamese reviewer | **Frontier-judge fallback with degradation calibration** | None available; fluency/register explicitly unverified |
| Reversed meanings | **Generate natively under English semantic constraint** | Translation makes translationese correlate with the reversed label |
| Knowledge split | Card facts → retrieval; style/code-mixing/safety → weights | Cache-stable prefix amortizes 1,050-token context; extensible deck free via KB path |
| Dataset | Tiered core (~3–5k) + bulk, size as ablation output | Under no-reviewer, quality-assurable size is the binding constraint |
| Safety ownership | `model` / `deterministic_validator` / `both_validator_wins` per category | Prevents double-refusal and uninterpretable eval numbers |
| Crisis routing | Time-aware data table with staleness fallback | Ngày Mai closed Mon/Tue + before 13:00; 3 a.m. Tuesday has no answer there |

---

## 10. Evidence ledger

| Evidence | Source |
|---|---|
| Corpus inventory (file-by-file, defects, join keys) | explore `ses_0444e4393ffeaZTLGpyIQ4olWZ` + tail extraction |
| Qwen3.5 specs (HF configs + model cards, fetched) | librarian `ses_0444e4443ffecy9ndKLn3kbOWe` |
| Vietnamese models + VMLU/SEA-HELM + tokenizer fertility | librarian `ses_0444e4400ffevPPeXAoaoWAAji` |
| CPU inference (runtimes, quants, throughput, imatrix, KV, spec-decode) | librarian `ses_0444e4426ffe89eZpgGKiIc236` + tail extraction |
| SFT/eval methodology (LoRA, synthetic data, judge, safety) | librarian `ses_0444e42d9ffe3ccMoqunSRdeko` + tail extraction |
| Architecture tradeoff resolution (7 questions) | oracle `ses_0443af680ffen2wFjDeLQCT1Et` |
| First-party tokenizer + sequence-budget measurements | this session (fert2.py, seqlen.py on real corpus) |
| Environment probe (11 items, real command output) | explore `ses_044011956ffeN2u7jgDd4rxHiN` |
| Pascal QLoRA + transformers v5 compatibility (binary-verified) | librarian `ses_043fd1685ffeAYDi3XXgmv16JR` |
| Metis gap analysis (7 blockers, 17 high, 8 medium) | metis `ses_043e8853effeHRI0Vl0ojxO3gX` |
| Dual high-accuracy review (3 rounds) | momus `ses_041851be3ffeQS2YvPZb15MrQj`, oracle `ses_04183822bffeliTBE6DHsiD79U` |

**Known gaps (research limitations, stated honestly):**

- No published Vietnamese benchmarks exist for any Qwen3.5 size; SEA-HELM has no model
  below 3B — our own eval is the only ground truth below 3B.
- No Vietnamese quantization study exists anywhere; the tone minimal-pair eval in the
  plan is that study.
- Judge-vs-native κ for Vietnamese is unmeasured in the literature.
- The Vietnamese esoteric book corpus may itself contain translation-derived text
  (native fraction unverified) — affects the function-word profile baseline.
- Vietnamese YouTube ASR destroys domain vocabulary ("tarot" → "cà rốt" 10× more often
  than correct) — usable for register only, not card-level ground truth.
