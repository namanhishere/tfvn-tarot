# vn-tarot-llm

Fine-tuned Qwen3-1.7B that reads Vietnamese tarot: deck fold, single-card
explanations, multi-card readings, extensible decks, and a safety stop with
time-aware crisis routing — served CPU-only via llama.cpp at Q5_K_M.

**Status:** pipeline W0–W7 complete and verified on smoke models. Production
training runs on rented GPU via `training/notebooks/` (Kaggle / vast.ai /
Colab); every downstream stage consumes the resulting checkpoint unchanged.

## Pipeline

```
data corpus (490 MB, gitignored) ──► W1 canonical KB ──► W2 Vietnamese layer
        │                                                │
        ├──► W3 SFT corpus (13.5k, filtered/split) ◄──────┘
        │                 │
        │                 ▼
        │         W4 SFT fine-tune (LoRA r=32)  ← training/notebooks/
        │                 ▼
        ├──► W5 merge → GGUF → imatrix → quant selection
        │                 ▼
        └──► W6 safety slice → continuation → labeled eval report
                          ▼
                   W7 serving: byte-stable RAG + validators + tool calls
```

## Quickstart

```bash
# tests (233 unit + integration)
.venv/bin/python -m pytest tests/ -q

# KB integrity assertions (stdlib only)
PYTHONPATH=src python3 -m tfvn.assert_kb

# end-to-end serving test (needs the smoke quant in /tmp/quants/)
RUN_E2E=1 .venv/bin/python -m pytest tests/test_e2e_serving.py -q
```

### Train

Open one of `training/notebooks/{kaggle,vast,colab}_train_sft.ipynb` on a
≥24 GB Turing/Ampere+ GPU. The committed config: LoRA r=32 α=32 all-linear,
LR 1e-4 cosine→1%, completions-only loss, seq 2048, epochs 2, fixed seed.
An **epoch-1 orientation tripwire** halts the run if ≥5% of card pairs exceed
the Jaccard threshold — failed runs cost 25% of budget, not 100%.

Then: `scripts/export_gguf.py` → `build_imatrix_corpus.py` +
`llama-imatrix` → `run_quant_comparison.py`.

### Serve

```bash
TAROT_MODEL=artifacts/quants/model.q5_k_m_imx.gguf scripts/serve.sh
uvicorn tfvn.serve:app --port 8078        # see serving/README.md
curl -s localhost:8078/reading -d '{"question_vi":"...","seed":42,"n_cards":3}'
```

Deterministic validators run post-generation; one constrained regeneration,
then an explicit `validation_warning`. Crisis phrasing routes through
`policy/crisis_routing.py` before the model ever sees the prompt.

## Repository map

| Path | Contents |
|---|---|
| `kb/` | frozen knowledge base (`cards.jsonl`, compact cards, spreads, whitelist) |
| `datasets/` | filtered SFT corpus, splits, hashes, safety slice |
| `evals/` | assertion suite, tone pairs, drift, faithfulness, safety XSTest, MCQ tripwire |
| `judge/`, `policy/` | judge calibration, safety policy, crisis routing |
| `src/tfvn/` | serialiser, validators, prompts, tools, reading assembly, serving |
| `scripts/` | build/train/eval/quant/serve entry points |
| `training/` | cloud notebooks |
| `.omo/plans/` | the work plan this repo implements |

## Integrity rules

- All JSONL is canonical (sorted keys, compact): byte-stable across processes;
  golden-file tested.
- Every generated artifact carries provenance and content hashes
  (`kb/CARDS_HASH.txt`, `datasets/DATASET_HASH.txt`).
- Claims discipline: no "native-verified" language without a native reviewer;
  frontier-assessed metrics are labelled as such
  (`evals/frontier_eval_protocol.md`). Baseline floors are recorded for every
  eval so fine-tune regressions are measurable, not assumed.

## License & data

Corpus sources are license-clean (public-domain Waite/Mathers texts, CC-BY
Vietnamese tarot prose, open card datasets). See `data/MANIFEST.md` provenance
index (corpus itself is gitignored).
