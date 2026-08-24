# vn-tarot serving — llama-server HTTP API (W7.4)

## Quick start

```bash
# 1. point at a GGUF (fine-tuned master or quant)
export TAROT_MODEL=artifacts/quants/model.q5_k_m_imx.gguf
export TAROT_PORT=8079

# 2. boot llama-server (physical-core detection, --mmap, ctx 4096)
scripts/serve.sh

# 3. serve the application API in another terminal
uvicorn tfvn.serve:app --host 127.0.0.1 --port 8078
```

`GET /health` → `{"ok": true, "deck_size": 78, "n_ctx": 4096}`.
`POST /reading` `{"question_vi": "...", "seed": 42, "n_cards": 3}`.

## Minimum requirements

- **RAM: 4 GB free** beyond the OS baseline. The Q5_K_M quant of Qwen3-1.7B
  is ~1.2 GB; mmap keeps cold pages on disk, budget ~2 GB resident under load.
- Disk: GGUF size + ~200 MB for KV cache spill headroom.

## Expected latency — ROOFLINE PROJECTIONS, NOT BENCHMARKS

These are **not first-party measurements**. They are roofline projections for
the reference deployment class; measure your own hardware before quoting SLAs.

| CPU class | Prefill (1k tok) | Decode (tok/s, est.) |
|---|---|---|
| 4-core DDR4-2400 (floor) | ~8–15 s | ~4–7 |
| 6-core DDR4-3200 (typical) | ~5–9 s | ~7–11 |
| 8-core DDR5-5600 (upside) | ~3–6 s | ~10–16 |

The local development box is SSE4.2-only (no AVX/AVX2 build); it is **slow but
correct** and used for correctness verification only.

## Prompt-cache discipline

- Assembly order is fixed: system block → card block (canonical JSON per drawn
  card) → position glosses → user question (`src/tfvn/reading.py`).
- The same draw produces a byte-identical prefix across turns (golden-file
  test); llama-server's prompt cache hits on turns 2+.
- Serving uses `/v1/chat/completions`; the message prefix matches the SFT
  training format (`train_sft.format_example`) so no distribution shift.

## Multi-turn truncation policy

`n_ctx = 4096`. When accumulated history exceeds the post-prefix budget,
**oldest user/assistant exchanges are dropped first**
(`tfvn.reading.truncate_history`); the stable prefix is never truncated.

## Validators in the serving path

Post-generation, deterministic checks run on every output:
containment (drawn cards present, nothing hallucinated), orientation
consistency, keyword-collision, position coverage. A failure triggers **one**
constrained regeneration; a second failure returns the original text with
`validation_warning: true` — never silent truncation, never an error.

## Safety stop

- Crisis phrasing routes through `policy/crisis_routing.py`
  (**validator-owned**, pre-model): open hours → Ngày mai hotline;
  closed/stale (>90 days since verification) → static always-valid fallback.
- Ambiguous requests ("bói đi") get ONE clarifying question instead of a
  guessed reading.
- Medical/legal/financial queries are answered-with-caveat per
  `policy/safety.md`; the full matrix lives there.

## Extensible decks

Drop a JSON file into `kb/decks/<name>.json`:

```json
{"deck": "my_deck", "canonical_names": ["Card One", "Card Two"]}
```

plus compact rows in the same schema as `kb/compact_cards.jsonl`. No code
changes; validators accept registered names automatically (proven by
`tests/test_extensibility.py`).
