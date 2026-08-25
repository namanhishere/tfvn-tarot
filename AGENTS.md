# AGENTS.md — tfvn-tarot agent handoff

**Status: all plan waves W0–W7 + F1–F4 executed and committed (2026-08-24).**
The authoritative plan is `.omo/plans/vn-tarot-llm.md`; machine audit lives in
`artifacts/final_verification.json` (8/8) with narrative in
`artifacts/plan_compliance_audit.md`. 141 tests (incl. 3 cross-process serving
E2E, run with `RUN_E2E=1`; smoke quant at `/tmp/quants/model.q5_k_m_imx.gguf`):
`.venv/bin/python -m pytest tests/ -q --ignore=tests/webapp`.
Local P106-100 (6 GB) pilots (2026-08-25): 600-row/1-epoch Qwen3-0.6B run
green (eval_loss 1.89); 2-epoch run proved the epoch-1 orientation tripwire
fires and halts (58% Jaccard rate on the undertrained pilot — expected).
Tripwire only arms when `--epochs > 1`; smoke mode disables it.
Pilot findings for the cloud run: `--lr 3e-4` beats the default `1e-4`
(eval 1.709 vs 1.893 @600 rows/1 epoch, 0.6B) — sweep LR first epoch on the
GPU box before committing to 2 full epochs. Seq 2048 does NOT fit 6 GB even
with expandable_segments (OOM at micro_batch 1); local pilots cap at seq 512.
Ablations @0.6B/600rows/1ep/lr3e-4/seq512 (same core eval): r32+core 1.709 <
r32+bulk 1.771 < r16+core 1.800 — plan defaults (r32, core-only) confirmed.
Deployment round-trip proven with trained (pilot) weights: merge → F16 GGUF
→ Q5_K_M → llama-server → Vietnamese reading with inline EN card name.
serve.sh now uses `--load-mode mmap` (old `--mmap` deprecated in llama.cpp).

---

## Repo conventions (unchanged, follow these)

- Canonical JSONL: `json.dumps(..., ensure_ascii=False, sort_keys=True,
  separators=(",", ":"))` — use `src/tfvn/serialise.py`
  (`dumps_canonical`, `write_jsonl`, `read_jsonl`).
- Vietnamese tokenizer for Jaccard/profile distance:
  import from `src/tfvn/w2_gates.py::simple_vi_tokens` (never redefine).
- Card ids 0–77; canonical names in `src/tfvn/aliases.py`. Page of Pentacles
  is card_id 74 in the VN source (defect fixed by name-keying).
- One commit per task, `feat:` / `fix:` / `test:` prefixes matching the plan.
  Never commit `.env` (gitignored), `data/`, `artifacts/` (gitignored),
  `.cache/`.
- LLM gateway: `src/tfvn/llm_client.py`, prompt-hash cached. The endpoint's
  model is a **reasoning chain consumer**: give `chat_json` calls
  `max_tokens >= 16000` or content comes back empty (`finish_reason=length`).

## Where everything lives now

| Layer | Path |
|---|---|
| Eval harness | `evals/` — assertions, tone pairs (300), drift, faithfulness, safety XSTest (300 rows), provider abstraction |
| Training | `scripts/train_sft.py` (platform-agnostic), `training/notebooks/{kaggle,vast,colab}_train_sft.ipynb` |
| Quantization | `scripts/export_gguf.py`, `build_imatrix_corpus.py`, `run_quant_comparison.py` |
| Safety | `policy/safety.md`, `policy/crisis_routing.py`, `scripts/build_safety_slice.py`, `augment_safety_slice.py`, `run_safety_eval.py` |
| Serving | `src/tfvn/tools.py` (deck/shuffle/draw/clarify), `reading.py` (byte-stable RAG), `serve.py` (validators + 1 constrained regen + crisis gate), `scripts/serve.sh`, `serving/README.md` |
| Extensibility | drop JSON into `kb/decks/*.json` — validators pick it up automatically |

## What is NOT done (billing-gated)

1. Real W4 training on a cloud GPU — notebooks are ready; every pipeline stage
   was smoke-verified locally on Qwen3-0.6B (CPU fp32 merge → GGUF → imatrix →
   quants → served evals).
2. Frontier-judge pairwise checkpoint selection (W4.4) — runs post-training.
3. Full-300 XSTest scoring run — CPU took >1h for 600 generations; use
   `--xstest-limit/--tone-limit` until a GPU box runs it.

## Verification habits that caught real bugs here

Run the smoke trainer after touching training code:
`python scripts/train_sft.py --smoke --out /tmp/smoke_run` (~30 s, uses
Qwen3-0.6B). It exercises tokenization, masking, LoRA attach, optimizer steps,
eval, save and metadata. Two silent-corruption bugs (NaN loss via truncation;
v5 API removals) were caught exactly this way.
