# AGENTS.md — tfvn-tarot agent handoff

**Status: all plan waves W0–W7 + F1–F4 executed and committed (2026-08-24).**
The authoritative plan is `.omo/plans/vn-tarot-llm.md`; machine audit lives in
`artifacts/final_verification.json` (8/8) with narrative in
`artifacts/plan_compliance_audit.md`. 129 tests: `.venv/bin/python -m pytest tests/ -q --ignore=tests/webapp`.

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
