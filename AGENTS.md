# AGENTS.md — tfvn-tarot Wave 2 execution guide

Handoff document for coding agents (Claude Code, Codex, OpenCode subagents) asked
to execute **Wave 2 — Vietnamese meaning layer (C2)** of the vn-tarot-llm plan
(`.omo/plans/vn-tarot-llm.md`, Wave 2 section, lines 142–163). Read that plan
section first; this file is the executable summary plus the repo-specific
mechanics you need to actually run it.

---

## 1. What Wave 2 is

Wave 1 produced a bilingual skeleton: a 156-row English semantic spine
(`kb/english_spine.jsonl`, 78 cards × upright/reversed) and a defective
Vietnamese upright layer (`kb/vn_upright.jsonl`, 77 real + 1 placeholder —
Page of Pentacles is `synthetic_no_anchor`). The Vietnamese source has a fatal
defect: **5 of its 6 prose fields are byte-identical between upright and
reversed** — the model would have no Vietnamese way to express a reversed card.

Wave 2 fixes that in three tasks:

| Task | Deliverable | Purpose |
|---|---|---|
| W2.1 | `kb/vn_orientation_attribution.json` | Decide per card: are the identical fields orientation-agnostic or upright-skewed? |
| W2.2 | `kb/vn_spine.jsonl` + `kb/w2_2_gate_report.json` | Generate the Vietnamese **reversed** meanings for all 78 cards, gated |
| W2.3 | `kb/cards.jsonl` + `kb/CARDS_HASH.txt` | Freeze the 156-row bilingual KB (single source of truth) |

Wave 2 feeds Wave 3 (SFT dataset) and Wave 4 (training). No GPU billing happens
until W2 completes.

## 2. The cardinal rule (do not violate)

**Generate native Vietnamese from the English semantic spine as constraints.
NEVER show the model English prose to translate.** Translationese that is
correlated with the reversed label is categorically worse than uniform
translationese — it lets the model "cheat" the orientation distinction. The
generation prompt embeds `keyword_atoms_en` + `polarity_axis` as semantic
constraints and style exemplars from the authentic Vietnamese `title_secondary`
register, never the English meaning text.

## 3. Two ways to execute Wave 2

### Approach A — LLM-API pipeline (recommended, already built and verified)

Working code exists and has been verified against the live endpoint on a
sample: **8/8 cards produced `synthetic` reversed rows, 0 failed gates, 100%
negative-control rejection** (including the previously-problematic Four of
Pentacles). Run it as-is:

```bash
cd /home/ubuntu/tfvn-tarot
cp .env.example .env        # if missing; fill LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
python3 scripts/test_wave2_api.py                 # smoke test (3 cards, ~16 API calls)
python3 scripts/build_wave2_api.py                # FULL run (all 78 cards)
python3 scripts/build_wave2_api.py --only w22 --cards 0,1,2   # partial / resume
```

Component map:
- `src/tfvn/llm_client.py` — OpenAI-protocol client; prompt-hash cache in
  `.cache/gen/` (a cache hit never re-bills). `load_env()` reads `.env`.
- `src/tfvn/w2_gates.py` — deterministic gates: G1 Vietnamese-ness profile
  check, G2 orientation Jaccard, G4 forbidden-claims, plus W2.1 polarity
  lexicon and the authentic-pair Jaccard distribution.
- `src/tfvn/w2_prompts.py` — generation + rubric prompts (native-Vietnamese
  instruction, JSON output contract).
- `scripts/build_wave2_api.py` — orchestrator (W2.1 → W2.2 → W2.3).
- `scripts/test_wave2_api.py` — end-to-end smoke test with pass/fail exit code.

The gates and their floors (plan W2.2):
- **G1 Vietnamese-ness**: function-word profile distance vs
  `kb/vn_register_profile.json`, calibrated to p90 of the authentic phatjkk
  register (≈5.2, recomputed on every run — do NOT hardcode). Floor: ≥75% of
  candidates pass.
- **G2 orientation Jaccard**: token-set Jaccard of generated prose vs the
  card's authentic upright Vietnamese prose; threshold = p90 of authentic
  upright/reversed `title_secondary` pairs from
  `data/vietnamese/Tarot-Vietnamese-API/data.txt` (≈0.24). ≥90% of pairs below.
- **G3 keyword back-check**: LLM rubric call — English atoms covered in the VI
  prose; recall ≥0.7 AND English card name inline. Negative control: prose from
  a DIFFERENT card's spine must be rejected ≥80% (measured on a sample).
- **G4 forbidden claims**: no literal death / diagnosis / legal advice
  (≤2 tolerated violations, each documented).

Generation is resilient to marginal gate misses: each card gets `--variants`
(default 2) candidates per round; if none passes every gate, the orchestrator
redraws fresh candidates up to `--max-retries` (default 3) more rounds. Each
round embeds the attempt number in the prompt so the prompt-hash cache misses
and genuinely new content is drawn (an identical cached response would be
useless). The gate report records `attempts_used` and `selected_variant` per
card. Retries cost ~1-2 API calls per attempt; the full 78-card run is
typically 350-400 calls (cache hits never re-bill).

Acceptance for W2.2: 156-row `kb/vn_spine.jsonl`, every reversed row
`vi_provenance: synthetic`. W2.3 refuses to write `cards.jsonl` unless all 78
reversed rows are `synthetic` (pass `--allow-incomplete` only for partial runs
that must not ship).

### Approach B — agent-executed manual path (no code changes)

If you are an agent WITHOUT API access or prefer a hands-on pass, do the tasks
by hand, in order, producing the same artifacts:

1. **W2.1** — For each card, read the 5 identical fields in `kb/vn_upright.jsonl`
   (`title_main/love/work/money/health`). Classify the identical prose as
   `vi_orientation_agnostic` (neutral/mixed polarity) or `vi_upright`
   (positive-skewed). Anchor: `title_secondary` describes card identity, not
   orientation. Write `kb/vn_orientation_attribution.json` (schema:
   `attributions[]` with `card_id`, `attribution`, `score`).
2. **W2.2** — For each of the 78 reversed rows in `kb/english_spine.jsonl`,
   write a new Vietnamese reversed meaning: 3–5 sentences, natural register,
   English card name inline, driven by `keyword_atoms_en` + `polarity_axis`.
   Two variants per card. Apply the four gates above. Output
   `kb/vn_spine.jsonl` (156 rows; reversed rows carry `vi_reversed_prose`,
   `vi_keywords_reversed`, `vi_provenance: synthetic`).
3. **W2.3** — Join `english_spine.jsonl` + `vn_upright.jsonl` +
   attribution + synthesis into `kb/cards.jsonl` (156 rows, field list in the
   plan lines 160). Assert: 156 rows; all 78 card_ids both orientations; Two of
   Cups reversed provenance `mathers`; Four of Pentacles reversed `waite`;
   Page of Pentacles `synthetic_no_anchor`; no `title_heath`; `sức khỏe`
   normalised (never `sức khoẻ`). Write `kb/CARDS_HASH.txt` (sha256 of the
   canonical serialisation).

## 4. Repo conventions (follow these)

- JSONL rows are canonical JSON: `json.dumps(..., ensure_ascii=False,
  sort_keys=True, separators=(",", ":"))` — use `src/tfvn/serialise.py`
  (`dumps_canonical`, `write_jsonl`, `read_jsonl`).
- Vietnamese tokeniser (for Jaccard / profile distance): the regex in
  `src/tfvn/register_profile.py::_simple_vi_tokens`; import it from
  `src/tfvn/w2_gates.py::simple_vi_tokens` instead of redefining.
- Card ids 0–77, canonical names from `src/tfvn/aliases.py`
  (`CANONICAL_NAMES`, `NAME_TO_ID`). Page of Pentacles is card_id 74 in
  `vn_upright.jsonl` (source id defect).
- One commit per task, message style: `feat: ...` / `fix: ...` matching the
  plan's `Commit:` lines. Never commit `.env` (gitignored) or `data/`.
- Every generation is cached by prompt hash — re-runs are cheap; do not
  disable the cache unless debugging prompt changes.

## 5. Environment & secrets

- `.env` (gitignored) holds `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. The
  template is `.env.example`. If `.env` is absent, set the same vars in the
  environment. NEVER echo or commit the key.
- The generation endpoint is OpenAI-protocol (`{base}/chat/completions`).
- The model is a **reasoning model**: it spends output tokens on a reasoning
  chain before the answer. The client sends `thinking: {"type": "disabled"}`
  by default (`LLM_THINKING=disabled`) so the chain is skipped — without it,
  some prompts exhaust the token budget and return empty `content`
  (`finish_reason: length`). If a call still returns empty content, raise
  `max_tokens` (the client/build script retry with doubled budgets
  automatically). Do not pass `response_format` to endpoints that don't
  support it — the client falls back automatically.

## 6. Verification checklist (before reporting done)

1. `python3 scripts/test_wave2_api.py` exits 0 (or, for Approach B, the
   equivalent assertions pass).
2. `kb/vn_orientation_attribution.json` has 78 entries.
3. `kb/vn_spine.jsonl` has 156 rows; all reversed rows `synthetic` (full run).
4. `kb/w2_2_gate_report.json` aggregate shows `failed_gate: 0` and
   `negative_control_rejection_rate >= 0.8`.
5. `kb/cards.jsonl` has 156 rows and `kb/CARDS_HASH.txt` exists (full run).
6. Spot-check 3 generated reversed rows: natural Vietnamese, English card name
   inline, reversed meaning distinct from the upright row's `title_secondary`.
7. `git status` clean of `.env`, `data/`, `.cache/`.

## 7. Out of scope for Wave 2 (do not start)

Wave 3 (SFT dataset generation), Wave 3e (eval harness), Wave 4 (training).
The only Wave-2-adjacent artifact that belongs here is the gate report
(`kb/w2_2_gate_report.json`) and the attribution report — those ARE W2 outputs.
