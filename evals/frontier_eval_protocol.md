# Frontier-Judge Eval Protocol — vn-tarot-llm

Status: W3e.6 artifact. Every quality claim that depends on a frontier judge is
**"frontier-judge–assessed, not native-verified"** — no native Vietnamese
reviewer is available for this project (plan decision, openly caveated).

## Judges

| Role | Model | Provider | Notes |
|---|---|---|---|
| Primary judge | `deepseek-v4-flash` via gateway | OpenAI protocol | Same gateway as generation; self-consistency risk acknowledged |
| Second judge | *(unprovisioned)* | — | If provisioned later, MUST be architecturally unrelated (different provider family). Until then layer-4-style judging is labelled **self-consistency only** |

Judge model versions are recorded per run in `evals/results/*.json`
(`provider` field) plus the gateway-reported model id in each raw response log.

## Pairwise protocol

Used in W4.4 checkpoint selection (fine-tune vs base win-rate):

1. Prompt set: 150–200 prompts drawn from `evals/suites/core_assertions.jsonl`
   categories {explanation, reading, safety, correction} + held-back prompts.
2. Both orderings judged (A-vs-B and B-vs-A); only **order-consistent**
   verdicts count as decisive. Inconsistent orders are discarded and reported.
3. Each verdict averages 3 rubric permutations to damp position/wording bias.
4. Cohen's κ reported per rubric criterion across permutations.
5. Minimum detectable effect at n=150 decisive pairs, α=0.05, power=0.8:
   ~±13 percentage points on win-rate. Claims smaller than this are reported
   as "no detectable difference".

## Rubric criteria and assessors

| Criterion | Assessor | Caveat level |
|---|---|---|
| Card containment, no hallucination | programmatic (`evals/faithfulness.py`) | none — deterministic |
| Orientation discrimination | programmatic (`tfvn.validators`) | none |
| Tone accuracy | programmatic (`evals/vn_tones.py`, mechanical gold) | none |
| Language drift | programmatic (`evals/drift.py`) | none |
| Refusal / over-refusal rates | programmatic (`evals/score_safety.py`) | keyword-rule based; stated method |
| Fluency / naturalness of Vietnamese | frontier judge (pairwise) | **frontier-judge–assessed, not native-verified** |
| Reading coherence & question subordination | frontier judge (pairwise) | **frontier-judge–assessed, self-consistency** |
| Cultural authenticity of refusal forms | frontier judge vs policy templates | template-anchored; unverified by natives |

## Structural anchor from authentic readings

The 30 held-back authentic Reddit readings are consumed ONLY as an abstracted
discourse schema (opening → card-by-card → synthesis → advice), never as
Vietnamese evaluation text. They appear here as a checklist item the judge rubric
references ("does the reading follow the canonical discourse schema?"), keeping
the authentic text out of the scoring path entirely.

## Degradation calibration linkage

Layer-4-style rubric filtering may only gate on axes whose detection rate was
calibrated above chance with CI lower bound > chance (`judge/calibration_report.md`,
W0.6). Axes marked `no-gate` there are excluded from every gate here; their
numbers are reported descriptively with the caveat attached.

## Forbidden claims

- "authentic native Vietnamese prose" — forbidden; say "synthesised Vietnamese,
  frontier-assessed".
- "validated fluency" — forbidden without a native reviewer.
- Any metric from an instrument without a measured Vietnamese detection rate is
  reported as descriptive, never as a gate.
