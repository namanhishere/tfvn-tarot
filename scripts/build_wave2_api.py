#!/usr/bin/env python3
"""Approach 1 — Wave 2 via the LLM API (OpenAI protocol).

Runs the three Wave 2 tasks using the configured generation endpoint
(LLM_BASE_URL / LLM_API_KEY / LLM_MODEL from .env or the environment):

  W2.1  Orientation attribution  -> kb/vn_orientation_attribution.json
  W2.2  Vietnamese reversed-meaning synthesis with 3-axis gate
                                  -> kb/vn_spine.jsonl + kb/w2_2_gate_report.json
  W2.3  Frozen 156-row bilingual KB
                                  -> kb/cards.jsonl + kb/CARDS_HASH.txt

Usage:
  python scripts/build_wave2_api.py                 # full run (all 78 cards)
  python scripts/build_wave2_api.py --only w21      # attribution only
  python scripts/build_wave2_api.py --limit 3       # first 3 cards (test)
  python scripts/build_wave2_api.py --cards 0,1,2   # specific cards
  python scripts/build_wave2_api.py --dry-run       # no API calls (pending rows)
  python scripts/build_wave2_api.py --neg-control 10 --no-neg-control

Every generation call is prompt-hash cached (LLMClient) — re-runs after a
partial failure only bill the missing prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.llm_client import LLMClient, LLMError, load_env  # noqa: E402
from tfvn.serialise import dumps_canonical, read_jsonl, write_jsonl  # noqa: E402
from tfvn.w2_gates import (  # noqa: E402
    authentic_pair_jaccard_distribution,
    authentic_profile_distances,
    card_name_inline,
    check_vietnamese_ness,
    forbidden_claims,
    jaccard,
    keyword_containment,
    percentile_90,
    polarity_lexicon_attribution,
)
from tfvn.w2_prompts import (  # noqa: E402
    build_generation_messages,
    build_rubric_messages,
)

KB = ROOT / "kb"
DATA_TXT = ROOT / "data/vietnamese/Tarot-Vietnamese-API/data.txt"

# The 5 byte-identical phatjkk prose fields (W2.1/H4).
IDENTICAL_FIELDS = ("title_main", "title_love", "title_work", "title_money", "title_health")


# ---------------------------------------------------------------- loading ---


def load_inputs() -> Dict[str, Any]:
    spine = read_jsonl(KB / "english_spine.jsonl")
    vn_upright = read_jsonl(KB / "vn_upright.jsonl")
    profile = json.loads((KB / "vn_register_profile.json").read_text(encoding="utf-8"))
    rev_rows = [r for r in spine if r["orientation"] == "reversed"]
    rev_rows.sort(key=lambda r: r["card_id"])
    up_rows = [r for r in spine if r["orientation"] == "upright"]
    up_rows.sort(key=lambda r: r["card_id"])
    vn_by_id = {int(r["card_id"]): r for r in vn_upright}
    return {
        "spine": spine,
        "rev_rows": rev_rows,
        "up_rows": up_rows,
        "vn_by_id": vn_by_id,
        "profile": profile,
    }


def upright_vn_prose(vn_row: Optional[dict]) -> str:
    """Anchor Vietnamese upright prose for a card (for G2 Jaccard)."""
    if not vn_row:
        return ""
    for key in ("title_secondary", "title_main"):
        val = (vn_row.get(key) or "").strip()
        if val:
            return val
    return ""


def exemplars_for(
    vn_by_id: Dict[int, dict], exclude_id: int, rng: random.Random, n: int = 4
) -> List[str]:
    pool = [
        (cid, upright_vn_prose(row))
        for cid, row in vn_by_id.items()
        if cid != exclude_id and upright_vn_prose(row)
    ]
    rng.shuffle(pool)
    return [prose for _, prose in pool[:n]]


# ------------------------------------------------------------------ W2.1 ----


def run_w21(inputs: Dict[str, Any], epsilon: float) -> Dict[str, Any]:
    print("== W2.1 orientation attribution (deterministic polarity-lexicon proxy)")
    vn_by_id = inputs["vn_by_id"]
    attributions: Dict[int, dict] = {}
    for cid in sorted(vn_by_id):
        vn = vn_by_id[cid]
        if vn.get("vi_provenance") != "source":
            attributions[cid] = {
                "card_id": cid,
                "name_en": vn["name_en"],
                "attribution": "vi_orientation_agnostic",
                "reason": "synthetic placeholder — no source prose to attribute",
            }
            continue
        text = " ".join((vn.get(k) or "") for k in IDENTICAL_FIELDS)
        res = polarity_lexicon_attribution(text, epsilon=epsilon)
        # Secondary anchor: the orientation-specific field should read neutral
        # (it describes card identity, not orientation) — if it skews strongly,
        # note it but keep the primary verdict from the identical fields.
        sec = polarity_lexicon_attribution(vn.get("title_secondary") or "", epsilon=epsilon)
        attributions[cid] = {
            "card_id": cid,
            "name_en": vn["name_en"],
            "attribution": res["attribution"],
            "score": res["score"],
            "hits": res["hits"],
            "upright_hits": res["upright_hits"],
            "reversed_hits": res["reversed_hits"],
            "epsilon": epsilon,
            "title_secondary_attribution": sec["attribution"],
        }
    agnostic = [c for c, a in attributions.items() if a["attribution"] == "vi_orientation_agnostic"]
    upright = [c for c, a in attributions.items() if a["attribution"] == "vi_upright"]
    skew = [c for c, a in attributions.items() if a["attribution"] == "vi_reversed_skew"]
    report = {
        "method": "polarity_lexicon_proxy (no /embeddings on endpoint; plan W2.1 fallback)",
        "epsilon": epsilon,
        "fields_used": list(IDENTICAL_FIELDS),
        "attributions": [attributions[c] for c in sorted(attributions)],
        "summary": {
            "vi_orientation_agnostic": len(agnostic),
            "vi_upright": len(upright),
            "vi_reversed_skew": len(skew),
            "synthesis_scope": "both_orientations"
            if agnostic
            else "reversed_only",
        },
    }
    out = KB / "vn_orientation_attribution.json"
    out.write_text(dumps_canonical(report) + "\n", encoding="utf-8")
    print(f"  wrote {out}: agnostic={len(agnostic)} upright={len(upright)} skew={len(skew)}")
    return {a["card_id"]: a for a in report["attributions"]}


# ------------------------------------------------------------------ W2.2 ----


def gate_one_variant(
    client: Optional[LLMClient],
    spine_row: Dict[str, Any],
    keywords_vi: List[str],
    prose: str,
    profile: Dict[str, Any],
    jaccard_threshold: float,
    upright_prose: str,
    use_rubric: bool,
    max_profile_distance: float = 2.5,
) -> Dict[str, Any]:
    g1 = check_vietnamese_ness(prose, profile, max_distance=max_profile_distance)
    jac = jaccard(prose, upright_prose) if upright_prose else 0.0
    g2 = {"pass": jac <= jaccard_threshold, "jaccard": round(jac, 4), "threshold": jaccard_threshold}
    g4 = {"pass": not forbidden_claims(prose), "matched": forbidden_claims(prose)}
    containment = keyword_containment(prose, keywords_vi)
    name_inline = card_name_inline(prose, spine_row["name_en"])

    g3: Dict[str, Any] = {
        "pass": False,
        "atoms_covered": 0,
        "atoms_total": len(spine_row.get("keyword_atoms_en") or []),
        "name_inline": name_inline,
        "keyword_containment": round(containment, 4),
        "recall": 0.0,
        "reason": "rubric skipped",
    }
    if use_rubric and client is not None:
        verdict: Optional[Dict[str, Any]] = None
        for budget in (700, 1400):
            try:
                verdict = client.chat_json(
                    build_rubric_messages(spine_row, prose, keywords_vi),
                    max_tokens=budget,
                    temperature=0.0,
                )
                break
            except LLMError:
                verdict = None
        if verdict is not None:
            covered = int(verdict.get("atoms_covered") or 0)
            total = int(verdict.get("atoms_total") or len(spine_row.get("keyword_atoms_en") or []))
            recall = covered / total if total else 0.0
            name_inline = bool(verdict.get("name_inline", name_inline))
            decision = str(verdict.get("decision", "fail")).lower() == "pass"
            g3 = {
                "pass": decision and recall >= 0.7 and name_inline,
                "atoms_covered": covered,
                "atoms_total": total,
                "name_inline": name_inline,
                "keyword_containment": round(containment, 4),
                "recall": round(recall, 4),
                "reason": str(verdict.get("reason") or "")[:200],
            }
        else:
            g3["reason"] = "rubric call failed twice"

    overall = g1["pass"] and g2["pass"] and g3["pass"] and g4["pass"]
    return {"pass": overall, "g1": g1, "g2": g2, "g3": g3, "g4": g4}


def generate_variant(
    client: Optional[LLMClient],
    spine_row: Dict[str, Any],
    exemplars: List[str],
    variant: int,
    temperature: float,
    attempt: int = 1,
) -> Optional[Dict[str, Any]]:
    if client is None:
        return None
    messages = build_generation_messages(spine_row, exemplars, variant=variant, attempt=attempt)
    data: Optional[Dict[str, Any]] = None
    for budget in (1600, 2800):  # reasoning models eat tokens; retry with headroom
        try:
            data = client.chat_json(messages, max_tokens=budget, temperature=temperature)
            break
        except LLMError:
            data = None
    if data is None:
        return None
    prose = str(data.get("prose") or "").strip()
    kws = [str(k).strip() for k in (data.get("keywords_vi") or []) if str(k).strip()]
    if not prose:
        return None
    return {"keywords_vi": kws, "prose": prose}


def run_w22(
    client: Optional[LLMClient],
    inputs: Dict[str, Any],
    attributions: Dict[int, dict],
    card_ids: Sequence[int],
    *,
    variants: int,
    neg_control: int,
    dry_run: bool,
    seed: int,
    temp_hi: float,
    temp_lo: float,
    max_retries: int = 3,
) -> Dict[str, Any]:
    print(f"== W2.2 reversed-meaning synthesis ({len(card_ids)} cards, {variants} variants/card)")
    rng = random.Random(seed)
    rev_rows = [r for r in inputs["rev_rows"] if r["card_id"] in card_ids]
    vn_by_id = inputs["vn_by_id"]
    profile = inputs["profile"]

    # G2 threshold: 90th percentile of authentic-pair Jaccard from the raw
    # phatjkk source (plan W2.2, H2). Fallback if the source is absent.
    if DATA_TXT.exists():
        dist = authentic_pair_jaccard_distribution(DATA_TXT, percentile=0.90)
        jac_threshold = dist["threshold"] if dist["threshold"] is not None else 0.55
        print(f"  authentic-pair Jaccard: n={dist['count']} "
              f"p90={dist['threshold']} max={dist.get('max')}")
    else:
        dist = {"count": 0, "threshold": None}
        jac_threshold = 0.55
        print(f"  {DATA_TXT} missing — G2 threshold fallback 0.55 (documented)")

    # G1 threshold calibration: the profile gate must be set to the AUTHENTIC
    # phatjkk register band, not an arbitrary constant. p90 of the authentic
    # distribution keeps the gate strict but lets the target register through.
    authentic_dists = authentic_profile_distances(list(vn_by_id.values()), profile)
    max_profile_distance = percentile_90(authentic_dists) if authentic_dists else 2.5
    print(f"  G1 calibrated: authentic n={len(authentic_dists)} p90={max_profile_distance:.3f}")

    per_card: Dict[str, Any] = {}
    for row in rev_rows:
        cid = row["card_id"]
        vn_row = vn_by_id.get(cid)
        upright_prose = upright_vn_prose(vn_row)
        ex = exemplars_for(vn_by_id, cid, rng)
        variants_out: List[Dict[str, Any]] = []
        selected: Optional[Dict[str, Any]] = None
        # Each attempt draws FRESH content (the prompt carries the attempt
        # number, so the prompt-hash cache misses) — a card that marginally
        # fails G2/G3 on one draw usually passes the next. Attempt rounds stop
        # early once a variant passes.
        for attempt in range(1, max_retries + 1):
            for v in range(1, variants + 1):
                temperature = (temp_hi + 0.1 * (attempt - 1)) if v % 2 == 1 else (
                    temp_lo + 0.1 * (attempt - 1)
                )
                gen = None if dry_run else generate_variant(
                    client, row, ex, v, temperature, attempt=attempt
                )
                if gen is None:
                    variants_out.append(
                        {
                            "variant": v,
                            "attempt": attempt,
                            "status": "pending" if dry_run else "generation_failed",
                        }
                    )
                    continue
                gate = gate_one_variant(
                    client,
                    row,
                    gen["keywords_vi"],
                    gen["prose"],
                    profile,
                    jac_threshold,
                    upright_prose,
                    use_rubric=not dry_run,
                    max_profile_distance=max_profile_distance,
                )
                variants_out.append(
                    {
                        "variant": v,
                        "attempt": attempt,
                        "temperature": round(temperature, 3),
                        "status": "pass" if gate["pass"] else "rejected",
                        "keywords_vi": gen["keywords_vi"],
                        "prose": gen["prose"],
                        "gate": gate,
                    }
                )
                if gate["pass"] and selected is None:
                    selected = variants_out[-1]
            if selected is not None:
                break
        if selected is not None:
            provenance = "synthetic"
        elif dry_run and all(v.get("status") == "pending" for v in variants_out):
            provenance = "synthetic_pending"
        else:
            provenance = "synthetic_failed_gate" if variants_out else "synthetic_pending"
        per_card[cid] = {
            "card_id": cid,
            "name_en": row["name_en"],
            "attribution": attributions.get(cid, {}).get("attribution"),
            "variants": variants_out,
            "selected_variant": selected["variant"] if selected else None,
            "attempts_used": max((v.get("attempt") or 1) for v in variants_out),
            "vi_provenance": provenance,
            "prose": selected["prose"] if selected else "",
            "keywords_vi": selected["keywords_vi"] if selected else [],
        }

    # Negative control: generate from a DIFFERENT card's spine, run the gate for
    # the target card — the gate must REJECT (plan: rejection floor >= 80%).
    neg_results: List[Dict[str, Any]] = []
    if client is not None and not dry_run and neg_control > 0:
        print(f"  negative control: {neg_control} wrong-card spine probes")
        sample = rng.sample(rev_rows, min(neg_control, len(rev_rows)))
        for target in sample:
            wrong = rng.choice([r for r in rev_rows if r["card_id"] != target["card_id"]])
            ex = exemplars_for(vn_by_id, wrong["card_id"], rng)
            gen = generate_variant(client, wrong, ex, variant=1, temperature=temp_hi)
            if gen is None:
                continue
            gate = gate_one_variant(
                client,
                target,
                gen["keywords_vi"],
                gen["prose"],
                profile,
                jac_threshold,
                upright_vn_prose(vn_by_id.get(target["card_id"])),
                use_rubric=True,
                max_profile_distance=max_profile_distance,
            )
            neg_results.append(
                {
                    "target": target["name_en"],
                    "spine_used": wrong["name_en"],
                    "rejected": not gate["pass"],
                    "gate": gate,
                }
            )
    neg_rate = (
        sum(1 for r in neg_results if r["rejected"]) / len(neg_results)
        if neg_results
        else None
    )

    # vn_spine.jsonl — 156 rows: upright populated from source (or placeholder),
    # reversed populated from synthesis (or pending/failed with reasons).
    vn_spine: List[Dict[str, Any]] = []
    for up in inputs["up_rows"]:
        vn = vn_by_id.get(up["card_id"])
        vn_spine.append(
            {
                "card_id": up["card_id"],
                "name_en": up["name_en"],
                "orientation": "upright",
                "vi_prose": upright_vn_prose(vn),
                "vi_upright_fields": {
                    k: (vn.get(k) or "") for k in IDENTICAL_FIELDS
                } if vn else {},
                "vi_provenance": vn.get("vi_provenance", "source") if vn else "source",
                "vi_orientation_attribution": attributions.get(up["card_id"], {}).get("attribution"),
            }
        )
    for cid in sorted(per_card):
        pc = per_card[cid]
        vn_spine.append(
            {
                "card_id": cid,
                "name_en": pc["name_en"],
                "orientation": "reversed",
                "vi_reversed_prose": pc["prose"],
                "vi_keywords_reversed": pc["keywords_vi"],
                "vi_provenance": pc["vi_provenance"],
                "vi_orientation_attribution": pc.get("attribution"),
                "gate_summary": {
                    "selected_variant": pc["selected_variant"],
                    "n_variants": len(pc["variants"]),
                },
            }
        )
    vn_spine.sort(key=lambda r: (r["card_id"], 0 if r["orientation"] == "upright" else 1))
    write_jsonl(KB / "vn_spine.jsonl", vn_spine)

    report = {
        "model": client.model if client else "dry-run",
        "seed": seed,
        "temperatures": {"hi": temp_hi, "lo": temp_lo},
        "max_retries": max_retries,
        "jaccard_threshold": jac_threshold,
        "authentic_pair_distribution": dist,
        "g1_calibration": {
            "authentic_profile_n": len(authentic_dists),
            "max_profile_distance": max_profile_distance,
            "method": "p90 of authentic phatjkk profile distances",
        },
        "per_card": per_card,
        "negative_control": neg_results,
        "negative_control_rejection_rate": round(neg_rate, 4) if neg_rate is not None else None,
        "negative_control_floor": 0.8,
        "aggregate": {
            "cards_processed": len(per_card),
            "synthetic": sum(1 for p in per_card.values() if p["vi_provenance"] == "synthetic"),
            "failed_gate": sum(
                1 for p in per_card.values() if p["vi_provenance"] == "synthetic_failed_gate"
            ),
            "pending": sum(1 for p in per_card.values() if p["vi_provenance"] == "synthetic_pending"),
        },
    }
    (KB / "w2_2_gate_report.json").write_text(dumps_canonical(report) + "\n", encoding="utf-8")
    print(f"  wrote kb/vn_spine.jsonl ({len(vn_spine)} rows) + kb/w2_2_gate_report.json")
    print(f"  aggregate: {report['aggregate']}")
    if neg_rate is not None:
        print(f"  negative-control rejection rate: {neg_rate:.2%} (floor 80%)")
    return {"vn_spine": vn_spine, "report": report}


# ------------------------------------------------------------------ W2.3 ----


def run_w23(
    inputs: Dict[str, Any],
    attributions: Dict[int, dict],
    vn_spine: List[Dict[str, Any]],
    *,
    allow_incomplete: bool,
) -> int:
    print("== W2.3 frozen 156-row bilingual KB")
    rev_by_id = {r["card_id"]: r for r in vn_spine if r["orientation"] == "reversed"}
    up_by_id = {r["card_id"]: r for r in vn_spine if r["orientation"] == "upright"}
    vn_by_id = inputs["vn_by_id"]
    by_key = {(r["card_id"], r["orientation"]): r for r in inputs["spine"]}

    rows: List[Dict[str, Any]] = []
    for (cid, orientation) in sorted(by_key):
        s = by_key[(cid, orientation)]
        vn = vn_by_id.get(cid)
        vi_row = up_by_id.get(cid) if orientation == "upright" else rev_by_id.get(cid)
        if orientation == "upright":
            meaning_vi = (vi_row or {}).get("vi_prose") or ""
            keywords_vi: List[str] = []
            provenance = (vi_row or {}).get("vi_provenance") or (
                vn.get("vi_provenance") if vn else "source"
            )
            source_ids = (vn or {}).get("source_ids") or []
        else:
            meaning_vi = (vi_row or {}).get("vi_reversed_prose") or ""
            keywords_vi = (vi_row or {}).get("vi_keywords_reversed") or []
            provenance = (vi_row or {}).get("vi_provenance") or "synthetic_pending"
            source_ids = []
        rows.append(
            {
                "card_id": cid,
                "name_en": s["name_en"],
                "name_vi_gloss": "",
                "arcana": s.get("arcana"),
                "suit": s.get("suit"),
                "rank": None,
                "number": s.get("number"),
                "element": s.get("element"),
                "planet": s.get("planet"),
                "zodiac": s.get("zodiac"),
                "orientation": orientation,
                "polarity_axis": s.get("polarity_axis"),
                "keywords_en": s.get("keyword_atoms_en"),
                "keywords_vi": keywords_vi,
                "meaning_en": s.get("meaning_summary_en"),
                "meaning_vi": meaning_vi,
                "domain_vi": (
                    {k: (vn.get(k) or "") for k in IDENTICAL_FIELDS}
                    if orientation == "upright" and vn
                    else {}
                ),
                "forbidden_claims": [],
                "yes_no": None,
                "source_ids": source_ids,
                "image_path": None,
                "vi_provenance": provenance,
                "vi_orientation_attribution": attributions.get(cid, {}).get("attribution"),
            }
        )

    # --- assertion suite (plan W2.3) ---
    rows.sort(key=lambda r: (r["card_id"], 0 if r["orientation"] == "upright" else 1))

    # --- assertion suite (plan W2.3) ---
    errors: List[str] = []
    if len(rows) != 156:
        errors.append(f"expected 156 rows, got {len(rows)}")
    for cid in range(78):
        for o in ("upright", "reversed"):
            if not any(r["card_id"] == cid and r["orientation"] == o for r in rows):
                errors.append(f"missing ({cid}, {o})")
    toc = next(
        (r for r in rows if r["name_en"] == "Two of Cups" and r["orientation"] == "reversed"), None
    )
    from tfvn.spine import NAME_TO_ID  # noqa: E402

    fop_id = NAME_TO_ID["Four of Pentacles"]
    fop = next((r for r in rows if r["card_id"] == fop_id and r["orientation"] == "reversed"), None)
    page = next(
        (r for r in rows if r["name_en"] == "Page of Pentacles" and r["orientation"] == "upright"),
        None,
    )
    if toc is None or (toc.get("meaning_en") or "").find("Mathers") == -1:
        errors.append("Two of Cups reversed must carry Mathers provenance in meaning_en")
    if fop is None or "Waite" not in (fop.get("meaning_en") or ""):
        errors.append("Four of Pentacles reversed provenance must be waite")
    if page is None or page.get("vi_provenance") != "synthetic_no_anchor":
        errors.append("Page of Pentacles vi_provenance must be synthetic_no_anchor")
    blob = dumps_canonical(rows)
    if "title_heath" in blob:
        errors.append("title_heath found (must be title_health)")
    if "sức khoẻ" in blob:
        errors.append("sức khoẻ found (must be sức khỏe)")

    incomplete = [r["name_en"] for r in rows if r["orientation"] == "reversed" and r["vi_provenance"] != "synthetic"]
    if incomplete and not allow_incomplete:
        errors.append(
            f"{len(incomplete)} reversed rows lack synthetic prose "
            f"({incomplete[:5]}...) — run a full W2.2 first or pass --allow-incomplete"
        )

    if errors:
        print("  ASSERTION FAILURES:")
        for e in errors:
            print("   -", e)
        return 1

    write_jsonl(KB / "cards.jsonl", rows)
    digest = hashlib.sha256(dumps_canonical(rows).encode("utf-8")).hexdigest()
    (KB / "CARDS_HASH.txt").write_text(digest + "\n", encoding="utf-8")
    print(f"  wrote kb/cards.jsonl ({len(rows)} rows)")
    print(f"  wrote kb/CARDS_HASH.txt ({digest})")
    return 0


# ------------------------------------------------------------------- main ---


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("w21", "w22", "w23"), help="run a single task")
    ap.add_argument("--limit", type=int, help="process only the first N cards")
    ap.add_argument("--cards", help="comma-separated card ids (e.g. 0,1,2)")
    ap.add_argument("--variants", type=int, default=2, help="variants per card (default 2)")
    ap.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="retry rounds for cards with no passing variant (default 3; each "
        "round redraws fresh content via a cache-missing prompt)",
    )
    ap.add_argument("--neg-control", type=int, default=20, help="negative-control probes (default 20)")
    ap.add_argument("--no-neg-control", action="store_true", help="skip the negative control")
    ap.add_argument("--dry-run", action="store_true", help="no API calls; reversed rows marked pending")
    ap.add_argument("--allow-incomplete", action="store_true", help="write cards.jsonl with non-synthetic reversed rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cache", action="store_true", help="bypass the prompt-hash cache")
    args = ap.parse_args()

    load_env()
    KB.mkdir(parents=True, exist_ok=True)

    # Build the client (None in --dry-run / missing key => offline mode).
    client: Optional[LLMClient]
    if args.dry_run:
        client = None
        print("DRY-RUN: no API calls will be made")
    else:
        client = LLMClient(seed=args.seed)
        if args.no_cache:
            client.cache_dir = Path("/dev/null")  # cache reads become misses
        try:
            models = client.available_models()
            print(f"connected to {client.base_url} — models: {models}")
            if client.model not in models:
                print(f"  WARNING: {client.model!r} not advertised; trying anyway")
        except LLMError as e:
            print(f"connection check failed: {e}", file=sys.stderr)
            return 2

    inputs = load_inputs()
    only = args.only

    card_ids: Optional[List[int]] = None
    if args.cards:
        card_ids = [int(x) for x in args.cards.split(",") if x.strip()]
    elif args.limit:
        card_ids = [r["card_id"] for r in inputs["rev_rows"][: args.limit]]

    # W2.1 always runs first (W2.2/W2.3 consume attributions).
    if only in (None, "w21"):
        attributions = run_w21(inputs, epsilon=0.15)
    else:
        attr_path = KB / "vn_orientation_attribution.json"
        if attr_path.exists():
            attributions = {
                a["card_id"]: a
                for a in json.loads(attr_path.read_text(encoding="utf-8"))["attributions"]
            }
        else:
            print("kb/vn_orientation_attribution.json missing — run W2.1 first", file=sys.stderr)
            return 1

    if only in (None, "w22"):
        targets = card_ids or [r["card_id"] for r in inputs["rev_rows"]]
        out22 = run_w22(
            client,
            inputs,
            attributions,
            targets,
            variants=args.variants,
            neg_control=0 if args.no_neg_control else args.neg_control,
            dry_run=args.dry_run,
            seed=args.seed,
            temp_hi=1.0,
            temp_lo=0.7,
            max_retries=args.max_retries,
        )
        vn_spine = out22["vn_spine"]
    else:
        vn_spine = read_jsonl(KB / "vn_spine.jsonl") if (KB / "vn_spine.jsonl").exists() else []
        if not vn_spine:
            print("kb/vn_spine.jsonl missing — run W2.2 first", file=sys.stderr)
            return 1

    if only in (None, "w23"):
        rc = run_w23(inputs, attributions, vn_spine, allow_incomplete=args.allow_incomplete)
        if rc:
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
