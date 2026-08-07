#!/usr/bin/env python3
"""W3.1 — Mine the Dendory corpus for STRUCTURAL distributions only.

Reads ``data/hf/Dendory__tarot/tarot_readings.csv`` (5,769 ChatGPT readings,
3-card draws, zero orientation encoding, English) and produces
``kb/dendory_structural_profile.json`` containing a CLOSED, machine-extractable
field list:

  - card-count distribution (how many cards per draw)
  - per-card sampling weights (card frequency in the corpus)
  - reading length-band distribution (word counts -> ngắn / đầy_đủ bands)
  - turn count (all corpus rows are single-turn)
  - question topic taxonomy (finite set) with Vietnamese labels
  - Vietnamese question-seed surface forms (inputs, NOT training targets)

Explicitly DO NOT extract paragraph templates, transition scaffolds, or
discourse structure (plan W3.1 / H5: importing English paragraph rhetoric as
Vietnamese calques is forbidden). The QA gate below greps the output for the
known Dendory prose scaffolds and fails if any leak through.

Usage:
  python scripts/build_wave3_w31.py [--csv PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.serialise import dumps_canonical  # noqa: E402

# Known Dendory paragraph-level scaffolds — MUST NOT appear in the profile.
DENDRORY_PROSE_SCAFFOLDS = [
    "Your tarot reading speaks of",
    "Your tarot reading is pointing to",
    "Your tarot reading suggests that",
    "The card signifies",
    "This card brings with it",
    "Based on the cards drawn",
    "is a reminder that you have",
    "With courage and determination",
]

# The corpus prompt pattern (make_dataset.py) — every row shares this intent.
CORPUS_PROMPT_PATTERN = (
    "Give me a one paragraph tarot reading if I pull the cards {}, {} and {}."
)

# W3.2's querent-context axis is sourced from the safety policy + W2 domain
# fields (the Dendory corpus itself is single-topic). This is the closed,
# finite topic set used for sampling weights — recorded here so the profile is
# the single source of truth.
TOPIC_TAXONOMY = [
    {"topic_id": "love", "label_en": "love & relationships", "label_vi": "tình yêu và các mối quan hệ"},
    {"topic_id": "career", "label_en": "career & work", "label_vi": "sự nghiệp và công việc"},
    {"topic_id": "money", "label_en": "money & finances", "label_vi": "tiền bạc và tài chính"},
    {"topic_id": "health", "label_en": "health & wellbeing", "label_vi": "sức khỏe và tinh thần"},
    {"topic_id": "spiritual", "label_en": "spiritual growth", "label_vi": "phát triển tâm linh"},
    {"topic_id": "decision", "label_en": "decision & path", "label_vi": "quyết định và hướng đi"},
]

# Natural Vietnamese renderings of the corpus question INTENT (written natively
# in Vietnamese, never translated from English prose). Used as input seeds.
VI_QUESTION_SEEDS = [
    "Xin hãy trải bài Tarot cho tôi với các lá {c1}, {c2} và {c3}.",
    "Hãy đọc cho tôi ý nghĩa khi rút được {c1}, {c2} và {c3}.",
    "Tôi vừa rút ba lá {c1}, {c2}, {c3} — bạn giải thích giúp tôi được không?",
    "Cho tôi một bài đọc với ba lá {c1}, {c2} và {c3} nhé.",
    "Giúp tôi hiểu bài trải này: {c1}, {c2}, {c3}.",
]


def read_corpus(csv_path: Path) -> List[Dict[str, str]]:
    """Read the CSV (column headers carry a leading space: ' Card 2' etc.)."""
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalise header names (strip whitespace) so DictReader keys are clean.
        reader.fieldnames = [h.strip() for h in reader.fieldnames or []]
        for raw in reader:
            cards = [
                (raw.get("Card 1") or "").strip(),
                (raw.get("Card 2") or "").strip(),
                (raw.get("Card 3") or "").strip(),
            ]
            reading = (raw.get("Reading") or "").strip()
            rows.append({"cards": [c for c in cards if c], "reading": reading})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=ROOT / "data/hf/Dendory__tarot/tarot_readings.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "kb/dendory_structural_profile.json")
    args = ap.parse_args()

    rows = read_corpus(args.csv)
    print(f"corpus rows: {len(rows)}")

    # --- card-count distribution ---
    card_counts = Counter(len(r["cards"]) for r in rows)
    n_readings = len(rows)

    # --- per-card sampling weights (card frequency) ---
    card_freq: Counter[str] = Counter()
    for r in rows:
        card_freq.update(r["cards"])
    card_weights = {
        name: round(freq / n_readings, 5) for name, freq in card_freq.most_common()
    }

    # --- length-band distribution (word counts of the reading) ---
    lengths = [len(r["reading"].split()) for r in rows if r["reading"]]
    lengths_sorted = sorted(lengths)
    p25 = lengths_sorted[int(0.25 * len(lengths_sorted))]
    p50 = lengths_sorted[int(0.50 * len(lengths_sorted))]
    p75 = lengths_sorted[int(0.75 * len(lengths_sorted))]
    p90 = lengths_sorted[int(0.90 * len(lengths_sorted))]

    def band_of(n: int) -> str:
        return "ngắn" if n <= p50 else "đầy_đủ"

    band_dist = Counter(band_of(n) for n in lengths)
    length_bands = {
        "ngắn": {"max_words": p50, "share": round(band_dist["ngắn"] / len(lengths), 4)}
        if lengths
        else 0.0,
        "đầy_đủ": {"min_words": p50 + 1, "share": round(band_dist["đầy_đủ"] / len(lengths), 4)}
        if lengths
        else 0.0,
    }

    # --- turn count (all corpus rows are single-turn: prompt -> reading) ---
    turn_count = {"distribution": {"1": 1.0}, "note": "corpus is single-turn by construction"}

    # --- topic taxonomy (single-topic corpus; axis sourced from policy) ---
    topic_dist = {
        t["topic_id"]: {"weight": round(1 / len(TOPIC_TAXONOMY), 5)}
        for t in TOPIC_TAXONOMY
    }

    profile: Dict[str, Any] = {
        "schema": "dendory_structural_profile",
        "schema_version": "1.0",
        "source": {
            "path": str(args.csv),
            "rows": n_readings,
            "prompt_pattern": CORPUS_PROMPT_PATTERN,
            "language": "en",
            "orientation_encoding": "none",
            "note": "H5: structure-only mining; no prose templates or discourse scaffolds extracted.",
        },
        "card_count_distribution": {
            str(k): v for k, v in sorted(card_counts.items())
        },
        "spread_id": "spread_three",
        "spread_note": "all corpus rows are flat 3-card draws; no position semantics in the source",
        "card_sampling_weights": card_weights,
        "length_band_distribution": length_bands,
        "length_stats": {
            "mean_words": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
        },
        "turn_count": turn_count,
        "topic_taxonomy": TOPIC_TAXONOMY,
        "topic_distribution": topic_dist,
        "topic_note": (
            "Dendory questions share one intent (3-card reading request); the "
            "finite topic axis above is sourced from the W0.5 safety policy + W2 "
            "domain fields and drives W3.2 sampling weights."
        ),
        "vi_question_seeds": VI_QUESTION_SEEDS,
        "qa": {
            "prose_scaffold_check": "grep for Dendory paragraph templates below",
            "scaffold_matches": [],
        },
    }

    # --- QA gate: no prose templates may leak through (plan W3.1 QA) ---
    blob = dumps_canonical(profile)
    leaks = [s for s in DENDRORY_PROSE_SCAFFOLDS if s.lower() in blob.lower()]
    if leaks:
        print("  QA FAIL — prose scaffolds leaked into the profile:")
        for s in leaks:
            print("   -", s)
        return 1
    profile["qa"]["scaffold_matches"] = leaks

    args.out.write_text(dumps_canonical(profile) + "\n", encoding="utf-8")
    print(f"  wrote {args.out}")
    print(f"  card-count distribution: {dict(sorted(card_counts.items()))}")
    print(f"  distinct cards: {len(card_weights)}")
    print(f"  length p50={p50} p90={p90} words; bands: {length_bands}")
    print("  QA: no prose scaffolds leaked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
