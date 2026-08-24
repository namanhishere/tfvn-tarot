#!/usr/bin/env python3
"""W5.2: assemble the Vietnamese imatrix calibration corpus.

Composition (plan H14):
  ~70% Vietnamese long-form tarot-register prose
       (authentic corpus prose + held-out generated readings, NOT the training set)
  ~20% English card names inline in Vietnamese sentences (+ English spine keywords)
  ~10% structural (system prompt template, spread position labels)

Output: plain-text chunks ready for `llama-imatrix -c 4096`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.serialise import read_jsonl  # noqa: E402

CHUNK_TOKENS = 4096


def vn_prose_pool(kb_rows, datasets_dir: Path) -> list:
    """Authentic KB domain prose + held-out anchor readings (never train rows)."""
    pool = []
    for r in kb_rows:
        dom = r.get("domain_vi") or {}
        pool += [t.strip() for t in dom.values() if isinstance(t, str) and len(t.split()) > 15]
    anchor = datasets_dir / "anchor/anchor_readings.jsonl"
    if anchor.exists():
        for r in read_jsonl(anchor):
            t = r.get("reading_vi") or r.get("target_vi") or ""
            if len(t.split()) > 30:
                pool.append(t.strip())
    return pool


def en_inline_sentences(cards, rng) -> list:
    templates = [
        "Trong trải bài này, lá {en} xuất hiện ở vị trí quan trọng nhất.",
        "Bạn rút được lá {en}; ý nghĩa của nó sẽ nói về giai đoạn tới.",
        "Khi lá {en} xuất hiện đảo ngược, thông điệp thay đổi đáng kể.",
        "The {suit} suit nói về những vấn đề liên quan tới đời sống hằng ngày, và lá {en} không ngoại lệ.",
    ]
    out = []
    for c in cards:
        out.append(rng.choice(templates).format(en=c["name_en"], suit=c.get("suit") or "tarot"))
    return out


STRUCTURAL = (
    "Bạn là chuyên gia đọc bài Tarot bằng tiếng Việt tự nhiên.\n"
    "CÁC VỊ TRÍ: quá khứ, hiện tại, tương lai\n"
    "TRẢ BÀI Celtic Cross. VỊ TRÍ 1: hiện tại. VỊ TRÍ 2: thách thức. "
    "VỊ TRÍ 3: quá khứ. VỊ TRÍ 4: tương lai.\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=str(ROOT / "kb/cards.jsonl"))
    ap.add_argument("--datasets-dir", default=str(ROOT / "datasets"))
    ap.add_argument("--chunks", type=int, default=300)
    ap.add_argument("--chunk-chars", type=int, default=12000,
                    help="~4096 tokens at ~3 chars/token Vietnamese")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "artifacts/imatrix_corpus.txt"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kb_rows = read_jsonl(Path(args.kb))
    cards = [{"name_en": r["name_en"], "suit": r.get("suit")}
             for r in kb_rows if r.get("orientation") == "upright"]

    vn = vn_prose_pool(kb_rows, Path(args.datasets_dir))
    en = en_inline_sentences(cards, rng)
    struct = STRUCTURAL.split("\n")
    rng.shuffle(vn)
    assert vn, "no Vietnamese prose found"

    parts = []  # (weight, source_list)
    parts.append((0.70, vn))
    parts.append((0.20, en))
    parts.append((0.10, struct * max(1, len(vn) // 10)))

    lines = []
    targets = {"vn": int(args.chunks * 0.7), "en": int(args.chunks * 0.2),
               "struct": args.chunks - int(args.chunks * 0.7) - int(args.chunks * 0.2)}
    made = {"vn": 0, "en": 0, "struct": 0}
    while sum(made.values()) < args.chunks:
        for kind, target in (("vn", targets["vn"]), ("en", targets["en"]),
                             ("struct", targets["struct"])):
            if made[kind] >= target:
                continue
            src = {"vn": vn, "en": en, "struct": struct}[kind]
            buf, size = [], 0
            while size < args.chunk_chars and src:
                piece = rng.choice(src)
                buf.append(piece)
                size += len(piece) + 1
            lines.append("<chunk kind=%s id=%d>\n%s\n</chunk>" %
                         (kind, made[kind], "\n".join(buf)))
            made[kind] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(lines), encoding="utf-8")
    total_bytes = out.stat().st_size
    meta = {"chunks": len(lines), "composition_targets": targets,
            "approx_tokens": total_bytes // 3, "bytes": total_bytes}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
