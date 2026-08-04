#!/usr/bin/env python3
"""Build all Wave 1 KB artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.aliases import alias_table_for_export, assert_alias_table_total_injective  # noqa: E402
from tfvn.register_profile import write_register_profile  # noqa: E402
from tfvn.serialise import (  # noqa: E402
    build_card_name_whitelist,
    build_compact_cards,
    dumps_canonical,
    mean_compact_tokens,
    read_jsonl,
    serialise_spine_document,
    try_load_qwen_tokenizer,
    write_jsonl,
)
from tfvn.spine import write_spine  # noqa: E402
from tfvn.spreads import write_spreads  # noqa: E402
from tfvn.vn_upright import write_vn_upright  # noqa: E402


def main() -> int:
    kb = ROOT / "kb"
    kb.mkdir(parents=True, exist_ok=True)

    print("Building english_spine.jsonl ...")
    spine_path = write_spine(out_path=kb / "english_spine.jsonl")
    print("  wrote", spine_path)

    print("Building vn_upright.jsonl ...")
    vn_path = write_vn_upright(out_path=kb / "vn_upright.jsonl")
    print("  wrote", vn_path)

    print("Building spreads.jsonl ...")
    sp_path, report = write_spreads(
        out_path=kb / "spreads.jsonl",
        report_path=kb / "spreads_discrimination_report.json",
    )
    print(
        "  wrote",
        sp_path,
        f"above_chance={report['spreads_above_chance']}/21",
        f"failing={report['failing_spreads']}",
    )

    print("Building vn_register_profile.json ...")
    reg_path = write_register_profile(out_path=kb / "vn_register_profile.json")
    print("  wrote", reg_path)

    print("Building compact_cards + whitelist ...")
    spine = read_jsonl(kb / "english_spine.jsonl")
    # Dual-process identity is tested separately; write once here
    doc_a = serialise_spine_document(spine)
    doc_b = serialise_spine_document(spine)
    assert doc_a == doc_b, "serialiser nondeterminism"
    (kb / "english_spine.canonical.json").write_bytes(doc_a + b"\n")

    compact = build_compact_cards(spine)
    write_jsonl(kb / "compact_cards.jsonl", compact)
    tok = try_load_qwen_tokenizer()
    mean_tok = mean_compact_tokens(compact, tokenizer=tok)
    method = "qwen" if tok else "proxy"
    print(f"  compact mean tokens={mean_tok:.2f} ({method})")
    if mean_tok > 65:
        # tighten meanings further
        for row in compact:
            m = row["meaning_summary_en"]
            if len(m) > 120:
                row["meaning_summary_en"] = m[:117].rstrip() + "..."
            row["keywords_en"] = (row.get("keywords_en") or [])[:4]
        write_jsonl(kb / "compact_cards.jsonl", compact)
        mean_tok = mean_compact_tokens(compact, tokenizer=tok)
        print(f"  tightened mean tokens={mean_tok:.2f}")
    assert mean_tok <= 65, mean_tok

    wl = build_card_name_whitelist()
    (kb / "card_name_whitelist.json").write_text(dumps_canonical(wl) + "\n", encoding="utf-8")
    assert_alias_table_total_injective()
    print(f"  whitelist entries={wl['entry_count']}")

    # alias table export
    (kb / "alias_table.json").write_text(
        dumps_canonical(alias_table_for_export()) + "\n", encoding="utf-8"
    )

    print("Wave 1 KB build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
