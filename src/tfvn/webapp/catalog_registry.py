"""Static artifact registry for the webapp catalog (todo A.2 helper).

One entry per tracked artifact in ``kb/`` and ``datasets/``; the catalog
module (``catalog.py``) inspects each against the live filesystem. Keeping
the table here keeps ``catalog.py`` under the 250-LOC ceiling.

``kb/README.md`` is deliberately omitted (documentation, not data).
``datasets/raw/*`` is gitignored — entries are still listed so the catalog
shows them present-if-exists rather than silently hiding regenerable files.
"""

# (id, path, kind, tag); kind ∈ {jsonl,json,txt,hash}, tag ∈ {kb,dataset,anchor,raw,report,hash}
REGISTRY: list[dict[str, str]] = [
    {"id": "kb/cards.jsonl", "path": "kb/cards.jsonl", "kind": "jsonl", "tag": "kb"},
    {"id": "kb/english_spine.jsonl", "path": "kb/english_spine.jsonl", "kind": "jsonl", "tag": "kb"},
    {"id": "kb/vn_spine.jsonl", "path": "kb/vn_spine.jsonl", "kind": "jsonl", "tag": "kb"},
    {"id": "kb/vn_upright.jsonl", "path": "kb/vn_upright.jsonl", "kind": "jsonl", "tag": "kb"},
    {"id": "kb/compact_cards.jsonl", "path": "kb/compact_cards.jsonl", "kind": "jsonl", "tag": "kb"},
    {"id": "kb/spreads.jsonl", "path": "kb/spreads.jsonl", "kind": "jsonl", "tag": "kb"},
    {"id": "kb/alias_table.json", "path": "kb/alias_table.json", "kind": "json", "tag": "kb"},
    {"id": "kb/card_name_whitelist.json", "path": "kb/card_name_whitelist.json", "kind": "json", "tag": "kb"},
    {"id": "kb/vn_register_profile.json", "path": "kb/vn_register_profile.json", "kind": "json", "tag": "kb"},
    {"id": "kb/vn_orientation_attribution.json", "path": "kb/vn_orientation_attribution.json", "kind": "json", "tag": "kb"},
    {"id": "kb/dendory_structural_profile.json", "path": "kb/dendory_structural_profile.json", "kind": "json", "tag": "kb"},
    {"id": "kb/english_spine.canonical.json", "path": "kb/english_spine.canonical.json", "kind": "json", "tag": "kb"},
    {"id": "kb/w2_2_gate_report.json", "path": "kb/w2_2_gate_report.json", "kind": "json", "tag": "report"},
    {"id": "kb/spreads_discrimination_report.json", "path": "kb/spreads_discrimination_report.json", "kind": "json", "tag": "report"},
    {"id": "kb/CARDS_HASH.txt", "path": "kb/CARDS_HASH.txt", "kind": "hash", "tag": "hash"},
    {"id": "datasets/filtered_core.jsonl", "path": "datasets/filtered_core.jsonl", "kind": "jsonl", "tag": "dataset"},
    {"id": "datasets/filtered_bulk.jsonl", "path": "datasets/filtered_bulk.jsonl", "kind": "jsonl", "tag": "dataset"},
    {"id": "datasets/splits.json", "path": "datasets/splits.json", "kind": "json", "tag": "dataset"},
    {"id": "datasets/split_stats.json", "path": "datasets/split_stats.json", "kind": "json", "tag": "dataset"},
    {"id": "datasets/filter_report.json", "path": "datasets/filter_report.json", "kind": "json", "tag": "report"},
    {"id": "datasets/coverage_report.json", "path": "datasets/coverage_report.json", "kind": "json", "tag": "report"},
    {"id": "datasets/ablation_report.json", "path": "datasets/ablation_report.json", "kind": "json", "tag": "report"},
    {"id": "datasets/base_diversity_baseline.json", "path": "datasets/base_diversity_baseline.json", "kind": "json", "tag": "dataset"},
    {"id": "datasets/DATASET_HASH.txt", "path": "datasets/DATASET_HASH.txt", "kind": "hash", "tag": "hash"},
    {"id": "datasets/anchor/anchor_readings.jsonl", "path": "datasets/anchor/anchor_readings.jsonl", "kind": "jsonl", "tag": "anchor"},
    {"id": "datasets/raw/generated.jsonl", "path": "datasets/raw/generated.jsonl", "kind": "jsonl", "tag": "raw"},
    {"id": "datasets/raw/generated_sep.jsonl", "path": "datasets/raw/generated_sep.jsonl", "kind": "jsonl", "tag": "raw"},
    {"id": "datasets/raw/ifd_scores.jsonl", "path": "datasets/raw/ifd_scores.jsonl", "kind": "jsonl", "tag": "raw"},
    {"id": "datasets/raw/purged_spread_context_mismatch_ids.txt", "path": "datasets/raw/purged_spread_context_mismatch_ids.txt", "kind": "txt", "tag": "raw"},
]
