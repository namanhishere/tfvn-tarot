# Wave 1 knowledge base artifacts

Built by `scripts/build_wave1.py`. Assert with:

```bash
PYTHONPATH=src python -m tfvn.assert_kb
PYTHONPATH=src pytest tests/ -q
```

| File | Rows / content |
|---|---|
| `english_spine.jsonl` | 156 (78×2) English semantic spine + reversed provenance |
| `vn_upright.jsonl` | 78 name-keyed Vietnamese uprights (Page = `synthetic_no_anchor`) |
| `spreads.jsonl` | 21 compressed spreads |
| `spreads_discrimination_report.json` | TF-IDF top-1 positional recovery |
| `vn_register_profile.json` | Function-word / particle profile |
| `compact_cards.jsonl` | 156 prompt-compact rows |
| `card_name_whitelist.json` | 78 canonical + aliases (shared by validators) |
| `alias_table.json` | Alias export |
| `english_spine.canonical.json` | Byte-stable spine document |
