"""Statistics aggregator tests (todo A.3 / C.1).

Verifies distributions, key-presence handling (``critique`` /
``reading_vi_original`` present on SOME rows of BOTH tiers and absent on
others), cache invalidation on (mtime,size) change, totals, the per-card
orientation matrix, the IFD histogram and the splits join — all against the
``fake_root`` fixture (7 SFT rows: 4 core + 3 bulk).
"""

from __future__ import annotations

import os

import pytest

from tfvn.webapp import stats


def test_source_and_tier_totals(fake_root):
    payload = stats.compute_stats(fake_root)
    assert payload["source"] == {"core_rows": 4, "bulk_rows": 3, "total": 7}
    assert payload["tier_counts"] == {"core": 4, "bulk": 3}


def test_task_type_distribution(fake_root):
    dist = stats.compute_stats(fake_root)["distributions"]
    assert dist["task_type"] == {"explanation": 1, "grounding": 1, "reading": 4, "safety": 1}


def test_register_length_querent_distributions(fake_root):
    dist = stats.compute_stats(fake_root)["distributions"]
    assert dist["register"] == {"casual": 5, "formal": 2}
    assert dist["length_band"] == {"long": 1, "medium": 5, "short": 1}
    assert dist["querent_context"] == {"career": 1, "crisis": 1, "general": 3, "love": 2}


def test_spread_and_cards_drawn_distributions(fake_root):
    dist = stats.compute_stats(fake_root)["distributions"]
    assert dist["spread_id"] == {"celtic": 1, "single": 6}
    # 000101 uses spread_id celtic but keeps the default spread_name_vi
    assert dist["spread_name_vi"] == {"Một lá": 7}
    # cards_used lengths: 2,1,1,1,2,0,1 -> drawn 0:1, 1:4, 2:2
    assert dist["cards_drawn"] == {"0": 1, "1": 4, "2": 2}


def test_critique_key_presence_both_tiers(fake_root):
    """critique exists on rows in BOTH tiers and is absent on rows in both —
    the verdict buckets must sum to the full total with no_critique filling
    the gaps (pass: 000001 core, 000101+000103 bulk; fix: 000003 core;
    no_critique: 000002+000004 core, 000102 bulk)."""
    dist = stats.compute_stats(fake_root)["distributions"]
    verdict = dist["critique_verdict"]
    assert verdict == {"fix": 1, "no_critique": 3, "pass": 3}
    assert sum(verdict.values()) == 7
    # critique_applied defaults to False on the other six rows
    assert dist["critique_applied"] == {"False": 6, "True": 1}


def test_safety_grounding_wrong_claim_distributions(fake_root):
    dist = stats.compute_stats(fake_root)["distributions"]
    assert dist["safety_category"] == {"general_advice": 1, "self_harm": 1}
    assert dist["grounding_defect"] == {"paraphrase": 1}
    assert dist["wrong_claim"] == {"True": 1}


def test_provenance_distribution(fake_root):
    prov = stats.compute_stats(fake_root)["distributions"]["provenance"]
    assert prov == {"gen_v4": 7, "scrubbed": 1}


def test_reading_vi_original_presence_does_not_affect_counts(fake_root):
    """reading_vi_original exists on core 000001 and bulk 000102 only; it is
    not a distribution field, and its presence must never change totals."""
    payload = stats.compute_stats(fake_root)
    assert payload["source"]["total"] == 7
    assert sum(payload["distributions"]["critique_verdict"].values()) == 7


def test_per_card_matrix_and_reversed_percent(fake_root):
    pc = stats.compute_stats(fake_root)["per_card"]
    # card 0: 000001 up, 000101 up, 000103 rev, 000004 up -> up3 rev1
    assert pc["frequency"]["0"] == 4
    assert pc["orientation_mix"]["0"] == {"reversed": 1, "upright": 3}
    # card 1: 000001 up, 000101 rev -> up1 rev1
    assert pc["orientation_mix"]["1"] == {"reversed": 1, "upright": 1}
    # card 2: 000002 rev; card 5: 000003 up
    assert pc["frequency"]["2"] == 1
    assert pc["frequency"]["5"] == 1
    assert pc["total_card_mentions"] == 8
    # reversed: card0 000103 + card1 000101 + card2 000002 = 3 of 8 -> 37.5
    assert stats.compute_stats(fake_root)["total_reversed_percent"] == 37.5


def test_ifd_stats_and_negative_values(fake_root):
    ifd = stats.compute_stats(fake_root)["ifd"]
    # values present: 0.5, -0.1, 0.9, 0.5, 0.2, 0.35 (w32_000102 has no ifd_score)
    assert ifd["count"] == 6
    assert ifd["min"] == -0.1
    assert ifd["max"] == 0.9
    assert round(ifd["mean"], 6) == round(0.3916666, 6)
    assert len(ifd["histogram"]) == 10
    assert len(ifd["bin_edges"]) == 11
    assert sum(ifd["histogram"]) == 6
    assert ifd["bin_edges"][0] == -0.1 and ifd["bin_edges"][-1] == 0.9


def test_splits_join(fake_root):
    splits = stats.compute_stats(fake_root)["splits"]
    assert splits["total_rows"] == 7
    assert splits["unmatched_rows"] == 1  # w32_000004 absent from splits.json
    assert splits["counts"] == {"test": 2, "train": 3, "val": 1}
    assert splits["by_task_type"]["train"] == {"grounding": 1, "reading": 2}
    assert splits["by_task_type"]["test"] == {"reading": 1, "safety": 1}
    assert splits["by_task_type"]["val"] == {"explanation": 1}


def test_kb_stats(fake_root):
    kb = stats.compute_stats(fake_root)["kb"]
    assert kb["total_rows"] == 4
    assert kb["arcana"] == {"major": 2, "minor": 2}
    assert kb["orientation"] == {"reversed": 2, "upright": 2}
    assert kb["vi_provenance"] == {"source": 2, "synthetic": 2}
    assert kb["polarity_axis"] == {"negative": 2, "positive": 2}
    assert kb["vi_orientation_attribution"] == {"vi_upright": 4}
    # meaning_en non-empty on all 4 rows
    assert kb["meaning_en"]["count"] == 4
    # meaning_vi non-empty on all 4 rows
    assert kb["meaning_vi"]["count"] == 4
    # domain_vi only on the two upright rows: love 2, work 1 (all upright)
    assert kb["domain_vi_coverage"] == {
        "love": {"upright": 2},
        "work": {"upright": 1},
    }


def test_spreads_stats(fake_root):
    spreads = stats.compute_stats(fake_root)["spreads"]
    assert spreads["count"] == 3
    assert spreads["cards_drawn"] == {"1": 1, "10": 1, "3": 1}
    assert spreads["difficulty"] == {"easy": 1, "hard": 1, "medium": 1}
    single = spreads["by_spread_id"]["single"]
    assert single["cards_drawn"] == 1 and single["name_en"] == "Single Card"
    assert single["num_positions"] == 1  # len(positions)
    assert spreads["by_spread_id"]["celtic"]["num_positions"] == 2


def test_anchor_stats(fake_root):
    anchor = stats.compute_stats(fake_root)["anchor"]
    assert anchor["count"] == 4
    assert anchor["by_card_id"] == {"0": 2, "1": 2}
    assert anchor["by_orientation"] == {"reversed": 2, "upright": 2}


def test_cache_serves_second_call(fake_root):
    before = stats.computation_count()
    a = stats.compute_stats(fake_root)
    assert stats.computation_count() == before + 1
    b = stats.compute_stats(fake_root)
    assert stats.computation_count() == before + 1
    assert a == b


def test_cache_invalidates_on_mtime_size_change(fake_root):
    before = stats.computation_count()
    stats.compute_stats(fake_root)
    assert stats.computation_count() == before + 1

    # Append an SFT row -> size + mtime change -> recompute.
    from tfvn.serialise import write_jsonl
    from tests.webapp.conftest import make_sft_row

    core_path = fake_root / "datasets" / "filtered_core.jsonl"
    from tfvn.serialise import read_jsonl

    rows = read_jsonl(core_path)
    rows.append(make_sft_row("w32_000099", task_type="grounding"))
    write_jsonl(core_path, rows)

    payload = stats.compute_stats(fake_root)
    assert stats.computation_count() == before + 2
    assert payload["source"]["core_rows"] == 5
    assert payload["source"]["total"] == 8

    # Same-size touch (mtime only) also invalidates.
    data = core_path.read_bytes()
    core_path.write_bytes(data)
    os.utime(
        core_path,
        ns=(os.stat(core_path).st_atime_ns + 10**9, os.stat(core_path).st_mtime_ns + 10**9),
    )
    stats.compute_stats(fake_root)
    assert stats.computation_count() == before + 3


def test_invalidate_drops_payload(fake_root):
    stats.compute_stats(fake_root)
    assert stats._cache_payload is not None
    stats.invalidate()
    assert stats._cache_payload is None and stats._cache_key is None


def test_missing_input_file_raises(tmp_path):
    """Stats is explicit: a missing input file is FileNotFoundError, never a
    silent partial result."""
    with pytest.raises(FileNotFoundError):
        stats.compute_stats(tmp_path)
    (tmp_path / "datasets").mkdir(parents=True)
    (tmp_path / "kb").mkdir()
    with pytest.raises(FileNotFoundError):
        stats.compute_stats(tmp_path)


def test_endpoint_returns_sorted_canonical_json(fake_root, monkeypatch):
    from tests.webapp.conftest import make_client

    real_compute = stats.compute_stats
    monkeypatch.setattr(stats, "compute_stats", lambda: real_compute(fake_root))
    client = make_client(stats.router)
    r = client.get("/api/stats")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["source"]["total"] == 7
    # byte-stable: re-serialising the payload with sorted keys reproduces bytes
    import json

    assert r.content == json.dumps(
        real_compute(fake_root), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
