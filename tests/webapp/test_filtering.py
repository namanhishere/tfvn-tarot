"""Filtered row / search / export tests (todo A.4 / C.1).

Covers filter combinations, ``q`` substring search including the per-dataset
fallback fields, pagination bounds, Pydantic 422s, the primary-key endpoint
for every dataset kind (example_id / card_id+orientation / card_id /
spread_id / anchor_id) and export-line validity via ``dumps_canonical``
round-trip. All against the ``fake_root`` fixture.
"""

from __future__ import annotations

import json

import pytest

from tfvn.serialise import dumps_canonical
from tfvn.webapp import filtering
from tfvn.webapp.filtering import RowsParams, get_row, get_rows, iter_export_lines


def _ids(root, dataset_id, **params):
    res = get_rows(root, dataset_id, RowsParams(**params))
    return [r["example_id"] for r in res.rows]


def test_all_sft_union_tier_tags(fake_root):
    res = get_rows(fake_root, "all_sft", RowsParams(page_size=200))
    assert res.total == 7
    assert {r["tier"] for r in res.rows} == {"core", "bulk"}
    assert sum(1 for r in res.rows if r["tier"] == "core") == 4
    assert sum(1 for r in res.rows if r["tier"] == "bulk") == 3


def test_single_tier_totals(fake_root):
    assert get_rows(fake_root, "filtered_core", RowsParams(page_size=200)).total == 4
    assert get_rows(fake_root, "filtered_bulk", RowsParams(page_size=200)).total == 3
    assert _ids(fake_root, "all_sft", tier="core") == [
        "w32_000001", "w32_000002", "w32_000003", "w32_000004",
    ]


def test_filter_combos(fake_root):
    assert _ids(fake_root, "all_sft", task_type="reading") == [
        "w32_000001", "w32_000002", "w32_000004", "w32_000103",
    ]
    assert _ids(fake_root, "all_sft", task_type="safety") == ["w32_000003"]
    # register goes through the pydantic alias
    assert _ids(fake_root, "all_sft", register="formal") == ["w32_000003", "w32_000101"]
    assert _ids(fake_root, "all_sft", length_band="long") == ["w32_000101"]
    assert _ids(fake_root, "all_sft", querent_context="crisis") == ["w32_000003"]
    assert _ids(fake_root, "all_sft", spread_id="celtic") == ["w32_000101"]
    # AND semantics across axes
    assert _ids(fake_root, "all_sft", task_type="reading", register="casual") == [
        "w32_000001", "w32_000002", "w32_000004", "w32_000103",
    ]
    assert _ids(fake_root, "all_sft", task_type="reading", querent_context="love") == [
        "w32_000001", "w32_000103",
    ]


def test_card_id_filter_list_membership(fake_root):
    """card_id matches the SFT card_ids LIST (int/str drift tolerated)."""
    assert _ids(fake_root, "all_sft", card_id=0) == [
        "w32_000001", "w32_000004", "w32_000101", "w32_000103",
    ]
    assert _ids(fake_root, "all_sft", card_id=1) == ["w32_000001", "w32_000101"]
    assert _ids(fake_root, "all_sft", card_id=5) == ["w32_000003"]
    assert _ids(fake_root, "all_sft", card_id=99) == []


def test_orientation_filter_list_membership(fake_root):
    assert _ids(fake_root, "all_sft", orientation="reversed") == [
        "w32_000002", "w32_000101", "w32_000103",
    ]
    # 000101 carries BOTH orientations in its list, so it matches upright too
    assert _ids(fake_root, "all_sft", orientation="upright") == [
        "w32_000001", "w32_000003", "w32_000004", "w32_000101",
    ]


def test_ifd_range_filters(fake_root):
    """ifd bounds apply only to rows that HAVE ifd_score; missing-key rows pass."""
    # rows with a score >= 0.5, plus 000102 which has no score (never excluded)
    assert _ids(fake_root, "all_sft", ifd_min=0.5) == [
        "w32_000001", "w32_000003", "w32_000004", "w32_000102",
    ]
    assert _ids(fake_root, "all_sft", ifd_max=0.2) == [
        "w32_000002", "w32_000101", "w32_000102",
    ]
    assert _ids(fake_root, "all_sft", ifd_min=0.0, ifd_max=0.6) == [
        "w32_000001", "w32_000004", "w32_000101", "w32_000102", "w32_000103",
    ]


def test_q_search_sft_fields(fake_root):
    assert _ids(fake_root, "all_sft", q="khủng hoảng") == ["w32_000003"]
    assert _ids(fake_root, "all_sft", q="công việc") == ["w32_000101"]
    assert _ids(fake_root, "all_sft", q="tình yêu") == ["w32_000001", "w32_000103"]
    assert _ids(fake_root, "all_sft", q="KHỦNG HOẢNG") == ["w32_000003"]


def test_q_search_cards_fields(fake_root):
    """cards search kind: name_en / meaning_en / meaning_vi only."""
    res = get_rows(fake_root, "cards", RowsParams(q="the fool", page_size=200))
    assert res.total == 2
    assert sorted(r["card_id"] for r in res.rows) == [0, 0]
    res = get_rows(fake_root, "cards", RowsParams(q="THE FOOL", page_size=200))
    assert res.total == 2


def test_q_search_fallback_fields(fake_root):
    """fallback search kind: any string (or str-list) value in the row."""
    # vn_spine: name_en "The Magician"
    res = get_rows(fake_root, "vn_spine", RowsParams(q="magician", page_size=200))
    assert res.total == 2
    # spreads: name_en / name_vi / spread_id strings
    res = get_rows(fake_root, "spreads", RowsParams(q="Single", page_size=200))
    assert res.total == 1 and res.rows[0]["spread_id"] == "single"
    # anchor: reading_vi strings
    res = get_rows(fake_root, "anchor", RowsParams(q="bước hụt", page_size=200))
    assert res.total == 1 and res.rows[0]["anchor_id"] == "anchor_0002"
    # fallback also covers str-list values (keywords_vi)
    res = get_rows(fake_root, "compact_cards", RowsParams(q="atom", page_size=200))
    assert res.total == 0  # compact_cards fixture rows carry no text list
    # vn_upright: title_main "Kẻ Khờ"
    res = get_rows(fake_root, "vn_upright", RowsParams(q="kẻ khờ", page_size=200))
    assert res.total == 1


def test_pagination_bounds(fake_root):
    res = get_rows(fake_root, "all_sft", RowsParams(page=1, page_size=3))
    assert res.total == 7 and len(res.rows) == 3
    res = get_rows(fake_root, "all_sft", RowsParams(page=2, page_size=3))
    assert res.total == 7 and len(res.rows) == 3
    res = get_rows(fake_root, "all_sft", RowsParams(page=3, page_size=3))
    assert res.total == 7 and len(res.rows) == 1
    res = get_rows(fake_root, "all_sft", RowsParams(page=4, page_size=3))
    assert res.total == 7 and res.rows == []


def test_pagination_unique_ids_across_pages(fake_root):
    seen = []
    for page in (1, 2, 3):
        res = get_rows(fake_root, "all_sft", RowsParams(page=page, page_size=3))
        seen.extend(r["example_id"] for r in res.rows)
    assert len(seen) == 7 and len(set(seen)) == 7


def test_primary_key_endpoints(fake_root):
    """Per dataset kind: example_id / card_id+orientation / card_id /
    spread_id / anchor_id."""
    # SFT: example_id
    assert get_row(fake_root, "filtered_core", "w32_000001")["example_id"] == "w32_000001"
    assert get_row(fake_root, "all_sft", "w32_000101")["tier"] == "bulk"
    # card_orientation datasets
    card = get_row(fake_root, "cards", "0/upright")
    assert card["card_id"] == 0 and card["orientation"] == "upright"
    assert get_row(fake_root, "vn_spine", "1/reversed")["orientation"] == "reversed"
    assert get_row(fake_root, "english_spine", "0/reversed")["card_id"] == 0
    assert get_row(fake_root, "compact_cards", "1/upright")["card_id"] == 1
    # vn_upright: card_id only (no orientation key on those rows)
    assert get_row(fake_root, "vn_upright", "1")["card_id"] == 1
    # spreads: spread_id
    assert get_row(fake_root, "spreads", "celtic")["name_en"] == "Celtic Cross"
    # anchor: anchor_id
    assert get_row(fake_root, "anchor", "anchor_0004")["card_id"] == 1


def test_primary_key_404s(fake_root):
    from fastapi import HTTPException

    for id_path in ("0/bogus", "abc/upright", "0"):
        with pytest.raises(HTTPException) as exc:
            get_row(fake_root, "cards", id_path)
        assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        get_row(fake_root, "filtered_core", "nope")
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        get_row(fake_root, "spreads", "missing_spread")
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        get_row(fake_root, "bogus_dataset", "x")
    assert exc.value.status_code == 404


def test_missing_raw_file_iterates_empty(fake_root):
    res = get_rows(fake_root, "raw_generated", RowsParams(page_size=200))
    assert res.total == 0 and res.rows == []
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        get_row(fake_root, "raw_generated", "anything")
    assert exc.value.status_code == 404


def test_export_lines_canonical_round_trip(fake_root):
    p = RowsParams(task_type="reading")
    lines = list(iter_export_lines(fake_root, "filtered_core", p))
    rows = get_rows(fake_root, "filtered_core", p).rows
    assert len(lines) == len(rows)
    for line, row in zip(lines, rows):
        parsed = json.loads(line)
        assert parsed == row
        assert line.rstrip("\n") == dumps_canonical(row)


def test_export_all_sft_tier_tag_lines(fake_root):
    lines = list(iter_export_lines(fake_root, "all_sft", RowsParams(tier="core")))
    assert len(lines) == 4
    for line in lines:
        assert json.loads(line)["tier"] == "core"


def test_export_lines_filters_match_rows_total(fake_root):
    p = RowsParams(querent_context="love")
    lines = list(iter_export_lines(fake_root, "all_sft", p))
    assert len(lines) == get_rows(fake_root, "all_sft", p).total == 2


def _client(fake_root, monkeypatch):
    from tests.webapp.conftest import make_client

    monkeypatch.setattr(filtering, "REPO_ROOT", fake_root)
    return make_client(filtering.router)


def test_router_422_on_bad_params(fake_root, monkeypatch):
    client = _client(fake_root, monkeypatch)
    assert client.get("/api/rows/all_sft?page=0").status_code == 422
    assert client.get("/api/rows/all_sft?page_size=999").status_code == 422
    assert client.get("/api/rows/all_sft?page_size=0").status_code == 422
    assert client.get("/api/rows/all_sft?card_id=abc").status_code == 422
    assert client.get("/api/rows/all_sft?tier=bogus").status_code == 422
    assert client.get("/api/rows/all_sft?ifd_min=abc").status_code == 422
    assert client.get("/api/rows/bogus_dataset").status_code == 404


def test_router_register_alias_and_combos(fake_root, monkeypatch):
    client = _client(fake_root, monkeypatch)
    r = client.get("/api/rows/all_sft?register=casual&page_size=200")
    assert r.status_code == 200
    assert r.json()["total"] == 5
    r = client.get("/api/rows/all_sft?task_type=safety&tier=core")
    assert r.status_code == 200 and r.json()["total"] == 1


def test_router_primary_key_routes(fake_root, monkeypatch):
    client = _client(fake_root, monkeypatch)
    assert client.get("/api/rows/cards/0/upright").json()["card_id"] == 0
    assert client.get("/api/rows/vn_upright/1").json()["card_id"] == 1
    assert client.get("/api/rows/spreads/single").json()["spread_id"] == "single"
    assert client.get("/api/rows/anchor/anchor_0001").json()["anchor_id"] == "anchor_0001"
    assert client.get("/api/rows/filtered_core/w32_000001").json()["example_id"] == "w32_000001"
    assert client.get("/api/rows/cards/0/bogus").status_code == 404


def test_router_export_headers_and_body(fake_root, monkeypatch):
    client = _client(fake_root, monkeypatch)
    r = client.get("/api/export/all_sft?tier=core")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert r.headers["content-disposition"] == "attachment; filename=all_sft.jsonl"
    lines = [ln for ln in r.text.splitlines() if ln]
    assert len(lines) == 4
    assert all(json.loads(ln)["tier"] == "core" for ln in lines)
    # HEAD mirror serves the same headers
    h = client.head("/api/export/filtered_core")
    assert h.status_code == 200
    assert h.headers["content-disposition"] == "attachment; filename=filtered_core.jsonl"


def test_export_422_on_bad_params(fake_root, monkeypatch):
    client = _client(fake_root, monkeypatch)
    assert client.get("/api/export/all_sft?tier=bogus").status_code == 422
