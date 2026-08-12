"""Report contents API tests (A.6 gap closure / C.1).

Small module — kept as its own file because the coverage gate spans all of
``src/tfvn/webapp/``. All reports read from the ``fake_root`` fixture.
"""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from tfvn.webapp import reports


def test_report_list_six_entries(fake_root):
    listing = reports.get_report_list(fake_root)
    assert [r.id for r in listing.reports] == [
        "filter_report", "coverage_report", "split_stats",
        "ablation_report", "w2_2_gate_report", "spreads_discrimination_report",
    ]
    split = next(r for r in listing.reports if r.id == "split_stats")
    assert split.splits_path == "datasets/splits.json"
    assert all(r.path for r in listing.reports)


def test_report_list_present_if_exists(tmp_path):
    from tests.webapp.conftest import build_fake_root

    root = build_fake_root(tmp_path)
    (root / "datasets" / "ablation_report.json").unlink()
    listing = reports.get_report_list(root)
    ids = [r.id for r in listing.reports]
    assert "ablation_report" not in ids and len(ids) == 5


def test_get_report_parsed_contents(fake_root):
    data = reports.get_report(fake_root, "filter_report")
    assert data["acceptance"] == {
        "l1_dedup": True, "l2_ifd": True, "l3_deita": True, "l4_judge": True,
    }
    w22 = reports.get_report(fake_root, "w2_2_gate_report")
    assert w22["aggregate"]["failed_gate"] == 0
    assert w22["negative_control_rejection_rate"] == 0.95


def test_get_report_404s(fake_root):
    with pytest.raises(HTTPException) as exc:
        reports.get_report(fake_root, "bogus_report")
    assert exc.value.status_code == 404
    # a root where the report file is absent -> 404 (present-if-exists)
    empty = fake_root / "missing_root"
    empty.mkdir()
    with pytest.raises(HTTPException) as exc:
        reports.get_report(empty, "filter_report")
    assert exc.value.status_code == 404


def test_get_report_500_on_invalid_json(tmp_path):
    sub = tmp_path / "datasets"
    sub.mkdir(parents=True)
    (sub / "filter_report.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        reports.get_report(tmp_path, "filter_report")
    assert exc.value.status_code == 500
    (sub / "filter_report.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        reports.get_report(tmp_path, "filter_report")
    assert exc.value.status_code == 500  # not a JSON object


def test_report_router(fake_root, monkeypatch):
    from tests.webapp.conftest import make_client

    real_list = reports.get_report_list
    real_get = reports.get_report
    monkeypatch.setattr(reports, "get_report_list", lambda *a: real_list(fake_root))
    monkeypatch.setattr(reports, "get_report", lambda *a: real_get(fake_root, a[-1]))
    client = make_client(reports.router)
    listing = client.get("/api/reports")
    assert listing.status_code == 200
    assert len(listing.json()["reports"]) == 6
    data = client.get("/api/reports/w2_2_gate_report")
    assert data.status_code == 200
    assert data.json()["aggregate"]["failed_gate"] == 0
    assert client.get("/api/reports/nope").status_code == 404
    assert client.get("/api/reports/split_stats").json()["counts"]["train"] == 4
