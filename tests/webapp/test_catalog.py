"""Catalog + hash-integrity tests (todo A.2 / C.1).

All tests operate on the ``fake_root`` fixture under ``tmp_path`` — real
``kb/`` and ``datasets/`` files are never read or mutated. The hash-drift
negative test mutates ONLY the fixture copy of ``cards.jsonl``.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from tfvn.serialise import dumps_canonical, read_jsonl, write_jsonl
from tfvn.webapp import catalog
from tfvn.webapp.catalog_registry import REGISTRY

# The scripts/ sys.path shim (conftest) makes build-script modules importable.
import build_wave2_api  # noqa: E402


def _artifact(cat: catalog.CatalogResponse, rel: str) -> catalog.ArtifactInfo:
    return next(a for a in cat.artifacts if a.id == rel)


def test_catalog_counts_and_registry_size(fake_root):
    cat = catalog.get_catalog(fake_root)
    assert len(cat.artifacts) == len(REGISTRY) == 29

    assert _artifact(cat, "kb/cards.jsonl").rows == 4
    assert _artifact(cat, "kb/english_spine.jsonl").rows == 4
    assert _artifact(cat, "kb/spreads.jsonl").rows == 3
    assert _artifact(cat, "datasets/anchor/anchor_readings.jsonl").rows == 4
    assert _artifact(cat, "datasets/filtered_core.jsonl").rows == 4
    assert _artifact(cat, "datasets/filtered_bulk.jsonl").rows == 3
    # rows is None for non-jsonl artifacts (json/txt/hash kinds don't count rows)
    assert _artifact(cat, "kb/card_name_whitelist.json").rows is None


def test_missing_raw_files_are_present_if_exists(fake_root):
    """datasets/raw/* is not created by the fixture — must not crash."""
    cat = catalog.get_catalog(fake_root)
    for rel in ("datasets/raw/generated.jsonl", "datasets/raw/generated_sep.jsonl",
                "datasets/raw/ifd_scores.jsonl",
                "datasets/raw/purged_spread_context_mismatch_ids.txt"):
        art = _artifact(cat, rel)
        assert art.rows == 0
        assert art.size_bytes == 0
        assert art.sha256 is None
        assert art.schema_keys is None
        assert art.tag == "raw"


def test_schema_keys_from_first_row(fake_root):
    cat = catalog.get_catalog(fake_root)
    keys = _artifact(cat, "kb/cards.jsonl").schema_keys
    assert keys is not None
    assert "card_id" in keys and "orientation" in keys and "meaning_en" in keys
    # vn_upright rows have no orientation key
    up_keys = _artifact(cat, "kb/vn_upright.jsonl").schema_keys
    assert up_keys is not None and "orientation" not in up_keys


def test_schema_keys_for_json_arrays(fake_root):
    """Array-valued JSON files report the first element's keys."""
    cat = catalog.get_catalog(fake_root)
    assert _artifact(cat, "kb/alias_table.json").schema_keys == ["alias", "canonical", "card_id"]
    assert _artifact(cat, "kb/english_spine.canonical.json").schema_keys == ["card_id", "orientation"]
    # dict-valued JSON reports its top-level keys
    assert _artifact(cat, "kb/card_name_whitelist.json").schema_keys == ["canonical_count"]


def test_hashcheck_matches_on_fixture(fake_root):
    hc = catalog.get_hashcheck(fake_root)
    assert hc.cards_match is True
    assert hc.dataset_match is True
    assert [c.id for c in hc.checks] == ["cards", "datasets"]
    assert all(c.method == "canonical" for c in hc.checks)
    cards, datasets = hc.checks
    assert cards.computed_digest == cards.recorded_digest
    assert len(cards.computed_digest) == 64
    assert datasets.computed_digest == datasets.recorded_digest


def test_cards_digest_replicates_build_wave2_api(fake_root):
    """Catalog's cards digest must be byte-identical to the build script's
    method (build_wave2_api.py:588): sha256(dumps_canonical(rows)) over parsed
    rows in file order. Requires the scripts/ sys.path shim."""
    rows = read_jsonl(fake_root / "kb" / "cards.jsonl")
    independent = hashlib.sha256(dumps_canonical(rows).encode("utf-8")).hexdigest()
    assert catalog._cards_digest(fake_root) == independent
    # Same serialiser object the build script uses -> identical bytes.
    assert build_wave2_api.dumps_canonical is dumps_canonical
    script_method = hashlib.sha256(
        build_wave2_api.dumps_canonical(rows).encode("utf-8")
    ).hexdigest()
    assert script_method == independent


def test_datasets_digest_replicates_build_wave3(fake_root):
    """Catalog's datasets digest must match the build_wave3.py:1377-1381
    method: concat tiers, sort by example_id, sha256 over joined canonical
    lines."""
    core = read_jsonl(fake_root / "datasets" / "filtered_core.jsonl")
    bulk = read_jsonl(fake_root / "datasets" / "filtered_bulk.jsonl")
    combined = core + bulk
    canonical = "\n".join(
        dumps_canonical(r) for r in sorted(combined, key=lambda r: r["example_id"])
    )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert catalog._datasets_digest(fake_root) == expected
    assert (fake_root / "datasets" / "DATASET_HASH.txt").read_text().strip() == expected


def test_hashcheck_matches_false_on_missing_hash_file(tmp_path):
    """No CARDS_HASH.txt -> recorded digest missing -> matches:false (no crash)."""
    from tests.webapp.conftest import build_fake_root

    root = build_fake_root(tmp_path)
    (root / "kb" / "CARDS_HASH.txt").unlink()
    hc = catalog.get_hashcheck(root)
    cards = hc.checks[0]
    assert cards.matches is False
    assert cards.computed_digest is not None
    assert cards.recorded_digest is None


def test_hash_drift_negative_only_mutates_fixture(fake_root):
    """Mutate a row in the FIXTURE cards.jsonl -> cards_match flips false.

    This is the plan's hash-drift negative test. Only the tmp_path copy is
    touched; the real repo kb/cards.jsonl must be byte-identical before and
    after (guarded here so the test fails loudly if it ever mutated the repo).
    """
    real_cards = catalog.REPO_ROOT / "kb" / "cards.jsonl"
    real_hash = catalog.REPO_ROOT / "kb" / "CARDS_HASH.txt"
    if not (real_cards.exists() and real_hash.exists()):
        pytest.skip("real repo kb artifacts absent — drift guard cannot run")
    before_real = (real_cards.read_bytes(), real_hash.read_bytes())

    hc = catalog.get_hashcheck(fake_root)
    assert hc.cards_match is True

    # Rewrite ONLY the fixture cards.jsonl with a drifted row.
    fixture_cards = fake_root / "kb" / "cards.jsonl"
    rows = read_jsonl(fixture_cards)
    rows[0] = {**rows[0], "meaning_en": "A TAMPERED meaning drift"}
    write_jsonl(fixture_cards, rows)

    catalog.invalidate()
    hc2 = catalog.get_hashcheck(fake_root)
    assert hc2.cards_match is False
    # datasets digest unaffected by the cards drift
    assert hc2.dataset_match is True

    # Guard: the real repo files were never touched.
    assert real_cards.read_bytes() == before_real[0]
    assert real_hash.read_bytes() == before_real[1]


def test_catalog_cache_invalidated_on_mtime_size_change(fake_root):
    cat1 = catalog.get_catalog(fake_root)
    assert _artifact(cat1, "kb/cards.jsonl").rows == 4
    key_before = catalog._CACHE["key"]

    # Append a row (size + mtime change) -> next call must recompute.
    fixture_cards = fake_root / "kb" / "cards.jsonl"
    rows = read_jsonl(fixture_cards)
    rows.append({**rows[0], "card_id": 9})
    write_jsonl(fixture_cards, rows)

    cat2 = catalog.get_catalog(fake_root)
    assert _artifact(cat2, "kb/cards.jsonl").rows == 5
    assert catalog._CACHE["key"] != key_before

    # Same-size touch (mtime change only) must also invalidate.
    data = fixture_cards.read_bytes()
    fixture_cards.write_bytes(data)
    os.utime(fixture_cards, ns=(os.stat(fixture_cards).st_atime_ns + 10**9,
                                os.stat(fixture_cards).st_mtime_ns + 10**9))
    key_before_touch = catalog._CACHE["key"]
    cat3 = catalog.get_catalog(fake_root)
    assert _artifact(cat3, "kb/cards.jsonl").rows == 5
    assert catalog._CACHE["key"] != key_before_touch


def test_invalidate_drops_cache(fake_root):
    catalog.get_catalog(fake_root)
    assert catalog._CACHE["key"] is not None
    catalog.invalidate()
    assert catalog._CACHE["key"] is None
    # recompute on demand
    cat = catalog.get_catalog(fake_root)
    assert len(cat.artifacts) == 29


def test_missing_source_yields_empty_hashcheck(tmp_path):
    """A bare root (no kb/, no datasets/) must not crash any endpoint."""
    hc = catalog.get_hashcheck(tmp_path)
    assert hc.cards_match is False
    assert hc.dataset_match is False
    # cards: missing source -> computed None; datasets: sha256 over zero rows
    assert hc.checks[0].computed_digest is None
    assert hc.checks[1].computed_digest == hashlib.sha256(b"").hexdigest()
    assert hc.checks[0].recorded_digest is None
    cat = catalog.get_catalog(tmp_path)
    assert len(cat.artifacts) == 29
    assert all(a.rows == 0 and a.size_bytes == 0 for a in cat.artifacts)


def test_catalog_router_via_client(fake_root, monkeypatch):
    """The endpoint functions default ``root=REPO_ROOT`` at def time, so the
    router is pointed at the fixture by wrapping the public functions."""
    real_catalog = catalog.get_catalog
    real_hashcheck = catalog.get_hashcheck
    monkeypatch.setattr(catalog, "get_catalog", lambda: real_catalog(fake_root))
    monkeypatch.setattr(catalog, "get_hashcheck", lambda: real_hashcheck(fake_root))
    client = catalog_router_client()
    r = client.get("/api/catalog")
    assert r.status_code == 200
    body = r.json()
    assert len(body["artifacts"]) == 29
    cards = next(a for a in body["artifacts"] if a["id"] == "kb/cards.jsonl")
    assert cards["rows"] == 4
    hc = client.get("/api/hashcheck")
    assert hc.status_code == 200
    assert hc.json()["cards_match"] is True
    assert hc.json()["dataset_match"] is True
    assert hc.json()["checks"][0]["method"] == "canonical"


def catalog_router_client():
    from tests.webapp.conftest import make_client

    return make_client(catalog.router)
