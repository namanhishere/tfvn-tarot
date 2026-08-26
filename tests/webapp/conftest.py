"""Shared fixtures for the webapp pytest suite (todo C.1).

The sys.path shim below mirrors ``scripts/test_wave2_api.py:21-22`` exactly:
pytest.ini only sets ``pythonpath = src``, so tests that import build-script
modules (hash-digest cross-checks) also need ``scripts/`` on the path.

The :func:`fake_root` fixture builds a tiny fake repo under ``tmp_path`` — a
few JSONL rows per dataset kind plus matching CARDS_HASH/DATASET_HASH files
and fake pipeline reports. All tests pass ``root=`` into the module functions;
real ``kb/`` and ``datasets/`` files are never touched.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tfvn.serialise import dumps_canonical, write_jsonl  # noqa: E402


# --------------------------------------------------------------------------- #
# Fake-data builders
# --------------------------------------------------------------------------- #

def make_sft_row(
    example_id: str,
    *,
    task_type: str = "reading",
    register: str = "casual",
    length_band: str = "medium",
    querent_context: str = "general",
    spread_id: str = "single",
    spread_name_vi: str = "Một lá",
    cards_used: Optional[list[dict[str, Any]]] = None,
    ifd_score: Optional[float] = None,
    critique: Optional[dict[str, Any]] = None,
    critique_applied: bool = False,
    safety_category: Optional[str] = None,
    grounding_defect: Optional[str] = None,
    wrong_claim: Optional[bool] = None,
    provenance: Optional[list[str]] = None,
    question_vi: str = "Chuyện tình tôi sẽ ra sao?",
    reading_vi: str = "Lá bài báo hiệu một khởi đầu mới.",
    reading_vi_original: Optional[str] = None,
    position_glosses: Optional[list[Any]] = None,
    card_ids: Optional[list[int]] = None,
    orientations: Optional[list[str]] = None,
    matched_pair_id: Optional[str] = None,
    cards_drawn_n: Optional[int] = None,
) -> dict[str, Any]:
    """One SFT row; optional keys are emitted only when provided (so tests
    exercise key-presence handling instead of a fixed schema)."""
    row: dict[str, Any] = {
        "example_id": example_id,
        "task_type": task_type,
        "register": register,
        "length_band": length_band,
        "querent_context": querent_context,
        "spread_id": spread_id,
        "spread_name_vi": spread_name_vi,
        "cards_used": cards_used
        if cards_used is not None
        else [{"card_id": 0, "orientation": "upright"}],
        "ifd_score": ifd_score,
        "question_vi": question_vi,
        "reading_vi": reading_vi,
        "critique_applied": critique_applied,
        "provenance": list(provenance or ["gen_v4"]),
        "matched_pair_id": matched_pair_id,
    }
    if cards_drawn_n is not None:
        row["cards_drawn"] = cards_drawn_n
    if critique is not None:
        row["critique"] = critique
    if safety_category is not None:
        row["safety_category"] = safety_category
    if grounding_defect is not None:
        row["grounding_defect"] = grounding_defect
    if wrong_claim is not None:
        row["wrong_claim"] = wrong_claim
    if reading_vi_original is not None:
        row["reading_vi_original"] = reading_vi_original
    if position_glosses is not None:
        row["position_glosses"] = position_glosses
    if card_ids is not None:
        row["card_ids"] = card_ids
    if orientations is not None:
        row["orientations"] = orientations
    return row


def _card_row(
    card_id: int,
    orientation: str,
    *,
    name_en: str,
    meaning_en: str,
    domain_vi: Optional[dict[str, str]] = None,
    meaning_vi: Optional[str] = "ý nghĩa tiếng Việt",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "card_id": card_id,
        "orientation": orientation,
        "arcana": "major" if card_id == 0 else "minor",
        "suit": None,
        "name_en": name_en,
        "name_vi": f"Tên {card_id}",
        "meaning_en": meaning_en,
        "meaning_vi": meaning_vi,
        "keywords_en": ["beginning", "innocence"],
        "keywords_vi": ["khởi đầu"],
        "polarity_axis": "positive" if orientation == "upright" else "negative",
        "vi_provenance": "source" if orientation == "upright" else "synthetic",
        "vi_orientation_attribution": "vi_upright",
        "title_secondary": "Secondary title",
    }
    if domain_vi is not None:
        row["domain_vi"] = domain_vi
    return row


def _compute_cards_digest(root: Path) -> str:
    from tfvn.serialise import read_jsonl

    rows = read_jsonl(root / "kb" / "cards.jsonl")
    return hashlib.sha256(dumps_canonical(rows).encode("utf-8")).hexdigest()


def _compute_datasets_digest(root: Path) -> str:
    from tfvn.serialise import read_jsonl

    def load(rel: str) -> list[dict[str, Any]]:
        p = root / rel
        return read_jsonl(p) if p.exists() else []

    combined = load("datasets/filtered_core.jsonl") + load(
        "datasets/filtered_bulk.jsonl"
    )
    canonical = "\n".join(
        dumps_canonical(r) for r in sorted(combined, key=lambda r: r["example_id"])
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_fake_root(root: Path) -> Path:
    """Write a tiny fake repo tree under *root*; returns the root."""
    # ---- kb/cards.jsonl (2 cards x 2 orientations) ---------------------------
    cards = [
        _card_row(
            0,
            "upright",
            name_en="The Fool",
            meaning_en="a fresh beginning full of promise",
            domain_vi={"love": "tình yêu mới", "work": "cơ hội mới"},
        ),
        _card_row(
            0,
            "reversed",
            name_en="The Fool",
            meaning_en="recklessness and a false start",
        ),
        _card_row(
            1,
            "upright",
            name_en="The Magician",
            meaning_en="willpower and manifestation",
            domain_vi={"love": "gặp gỡ"},
        ),
        _card_row(
            1,
            "reversed",
            name_en="The Magician",
            meaning_en="trickery and wasted talent",
        ),
    ]
    write_jsonl(root / "kb" / "cards.jsonl", cards)

    # ---- kb/spines (4 rows each for the card_orientation pk datasets) --------
    spine_rows = [
        {
            "card_id": cid,
            "orientation": orient,
            "name_en": name,
            "meaning_en": meaning,
            "keyword_atoms_en": ["atom"],
        }
        for cid, orient, name, meaning in (
            (0, "upright", "The Fool", "beginning"),
            (0, "reversed", "The Fool", "fall"),
            (1, "upright", "The Magician", "will"),
            (1, "reversed", "The Magician", "trick"),
        )
    ]
    write_jsonl(root / "kb" / "english_spine.jsonl", spine_rows)
    write_jsonl(root / "kb" / "vn_spine.jsonl", spine_rows)
    write_jsonl(
        root / "kb" / "compact_cards.jsonl",
        [{"card_id": r["card_id"], "orientation": r["orientation"]} for r in spine_rows],
    )
    # vn_upright has NO orientation key (78 upright-only rows -> pk = card_id)
    write_jsonl(
        root / "kb" / "vn_upright.jsonl",
        [
            {"card_id": 0, "name_en": "The Fool", "title_main": "Kẻ Khờ"},
            {"card_id": 1, "name_en": "The Magician", "title_main": "Phù Thủy"},
        ],
    )

    # ---- kb JSON files (dicts and one array for list schema_keys) ------------
    _write_json(root / "kb" / "alias_table.json", [{"alias": "the fool", "canonical": "The Fool", "card_id": 0}])
    _write_json(root / "kb" / "card_name_whitelist.json", {"canonical_count": 78})
    _write_json(root / "kb" / "vn_register_profile.json", {"threshold": 5.2})
    _write_json(root / "kb" / "vn_orientation_attribution.json", {"attributions": []})
    _write_json(root / "kb" / "dendory_structural_profile.json", {"profile": "x"})
    _write_json(
        root / "kb" / "english_spine.canonical.json",
        [{"card_id": 0, "orientation": "upright"}, {"card_id": 0, "orientation": "reversed"}],
    )

    # ---- kb reports + hash ---------------------------------------------------
    _write_json(
        root / "kb" / "w2_2_gate_report.json",
        {
            "aggregate": {"failed_gate": 0},
            "negative_control_rejection_rate": 0.95,
            "total_cards": 78,
        },
    )
    _write_json(
        root / "kb" / "spreads_discrimination_report.json",
        {"spreads": 21, "above_chance": 20, "failing_spreads": []},
    )
    (root / "kb" / "CARDS_HASH.txt").write_text(
        _compute_cards_digest(root) + "\n", encoding="utf-8"
    )

    # ---- kb/spreads.jsonl ----------------------------------------------------
    write_jsonl(
        root / "kb" / "spreads.jsonl",
        [
            {
                "spread_id": "single",
                "name_en": "Single Card",
                "name_vi": "Một lá",
                "cards_drawn": 1,
                "difficulty": "easy",
                "positions": [{"id": 1, "gloss_en": "advice"}],
            },
            {
                "spread_id": "three",
                "name_en": "Three Cards",
                "name_vi": "Ba lá",
                "cards_drawn": 3,
                "difficulty": "medium",
                "positions": [],
            },
            {
                "spread_id": "celtic",
                "name_en": "Celtic Cross",
                "name_vi": "Thập tự Celtic",
                "cards_drawn": 10,
                "difficulty": "hard",
                "positions": [{"id": 1}, {"id": 2}],
            },
        ],
    )

    # ---- datasets: SFT tiers -------------------------------------------------
    core = [
        make_sft_row(
            "w32_000001",
            task_type="reading",
            register="casual",
            querent_context="love",
            cards_used=[
                {"card_id": 0, "orientation": "upright"},
                {"card_id": 1, "orientation": "upright"},
            ],
            card_ids=[0, 1],
            orientations=["upright", "upright"],
            ifd_score=0.5,
            critique={
                "verdict": "pass",
                "answers_question": True,
                "vietnamese_natural": True,
            },
            critique_applied=True,
            matched_pair_id="mp_001",
            reading_vi="Lá The Fool báo hiệu một khởi đầu mới trong tình yêu.",
            reading_vi_original="Bản gốc câu trả lời tiếng Việt.",
        ),
        make_sft_row(
            "w32_000002",
            task_type="reading",
            querent_context="general",
            cards_used=[{"card_id": 2, "orientation": "reversed"}],
            card_ids=[2],
            orientations=["reversed"],
            ifd_score=-0.1,  # IFD can be negative
            provenance=["gen_v4", "scrubbed"],
            safety_category="general_advice",
            reading_vi="Lá bài lộn ngược cảnh báo sự hấp tấp.",
        ),
        make_sft_row(
            "w32_000003",
            task_type="safety",
            register="formal",
            length_band="short",
            querent_context="crisis",
            cards_used=[{"card_id": 5, "orientation": "upright"}],
            card_ids=[5],
            orientations=["upright"],
            ifd_score=0.9,
            critique={"verdict": "fix", "faithful": False},
            safety_category="self_harm",
            grounding_defect="paraphrase",
            wrong_claim=True,
            matched_pair_id="safe_self_harm_crisis_0_s",
            reading_vi="Trong tình huống khủng hoảng, hãy gọi giúp đỡ ngay.",
        ),
        make_sft_row(
            "w32_000004",  # deliberately absent from splits.json -> unmatched
            task_type="reading",
            cards_used=[{"card_id": 0, "orientation": "upright"}],
            card_ids=[0],
            orientations=["upright"],
            ifd_score=0.5,
            reading_vi="Một ngày bình thường, không có biến cố.",
        ),
    ]
    bulk = [
        make_sft_row(
            "w32_000101",
            task_type="explanation",
            register="formal",
            length_band="long",
            querent_context="career",
            spread_id="celtic",
            cards_used=[
                {"card_id": 0, "orientation": "upright"},
                {"card_id": 1, "orientation": "reversed"},
            ],
            card_ids=[0, 1],
            orientations=["upright", "reversed"],
            ifd_score=0.2,
            critique={"verdict": "pass"},
            matched_pair_id="mp_101",
            reading_vi="Giải thích dài về công việc và sự nghiệp.",
        ),
        make_sft_row(
            "w32_000102",
            task_type="grounding",
            querent_context="general",
            cards_used=[],
            card_ids=[],
            orientations=[],
            reading_vi="Căn cứ kiến thức nền về tarot.",
            reading_vi_original="Bản gốc từ tier bulk.",
        ),  # no ifd_score key at all
        make_sft_row(
            "w32_000103",
            task_type="reading",
            register="casual",
            querent_context="love",
            spread_id="single",
            cards_used=[{"card_id": 0, "orientation": "reversed"}],
            card_ids=[0],
            orientations=["reversed"],
            ifd_score=0.35,
            critique={"verdict": "pass"},
            reading_vi="Tình yêu hiện tại cần sự chín chắn.",
        ),
    ]
    write_jsonl(root / "datasets" / "filtered_core.jsonl", core)
    write_jsonl(root / "datasets" / "filtered_bulk.jsonl", bulk)

    # ---- datasets: splits + reports + hash -----------------------------------
    _write_json(
        root / "datasets" / "splits.json",
        {
            "w32_000001": "train",
            "w32_000002": "train",
            "w32_000003": "test",
            "w32_000101": "val",
            "w32_000102": "train",
            "w32_000103": "test",
        },
    )
    _write_json(
        root / "datasets" / "split_stats.json",
        {"acceptance": {"splits_ok": True}, "counts": {"train": 4, "val": 1, "test": 2}},
    )
    _write_json(
        root / "datasets" / "filter_report.json",
        {
            "acceptance": {
                "l1_dedup": True,
                "l2_ifd": True,
                "l3_deita": True,
                "l4_judge": True,
            },
            "input_rows": 10,
            "output_rows": 7,
        },
    )
    _write_json(
        root / "datasets" / "coverage_report.json",
        {
            "acceptance": {"coverage_universe_ok": True, "safety_pairs_ge_5_each": True},
            "rows_in": 7,
        },
    )
    _write_json(
        root / "datasets" / "ablation_report.json",
        {"acceptance": {"ablation_ok": True}, "floor_verdict": "PASS"},
    )
    _write_json(root / "datasets" / "base_diversity_baseline.json", {"distinct_2": 0.12})
    (root / "datasets" / "DATASET_HASH.txt").write_text(
        _compute_datasets_digest(root) + "\n", encoding="utf-8"
    )

    # ---- datasets: anchor ----------------------------------------------------
    write_jsonl(
        root / "datasets" / "anchor" / "anchor_readings.jsonl",
        [
            {"anchor_id": "anchor_0001", "card_id": 0, "orientation": "upright",
             "question_vi": "Hỏi?", "reading_vi": "Một khởi đầu mới."},
            {"anchor_id": "anchor_0002", "card_id": 0, "orientation": "reversed",
             "question_vi": "Hỏi?", "reading_vi": "Cẩn thận bước hụt."},
            {"anchor_id": "anchor_0003", "card_id": 1, "orientation": "upright",
             "question_vi": "Hỏi?", "reading_vi": "Ý chí mạnh mẽ."},
            {"anchor_id": "anchor_0004", "card_id": 1, "orientation": "reversed",
             "question_vi": "Hỏi?", "reading_vi": "Đừng tin lời hứa suông."},
        ],
    )

    # datasets/raw/* deliberately NOT created — catalog/stats tests verify
    # present-if-exists / missing-file handling against these.
    return root


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A complete tiny fake repo tree; never touches real kb/ or datasets/."""
    return build_fake_root(tmp_path)


@pytest.fixture(autouse=True)
def _reset_webapp_state():
    """Drop module-level caches / runs state after every test.

    The catalog/stats caches and the runs single-flight globals live at module
    scope across the whole pytest process — without a reset, a test that
    rewrote a fixture file or started a fake run would poison the next test.
    """
    yield
    from tfvn.reading_stream import _SESSIONS as _READING_SESSIONS
    from tfvn.webapp import catalog, runs, stats

    catalog.invalidate()
    stats.invalidate()
    runs._CURRENT = None
    runs._HANDLES = {}
    runs._RECONCILED = False
    runs._ORPHANED = []
    runs._HISTORY = []
    _READING_SESSIONS.clear()


def make_client(router) -> Any:
    """A TestClient over just *router* (no lifespan, no static mount)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)
