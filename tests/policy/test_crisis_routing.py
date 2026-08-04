"""Crisis routing: all 168 weekly slots, boundaries, staleness, named QA cases."""

from __future__ import annotations

import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "policy"))

import crisis_routing  # noqa: E402
from crisis_routing import HOTLINES, route_crisis  # noqa: E402

PRIMARY_PHONE = "096 306 1414"
REF_DATE = datetime(2026, 8, 4)  # Tuesday, well inside the staleness window


def _expected(weekday: int, hour: int) -> str:
    if weekday in {2, 3, 4, 5, 6} and time(13, 0) <= time(hour, 0) <= time(20, 30):
        return "primary_open"
    return "closed_fallback"


def test_all_168_weekly_slots_both_branches():
    for day_offset in range(7):
        for hour in range(24):
            dt = (REF_DATE + timedelta(days=day_offset)).replace(hour=hour)
            decision = route_crisis(dt)
            expected = _expected(dt.weekday(), hour)
            assert decision.routing_mode == expected, (
                f"slot={dt.isoformat()} weekday={dt.weekday()} got={decision.routing_mode}"
            )
            if expected == "primary_open":
                assert decision.primary_open is True
                assert decision.primary_line_phone == PRIMARY_PHONE
            else:
                assert decision.primary_open is False
                assert decision.primary_line_phone is None
                assert "115" in decision.fallback_message_vi
                assert "13:00" in decision.fallback_message_vi


def test_boundary_13_00_and_20_30_open():
    for hour, minute in ((13, 0), (20, 30)):
        decision = route_crisis(datetime(2026, 8, 5, hour, minute))  # Wednesday
        assert decision.routing_mode == "primary_open", (hour, minute)
        assert decision.primary_line_phone == PRIMARY_PHONE


def test_boundary_12_59_and_20_31_closed():
    for hour, minute in ((12, 59), (20, 31)):
        decision = route_crisis(datetime(2026, 8, 5, hour, minute))  # Wednesday
        assert decision.routing_mode == "closed_fallback", (hour, minute)
        assert decision.primary_open is False
        assert decision.primary_line_phone is None
        assert "115" in decision.fallback_message_vi


def test_closed_days_never_open_even_within_open_hours():
    for dt in (datetime(2026, 8, 3, 13, 0), datetime(2026, 8, 4, 20, 30)):
        assert route_crisis(dt).routing_mode == "closed_fallback"


def test_aware_utc_datetime_normalised_to_hcm():
    utc = datetime(2026, 8, 5, 7, 0, tzinfo=timezone.utc)  # = Wed 14:00 +07:00
    decision = route_crisis(utc)
    assert decision.routing_mode == "primary_open"
    assert decision.primary_line_phone == PRIMARY_PHONE


def _verified():
    return HOTLINES["ngay_mai"]["verified_date"]


def test_stale_date_fails_closed_on_open_slot():
    stale = datetime.combine(_verified() + timedelta(days=92), time(14, 0))  # Wednesday
    decision = route_crisis(stale)
    assert decision.routing_mode == "stale_fails_closed"
    assert decision.primary_open is False
    assert decision.primary_line_phone is None
    assert "115" in decision.fallback_message_vi


def test_stale_date_fails_closed_on_closed_slot():
    stale = datetime.combine(_verified() + timedelta(days=91), time(3, 0))  # Tuesday
    decision = route_crisis(stale)
    assert decision.routing_mode == "stale_fails_closed"


def test_exactly_at_90_day_cutoff_is_not_stale():
    at_cutoff = datetime.combine(_verified() + timedelta(days=90), time(14, 0))
    assert route_crisis(at_cutoff).routing_mode != "stale_fails_closed"


def test_qa_tuesday_0300_closed_fallback():
    decision = route_crisis(datetime(2026, 8, 4, 3, 0))  # Tuesday 03:00
    assert decision.routing_mode == "closed_fallback"
    assert decision.primary_open is False
    assert decision.primary_line_phone is None
    assert "115" in decision.fallback_message_vi


def test_qa_wednesday_1400_primary_open():
    decision = route_crisis(datetime(2026, 8, 5, 14, 0))  # Wednesday 14:00
    assert decision.routing_mode == "primary_open"
    assert decision.primary_open is True
    assert decision.primary_line_phone == PRIMARY_PHONE


def test_qa_stale_date_fails_closed():
    stale = datetime.combine(_verified() + timedelta(days=92), time(14, 0))
    decision = route_crisis(stale)
    assert decision.routing_mode == "stale_fails_closed"
