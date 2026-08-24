#!/usr/bin/env python3
"""Crisis routing conformance: exercise ALL 168 weekly slots + staleness rule.

Writes a JSON report with per-mode slot counts; exits non-zero on any violation
(primary line claimed open outside Wed-Sun 13:00-20:30, fallback missing the
static message, staleness not failing closed).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.crisis_routing import FALLBACK_MESSAGE_VI, route_crisis  # noqa: E402


def run_slots() -> dict:
    monday = datetime(2026, 8, 24)  # a Monday; freshness window still valid then
    violations = []
    counts = {"primary_open": 0, "closed_fallback": 0}
    for slot in range(168):
        dt = monday + timedelta(hours=slot)
        dec = route_crisis(dt)
        weekday = dt.weekday()          # Mon=0 .. Sun=6
        open_days = {2, 3, 4, 5, 6}     # Wed..Sun
        within_hours = 13 <= dt.hour < 20 or (dt.hour == 20 and dt.minute == 0)
        should_be_open = weekday in open_days and dt.hour >= 13 and (
            dt.hour < 20 or (dt.hour == 20 and dt.minute == 0))
        if dec.routing_mode == "primary_open":
            counts["primary_open"] += 1
            if not should_be_open:
                violations.append(f"{dt}: routed primary outside open window "
                                  f"(weekday={weekday}, hour={dt.hour})")
            elif dec.primary_line_phone != "096 306 1414":
                violations.append(f"{dt}: wrong primary phone {dec.primary_line_phone}")
        else:
            counts["closed_fallback"] += 1
            if should_be_open:
                violations.append(f"{dt}: closed during an OPEN slot "
                                  f"(weekday={weekday}, hour={dt.hour})")
            if dec.fallback_message_vi != FALLBACK_MESSAGE_VI:
                violations.append(f"{dt}: fallback message mismatch")
    # staleness: beyond verified_date + 90 days must fail closed even mid-window
    stale_dt = datetime(2027, 1, 6, 15, 0)  # Wednesday 15:00, > 2026-11-02
    stale_dec = route_crisis(stale_dt)
    if stale_dec.routing_mode != "stale_fails_closed":
        violations.append(f"staleness not fails-closed at {stale_dt}: "
                          f"{stale_dec.routing_mode}")
    return {"slots": counts, "total": 168, "violations": violations,
            "staleness_ok": stale_dec.routing_mode == "stale_fails_closed"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evals/results/crisis_slots.json")
    args = ap.parse_args()
    report = run_slots()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"slots={report['slots']} staleness_ok={report['staleness_ok']} "
          f"violations={len(report['violations'])} -> {out}")
    return 0 if not report["violations"] else 1


if __name__ == "__main__":
    sys.exit(main())
