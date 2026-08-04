"""Time-aware crisis routing for tfvn-tarot.

Deterministic over the local Asia/Ho_Chi_Minh clock: the caller passes the
current local time explicitly; this module never reads the wall clock.

Hotline source (verified 2026-08-04):
  - duongdaynongngaymai.vn (official): "(13h00 - 20h30) Thứ 4, Thứ 5, Thứ 6,
    Thứ 7 & Chủ Nhật" -> open Wed-Sun, closed Mon-Tue.
  - svvn.tienphong.vn (Tiền Phong, 2021-05-30): confirms 13:00-20:30 hours.
  - ketoananpha.vn hotline roundup.
Emergency: 115 (cấp cứu y tế), 24/7, always valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

TZ_HCM = ZoneInfo("Asia/Ho_Chi_Minh")

# Weekday is Monday=0 ... Sunday=6 (Python datetime convention). A wrong reading
# of this convention would route a suicidal user to a dead line, so it is stated
# once here and everywhere the table is read.
OPEN_DAYS = {2, 3, 4, 5, 6}  # Wednesday..Sunday

STALENESS_WINDOW_DAYS = 90
PRIMARY_LINE_ID = "ngay_mai"

FALLBACK_MESSAGE_VI = (
    "Nếu đây là tình huống khẩn cấp, hãy gọi ngay 115 "
    "(cấp cứu y tế, hoạt động 24/7). "
    "Đường dây nóng Ngày mai (096 306 1414) hiện ngoài giờ hỗ trợ; "
    "đường dây mở lại lúc 13:00."
)

HOTLINES: dict[str, dict] = {
    "ngay_mai": {
        "phone": "096 306 1414",
        "name_vi": "Đường dây nóng Ngày mai",
        "open_days": OPEN_DAYS,
        "open_start": time(13, 0),
        "open_end": time(20, 30),
        "source": (
            "duongdaynongngaymai.vn (official); "
            "svvn.tienphong.vn (Tiền Phong 2021-05-30); ketoananpha.vn roundup"
        ),
        "verified_date": date(2026, 8, 4),
    },
}


@dataclass(frozen=True)
class RoutingDecision:
    primary_line_phone: str | None
    primary_open: bool
    fallback_message_vi: str
    routing_mode: str


def _localize(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TZ_HCM)
    return dt.astimezone(TZ_HCM)


def route_crisis(dt_naive_local: datetime) -> RoutingDecision:
    """Route to the primary hotline or fail closed.

    A naive datetime is interpreted as Asia/Ho_Chi_Minh local time; an aware
    datetime is converted to it. Routing fails closed once the local date is
    past the line's verified date plus STALENESS_WINDOW_DAYS, regardless of
    day or hour.
    """
    local = _localize(dt_naive_local)
    line = HOTLINES[PRIMARY_LINE_ID]
    if local.date() > line["verified_date"] + timedelta(days=STALENESS_WINDOW_DAYS):
        return RoutingDecision(
            primary_line_phone=None,
            primary_open=False,
            fallback_message_vi=FALLBACK_MESSAGE_VI,
            routing_mode="stale_fails_closed",
        )
    local_time = local.time()
    open_now = (
        local.weekday() in line["open_days"]
        and line["open_start"] <= local_time <= line["open_end"]
    )
    if open_now:
        return RoutingDecision(
            primary_line_phone=line["phone"],
            primary_open=True,
            fallback_message_vi=FALLBACK_MESSAGE_VI,
            routing_mode="primary_open",
        )
    return RoutingDecision(
        primary_line_phone=None,
        primary_open=False,
        fallback_message_vi=FALLBACK_MESSAGE_VI,
        routing_mode="closed_fallback",
    )
