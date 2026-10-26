"""Scan window (Asia/Colombo): 12:00 inclusive – 19:00 exclusive; scans on every 15-minute boundary."""
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import TIMEZONE, TRADING_WEEKDAYS, WINDOW_START, WINDOW_END, SCAN_OFFSET_SEC

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(TIMEZONE)
except Exception:                         # tzdata missing → Sri Lanka has had no DST since 2006
    TZ = timezone(timedelta(hours=5, minutes=30))


def now_local() -> datetime:
    return datetime.now(TZ)


def _aware(now: Optional[datetime]) -> datetime:
    if now is None:
        return now_local()
    if now.tzinfo is None:
        raise ValueError("naive datetime not allowed")
    return now.astimezone(TZ)


def in_window(now: Optional[datetime] = None) -> bool:
    n = _aware(now)
    return n.weekday() in TRADING_WEEKDAYS and WINDOW_START <= n.time() < WINDOW_END


def next_window_start(now: Optional[datetime] = None) -> datetime:
    n = _aware(now)
    for add in range(0, 9):
        d = (n + timedelta(days=add)).date()
        cand = datetime.combine(d, WINDOW_START, tzinfo=TZ)
        if d.weekday() in TRADING_WEEKDAYS and cand > n:
            return cand
    raise RuntimeError("no trading weekday configured")


def secs_to_next_scan(interval_min: int, now: Optional[datetime] = None, offset_sec: int = SCAN_OFFSET_SEC) -> float:
    """Seconds until the next wall-clock multiple of `interval_min` (+offset so the candle is surely closed)."""
    n = _aware(now)
    m = interval_min - (n.minute % interval_min)
    nxt = (n + timedelta(minutes=m)).replace(second=0, microsecond=0) + timedelta(seconds=offset_sec)
    if (nxt - n).total_seconds() > interval_min * 60:          # already past this boundary's offset → fine, next one
        nxt -= timedelta(minutes=interval_min)
    return max((nxt - n).total_seconds(), 1.0)
