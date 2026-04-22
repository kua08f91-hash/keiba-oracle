"""Timezone helpers — single source of truth for JST/UTC handling.

Rules:
  - DB storage: always UTC (now_utc())
  - Schedule/display: always JST (now_jst())
  - Convert at boundaries (to_jst())
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9), name="JST")


def now_jst() -> datetime:
    """Current time in JST (aware)."""
    return datetime.now(JST)


def now_utc() -> datetime:
    """Current time in UTC (aware). For DB storage."""
    return datetime.now(timezone.utc)


def to_jst(dt: datetime) -> datetime:
    """Convert naive (assumed UTC) or aware datetime to JST."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(JST)
