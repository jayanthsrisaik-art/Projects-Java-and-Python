"""UTC timestamp helpers. Everything internal is integer epoch seconds."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .config import HOUR


def parse_ts(value: str | int | float) -> int:
    """Parse an ISO-8601 timestamp (``Z`` or offset) or epoch seconds to epoch seconds."""
    if isinstance(value, (int, float)):
        return int(value)
    s = value.strip()
    if s.lstrip("-").isdigit():
        return int(s)
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {value!r}")
    return int(dt.timestamp())


def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def floor_hour(ts: int) -> int:
    return int(ts) // HOUR * HOUR


def ceil_hour(ts: int) -> int:
    return -(-int(ts) // HOUR) * HOUR


def hour_of_day(h0: int, nh: int) -> np.ndarray:
    return ((h0 // HOUR + np.arange(nh)) % 24).astype(np.int64)


def day_of_week(h0: int, nh: int) -> np.ndarray:
    """Monday=0. 1970-01-01 was a Thursday (3)."""
    days = (h0 // HOUR + np.arange(nh)) // 24
    return ((days + 3) % 7).astype(np.int64)


def overlaps(t0: int, t1: int, h0: int, nh: int) -> tuple[np.ndarray, np.ndarray]:
    """Hour indices of the window [h0, h0+nh*3600) overlapped by [t0, t1) and the overlap in seconds."""
    lo = max(t0, h0)
    hi = min(t1, h0 + nh * HOUR)
    if hi <= lo:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    i0 = (lo - h0) // HOUR
    i1 = (hi - 1 - h0) // HOUR
    idx = np.arange(i0, i1 + 1, dtype=np.int64)
    starts = h0 + idx * HOUR
    sec = np.minimum(starts + HOUR, hi) - np.maximum(starts, lo)
    return idx, sec.astype(np.float64)
