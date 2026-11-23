"""Time handling for GTFS schedules.

GTFS stop times are not wall-clock times. The spec defines them as elapsed time
measured from "noon minus 12h" on the service day, in the agency timezone. That
reference instant is local midnight on every normal day, but it is 23:00 of the
previous day on a spring-forward day and 01:00 on a fall-back day (for zones
that switch at 02:00). Times can also exceed 24:00:00 for trips that run past
midnight but belong to the previous service day.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

_TIME_RE = re.compile(r"^(\d{1,3}):([0-5]\d):([0-5]\d)$")


class TimeFormatError(ValueError):
    pass


def parse_gtfs_time(text: str) -> int:
    """Parse 'H:MM:SS' / 'HH:MM:SS' (hours may be >= 24) into seconds."""
    m = _TIME_RE.match(text.strip())
    if not m:
        raise TimeFormatError(f"not a GTFS time: {text!r}")
    h, mi, s = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s


def format_gtfs_time(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    mi, s = divmod(rem, 60)
    return f"{h:02d}:{mi:02d}:{s:02d}"


def parse_gtfs_date(text: str) -> date:
    t = text.strip()
    if not re.fullmatch(r"\d{8}", t):
        raise ValueError(f"not a GTFS date (YYYYMMDD): {text!r}")
    return date(int(t[:4]), int(t[4:6]), int(t[6:]))


def service_day_reference(service_date: date, tz: ZoneInfo) -> datetime:
    """UTC instant of 'noon minus 12h' on the service date (GTFS reference)."""
    noon = datetime(service_date.year, service_date.month, service_date.day, 12, tzinfo=tz)
    return noon.astimezone(timezone.utc) - timedelta(hours=12)


def reference_epoch(service_date: date, tz: ZoneInfo) -> int:
    return int(service_day_reference(service_date, tz).timestamp())


def epoch_to_local(epoch: int, tz: ZoneInfo) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone(tz)


def epoch_to_utc_iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)
