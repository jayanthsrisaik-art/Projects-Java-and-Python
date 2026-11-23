"""Departure board queries over the built stop_event table.

A board for a real instant must look at every service date whose events can
reach that instant, not just the calendar date of the query: at 00:40 on a
Saturday, the 24:40:00 trip belongs to Friday's service date. Queries are by
absolute instant (departure_epoch), and the needed service dates are checked
against what has been built so a missing day is an error, not a silent gap.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .expand import built_dates
from .timeutil import epoch_to_local, reference_epoch


class CoverageError(Exception):
    pass


def store_tz(conn: sqlite3.Connection) -> ZoneInfo:
    return ZoneInfo(conn.execute("SELECT value FROM meta WHERE key='timezone'").fetchone()[0])


def required_service_dates(conn: sqlite3.Connection, at: datetime, minutes: int) -> list[date]:
    """Service dates whose events can fall in [at, at + minutes): those whose
    reference instant is before the window end and whose latest event
    (reference + max_time_s) is not before the window start."""
    tz = store_tz(conn)
    max_s = int(conn.execute("SELECT value FROM meta WHERE key='max_time_s'").fetchone()[0])
    lo = int(at.timestamp())
    hi = lo + minutes * 60
    d = at.astimezone(tz).date() - timedelta(days=math.ceil(max_s / 86400) + 1)
    last = (at + timedelta(minutes=minutes)).astimezone(tz).date() + timedelta(days=1)
    out = []
    while d <= last:
        ref = reference_epoch(d, tz)
        if ref < hi and ref + max_s >= lo:
            out.append(d)
        d += timedelta(days=1)
    return out


def departures(conn: sqlite3.Connection, stop_id: str, at: datetime, minutes: int = 60,
               same_date_only: bool = False, check_coverage: bool = True) -> list[dict]:
    """Departures from stop_id in [at, at + minutes). `at` must be timezone-aware.

    same_date_only=True reproduces the common bug of filtering on
    service_date == local date of the query (used as a baseline).
    """
    if at.tzinfo is None:
        raise ValueError("'at' must be timezone-aware")
    tz = store_tz(conn)
    if check_coverage:
        missing = [d for d in required_service_dates(conn, at, minutes) if d not in built_dates(conn)]
        if missing:
            raise CoverageError("service dates not built: " + ", ".join(d.isoformat() for d in missing))
    lo = int(at.timestamp())
    hi = lo + minutes * 60
    q = """SELECT e.departure_epoch, e.route_id, t.headsign, e.trip_id, e.instance_id, e.service_date,
                  e.stop_sequence, e.frequency_estimated
           FROM stop_event e JOIN stg_trip t ON t.trip_id = e.trip_id
           WHERE e.stop_id = ? AND e.is_last = 0 AND e.departure_epoch >= ? AND e.departure_epoch < ?"""
    params: list = [stop_id, lo, hi]
    if same_date_only:
        q += " AND e.service_date = ?"
        params.append(at.astimezone(tz).date().isoformat())
    rows = conn.execute(q + " ORDER BY e.departure_epoch, e.instance_id", params).fetchall()
    return [
        {
            "departure_local": epoch_to_local(r[0], tz).isoformat(),
            "departure_epoch": r[0],
            "route_id": r[1],
            "headsign": r[2],
            "trip_id": r[3],
            "instance_id": r[4],
            "service_date": r[5],
            "stop_sequence": r[6],
            "frequency_estimated": bool(r[7]),
        }
        for r in rows
    ]
