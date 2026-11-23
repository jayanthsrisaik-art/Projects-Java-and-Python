"""Load a GTFS directory, validate it, and produce a cleaned, interpolated feed.

Policy:
- Feed-level problems (missing file/column, bad or mixed timezone) are fatal:
  `load_feed` raises FeedError.
- Row-level problems are reported as Issues with file and line number. A bad
  trip is quarantined (left out of the output, never half-loaded). Bad calendar
  rows are dropped, which in turn quarantines trips that only referenced them.
- Warnings never drop data.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .timeutil import TimeFormatError, parse_gtfs_date, parse_gtfs_time

REQUIRED = {
    "agency.txt": ["agency_timezone"],
    "stops.txt": ["stop_id", "stop_name"],
    "routes.txt": ["route_id"],
    "trips.txt": ["route_id", "service_id", "trip_id"],
    "stop_times.txt": ["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"],
}
OPTIONAL = {
    "calendar.txt": ["service_id", "monday", "tuesday", "wednesday", "thursday",
                     "friday", "saturday", "sunday", "start_date", "end_date"],
    "calendar_dates.txt": ["service_id", "date", "exception_type"],
    "frequencies.txt": ["trip_id", "start_time", "end_time", "headway_secs"],
}
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class FeedError(Exception):
    pass


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" | "warning"
    file: str
    line: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity.upper():7} {self.file}:{self.line} [{self.code}] {self.message}"


@dataclass
class StopTime:
    stop_sequence: int
    stop_id: str
    arrival_s: int
    departure_s: int
    interpolated: bool


@dataclass
class Trip:
    trip_id: str
    route_id: str
    service_id: str
    headsign: str
    direction_id: str
    stop_times: list[StopTime] = field(default_factory=list)


@dataclass
class Calendar:
    service_id: str
    days: tuple[int, ...]  # Monday=0 .. Sunday=6, values 0/1
    start: date
    end: date


@dataclass
class Frequency:
    trip_id: str
    start_s: int
    end_s: int
    headway_s: int
    exact_times: int


@dataclass
class Feed:
    timezone: str
    stops: dict[str, dict]
    routes: dict[str, dict]
    trips: dict[str, Trip]
    calendars: dict[str, Calendar]
    calendar_dates: dict[tuple[str, date], int]  # (service_id, date) -> 1 add / 2 remove
    frequencies: list[Frequency]
    issues: list[Issue]
    quarantined_trips: set[str]

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def max_time_s(self) -> int:
        times = [st.departure_s for t in self.trips.values() for st in t.stop_times]
        extra = max((f.end_s for f in self.frequencies), default=0)
        span = max((t.stop_times[-1].arrival_s - t.stop_times[0].departure_s
                    for t in self.trips.values() if t.stop_times), default=0)
        return max(max(times, default=0), extra + span)


def _read(path: str, fname: str, required_cols: list[str]) -> list[dict]:
    full = os.path.join(path, fname)
    with open(full, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise FeedError(f"{fname}: empty file") from None
        missing = [c for c in required_cols if c not in header]
        if missing:
            raise FeedError(f"{fname}: missing required column(s) {missing}")
        rows = []
        for lineno, raw in enumerate(reader, start=2):
            if not any(cell.strip() for cell in raw):
                continue
            raw = raw + [""] * (len(header) - len(raw))
            row = {h: raw[i].strip() for i, h in enumerate(header)}
            row["_line"] = lineno
            rows.append(row)
        return rows


def load_feed(path: str) -> Feed:
    for fname, cols in REQUIRED.items():
        if not os.path.exists(os.path.join(path, fname)):
            raise FeedError(f"missing required file {fname}")
    tables = {f: _read(path, f, c) for f, c in REQUIRED.items()}
    for fname, cols in OPTIONAL.items():
        tables[fname] = _read(path, fname, cols) if os.path.exists(os.path.join(path, fname)) else []
    if not tables["calendar.txt"] and not tables["calendar_dates.txt"]:
        raise FeedError("feed needs calendar.txt and/or calendar_dates.txt")

    issues: list[Issue] = []

    def err(file, line, code, msg):
        issues.append(Issue("error", file, line, code, msg))

    def warn(file, line, code, msg):
        issues.append(Issue("warning", file, line, code, msg))

    # --- agency timezone (fatal if wrong) ---
    tzs = {r["agency_timezone"] for r in tables["agency.txt"]}
    if len(tzs) != 1:
        raise FeedError(f"agency.txt: need exactly one agency_timezone, found {sorted(tzs)}")
    tzname = tzs.pop()
    try:
        ZoneInfo(tzname)
    except (ZoneInfoNotFoundError, ValueError):
        raise FeedError(f"agency.txt: unknown timezone {tzname!r}") from None

    # --- simple keyed tables ---
    def keyed(fname, key):
        out = {}
        for r in tables[fname]:
            k = r[key]
            if not k:
                err(fname, r["_line"], "empty_key", f"empty {key}")
            elif k in out:
                err(fname, r["_line"], "duplicate_key", f"duplicate {key} {k!r} (first on line {out[k]['_line']})")
            else:
                out[k] = r
        return out

    stops = keyed("stops.txt", "stop_id")
    routes = keyed("routes.txt", "route_id")

    # --- calendar ---
    calendars: dict[str, Calendar] = {}
    for r in tables["calendar.txt"]:
        sid, line = r["service_id"], r["_line"]
        try:
            days = tuple(int(r[d]) for d in DAYS)
            if any(v not in (0, 1) for v in days):
                raise ValueError("day flags must be 0 or 1")
            start, end = parse_gtfs_date(r["start_date"]), parse_gtfs_date(r["end_date"])
        except ValueError as e:
            err("calendar.txt", line, "bad_calendar", f"service {sid!r}: {e}")
            continue
        if start > end:
            err("calendar.txt", line, "inverted_range", f"service {sid!r}: start_date after end_date")
            continue
        if sid in calendars:
            err("calendar.txt", line, "duplicate_key", f"duplicate service_id {sid!r}")
            continue
        if not any(days):
            warn("calendar.txt", line, "no_weekdays", f"service {sid!r} has no active weekdays")
        calendars[sid] = Calendar(sid, days, start, end)

    cal_dates: dict[tuple[str, date], int] = {}
    cd_services: set[str] = set()
    for r in tables["calendar_dates.txt"]:
        sid, line = r["service_id"], r["_line"]
        try:
            d = parse_gtfs_date(r["date"])
            et = int(r["exception_type"])
            if et not in (1, 2):
                raise ValueError("exception_type must be 1 or 2")
        except ValueError as e:
            err("calendar_dates.txt", line, "bad_exception", f"service {sid!r}: {e}")
            continue
        if (sid, d) in cal_dates:
            err("calendar_dates.txt", line, "duplicate_key",
                f"service {sid!r} has more than one exception on {d.isoformat()}")
            continue
        cal_dates[(sid, d)] = et
        cd_services.add(sid)
        if et == 2 and sid not in calendars:
            warn("calendar_dates.txt", line, "remove_without_calendar",
                 f"service {sid!r} removes {d.isoformat()} but has no calendar.txt row")
    known_services = set(calendars) | cd_services

    # --- trips ---
    trips: dict[str, Trip] = {}
    quarantined: set[str] = set()
    trip_lines: dict[str, int] = {}
    for r in tables["trips.txt"]:
        tid, line = r["trip_id"], r["_line"]
        if not tid:
            err("trips.txt", line, "empty_key", "empty trip_id")
            continue
        if tid in trips or tid in quarantined:
            err("trips.txt", line, "duplicate_key", f"duplicate trip_id {tid!r}")
            quarantined.add(tid)
            trips.pop(tid, None)
            continue
        trip_lines[tid] = line
        if r["route_id"] not in routes:
            err("trips.txt", line, "unknown_route", f"trip {tid!r}: unknown route_id {r['route_id']!r}")
            quarantined.add(tid)
            continue
        if r["service_id"] not in known_services:
            err("trips.txt", line, "unknown_service", f"trip {tid!r}: unknown service_id {r['service_id']!r}")
            quarantined.add(tid)
            continue
        trips[tid] = Trip(tid, r["route_id"], r["service_id"], r.get("trip_headsign", ""),
                          r.get("direction_id", ""))

    # --- stop_times ---
    raw_st: dict[str, list[dict]] = {}
    for r in tables["stop_times.txt"]:
        tid, line = r["trip_id"], r["_line"]
        if tid not in trips:
            if tid not in quarantined:
                err("stop_times.txt", line, "unknown_trip", f"unknown trip_id {tid!r}")
            continue
        raw_st.setdefault(tid, []).append(r)

    for tid, rows in raw_st.items():
        bad = _build_stop_times(trips[tid], rows, stops, err, warn)
        if bad:
            quarantined.add(tid)
    for tid, trip in list(trips.items()):
        if tid in quarantined:
            del trips[tid]
        elif not trip.stop_times:
            warn("trips.txt", trip_lines[tid], "no_stop_times", f"trip {tid!r} has no stop_times; skipped")
            del trips[tid]

    # --- frequencies ---
    freqs: list[Frequency] = []
    by_trip: dict[str, list[Frequency]] = {}
    for r in tables["frequencies.txt"]:
        tid, line = r["trip_id"], r["_line"]
        if tid not in trips:
            if tid not in quarantined:
                err("frequencies.txt", line, "unknown_trip", f"unknown or skipped trip_id {tid!r}")
            continue
        try:
            s, e = parse_gtfs_time(r["start_time"]), parse_gtfs_time(r["end_time"])
            h = int(r["headway_secs"])
            exact = int(r.get("exact_times") or 0)
        except (TimeFormatError, ValueError) as ex:
            err("frequencies.txt", line, "bad_frequency", f"trip {tid!r}: {ex}")
            continue
        if h <= 0 or s >= e or exact not in (0, 1):
            err("frequencies.txt", line, "bad_frequency",
                f"trip {tid!r}: need headway_secs > 0, start_time < end_time, exact_times in (0,1)")
            continue
        f = Frequency(tid, s, e, h, exact)
        for other in by_trip.get(tid, []):
            if s < other.end_s and other.start_s < e:
                err("frequencies.txt", line, "overlapping_frequency",
                    f"trip {tid!r}: window overlaps another frequencies row")
                break
        else:
            by_trip.setdefault(tid, []).append(f)
            freqs.append(f)

    return Feed(tzname, stops, routes, trips, calendars, cal_dates, freqs, issues, quarantined)


def _build_stop_times(trip: Trip, rows: list[dict], stops, err, warn) -> bool:
    """Parse, sort, check and interpolate one trip. Returns True if the trip is bad."""
    F = "stop_times.txt"
    parsed = []
    bad = False
    seen_seq: dict[int, int] = {}
    for r in rows:
        line = r["_line"]
        try:
            seq = int(r["stop_sequence"])
        except ValueError:
            err(F, line, "bad_sequence", f"trip {trip.trip_id!r}: stop_sequence {r['stop_sequence']!r}")
            bad = True
            continue
        if seq in seen_seq:
            err(F, line, "duplicate_key", f"trip {trip.trip_id!r}: duplicate stop_sequence {seq}")
            bad = True
            continue
        seen_seq[seq] = line
        if r["stop_id"] not in stops:
            err(F, line, "unknown_stop", f"trip {trip.trip_id!r}: unknown stop_id {r['stop_id']!r}")
            bad = True
        arr = dep = None
        try:
            if r["arrival_time"]:
                arr = parse_gtfs_time(r["arrival_time"])
            if r["departure_time"]:
                dep = parse_gtfs_time(r["departure_time"])
        except TimeFormatError as e:
            err(F, line, "bad_time", f"trip {trip.trip_id!r}: {e}")
            bad = True
            continue
        if (arr is None) != (dep is None):
            warn(F, line, "half_timepoint",
                 f"trip {trip.trip_id!r} seq {seq}: only one of arrival/departure given; copied to the other")
            arr = dep if arr is None else arr
            dep = arr if dep is None else dep
        if arr is not None and dep < arr:
            err(F, line, "departs_before_arrival", f"trip {trip.trip_id!r} seq {seq}: departure before arrival")
            bad = True
        dist = r.get("shape_dist_traveled", "")
        try:
            dist_v = float(dist) if dist else None
        except ValueError:
            warn(F, line, "bad_distance", f"trip {trip.trip_id!r} seq {seq}: shape_dist_traveled {dist!r} ignored")
            dist_v = None
        parsed.append((seq, r["stop_id"], arr, dep, dist_v, line))
    if bad or not parsed:
        return bad
    parsed.sort(key=lambda p: p[0])
    first, last = parsed[0], parsed[-1]
    for p in (first, last):
        if p[2] is None:
            err(F, p[5], "missing_endpoint_time",
                f"trip {trip.trip_id!r}: first and last stop must have times (seq {p[0]})")
            return True
    # monotonic check over timepoints
    prev = None
    for p in parsed:
        if p[2] is None:
            continue
        if prev is not None and p[2] < prev[3]:
            err(F, p[5], "time_travel",
                f"trip {trip.trip_id!r}: seq {p[0]} arrives before seq {prev[0]} departs")
            return True
        prev = p
    trip.stop_times = _interpolate(parsed)
    return False


def _interpolate(parsed) -> list[StopTime]:
    """Fill blank times between timepoints, by distance if available, else by stop count."""
    out: list[StopTime] = []
    idx_tp = [i for i, p in enumerate(parsed) if p[2] is not None]
    for a, b in zip(idx_tp, idx_tp[1:]):
        pa, pb = parsed[a], parsed[b]
        if not out:
            out.append(StopTime(pa[0], pa[1], pa[2], pa[3], False))
        t0, t1 = pa[3], pb[2]
        dists = [parsed[k][4] for k in range(a, b + 1)]
        use_dist = all(d is not None for d in dists) and all(x < y for x, y in zip(dists, dists[1:]))
        for k in range(a + 1, b):
            if use_dist:
                frac = (dists[k - a] - dists[0]) / (dists[-1] - dists[0])
            else:
                frac = (k - a) / (b - a)
            t = int(round(t0 + frac * (t1 - t0)))
            out.append(StopTime(parsed[k][0], parsed[k][1], t, t, True))
        out.append(StopTime(pb[0], pb[1], pb[2], pb[3], False))
    if not out:  # single timepoint trip
        p = parsed[0]
        out.append(StopTime(p[0], p[1], p[2], p[3], False))
    return out
