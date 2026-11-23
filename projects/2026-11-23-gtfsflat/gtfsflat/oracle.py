"""Row-at-a-time reference expansion, plus the naive shortcuts used as baselines.

`expand_python(feed, start, end)` with default flags is an independent
implementation of the same rules as the SQL engine (per-date loops, datetime
arithmetic instead of epoch joins) and serves as the oracle in tests.

Each flag switches on one common shortcut seen in hand-rolled GTFS loaders:
  ignore_exceptions  use calendar.txt only, ignore calendar_dates.txt
  wrap_24h           read 25:10:00 as 01:10 on the *same* date (hour % 24)
  wallclock          localise HH:MM:SS as wall time on the service date
                     (with day rollover) instead of noon-minus-12h elapsed time
  ignore_frequencies run each frequency template trip once, as a normal trip
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from .feed import Feed
from .timeutil import daterange, format_gtfs_time, service_day_reference


@dataclass(frozen=True)
class Shortcuts:
    ignore_exceptions: bool = False
    wrap_24h: bool = False
    wallclock: bool = False
    ignore_frequencies: bool = False


CORRECT = Shortcuts()


def service_active(feed: Feed, service_id: str, d: date, ignore_exceptions: bool = False) -> bool:
    if not ignore_exceptions:
        et = feed.calendar_dates.get((service_id, d))
        if et == 1:
            return True
        if et == 2:
            return False
    cal = feed.calendars.get(service_id)
    return bool(cal and cal.start <= d <= cal.end and cal.days[d.weekday()])


def _instant(d: date, seconds: int, feed: Feed, sc: Shortcuts) -> int:
    tz = feed.tz
    if sc.wrap_24h:
        seconds = seconds % 86400
    if sc.wallclock:
        days, rem = divmod(seconds, 86400)
        wall = datetime.combine(d + timedelta(days=days), time()) + timedelta(seconds=rem)
        return int(wall.replace(tzinfo=tz).timestamp())
    return int((service_day_reference(d, tz) + timedelta(seconds=seconds)).timestamp())


def expand_python(feed: Feed, start: date, end: date, sc: Shortcuts = CORRECT) -> list[tuple]:
    """Return sorted (instance_id, stop_sequence, stop_id, arrival_epoch, departure_epoch)."""
    freq_by_trip: dict[str, list] = {}
    if not sc.ignore_frequencies:
        for f in feed.frequencies:
            freq_by_trip.setdefault(f.trip_id, []).append(f)
    out = []
    for d in daterange(start, end):
        for trip in feed.trips.values():
            if not service_active(feed, trip.service_id, d, sc.ignore_exceptions):
                continue
            if trip.trip_id in freq_by_trip:
                base = trip.stop_times[0].departure_s
                instances = []
                for f in freq_by_trip[trip.trip_id]:
                    s = f.start_s
                    while s < f.end_s:
                        instances.append((f"{trip.trip_id}@{d.isoformat()}+{format_gtfs_time(s)}", s - base))
                        s += f.headway_s
            else:
                instances = [(f"{trip.trip_id}@{d.isoformat()}", 0)]
            for iid, off in instances:
                for st in trip.stop_times:
                    out.append((iid, st.stop_sequence, st.stop_id,
                                _instant(d, st.arrival_s + off, feed, sc),
                                _instant(d, st.departure_s + off, feed, sc)))
    out.sort()
    return out


def compare(expected: list[tuple], actual: list[tuple]) -> dict:
    """Event-level diff keyed by (instance_id, stop_sequence)."""
    exp = {(r[0], r[1]): r[2:] for r in expected}
    act = {(r[0], r[1]): r[2:] for r in actual}
    common = exp.keys() & act.keys()
    wrong = sum(1 for k in common if exp[k] != act[k])
    return {
        "expected": len(exp),
        "produced": len(act),
        "exact": len(common) - wrong,
        "wrong_time": wrong,
        "missing": len(exp.keys() - act.keys()),
        "extra": len(act.keys() - exp.keys()),
    }


def utc(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)
