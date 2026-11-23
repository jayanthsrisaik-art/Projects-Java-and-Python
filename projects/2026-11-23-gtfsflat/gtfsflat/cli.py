"""Command line interface: validate, build, board, summary, export, synth."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime

from .board import CoverageError, departures, store_tz
from .expand import build, connect
from .feed import FeedError, load_feed
from .synth import generate
from .timeutil import epoch_to_local, epoch_to_utc_iso, format_gtfs_time


def _date(s: str) -> date:
    return date.fromisoformat(s)


def cmd_validate(a) -> int:
    try:
        feed = load_feed(a.feed)
    except FeedError as e:
        print(f"FATAL {e}")
        return 2
    if a.json:
        print(json.dumps([i.__dict__ for i in feed.issues], indent=2))
    else:
        for i in feed.issues:
            print(i)
        print(f"{len(feed.errors)} error(s), {len(feed.issues) - len(feed.errors)} warning(s); "
              f"{len(feed.trips)} trips usable, {len(feed.quarantined_trips)} quarantined")
    return 1 if feed.errors else 0


def cmd_build(a) -> int:
    try:
        feed = load_feed(a.feed)
    except FeedError as e:
        print(f"FATAL {e}")
        return 2
    for i in feed.issues:
        print(i, file=sys.stderr)
    if feed.errors and a.strict:
        print(f"{len(feed.errors)} error(s); --strict set, nothing built")
        return 1
    conn, counts = build(feed, a.db, _date(a.start), _date(a.end))
    conn.close()
    print(json.dumps({"db": a.db, "start": a.start, "end": a.end, **counts,
                      "quarantined_trips": sorted(feed.quarantined_trips)}))
    return 0


def _parse_at(conn, text: str) -> datetime:
    at = datetime.fromisoformat(text)
    if at.tzinfo is None:
        at = at.replace(tzinfo=store_tz(conn))  # ambiguous fall-back times resolve to the first (fold=0)
    return at


def cmd_board(a) -> int:
    conn = connect(a.db)
    try:
        rows = departures(conn, a.stop, _parse_at(conn, a.at), a.minutes)
    except CoverageError as e:
        print(f"ERROR {e}")
        return 1
    if a.json:
        print(json.dumps(rows, indent=2))
        return 0
    for r in rows:
        flag = " ~" if r["frequency_estimated"] else ""
        print(f"{r['departure_local'][11:]}  {r['route_id']:<4} {r['headsign']:<12} "
              f"trip {r['trip_id']} (service {r['service_date']}){flag}")
    if not rows:
        print("no departures")
    return 0


def cmd_summary(a) -> int:
    conn = connect(a.db)
    tz = store_tz(conn)
    rows = conn.execute("SELECT * FROM daily_route_summary WHERE service_date = ? ORDER BY route_id",
                        (a.date,)).fetchall()
    print(f"{'route':<6}{'trips':>6}  {'first departure':<26}{'last arrival':<26}{'interp':>7}{'freq~':>7}")
    for _, route, trips, first, last, interp, est in rows:
        print(f"{route:<6}{trips:>6}  {epoch_to_local(first, tz).isoformat():<26}"
              f"{epoch_to_local(last, tz).isoformat():<26}{interp:>7}{est:>7}")
    return 0


def cmd_export(a) -> int:
    conn = connect(a.db)
    tz = store_tz(conn)
    q = """SELECT service_date, instance_id, route_id, stop_sequence, stop_id, arrival_s, departure_s,
                  arrival_epoch, departure_epoch, interpolated, frequency_estimated
           FROM stop_event WHERE service_date BETWEEN ? AND ? ORDER BY service_date, instance_id, stop_sequence"""
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["service_date", "instance_id", "route_id", "stop_sequence", "stop_id",
                    "gtfs_arrival", "gtfs_departure", "arrival_utc", "departure_utc", "departure_local",
                    "interpolated", "frequency_estimated"])
        n = 0
        for r in conn.execute(q, (a.start, a.end)):
            w.writerow([r[0], r[1], r[2], r[3], r[4], format_gtfs_time(r[5]), format_gtfs_time(r[6]),
                        epoch_to_utc_iso(r[7]), epoch_to_utc_iso(r[8]), epoch_to_local(r[8], tz).isoformat(), r[9], r[10]])
            n += 1
    print(f"wrote {n} events to {a.out}")
    return 0


def cmd_synth(a) -> int:
    print(json.dumps(generate(a.out, a.seed)))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gtfsflat", description="GTFS feed -> validated, timezone-correct stop events")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("validate", help="report feed problems with file:line")
    s.add_argument("feed")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_validate)
    s = sub.add_parser("build", help="expand service dates [start, end] into a SQLite store")
    s.add_argument("feed")
    s.add_argument("--db", required=True)
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--strict", action="store_true", help="refuse to build if the feed has errors")
    s.set_defaults(fn=cmd_build)
    s = sub.add_parser("board", help="departures from a stop at a local or offset-qualified time")
    s.add_argument("--db", required=True)
    s.add_argument("--stop", required=True)
    s.add_argument("--at", required=True, help="e.g. 2026-11-01T01:30 (agency time) or with +HH:MM offset")
    s.add_argument("--minutes", type=int, default=60)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_board)
    s = sub.add_parser("summary", help="per-route trips and service span for one service date")
    s.add_argument("--db", required=True)
    s.add_argument("--date", required=True)
    s.set_defaults(fn=cmd_summary)
    s = sub.add_parser("export", help="write stop events to CSV")
    s.add_argument("--db", required=True)
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("synth", help="write a seeded synthetic feed")
    s.add_argument("out")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
