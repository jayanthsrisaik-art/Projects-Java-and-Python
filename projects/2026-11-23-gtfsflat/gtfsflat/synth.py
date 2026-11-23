"""Seeded synthetic GTFS feed with the features loaders tend to get wrong.

- weekday / Saturday / Sunday services over calendar year 2026
- holidays where weekday service is removed and Sunday service added
- an event service that exists only in calendar_dates.txt
- a night-owl route with trips after midnight written as 24:xx-26:xx
  and early-morning trips (00:xx-02:xx) that cross DST transitions
- one exact_times=1 and one exact_times=0 frequency route
- non-timepoint stops with blank times and shape_dist_traveled
"""
from __future__ import annotations

import csv
import os
import random

from .timeutil import format_gtfs_time

TZ = "America/New_York"
HOLIDAYS = ["20260101", "20260525", "20260704", "20260907", "20261126", "20261225"]
EVENT_DATES = ["20261128", "20260307"]


def _w(path, name, header, rows):
    with open(os.path.join(path, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def generate(path: str, seed: int = 0) -> dict:
    rng = random.Random(seed)
    os.makedirs(path, exist_ok=True)
    n_stops = 40
    stops = [(f"S{i:02d}", f"Stop {i:02d}", f"{40.70 + i * 0.002:.4f}", f"{-74.00 + i * 0.003:.4f}")
             for i in range(1, n_stops + 1)]
    _w(path, "agency.txt", ["agency_id", "agency_name", "agency_url", "agency_timezone"],
       [["A", "Synthetic Transit", "https://example.invalid", TZ]])
    _w(path, "stops.txt", ["stop_id", "stop_name", "stop_lat", "stop_lon"], stops)

    _w(path, "calendar.txt",
       ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "start_date", "end_date"],
       [["WK", 1, 1, 1, 1, 1, 0, 0, "20260101", "20261231"],
        ["SA", 0, 0, 0, 0, 0, 1, 0, "20260101", "20261231"],
        ["SU", 0, 0, 0, 0, 0, 0, 1, "20260101", "20261231"],
        ["DAILY", 1, 1, 1, 1, 1, 1, 1, "20260101", "20261231"]])
    cd = []
    for h in HOLIDAYS:
        cd.append(["WK", h, 2])
        cd.append(["SU", h, 1])
    for e in EVENT_DATES:
        cd.append(["EVENT", e, 1])
    _w(path, "calendar_dates.txt", ["service_id", "date", "exception_type"], cd)

    routes, trips, stop_times, freqs = [], [], [], []

    def route_stops(k):
        start = rng.randrange(0, n_stops - k)
        return [s[0] for s in stops[start:start + k]]

    def add_trip(route_id, service_id, tid, seq_stops, dep_s, blank_every=0, headsign=""):
        trips.append([route_id, service_id, tid, headsign or seq_stops[-1], "0"])
        t, dist = dep_s, 0.0
        for i, sid in enumerate(seq_stops):
            if i:
                hop = rng.randint(90, 240)
                t += hop
                dist += hop * rng.uniform(5.0, 9.0)
            dwell = 30 if 0 < i < len(seq_stops) - 1 else 0
            blank = blank_every and 0 < i < len(seq_stops) - 1 and i % blank_every == 0
            arr = "" if blank else format_gtfs_time(t)
            dep = "" if blank else format_gtfs_time(t + dwell)
            stop_times.append([tid, arr, dep, sid, i + 1, f"{dist:.1f}"])
            t += dwell

    # R1: weekday all-day route with non-timepoints
    routes.append(["R1", "A", "1", "Crosstown", 3])
    r1 = route_stops(12)
    for k, dep in enumerate(range(6 * 3600, 23 * 3600, 15 * 60)):
        add_trip("R1", "WK", f"R1_WK_{k:03d}", r1, dep, blank_every=3)
    # R2: weekday + Saturday, different frequencies
    routes.append(["R2", "A", "2", "Harbor", 3])
    r2 = route_stops(10)
    for k, dep in enumerate(range(5 * 3600 + 1800, 24 * 3600 + 1800, 20 * 60)):
        add_trip("R2", "WK", f"R2_WK_{k:03d}", r2, dep)
    for k, dep in enumerate(range(7 * 3600, 23 * 3600, 30 * 60)):
        add_trip("R2", "SA", f"R2_SA_{k:03d}", r2, dep)
    # R3: Sunday (and holiday) service
    routes.append(["R3", "A", "3", "Parkway", 3])
    r3 = route_stops(9)
    for k, dep in enumerate(range(8 * 3600, 22 * 3600, 40 * 60)):
        add_trip("R3", "SU", f"R3_SU_{k:03d}", r3, dep)
    # N1: night owl, every night; late trips written past 24:00 and early ones before 03:00
    routes.append(["N1", "A", "N1", "Night Owl", 3])
    n1 = route_stops(8)
    for k, dep in enumerate(range(23 * 3600 + 20 * 60, 27 * 3600, 30 * 60)):
        add_trip("N1", "DAILY", f"N1_LATE_{k:03d}", n1, dep)
    for k, dep in enumerate(range(10 * 60, 3 * 3600, 30 * 60)):
        add_trip("N1", "DAILY", f"N1_EARLY_{k:03d}", n1, dep)
    # F1: exact-times frequency route (weekday rush); F2: headway-based route (daily)
    routes.append(["F1", "A", "F1", "Rush Shuttle", 3])
    add_trip("F1", "WK", "F1_TEMPLATE", route_stops(6), 7 * 3600)
    freqs.append(["F1_TEMPLATE", "07:00:00", "09:30:00", 360, 1])
    routes.append(["F2", "A", "F2", "Museum Loop", 3])
    add_trip("F2", "DAILY", "F2_TEMPLATE", route_stops(7), 10 * 3600)
    freqs.append(["F2_TEMPLATE", "10:00:00", "13:00:00", 600, 0])
    freqs.append(["F2_TEMPLATE", "13:00:00", "16:00:00", 900, 0])
    # EV: calendar_dates-only event service, runs past midnight
    routes.append(["EV", "A", "EV", "Stadium Express", 3])
    ev = route_stops(5)
    for k, dep in enumerate(range(22 * 3600, 26 * 3600, 20 * 60)):
        add_trip("EV", "EVENT", f"EV_{k:03d}", ev, dep)

    _w(path, "routes.txt", ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"], routes)
    _w(path, "trips.txt", ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id"], trips)
    _w(path, "stop_times.txt",
       ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence", "shape_dist_traveled"],
       stop_times)
    _w(path, "frequencies.txt", ["trip_id", "start_time", "end_time", "headway_secs", "exact_times"], freqs)
    return {"trips": len(trips), "stop_times": len(stop_times), "routes": len(routes), "stops": len(stops)}
