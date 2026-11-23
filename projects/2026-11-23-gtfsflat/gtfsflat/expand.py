"""Set-based SQL expansion of a cleaned feed into scheduled stop events.

The cleaned feed is staged into SQLite, then three SQL steps run:
  1. service_dates   calendar weekday ranges + calendar_dates adds - removes
  2. trip_instances  one row per (trip, service date), or one per frequency
                     start for frequency-based trips
  3. stop_event      template stop times shifted by the instance offset and
                     anchored at the service day's "noon minus 12h" instant

Timezone math is the only step SQLite cannot do; it is done once per service
date in Python and joined in as a small lookup table (service_day_ref).
The build is idempotent: rebuilding a date range replaces it in one transaction.
"""
from __future__ import annotations

import sqlite3
from datetime import date

from .feed import Feed
from .timeutil import daterange, reference_epoch

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dim_stop(stop_id TEXT PRIMARY KEY, stop_name TEXT);
CREATE TABLE IF NOT EXISTS dim_route(route_id TEXT PRIMARY KEY, route_short_name TEXT, route_long_name TEXT);
CREATE TABLE IF NOT EXISTS stg_trip(trip_id TEXT PRIMARY KEY, route_id TEXT, service_id TEXT,
    headsign TEXT, direction_id TEXT);
CREATE TABLE IF NOT EXISTS stg_stop_time(trip_id TEXT, stop_sequence INTEGER, stop_id TEXT,
    arrival_s INTEGER, departure_s INTEGER, interpolated INTEGER,
    PRIMARY KEY(trip_id, stop_sequence));
CREATE TABLE IF NOT EXISTS stg_calendar(service_id TEXT PRIMARY KEY, mon INT, tue INT, wed INT,
    thu INT, fri INT, sat INT, sun INT, start_date TEXT, end_date TEXT);
CREATE TABLE IF NOT EXISTS stg_calendar_date(service_id TEXT, date TEXT, exception_type INT,
    PRIMARY KEY(service_id, date));
CREATE TABLE IF NOT EXISTS stg_frequency(trip_id TEXT, start_s INT, end_s INT, headway_s INT, exact_times INT);
CREATE TABLE IF NOT EXISTS service_day_ref(service_date TEXT PRIMARY KEY, ref_epoch INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS service_date(service_id TEXT, service_date TEXT, PRIMARY KEY(service_id, service_date));
CREATE TABLE IF NOT EXISTS trip_instance(instance_id TEXT PRIMARY KEY, trip_id TEXT, service_date TEXT,
    offset_s INTEGER, frequency_start_s INTEGER, exact_times INTEGER);
CREATE TABLE IF NOT EXISTS stop_event(
    service_date TEXT NOT NULL, instance_id TEXT NOT NULL, trip_id TEXT NOT NULL, route_id TEXT NOT NULL,
    stop_sequence INTEGER NOT NULL, stop_id TEXT NOT NULL,
    arrival_s INTEGER NOT NULL, departure_s INTEGER NOT NULL,
    arrival_epoch INTEGER NOT NULL, departure_epoch INTEGER NOT NULL,
    is_first INTEGER NOT NULL, is_last INTEGER NOT NULL,
    interpolated INTEGER NOT NULL, frequency_estimated INTEGER NOT NULL,
    PRIMARY KEY(instance_id, stop_sequence));
CREATE INDEX IF NOT EXISTS ix_event_stop_dep ON stop_event(stop_id, departure_epoch);
CREATE INDEX IF NOT EXISTS ix_event_date ON stop_event(service_date);
CREATE VIEW IF NOT EXISTS daily_route_summary AS
    SELECT service_date, route_id,
           COUNT(DISTINCT instance_id) AS trips,
           MIN(CASE WHEN is_first THEN departure_epoch END) AS first_departure_epoch,
           MAX(CASE WHEN is_last THEN arrival_epoch END) AS last_arrival_epoch,
           SUM(interpolated) AS interpolated_events,
           SUM(frequency_estimated) AS frequency_estimated_events
    FROM stop_event GROUP BY service_date, route_id;
"""

SERVICE_DATES_SQL = """
WITH RECURSIVE days(d) AS (
    SELECT :start UNION ALL SELECT date(d, '+1 day') FROM days WHERE d < :end
),
base AS (
    SELECT c.service_id, days.d
    FROM stg_calendar c JOIN days ON days.d BETWEEN c.start_date AND c.end_date
    WHERE CASE CAST(strftime('%w', days.d) AS INTEGER)
            WHEN 0 THEN c.sun WHEN 1 THEN c.mon WHEN 2 THEN c.tue WHEN 3 THEN c.wed
            WHEN 4 THEN c.thu WHEN 5 THEN c.fri ELSE c.sat END = 1
),
added AS (
    SELECT service_id, date AS d FROM stg_calendar_date
    WHERE exception_type = 1 AND date BETWEEN :start AND :end
),
removed AS (
    SELECT service_id, date AS d FROM stg_calendar_date
    WHERE exception_type = 2 AND date BETWEEN :start AND :end
)
INSERT INTO service_date(service_id, service_date)
SELECT service_id, d FROM (SELECT * FROM base UNION SELECT * FROM added)
EXCEPT SELECT service_id, d FROM removed
"""

INSTANCES_SQL = """
WITH RECURSIVE
first_dep AS (
    SELECT st.trip_id, st.departure_s FROM stg_stop_time st
    WHERE st.stop_sequence = (SELECT MIN(stop_sequence) FROM stg_stop_time x WHERE x.trip_id = st.trip_id)
),
starts(trip_id, s, end_s, headway_s, exact_times) AS (
    SELECT trip_id, start_s, end_s, headway_s, exact_times FROM stg_frequency
    UNION ALL
    SELECT trip_id, s + headway_s, end_s, headway_s, exact_times FROM starts WHERE s + headway_s < end_s
)
INSERT INTO trip_instance(instance_id, trip_id, service_date, offset_s, frequency_start_s, exact_times)
SELECT t.trip_id || '@' || sd.service_date, t.trip_id, sd.service_date, 0, NULL, 1
FROM stg_trip t JOIN service_date sd ON sd.service_id = t.service_id
WHERE sd.service_date BETWEEN :start AND :end
  AND t.trip_id NOT IN (SELECT trip_id FROM stg_frequency)
UNION ALL
SELECT t.trip_id || '@' || sd.service_date || '+' ||
         printf('%02d:%02d:%02d', st.s / 3600, (st.s % 3600) / 60, st.s % 60),
       t.trip_id, sd.service_date, st.s - fd.departure_s, st.s, st.exact_times
FROM starts st
JOIN stg_trip t ON t.trip_id = st.trip_id
JOIN first_dep fd ON fd.trip_id = st.trip_id
JOIN service_date sd ON sd.service_id = t.service_id
WHERE sd.service_date BETWEEN :start AND :end
"""

EVENTS_SQL = """
WITH bounds AS (
    SELECT trip_id, MIN(stop_sequence) AS lo, MAX(stop_sequence) AS hi FROM stg_stop_time GROUP BY trip_id
)
INSERT INTO stop_event
SELECT i.service_date, i.instance_id, i.trip_id, t.route_id, st.stop_sequence, st.stop_id,
       st.arrival_s + i.offset_s, st.departure_s + i.offset_s,
       r.ref_epoch + st.arrival_s + i.offset_s, r.ref_epoch + st.departure_s + i.offset_s,
       st.stop_sequence = b.lo, st.stop_sequence = b.hi,
       st.interpolated, 1 - i.exact_times
FROM trip_instance i
JOIN stg_trip t ON t.trip_id = i.trip_id
JOIN stg_stop_time st ON st.trip_id = i.trip_id
JOIN bounds b ON b.trip_id = i.trip_id
JOIN service_day_ref r ON r.service_date = i.service_date
WHERE i.service_date BETWEEN :start AND :end
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def stage(conn: sqlite3.Connection, feed: Feed) -> None:
    """Replace staged feed tables with the cleaned feed (one transaction)."""
    with conn:
        for t in ("dim_stop", "dim_route", "stg_trip", "stg_stop_time", "stg_calendar",
                  "stg_calendar_date", "stg_frequency", "service_day_ref", "service_date",
                  "trip_instance", "stop_event"):
            conn.execute(f"DELETE FROM {t}")
        conn.execute("INSERT OR REPLACE INTO meta VALUES('timezone', ?)", (feed.timezone,))
        conn.execute("INSERT OR REPLACE INTO meta VALUES('max_time_s', ?)", (str(feed.max_time_s),))
        conn.executemany("INSERT INTO dim_stop VALUES(?,?)",
                         [(k, v.get("stop_name", "")) for k, v in feed.stops.items()])
        conn.executemany("INSERT INTO dim_route VALUES(?,?,?)",
                         [(k, v.get("route_short_name", ""), v.get("route_long_name", ""))
                          for k, v in feed.routes.items()])
        conn.executemany("INSERT INTO stg_trip VALUES(?,?,?,?,?)",
                         [(t.trip_id, t.route_id, t.service_id, t.headsign, t.direction_id)
                          for t in feed.trips.values()])
        conn.executemany("INSERT INTO stg_stop_time VALUES(?,?,?,?,?,?)",
                         [(t.trip_id, s.stop_sequence, s.stop_id, s.arrival_s, s.departure_s, int(s.interpolated))
                          for t in feed.trips.values() for s in t.stop_times])
        conn.executemany("INSERT INTO stg_calendar VALUES(?,?,?,?,?,?,?,?,?,?)",
                         [(c.service_id, *c.days, c.start.isoformat(), c.end.isoformat())
                          for c in feed.calendars.values()])
        conn.executemany("INSERT INTO stg_calendar_date VALUES(?,?,?)",
                         [(sid, d.isoformat(), et) for (sid, d), et in feed.calendar_dates.items()])
        conn.executemany("INSERT INTO stg_frequency VALUES(?,?,?,?,?)",
                         [(f.trip_id, f.start_s, f.end_s, f.headway_s, f.exact_times) for f in feed.frequencies])


def expand(conn: sqlite3.Connection, start: date, end: date, tz) -> dict:
    """(Re)build service dates [start, end] idempotently. Returns row counts."""
    if start > end:
        raise ValueError("start after end")
    p = {"start": start.isoformat(), "end": end.isoformat()}
    with conn:
        conn.execute("DELETE FROM stop_event WHERE service_date BETWEEN :start AND :end", p)
        conn.execute("DELETE FROM trip_instance WHERE service_date BETWEEN :start AND :end", p)
        conn.execute("DELETE FROM service_date WHERE service_date BETWEEN :start AND :end", p)
        conn.executemany("INSERT OR REPLACE INTO service_day_ref VALUES(?,?)",
                         [(d.isoformat(), reference_epoch(d, tz)) for d in daterange(start, end)])
        conn.execute(SERVICE_DATES_SQL, p)
        conn.execute(INSTANCES_SQL, p)
        conn.execute(EVENTS_SQL, p)
    return {
        "service_dates": conn.execute("SELECT COUNT(*) FROM service_date WHERE service_date BETWEEN :start AND :end", p).fetchone()[0],
        "trip_instances": conn.execute("SELECT COUNT(*) FROM trip_instance WHERE service_date BETWEEN :start AND :end", p).fetchone()[0],
        "stop_events": conn.execute("SELECT COUNT(*) FROM stop_event WHERE service_date BETWEEN :start AND :end", p).fetchone()[0],
    }


def built_dates(conn: sqlite3.Connection) -> set[date]:
    return {date.fromisoformat(r[0]) for r in conn.execute("SELECT service_date FROM service_day_ref")}


def fetch_events(conn: sqlite3.Connection, start: date | None = None, end: date | None = None) -> list[tuple]:
    """Events as (instance_id, stop_sequence, stop_id, arrival_epoch, departure_epoch), sorted."""
    q = "SELECT instance_id, stop_sequence, stop_id, arrival_epoch, departure_epoch FROM stop_event"
    params: tuple = ()
    if start and end:
        q += " WHERE service_date BETWEEN ? AND ?"
        params = (start.isoformat(), end.isoformat())
    return sorted(conn.execute(q + " ORDER BY instance_id, stop_sequence", params).fetchall())


def build(feed: Feed, db_path: str, start: date, end: date) -> tuple[sqlite3.Connection, dict]:
    conn = connect(db_path)
    stage(conn, feed)
    counts = expand(conn, start, end, feed.tz)
    return conn, counts

