import csv
import os
from datetime import date, datetime, timezone

import pytest

from gtfsflat.expand import build, expand, fetch_events
from gtfsflat.feed import load_feed
from gtfsflat.oracle import Shortcuts, compare, expand_python
from gtfsflat.synth import generate

START, END = date(2026, 10, 30), date(2026, 11, 3)


def utc(*a):
    return int(datetime(*a, tzinfo=timezone.utc).timestamp())


@pytest.fixture
def built(tmp_path, sample_feed):
    feed = load_feed(sample_feed)
    conn, counts = build(feed, str(tmp_path / "s.db"), START, END)
    yield feed, conn, counts
    conn.close()


def event(conn, instance_id, seq):
    return conn.execute("SELECT stop_id, arrival_epoch, departure_epoch FROM stop_event "
                        "WHERE instance_id = ? AND stop_sequence = ?", (instance_id, seq)).fetchone()


def test_calendar_exceptions(built):
    _, conn, _ = built
    active = set(conn.execute("SELECT service_id, service_date FROM service_date"))
    # Mon 2026-11-02 swaps weekday service for weekend service via calendar_dates
    assert ("WKDY", "2026-11-02") not in active and ("WKND", "2026-11-02") in active
    assert ("WKDY", "2026-11-03") in active and ("WKND", "2026-10-31") in active


def test_times_past_midnight_belong_to_previous_service_day(built):
    _, conn, _ = built
    # Sat 2026-10-31 service, 24:30:00 -> Sun 00:30 EDT = 04:30Z
    assert event(conn, "OWL_LATE@2026-10-31", 1) == ("A", utc(2026, 11, 1, 4, 30), utc(2026, 11, 1, 4, 30))


def test_fall_back_day_uses_noon_minus_12h(built):
    _, conn, _ = built
    # Sun 2026-11-01: reference is 01:00 EDT (05:00Z). 00:30:00 -> 05:30Z = 01:30 EDT, 01:30:00 -> 06:30Z = 01:30 EST
    assert event(conn, "OWL_EARLY@2026-11-01", 1)[1] == utc(2026, 11, 1, 5, 30)
    assert event(conn, "OWL_EARLY@2026-11-01", 2)[1] == utc(2026, 11, 1, 6, 30)
    # a trip's elapsed time is preserved across the transition (60 min), wall clock would say 0 / 120
    # after the transition, times read like wall clock again: 08:00 EST = 13:00Z
    assert event(conn, "L1_1@2026-11-01", 1)[1] == utc(2026, 11, 1, 13, 0)


def test_frequency_expansion(built):
    _, conn, _ = built
    rows = conn.execute("SELECT instance_id, offset_s FROM trip_instance WHERE trip_id='SH_T' "
                        "AND service_date='2026-11-03' ORDER BY instance_id").fetchall()
    # 07:00-07:30 every 600 s, end exclusive -> 3 departures
    assert rows == [("SH_T@2026-11-03+07:00:00", 0), ("SH_T@2026-11-03+07:10:00", 600),
                    ("SH_T@2026-11-03+07:20:00", 1200)]
    assert event(conn, "SH_T@2026-11-03+07:20:00", 2)[1] == utc(2026, 11, 3, 12, 30)  # 07:30 EST
    # template trip is not also run as a standalone trip
    assert conn.execute("SELECT COUNT(*) FROM trip_instance WHERE instance_id='SH_T@2026-11-03'").fetchone()[0] == 0


def test_matches_expected_sample_output(built):
    _, conn, _ = built
    with open(os.path.join(os.path.dirname(__file__), "..", "samples", "expected", "events.csv")) as fh:
        expected = list(csv.DictReader(fh))
    got = conn.execute("SELECT instance_id, stop_sequence, departure_epoch FROM stop_event "
                       "ORDER BY service_date, instance_id, stop_sequence").fetchall()
    assert len(got) == len(expected) == 36
    for (iid, seq, dep), row in zip(got, expected):
        assert (iid, str(seq)) == (row["instance_id"], row["stop_sequence"])
        assert datetime.fromtimestamp(dep, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") == row["departure_utc"]


def test_rebuild_is_idempotent(built):
    feed, conn, counts = built
    before = fetch_events(conn)
    again = expand(conn, START, END, feed.tz)
    assert again == counts and fetch_events(conn) == before
    # rebuilding a sub-range only replaces that range
    expand(conn, date(2026, 11, 1), date(2026, 11, 1), feed.tz)
    assert fetch_events(conn) == before


def test_sql_matches_oracle_on_sample(built):
    feed, conn, _ = built
    diff = compare(expand_python(feed, START, END), fetch_events(conn))
    assert diff["exact"] == diff["expected"] == diff["produced"] == 36


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("window", [(date(2026, 3, 5), date(2026, 3, 10)), (date(2026, 11, 24), date(2026, 11, 29))])
def test_sql_matches_oracle_on_synthetic(tmp_path, seed, window):
    generate(str(tmp_path / "feed"), seed)
    feed = load_feed(str(tmp_path / "feed"))
    conn, _ = build(feed, str(tmp_path / "s.db"), *window)
    diff = compare(expand_python(feed, *window), fetch_events(conn))
    conn.close()
    assert diff["expected"] > 0
    assert diff["exact"] == diff["expected"] == diff["produced"]


def test_shortcuts_are_detectably_wrong(tmp_path):
    """Each baseline shortcut produces a nonzero, specific kind of error on the synthetic feed."""
    generate(str(tmp_path / "feed"), 0)
    feed = load_feed(str(tmp_path / "feed"))
    s, e = date(2026, 10, 30), date(2026, 11, 28)
    truth = expand_python(feed, s, e)
    d = {k: compare(truth, expand_python(feed, s, e, Shortcuts(**{k: True})))
         for k in ["ignore_exceptions", "wrap_24h", "wallclock", "ignore_frequencies"]}
    assert d["ignore_exceptions"]["missing"] > 0 and d["ignore_exceptions"]["extra"] > 0  # Thanksgiving + event
    assert d["wrap_24h"]["wrong_time"] > 0 and d["wrap_24h"]["missing"] == 0
    assert d["wallclock"]["wrong_time"] > 0 and d["wallclock"]["missing"] == 0
    assert d["ignore_frequencies"]["missing"] > 0
