import json
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from gtfsflat.board import CoverageError, departures
from gtfsflat.cli import main
from gtfsflat.expand import build
from gtfsflat.feed import load_feed

NY = ZoneInfo("America/New_York")
EXPECTED = os.path.join(os.path.dirname(__file__), "..", "samples", "expected")


@pytest.fixture
def conn(tmp_path, sample_feed):
    c, _ = build(load_feed(sample_feed), str(tmp_path / "s.db"), date(2026, 10, 30), date(2026, 11, 3))
    yield c
    c.close()


def test_board_includes_previous_service_day(conn):
    rows = departures(conn, "A", datetime(2026, 11, 1, 0, 0, tzinfo=NY), 120)
    assert [(r["trip_id"], r["service_date"], r["departure_local"]) for r in rows] == [
        ("OWL_LATE", "2026-10-31", "2026-11-01T00:30:00-04:00"),
        ("OWL_EARLY", "2026-11-01", "2026-11-01T01:30:00-04:00"),
    ]
    # the common bug: filter on service_date == calendar date of the query
    naive = departures(conn, "A", datetime(2026, 11, 1, 0, 0, tzinfo=NY), 120, same_date_only=True)
    assert [r["trip_id"] for r in naive] == ["OWL_EARLY"]


def test_board_excludes_last_stop(conn):
    # C is the last stop of every trip that reaches it: arrivals only, no departures
    assert departures(conn, "C", datetime(2026, 10, 31, 0, 0, tzinfo=NY), 24 * 60) == []


def test_board_refuses_unbuilt_dates(conn):
    # service day 2026-10-29 can still have events until 25:15:00 = 01:15 on 10-30
    with pytest.raises(CoverageError, match="2026-10-29"):
        departures(conn, "A", datetime(2026, 10, 30, 0, 30, tzinfo=NY), 60)
    # by 06:00 it no longer matters that 10-29 was not built
    assert departures(conn, "A", datetime(2026, 10, 30, 6, 0, tzinfo=NY), 60) == []
    with pytest.raises(ValueError):
        departures(conn, "A", datetime(2026, 11, 1, 6, 0), 60)


def test_cli_validate_exit_codes(sample_feed, broken_feed, capsys):
    assert main(["validate", sample_feed]) == 0
    assert main(["validate", broken_feed]) == 1
    with open(os.path.join(EXPECTED, "validate_broken_feed.txt")) as fh:
        assert capsys.readouterr().out.splitlines()[-11:] == fh.read().splitlines()


def test_cli_build_strict_and_outputs(tmp_path, sample_feed, broken_feed, capsys):
    db = str(tmp_path / "x.db")
    assert main(["build", broken_feed, "--db", db, "--start", "2026-11-02", "--end", "2026-11-03", "--strict"]) == 1
    assert not os.path.exists(db)
    assert main(["build", sample_feed, "--db", db, "--start", "2026-10-30", "--end", "2026-11-03"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (out["service_dates"], out["trip_instances"], out["stop_events"]) == (5, 15, 36)

    assert main(["board", "--db", db, "--stop", "A", "--at", "2026-11-01T00:00", "--minutes", "120"]) == 0
    with open(os.path.join(EXPECTED, "board_A_2026-11-01T0000.txt")) as fh:
        assert capsys.readouterr().out == fh.read()

    assert main(["summary", "--db", db, "--date", "2026-11-01"]) == 0
    with open(os.path.join(EXPECTED, "summary_2026-11-01.txt")) as fh:
        assert capsys.readouterr().out == fh.read()

    csv_out = str(tmp_path / "e.csv")
    assert main(["export", "--db", db, "--start", "2026-10-30", "--end", "2026-11-03", "--out", csv_out]) == 0
    with open(csv_out) as a, open(os.path.join(EXPECTED, "events.csv")) as b:
        assert a.read() == b.read()


def test_cli_build_non_strict_quarantines(tmp_path, broken_feed, capsys):
    db = str(tmp_path / "y.db")
    assert main(["build", broken_feed, "--db", db, "--start", "2026-11-02", "--end", "2026-11-03"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["quarantined_trips"] == ["BAD_ROUTE", "BAD_TIME", "NO_END", "TIME_TRAVEL", "USES_BAD_CAL"]
    # broken feed: duplicate exception on 11-02 dropped -> 11-02 keeps first row (removed); 11-03 runs GOOD + 3 SH_T
    assert out["trip_instances"] == 4
