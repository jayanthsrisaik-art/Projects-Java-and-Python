import pytest

from gtfsflat.feed import FeedError, load_feed


def codes(feed):
    return sorted((i.file, i.line, i.code) for i in feed.issues)


def test_clean_sample_has_no_issues(sample_feed):
    feed = load_feed(sample_feed)
    assert feed.issues == []
    assert set(feed.trips) == {"L1_1", "OWL_LATE", "OWL_EARLY", "SH_T"}


def test_interpolation_uses_shape_distance(sample_feed):
    st = load_feed(sample_feed).trips["L1_1"].stop_times
    # 08:00 -> 08:20 over 4000 units, B at 3000 -> 75% -> 08:15:00
    assert [(s.stop_id, s.arrival_s, s.interpolated) for s in st] == [
        ("A", 8 * 3600, False), ("B", 8 * 3600 + 900, True), ("C", 8 * 3600 + 1200, False)]


def test_interpolation_falls_back_to_stop_count(feed_copy):
    p = feed_copy / "stop_times.txt"
    p.write_text(p.read_text().replace("L1_1,,,B,2,3000", "L1_1,,,B,2,"))
    st = load_feed(str(feed_copy)).trips["L1_1"].stop_times
    assert st[1].arrival_s == 8 * 3600 + 600  # halfway by index


def test_broken_feed_issues_with_line_numbers(broken_feed):
    feed = load_feed(broken_feed)
    assert codes(feed) == sorted([
        ("calendar.txt", 3, "inverted_range"),
        ("calendar_dates.txt", 3, "duplicate_key"),
        ("trips.txt", 3, "unknown_service"),
        ("trips.txt", 4, "unknown_route"),
        ("stop_times.txt", 11, "unknown_trip"),
        ("stop_times.txt", 3, "half_timepoint"),
        ("stop_times.txt", 6, "time_travel"),
        ("stop_times.txt", 7, "bad_time"),
        ("stop_times.txt", 8, "unknown_stop"),
        ("stop_times.txt", 10, "missing_endpoint_time"),
    ])
    # bad trips are quarantined whole, good ones survive
    assert set(feed.trips) == {"GOOD", "SH_T"}
    assert feed.quarantined_trips == {"USES_BAD_CAL", "BAD_ROUTE", "TIME_TRAVEL", "BAD_TIME", "NO_END"}
    assert [i.severity for i in feed.issues if i.code == "half_timepoint"] == ["warning"]


def test_missing_required_file_is_fatal(feed_copy):
    (feed_copy / "stops.txt").unlink()
    with pytest.raises(FeedError, match="stops.txt"):
        load_feed(str(feed_copy))


def test_missing_required_column_is_fatal(feed_copy):
    (feed_copy / "trips.txt").write_text("route_id,trip_id\nL1,L1_1\n")
    with pytest.raises(FeedError, match="service_id"):
        load_feed(str(feed_copy))


def test_unknown_timezone_is_fatal(feed_copy):
    p = feed_copy / "agency.txt"
    p.write_text(p.read_text().replace("America/New_York", "Mars/Olympus_Mons"))
    with pytest.raises(FeedError, match="timezone"):
        load_feed(str(feed_copy))


def test_bom_and_whitespace_tolerated(feed_copy):
    p = feed_copy / "routes.txt"
    p.write_text("﻿" + p.read_text().replace("L1,A", " L1 ,A"), encoding="utf-8")
    assert "L1" in load_feed(str(feed_copy)).routes


def test_frequency_validation(feed_copy):
    p = feed_copy / "frequencies.txt"
    p.write_text(p.read_text() + "SH_T,07:20:00,08:00:00,600,1\nSH_T,09:00:00,08:00:00,600,1\nSH_T,10:00:00,11:00:00,0,1\n")
    feed = load_feed(str(feed_copy))
    assert [(i.line, i.code) for i in feed.issues] == [
        (3, "overlapping_frequency"), (4, "bad_frequency"), (5, "bad_frequency")]
    assert len(feed.frequencies) == 1


def test_departure_before_arrival_quarantines_trip(feed_copy):
    p = feed_copy / "stop_times.txt"
    p.write_text(p.read_text().replace("OWL_LATE,24:45:00,24:45:00", "OWL_LATE,24:45:00,24:40:00"))
    feed = load_feed(str(feed_copy))
    assert [i.code for i in feed.errors] == ["departs_before_arrival"]
    assert "OWL_LATE" not in feed.trips
