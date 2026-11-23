from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from gtfsflat.timeutil import (TimeFormatError, format_gtfs_time, parse_gtfs_date, parse_gtfs_time,
                               service_day_reference)

NY = ZoneInfo("America/New_York")


@pytest.mark.parametrize("text,seconds", [
    ("00:00:00", 0), ("7:05:09", 7 * 3600 + 309), ("24:30:00", 88200), ("25:15:00", 90900), ("100:00:00", 360000),
])
def test_parse_times_including_past_midnight(text, seconds):
    assert parse_gtfs_time(text) == seconds
    assert parse_gtfs_time(format_gtfs_time(seconds)) == seconds


@pytest.mark.parametrize("bad", ["9:5:00", "09:60:00", "", "ab:cd:ef", "-1:00:00", "09:00"])
def test_reject_malformed_times(bad):
    with pytest.raises(TimeFormatError):
        parse_gtfs_time(bad)


def test_parse_date():
    assert parse_gtfs_date("20261126") == date(2026, 11, 26)
    with pytest.raises(ValueError):
        parse_gtfs_date("2026-11-26")


# Hand-computed from the spec: reference = local noon - 12h.
@pytest.mark.parametrize("d,expected_utc", [
    (date(2026, 7, 1), datetime(2026, 7, 1, 4, tzinfo=timezone.utc)),     # EDT midnight
    (date(2026, 1, 15), datetime(2026, 1, 15, 5, tzinfo=timezone.utc)),   # EST midnight
    (date(2026, 3, 8), datetime(2026, 3, 8, 4, tzinfo=timezone.utc)),     # spring forward: 23:00 EST on Mar 7
    (date(2026, 11, 1), datetime(2026, 11, 1, 5, tzinfo=timezone.utc)),   # fall back: 01:00 EDT
])
def test_service_day_reference_on_dst_days(d, expected_utc):
    assert service_day_reference(d, NY) == expected_utc


def test_reference_in_southern_hemisphere_zone():
    # Sydney DST starts 2026-10-04 at 02:00 (UTC+10 -> +11): noon AEDT = 01:00 UTC, minus 12h = 13:00 UTC Oct 3
    assert service_day_reference(date(2026, 10, 4), ZoneInfo("Australia/Sydney")) == \
        datetime(2026, 10, 3, 13, tzinfo=timezone.utc)
