from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from sniffload import values as V


@pytest.mark.parametrize("text,dec,expected", [
    ("1,234.56", ".", Decimal("1234.56")),
    ("1.234,56", ",", Decimal("1234.56")),
    ("1 234,5", ",", Decimal("1234.5")),
    ("1 234,5", ",", Decimal("1234.5")),
    ("1'234.50", ".", Decimal("1234.50")),
    ("(1,234.50)", ".", Decimal("-1234.50")),
    ("$-12.00", ".", Decimal("-12.00")),
    ("-€5", ".", Decimal("-5")),
    ("12 €", ",", Decimal("12")),
    (".5", ".", Decimal("0.5")),
    ("0,125", ",", Decimal("0.125")),
    ("1e3", ".", Decimal("1000")),
    ("1.234.567", ",", Decimal("1234567")),
])
def test_parse_number_accepts(text, dec, expected):
    p = V.parse_number(text, dec)
    assert p is not None and p.value == expected


@pytest.mark.parametrize("text,dec", [
    ("1,23", "."),        # thousands groups must have 3 digits
    ("0,125", "."),       # a leading 0 group is not a thousands group
    ("1.234.567", "."),   # two decimal points
    ("1,234.56", ","),    # mixed separators in the wrong roles
    ("1,234 567", "."),   # inconsistent thousands separators
    ("12..5", "."),
    ("abc", "."),
    ("", "."),
])
def test_parse_number_rejects(text, dec):
    assert V.parse_number(text, dec) is None


def test_ambiguous_values_parse_differently_per_locale():
    assert V.parse_number("1,234", ".").value == 1234
    assert V.parse_number("1,234", ",").value == Decimal("1.234")


def test_leading_zero_flag():
    assert V.parse_number("02134", ".").leading_zero
    assert not V.parse_number("0.5", ".").leading_zero
    assert not V.parse_number("0", ".").leading_zero


def test_nulls_and_bools():
    for t in ["", " ", "NA", "n/a", "NULL", "-", "nan"]:
        assert V.is_null(t)
    assert not V.is_null("0")
    assert V.parse_bool("Yes") is True and V.parse_bool("nein") is False
    assert V.parse_bool("maybe") is None


def test_dates_and_datetimes():
    assert V.parse_date("31/01/2024", "%d/%m/%Y") == date(2024, 1, 31)
    assert V.parse_date("31/01/2024", "%m/%d/%Y") is None
    assert V.parse_date("05/03/24", "%d/%m/%Y") is None  # %Y needs 4 digits
    assert V.parse_date("05 Mar 2024", "%d %b %Y") == date(2024, 3, 5)
    assert V.parse_datetime("2024-03-01T08:25:00Z", "iso") == datetime(2024, 3, 1, 8, 25, tzinfo=timezone.utc)
    assert V.parse_datetime("2024-03-01T08:35+01:00", "iso").utcoffset() == timedelta(hours=1)
    assert V.parse_datetime("2024-03-01", "iso") is None  # a date is not a datetime
