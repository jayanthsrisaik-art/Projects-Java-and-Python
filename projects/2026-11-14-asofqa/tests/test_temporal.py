from datetime import date

import pytest

from asofqa.temporal import parse, strip_temporal

REF = date(2026, 11, 14)


@pytest.mark.parametrize("q,kind,start,end", [
    ("What is the PTO allowance?", "point", REF, REF),
    ("What is the PTO allowance right now?", "point", REF, REF),
    ("What was it on 2023-05-10?", "point", date(2023, 5, 10), date(2023, 5, 10)),
    ("As of March 3, 2022, what was it?", "point", date(2022, 3, 3), date(2022, 3, 3)),
    ("What was in effect on 23 October 2025?", "point", date(2025, 10, 23), date(2025, 10, 23)),
    ("Looking back to 3/23/2026, what was it?", "point", date(2026, 3, 23), date(2026, 3, 23)),
    ("What was it in 2023?", "range", date(2023, 1, 1), date(2023, 12, 31)),
    ("Throughout 2021, what was it?", "range", date(2021, 1, 1), date(2021, 12, 31)),
    ("What was it during February 2024?", "range", date(2024, 2, 1), date(2024, 2, 29)),
    ("What was it for July of 2023?", "range", date(2023, 7, 1), date(2023, 7, 31)),
    ("What was it in Q4 2022?", "range", date(2022, 10, 1), date(2022, 12, 31)),
    ("What applied in the second half of 2024?", "range", date(2024, 7, 1), date(2024, 12, 31)),
    ("What applied in H1 2024?", "range", date(2024, 1, 1), date(2024, 6, 30)),
    ("What was it last year?", "range", date(2025, 1, 1), date(2025, 12, 31)),
    ("What has it been this year?", "range", date(2026, 1, 1), REF),
    ("What was it last month?", "range", date(2026, 10, 1), date(2026, 10, 31)),
    ("What was it two years ago?", "point", date(2024, 11, 14), date(2024, 11, 14)),
    ("What was it a year ago?", "point", date(2025, 11, 14), date(2025, 11, 14)),
    ("What was it 18 months ago?", "point", date(2025, 5, 14), date(2025, 5, 14)),
])
def test_dates_and_periods(q, kind, start, end):
    it = parse(q, REF)
    assert (it.kind, it.start, it.end) == (kind, start, end)


def test_previous_and_when():
    assert parse("What was it before the most recent change?", REF).kind == "previous"
    assert parse("What did it used to be before it was last revised?", REF).kind == "previous"
    w = parse("When did the learning budget become $2,000?", REF)
    assert (w.kind, w.value) == ("when", "2000")
    w = parse("Since when has the 401(k) match been 6%?", REF)
    assert (w.kind, w.value) == ("when", "6")


def test_money_is_not_a_year():
    # "$2000" must not be read as the year 2000
    assert parse("Is the budget $2000?", REF).kind == "point"
    assert parse("Is the budget 2020 days?", REF).start == REF


def test_impossible_date_falls_through():
    # 30 February does not exist; the parser falls back to the year it can still see
    it = parse("What was it on 2/30/2024?", REF)
    assert (it.kind, it.start, it.end) == ("range", date(2024, 1, 1), date(2024, 12, 31))


def test_strip_temporal_removes_phrase():
    q = "What was the meal allowance in Q2 2023?"
    assert strip_temporal(q, parse(q, REF)) == "What was the meal allowance in  ?"
