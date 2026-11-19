import datetime as dt
import random

import pytest

from safeshare.qi import QIError, QISpec


def test_date_format_parse_roundtrip():
    s = QISpec("dob", "date").fit([dt.date(1950, 1, 1).toordinal(),
                                   dt.date(2000, 1, 1).toordinal()])
    rng = random.Random(1)
    for _ in range(200):
        a = rng.randint(s.lo, s.hi)
        b = rng.randint(a, s.hi)
        assert s.parse(s.format((a, b))) == (a, b)
    assert s.format((s.lo, s.lo)) == "1950-01-01"


def test_zip_prefix_and_range_formats():
    s = QISpec("zip", "zip").fit([2100, 2499])
    assert s.format((2100, 2199)) == "021**"
    assert s.format((2130, 2139)) == "0213*"
    assert s.format((2130, 2145)) == "02130-02145"
    assert s.format((2134, 2134)) == "02134"
    assert s.format((0, 99999)) == "*****"
    for cell in [(2100, 2199), (2130, 2139), (2130, 2145), (2134, 2134), (0, 99999)]:
        assert s.parse(s.format(cell)) == cell


def test_numeric_parse_handles_negatives():
    s = QISpec("x", "numeric").fit([-20, 20])
    for cell in [(-5, -3), (-5, 7), (10, 20), (4, 4), (-4, -4)]:
        assert s.parse(s.format(cell)) == cell


def test_categorical_roundtrip_and_ncp():
    s = QISpec("sex", "categorical").fit(["F", "M", "X"])
    cell = frozenset({"F", "X"})
    assert s.parse(s.format(cell)) == cell
    assert s.ncp(frozenset({"F"})) == 0.0
    assert s.ncp(cell) == 0.5
    assert s.ncp(s.domain) == 1.0


@pytest.mark.parametrize("kind,values", [
    ("date", [dt.date(1931, 5, 17).toordinal(), dt.date(2007, 12, 2).toordinal()]),
    ("zip", [1701, 2499]),
    ("numeric", [18, 95]),
])
def test_hierarchy_levels_contain_value_and_coarsen(kind, values):
    s = QISpec("q", kind).fit(values)
    rng = random.Random(3)
    for _ in range(100):
        v = rng.randint(s.lo, s.hi)
        prev = None
        for level in range(s.levels()):
            cell = s.generalise(v, level)
            assert s.contains(cell, v)
            if prev is not None:   # each level is at least as coarse as the last
                assert cell[0] <= prev[0] and cell[1] >= prev[1]
            prev = cell
        assert s.ncp(s.generalise(v, s.levels() - 1)) == 1.0


def test_padded_bounds_cover_domain():
    s = QISpec("dob", "date").fit([dt.date(1931, 5, 17).toordinal(),
                                   dt.date(2007, 12, 2).toordinal()])
    lo, hi = s.padded_bounds()
    assert dt.date.fromordinal(lo) == dt.date(1931, 1, 1)
    assert dt.date.fromordinal(hi) == dt.date(2007, 12, 31)


def test_encode_rejects_bad_values():
    with pytest.raises(QIError):
        QISpec("zip", "zip").encode("2134")
    with pytest.raises(QIError):
        QISpec("dob", "date").encode("31/12/1990")
    with pytest.raises(QIError):
        QISpec("sex", "categorical").encode("  ")
    with pytest.raises(QIError):
        QISpec("x", "bogus")
