from conftest import H, T0

from meterfill.clean import clean_meter, naive_intervals, step
from meterfill.config import REGISTER_MAX


def series(values, start=T0, every=1800):
    return [(start + i * every, v) for i, v in enumerate(values)]


def statuses(res):
    return [s for _, _, s in res.flags]


def test_clean_monotone_series_is_all_ok_and_conserves_energy():
    rows = series([100.0, 100.5, 101.25, 102.0, 103.5])
    res = clean_meter(rows)
    assert statuses(res) == ["first", "ok", "ok", "ok", "ok"]
    assert len(res.segments) == 1
    assert abs(sum(iv.kwh for iv in res.intervals) - 3.5) < 1e-9


def test_input_order_does_not_matter():
    rows = series([10.0, 10.4, 11.0, 11.9, 12.2])
    assert clean_meter(rows).flags == clean_meter(list(reversed(rows))).flags


def test_rollover_is_unwrapped_not_treated_as_reset():
    rows = series([99_998.5, 99_999.4, 0.3, 1.1])
    res = clean_meter(rows)
    assert statuses(res) == ["first", "ok", "rollover", "ok"]
    assert len(res.segments) == 1
    total = sum(iv.kwh for iv in res.intervals)
    assert abs(total - (REGISTER_MAX - 99_998.5 + 1.1)) < 1e-6


def test_high_and_low_misreads_are_skipped():
    rows = series([500.0, 500.6, 9_500.9, 501.5, 502.0, 2.4, 503.1, 503.6])
    res = clean_meter(rows)
    assert statuses(res) == ["first", "ok", "misread", "ok", "ok", "misread", "ok", "ok"]
    assert abs(sum(iv.kwh for iv in res.intervals) - 3.6) < 1e-9


def test_misread_that_looks_plausible_over_a_long_gap_is_caught_by_lookahead():
    # 48 h outage, then a reading +1000 kWh (21 kW average: plausible on its own),
    # but the next reading agrees with the pre-outage value, not with it.
    rows = [(T0, 2000.0), (T0 + 1800, 2000.4), (T0 + 1800 + 48 * H, 3070.0), (T0 + 1800 + 48 * H + 1800, 2070.6), (T0 + 1800 + 49 * H, 2071.0)]
    res = clean_meter(rows)
    assert statuses(res) == ["first", "ok", "misread", "ok", "ok"]
    assert res.intervals[1].is_gap


def test_sustained_drop_starts_new_segment():
    rows = series([7_000.0, 7_000.5, 7_001.0, 0.2, 0.9, 1.5])
    res = clean_meter(rows)
    assert statuses(res) == ["first", "ok", "ok", "reset", "ok", "ok"]
    assert len(res.segments) == 2
    kinds = [d["kind"] for d in res.issues()]
    assert "reset" in kinds and "unmetered" in kinds
    # energy across the exchange is unknown, so no interval spans it
    assert all(iv.t1 <= rows[2][0] or iv.t0 >= rows[3][0] for iv in res.intervals)


def test_step_plausibility_uses_elapsed_time():
    assert step(T0, 100.0, T0 + H, 120.0)[0]  # 20 kW for an hour: ok
    assert not step(T0, 100.0, T0 + H, 130.0)[0]  # 30 kW: implausible
    assert step(T0, 100.0, T0 + 10 * H, 300.0)[0]  # 20 kW over 10 h: ok
    assert step(T0, 100.0, T0 + 600, 99.995)[0]  # within rounding tolerance
    assert not step(T0, 100.0, T0 + 600, 99.9)[0]


def test_gap_issue_reports_metered_energy():
    rows = [(T0, 10.0), (T0 + 1800, 10.5), (T0 + 1800 + 5 * H, 16.5), (T0 + 1800 + 5 * H + 1800, 17.0)]
    gaps = [d for d in clean_meter(rows).issues() if d["kind"] == "gap"]
    assert gaps == [{"kind": "gap", "start_ts": T0 + 1800, "end_ts": T0 + 1800 + 5 * H, "value": 6.0}]


def test_naive_baseline_is_inflated_by_a_misread():
    rows = series([500.0, 500.6, 9_500.9, 501.5])
    assert sum(iv.kwh for iv in naive_intervals(rows)) > 9_000


def test_exchange_of_a_meter_near_wrap_is_a_reset_not_misread_plus_rollover():
    # Old register 99,916; new meter starts near 0. The first new value is too far for a
    # rollover (29 kW), the second is just plausible as one (24 kW) - but both new values
    # agree with each other, so this is a new regime, not an outlier.
    t = [1769571008, 1769572636, 1769574732, 1769585154, 1769587272, 1769589201]
    v = [99916.32, 99916.44, 99916.62, 0.16, 0.55, 0.78]
    res = clean_meter(list(zip(t, v)))
    assert statuses(res) == ["first", "ok", "ok", "reset", "ok", "ok"]
