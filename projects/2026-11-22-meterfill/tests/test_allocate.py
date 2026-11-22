import numpy as np
from conftest import H, T0

from meterfill.allocate import coverage, fill
from meterfill.clean import clean_meter
from meterfill.config import ACTUAL, ALLOCATED, ESTIMATED


def gap_meter():
    # readings every 30 min for 6 h, a 10 h gap carrying 20 kWh, then 6 h more
    rows, v, t = [], 1000.0, T0 + 900
    for _ in range(12):
        rows.append((t, v)); t += 1800; v += 0.4
    t += 10 * H - 1800
    v += 20.0 - 0.4
    for _ in range(12):
        rows.append((t, round(v, 2))); t += 1800; v += 0.4
    return rows


def test_short_intervals_spread_at_constant_rate():
    rows = [(T0, 0.0), (T0 + 1800, 1.0), (T0 + 3 * 1800, 2.0)]
    cov = coverage(clean_meter(rows), T0, 2)
    assert np.allclose(cov.actual_kwh, [1.5, 0.5])
    assert np.allclose(cov.actual_sec, [3600, 1800])


def test_gap_total_is_conserved_for_any_shape():
    rows = gap_meter()
    nh = 24
    cov = coverage(clean_meter(rows), T0, nh)
    assert len(cov.gaps) == 1
    g = cov.gaps[0]
    for pred in (np.ones(nh), np.linspace(0.1, 5, nh), np.random.default_rng(0).gamma(2.0, 1.0, nh)):
        f = fill(cov, pred)
        # total over the window equals everything metered + estimated edges
        gap_part = f.kwh[f.gap_id == 0].sum() - cov.actual_kwh[f.gap_id == 0].sum()
        assert abs(gap_part - g.kwh) < 1e-9


def test_quality_codes():
    rows = gap_meter()
    cov = coverage(clean_meter(rows), T0, 24)
    f = fill(cov, np.ones(24))
    assert f.quality[0] == ESTIMATED  # before the first reading at 00:15
    assert f.quality[3] == ACTUAL
    assert (f.quality == ALLOCATED).sum() >= 10
    assert f.quality[-1] == ESTIMATED  # after the last reading


def test_shape_moves_energy_inside_the_gap():
    rows = gap_meter()
    cov = coverage(clean_meter(rows), T0, 24)
    pred = np.ones(24)
    pred[10] = 10.0
    flat, peaked = fill(cov, np.ones(24)), fill(cov, pred)
    assert peaked.kwh[10] > 3 * flat.kwh[10]
    assert abs(peaked.kwh[peaked.gap_id == 0].sum() - flat.kwh[flat.gap_id == 0].sum()) < 1e-9


def test_unscaled_ablation_ignores_metered_total():
    rows = gap_meter()
    cov = coverage(clean_meter(rows), T0, 24)
    f = fill(cov, np.full(24, 0.1), scale=False)
    assert f.kwh[f.gap_id == 0].sum() < 5  # far below the 20 kWh the register says


def test_gap_partly_outside_window_is_scaled_by_share():
    rows = [(T0 - 4 * H, 0.0), (T0 + 4 * H, 8.0), (T0 + 4 * H + 1800, 8.5)]
    cov = coverage(clean_meter(rows), T0, 6)
    f = fill(cov, np.ones(6))
    assert abs(f.kwh[:4].sum() - 4.0) < 1e-9


def test_reset_span_is_estimated():
    rows = [(T0, 50.0), (T0 + 1800, 50.5), (T0 + 5 * H, 0.1), (T0 + 5 * H + 1800, 0.6)]
    cov = coverage(clean_meter(rows), T0, 8)
    assert cov.open_spans[0] == (T0 + 1800, T0 + 5 * H)
    f = fill(cov, np.full(8, 0.7))
    assert all(f.quality[1:5] == ESTIMATED)
    assert abs(f.kwh[2] - 0.7) < 1e-9
