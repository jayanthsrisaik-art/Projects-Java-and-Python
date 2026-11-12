import math
import random
import sqlite3
from dataclasses import replace
from datetime import date, timedelta

import pytest

from dqwatch.config import DEFAULT_SPECS, DetectorParams, config_from_dict
from dqwatch.detect import ALERT_STATUSES, detect_all, detect_series
from dqwatch.generate import DEMO_CONFIG, generate
from dqwatch.profile import profile

P = DetectorParams()
WEEK = [1.0, 1.0, 1.05, 1.05, 1.1, 0.6, 0.5]  # strong weekend dip


def dates(n, start=date(2025, 1, 6)):
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def counts(n, seed=0, level=1000.0, noise=0.02):
    rng = random.Random(seed)
    return [float(round(level * WEEK[i % 7] * math.exp(rng.gauss(0, noise)))) for i in range(n)]


def run(metric, values, ns=None, params=P, spec=None, **kw):
    spec = spec or DEFAULT_SPECS[metric.split(":")[0]]
    ns = ns or [int(v) if v else 0 for v in values]
    return detect_series("t", metric, dates(len(values)), values, ns, spec, params, **kw)


def flagged(res):
    return [i for i, r in enumerate(res) if r.status in ALERT_STATUSES]


def test_weekly_seasonality_is_not_an_anomaly_but_a_drop_is():
    v = counts(84)
    v[70] *= 0.6  # a Monday at weekend level: only wrong given the weekday
    res = run("row_count", v)
    assert flagged(res) == [70]
    assert res[70].status == "alert" and res[70].score < -4
    # Without seasonality the weekend dips themselves look like incidents
    no_season = run("row_count", v, params=replace(P, use_seasonal=False))
    assert len(flagged(no_season)) > 5


def test_warmup_then_ok():
    res = run("row_count", counts(60))
    assert res[0].status == "warmup" and res[13].status == "warmup"
    assert all(r.status == "ok" for r in res[40:])


def test_empty_partition_is_missing_not_log_of_zero():
    v = counts(60)
    v[50] = 0.0
    res = run("row_count", v)
    assert res[50].status == "missing"
    assert flagged(res) == [50]


def test_trend_is_absorbed_by_drift_correction():
    v = [x * math.exp(0.004 * i) for i, x in enumerate(counts(200, seed=3))]  # ~+330%/year
    assert flagged(run("row_count", v)) == []


def test_persistent_shift_is_accepted_as_level_shift():
    v = counts(120, seed=5)
    v = [x * (1.5 if i >= 80 else 1.0) for i, x in enumerate(v)]
    for params in (P, replace(P, exclude_anomalies=True)):
        res = run("row_count", v, params=params)
        idx = flagged(res)
        assert idx and idx[0] == 80
        assert all(80 <= i < 90 for i in idx), idx  # quiet again soon after the shift
        statuses = [res[i].status for i in idx]
        assert "level_shift" in statuses or len(idx) < P.persist


def test_rate_band_widens_for_small_samples():
    rng = random.Random(1)
    ns = [800] * 70
    v = [sum(rng.random() < 0.02 for _ in range(800)) / 800 for _ in range(70)]
    v[60], ns[60] = 3 / 40, 40  # 7.5% of 40 rows: within binomial noise
    v[65] = 0.075  # 7.5% of 800 rows: a real spike
    res = run("null_rate:x", v, ns)
    assert flagged(res) == [65]
    no_sampling = run("null_rate:x", v, ns, params=replace(P, use_sampling=False))
    assert 60 in flagged(no_sampling)


def test_rate_drops_and_tiny_shifts_are_ignored():
    v = [0.05] * 40 + [0.0] + [0.05] * 5 + [0.055]
    ns = [100000] * len(v)
    assert flagged(run("null_rate:x", v, ns)) == []  # down is good news; +0.5pp < min_delta


def test_duplicates_from_zero_baseline():
    v = [0.0] * 40 + [1 / 1000] + [0.0] * 5 + [0.2]
    ns = [1000] * len(v)
    assert flagged(run("dup_rate", v, ns)) == [46]  # a single duplicated row is below min_delta


def test_mean_standard_error_floor():
    # a small table whose mean is noisy: sd 30 over 20 rows -> SE ~ 6.7 on a mean of ~25
    rng = random.Random(2)
    v = [25 + rng.gauss(0, 1) for _ in range(60)]
    v[50] = 33.0  # +28%, within ~1.2 standard errors given n=20, sd=30
    sds = [30.0] * 60
    res = run("mean:amount", v, ns=[20] * 60, sds=sds)
    assert 50 not in flagged(res)
    res_big = run("mean:amount", v, ns=[20000] * 60, sds=sds)
    assert 50 in flagged(res_big)


def _cats(rng, n, w):
    out = {"a": 0, "b": 0, "c": 0}
    for _ in range(n):
        out[rng.choices("abc", w)[0]] += 1
    return out


def test_psi_requires_more_than_sampling_noise():
    rng = random.Random(4)
    w = [0.5, 0.35, 0.15]
    cats = [_cats(rng, 60, w) for _ in range(70)]
    cats[60] = {"a": 30, "b": 30, "c": 0}  # moderate change on 60 rows
    cats[66] = {"a": 0, "b": 50, "c": 10}  # a whole category mislabelled
    res = run("psi:k", [None] * 70, ns=[60] * 70, categories=cats)
    assert 66 in flagged(res)
    assert len(flagged(res)) <= 2
    assert res[66].upper is not None and res[66].value > res[66].upper


def test_freshness_alerts_only_late_loads():
    rng = random.Random(6)
    v = [1.5 + rng.expovariate(1 / 0.6) for _ in range(60)]
    v[50] = 0.1  # early is fine
    v[55] = 14.0
    assert flagged(run("freshness_lag_h", v, ns=[100] * 60)) == [55]


def test_none_values_are_no_data():
    v = counts(40)
    v[30] = None
    res = run("row_count", v)
    assert res[30].status == "no_data"


@pytest.mark.parametrize("exclude", [False, True])
def test_no_look_ahead(tmp_path, exclude):
    """Results for a date are identical whether or not later partitions exist."""
    db = tmp_path / "wh.db"
    generate(str(db), days=110, seed=3)
    cfg = config_from_dict(DEMO_CONFIG)
    conn = sqlite3.connect(db)
    profile(conn, cfg)
    params = replace(cfg.detector, exclude_anomalies=exclude)
    full = {(r.table, r.date, r.metric): r for r in detect_all(conn, cfg, params=params)}
    cut = "2025-03-20"
    part = detect_all(conn, cfg, until=cut, params=params)
    assert part and all(r.date <= cut for r in part)
    for r in part:
        assert full[(r.table, r.date, r.metric)] == r
