import json
import sqlite3

from dqwatch.baselines import static_alerts, zscore_alerts
from dqwatch.config import TableConfig
from dqwatch.evaluate import METHODS, evaluate, score
from dqwatch.generate import generate
from dqwatch.profile import Series

T = TableConfig(name="t", partition_column="pd", key="id", null_checks=("e",))


def series(row_counts, dup=None, nulls=None):
    n = len(row_counts)
    dates = [f"2025-01-{i + 1:02d}" for i in range(n)]
    values = {"row_count": row_counts, "dup_rate": dup or [0.0] * n, "null_rate:e": nulls or [0.0] * n}
    return Series("t", dates, values, {m: [100] * n for m in values}, {m: [None] * n for m in values}, {})


def test_static_rules():
    rc = [100.0] * 7 + [100.0, 60.0, 100.0, 0.0, 100.0, 100.0, 100.0]
    nulls = [0.0] * 13 + [0.05]
    got = static_alerts(series(rc, nulls=nulls), T)
    assert ("t", "2025-01-09", "row_count") in got  # -40% week over week
    assert ("t", "2025-01-11", "row_count") in got  # empty partition
    assert ("t", "2025-01-14", "null_rate:e") in got
    assert ("t", "2025-01-08", "row_count") not in got


def test_zscore_needs_history_and_flags_outliers():
    rc = [100.0, 101.0, 99.0, 100.0, 102.0, 98.0, 100.0, 101.0, 50.0]
    got = zscore_alerts(series(rc), T)
    assert got == [("t", "2025-01-09", "row_count")]
    assert zscore_alerts(series(rc[:6] + [10.0]), T) == []  # < 7 points of history


def test_score_counts_explained_and_false_alerts():
    truth = {"eval_start": "2025-01-05",
             "incidents": [{"table": "t", "dates": ["2025-01-06", "2025-01-07"], "type": "null_spike", "params": {},
                            "metrics": ["null_rate:e"]},
                           {"table": "t", "dates": ["2025-01-09"], "type": "volume_drop",
                            "params": {"keep": 0.5}, "metrics": ["row_count"]}],
             "benign_events": [{"table": "t", "date": "2025-01-20", "type": "level_shift"}]}
    alerts = {("t", "2025-01-07", "null_rate:e"),  # explained (day 2 of the incident)
              ("t", "2025-01-06", "null_rate:e"),  # explained, same incident
              ("t", "2025-01-09", "dup_rate"),  # wrong metric on an incident day: false
              ("t", "2025-01-02", "row_count"),  # before eval_start: ignored
              ("t", "2025-01-21", "row_count")}  # after the benign shift
    s = score(alerts, truth)
    assert s["alerts"] == 4
    assert s["explained_alerts"] == 2
    assert s["false_alerts"] == 2
    assert s["caught"] == 1 and s["incidents"] == 2
    assert s["by_type"]["null_spike"] == [1, 1] and s["by_type"]["volume_drop"] == [0, 1]
    assert s["benign_shift_alerts"] == 1
    assert s["missed"] == [{"table": "t", "dates": ["2025-01-09"], "type": "volume_drop", "params": {"keep": 0.5}}]


def test_generator_is_deterministic(tmp_path):
    t1 = generate(str(tmp_path / "a.db"), days=80, seed=9)
    t2 = generate(str(tmp_path / "b.db"), days=80, seed=9)
    assert t1 == t2
    q = "SELECT COUNT(*), SUM(amount) FROM orders"
    assert (sqlite3.connect(tmp_path / "a.db").execute(q).fetchone()
            == sqlite3.connect(tmp_path / "b.db").execute(q).fetchone())
    assert t1["incidents"] and all(i["dates"][0] >= t1["eval_start"] for i in t1["incidents"])


def test_generator_injects_what_truth_says(tmp_path):
    db = tmp_path / "w.db"
    truth = generate(str(db), days=200, seed=4)
    conn = sqlite3.connect(db)
    for inc in truth["incidents"]:
        tbl, d = inc["table"], inc["dates"][0]
        key = "order_id" if tbl == "orders" else "refund_id"
        n, distinct = conn.execute(f"SELECT COUNT(*), COUNT(DISTINCT {key}) FROM {tbl} WHERE partition_date=?",
                                   (d,)).fetchone()
        if inc["type"] == "missing_partition":
            assert n == 0
        if inc["type"] == "duplicate_load":
            assert n > distinct
        else:
            assert n == distinct


def test_small_evaluation_runs_and_dqwatch_beats_static_precision(tmp_path):
    res = evaluate([21], days=150, workdir=str(tmp_path))
    s = res["summary"]
    assert set(s) == set(METHODS)
    for m in METHODS:
        assert 0 <= s[m]["caught"] <= s[m]["incidents"]
        assert s[m]["explained_alerts"] + s[m]["false_alerts"] == s[m]["alerts"]
    assert s["dqwatch"]["false_alerts"] < s["static_rules"]["false_alerts"]
    json.dumps(res)  # serialisable
