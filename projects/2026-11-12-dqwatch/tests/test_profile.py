import math
import sqlite3
from datetime import date

import pytest

from dqwatch.config import TableConfig, check_ident
from dqwatch.profile import NULL_CATEGORY, load_series, profile_table

T = TableConfig(name="t", partition_column="pd", key="id", loaded_at="loaded_at",
                null_checks=("email",), mean_checks=("amount",), category_checks=("kind",))


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE t (id INTEGER, pd TEXT, email TEXT, amount REAL, kind TEXT, loaded_at TEXT)")
    rows = [
        (1, "2025-01-01", "a@x", 10.0, "a", "2025-01-02 01:00:00"),
        (2, "2025-01-01", None, 20.0, "a", "2025-01-02 03:30:00"),
        (2, "2025-01-01", None, 30.0, "b", "2025-01-02 02:00:00"),  # duplicate key
        (3, "2025-01-01", "c@x", 40.0, None, "2025-01-02 01:00:00"),
        # 2025-01-02 has no rows at all
        (4, "2025-01-03", "d@x", 5.0, "b", "2025-01-03 23:00:00"),  # loaded before partition end
    ]
    c.executemany("INSERT INTO t VALUES (?,?,?,?,?,?)", rows)
    return c


def test_metrics_match_hand_computation(conn):
    profile_table(conn, T, date(2025, 1, 1), date(2025, 1, 3))
    s = load_series(conn, T)
    assert s.dates == ["2025-01-01", "2025-01-02", "2025-01-03"]
    v = s.values
    assert v["row_count"] == [4, 0, 1]
    assert v["dup_rate"][0] == pytest.approx(1 / 4)
    assert v["null_rate:email"][0] == pytest.approx(2 / 4)
    assert v["mean:amount"][0] == pytest.approx(25.0)
    assert s.sd["mean:amount"][0] == pytest.approx(math.sqrt(125.0))  # population sd of 10,20,30,40
    assert v["freshness_lag_h"][0] == pytest.approx(3.5)
    assert v["freshness_lag_h"][2] == pytest.approx(-1.0)
    # the empty day exists with row_count 0 and undefined everything else
    assert [v[m][1] for m in T.metrics() if m != "row_count"] == [None] * (len(T.metrics()) - 1)
    assert s.categories["kind"][0] == {"a": 2, "b": 1, NULL_CATEGORY: 1}
    assert s.categories["kind"][1] == {}
    assert s.n["mean:amount"] == [4, 0, 1]


def test_reprofile_is_idempotent_and_picks_up_reloads(conn):
    profile_table(conn, T, date(2025, 1, 1), date(2025, 1, 3))
    first = conn.execute("SELECT * FROM dq_metrics ORDER BY 1,2,3").fetchall()
    profile_table(conn, T, date(2025, 1, 1), date(2025, 1, 3))
    assert conn.execute("SELECT * FROM dq_metrics ORDER BY 1,2,3").fetchall() == first
    # the duplicate is removed upstream and the partition re-profiled
    conn.execute("DELETE FROM t WHERE id = 2 AND amount = 30.0")
    profile_table(conn, T, date(2025, 1, 1), date(2025, 1, 1))
    s = load_series(conn, T)
    assert s.values["dup_rate"][0] == 0.0
    assert s.categories["kind"][0] == {"a": 2, NULL_CATEGORY: 1}
    assert s.values["row_count"] == [3, 0, 1]  # other dates untouched


def test_until_truncates(conn):
    profile_table(conn, T, date(2025, 1, 1), date(2025, 1, 3))
    s = load_series(conn, T, until="2025-01-02")
    assert s.dates == ["2025-01-01", "2025-01-02"]


@pytest.mark.parametrize("bad", ["x; DROP TABLE t", "a b", "1abc", "", "t--"])
def test_identifiers_are_validated(bad):
    with pytest.raises(ValueError):
        check_ident(bad)
    with pytest.raises(ValueError):
        TableConfig(name="t", partition_column="pd", null_checks=(bad,))
