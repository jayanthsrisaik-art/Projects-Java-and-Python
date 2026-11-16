import pytest

from colineage.analyzer import AGGREGATE, IDENTITY, TRANSFORM, WINDOW, LineageError, analyze_query

CAT = {
    "a": {"id": "INTEGER", "x": "INTEGER", "k": "TEXT"},
    "b": {"id": "INTEGER", "y": "TEXT", "a_id": "INTEGER"},
}
KEYS = {"a": {"id"}, "b": {"id"}}


def q(sql, keys=None):
    return analyze_query(sql, CAT, keys=keys)


def test_identity_transform_aggregate_window_kinds():
    r = q(
        "SELECT x, x + 1 AS x1, (x) AS px, sum(x) AS s, "
        "rank() OVER (PARTITION BY k ORDER BY x) AS rk FROM a GROUP BY x, k"
    )
    assert r.columns["x"] == {"a.x": IDENTITY}
    assert r.columns["x1"] == {"a.x": TRANSFORM}
    assert r.columns["px"] == {"a.x": IDENTITY}
    assert r.columns["s"] == {"a.x": AGGREGATE}
    assert r.columns["rk"] == {"a.k": WINDOW, "a.x": WINDOW}


def test_aggregate_inside_window_is_window_not_aggregate():
    r = q("SELECT sum(x) OVER (ORDER BY id) AS run FROM a")
    assert r.columns["run"] == {"a.x": WINDOW, "a.id": WINDOW}


def test_control_from_where_join_group_having():
    r = q(
        "SELECT a.k, count(*) AS n FROM a JOIN b ON b.a_id = a.id "
        "WHERE b.y = 'z' GROUP BY a.k HAVING max(a.x) > 3"
    )
    assert r.columns["n"] == {}
    assert r.control == {"b.a_id", "a.id", "b.y", "a.k", "a.x"}


def test_cte_chain_and_rename_follow_through():
    r = q("WITH c1 AS (SELECT id AS key, x * 2 AS dbl FROM a WHERE k = 'p'), "
          "c2 AS (SELECT key, dbl FROM c1) SELECT key AS final_key, dbl FROM c2")
    assert r.columns == {"final_key": {"a.id": IDENTITY}, "dbl": {"a.x": TRANSFORM}}
    assert r.control == {"a.k"}  # CTE filter controls the outer rows


def test_star_expansion_uses_catalog():
    r = q("SELECT * FROM a")
    assert list(r.columns) == ["id", "x", "k"]


def test_derived_table_alias():
    r = q("SELECT t.v FROM (SELECT x + id AS v FROM a) AS t")
    assert r.columns == {"v": {"a.x": TRANSFORM, "a.id": TRANSFORM}}


def test_union_all_merges_positionally_union_distinct_adds_control():
    r = q("SELECT x AS v, k FROM a UNION ALL SELECT a_id, y FROM b")
    assert r.columns == {"v": {"a.x": IDENTITY, "b.a_id": IDENTITY}, "k": {"a.k": IDENTITY, "b.y": IDENTITY}}
    assert r.control == set()
    r = q("SELECT x AS v FROM a UNION SELECT a_id FROM b")
    assert r.control == {"a.x", "b.a_id"}


def test_except_right_side_is_control_only():
    r = q("SELECT id FROM a EXCEPT SELECT a_id FROM b WHERE y = 'q'")
    assert r.columns == {"id": {"a.id": IDENTITY}}
    assert r.control == {"a.id", "b.a_id", "b.y"}


def test_correlated_exists_is_control():
    r = q("SELECT a.x FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.a_id = a.id AND b.y = 'r')")
    assert r.columns == {"x": {"a.x": IDENTITY}}
    assert r.control == {"b.a_id", "a.id", "b.y"}


def test_scalar_subquery_in_select_is_value():
    r = q("SELECT x / (SELECT sum(x) FROM a WHERE k = 'p') AS share FROM a")
    assert r.columns["share"] == {"a.x": AGGREGATE, "a.k": TRANSFORM}
    assert r.control == set()


def test_distinct_and_order_under_limit():
    r = q("SELECT DISTINCT k FROM a")
    assert r.control == {"a.k"}
    r = q("SELECT k FROM a ORDER BY x DESC LIMIT 3")
    assert r.control == {"a.x"} and r.warnings == []
    r = q("SELECT k FROM a ORDER BY x")  # ordering alone changes no values
    assert r.control == set()
    assert q("SELECT k FROM a LIMIT 3").warnings  # nondeterministic


def test_order_by_output_alias_resolves_to_projection():
    r = q("SELECT x * 2 AS dbl FROM a ORDER BY dbl LIMIT 1")
    assert r.control == {"a.x"}


def test_using_join():
    r = q("SELECT a.x, b.y FROM a JOIN b USING (id)")
    assert r.control == {"a.id", "b.id"}


def test_key_preserved_left_join_group_by_pk():
    sql = ("SELECT a.id, a.k, count(b.id) AS n FROM a LEFT JOIN b ON b.a_id = a.id "
           "GROUP BY a.id, a.k")
    r = q(sql, KEYS)
    assert r.control == set()  # one row per row of a
    assert r.column_control.get("k") is None
    assert r.column_control["n"] == {"b.a_id", "a.id"}
    # without the declared key the analysis stays conservative
    r = q(sql)
    assert r.control == {"b.a_id", "a.id", "a.k"}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a.id, a.k, count(b.id) AS n FROM a JOIN b ON b.a_id = a.id GROUP BY a.id, a.k",
        "SELECT a.id, a.k, count(b.id) AS n FROM a LEFT JOIN b ON b.a_id = a.id WHERE a.x > 0 GROUP BY a.id, a.k",
        "SELECT a.id, b.y, count(*) AS n FROM a LEFT JOIN b ON b.a_id = a.id GROUP BY a.id, b.y",
        "SELECT a.k, count(*) AS n FROM a LEFT JOIN b ON b.a_id = a.id GROUP BY a.k",
    ],
)
def test_key_preserved_rule_does_not_fire_when_unsafe(sql):
    r = q(sql, KEYS)
    assert r.control, "inner join / filter / foreign grouping / non-key grouping must keep control"


@pytest.mark.parametrize(
    "sql,msg",
    [
        ("SELECT id FROM a JOIN b ON a.id = b.id", "ambiguous column 'id'"),
        ("SELECT zz FROM a", "unknown column 'zz'"),
        ("SELECT * FROM nope", "unknown relation(s): nope"),
        ("SELECT 1; SELECT 2", "exactly one statement"),
        ("DELETE FROM a", "must be a SELECT"),
        ("SELECT x, x FROM a", "duplicate output column"),
        ("SELECT x FROM a UNION ALL SELECT id, y FROM b", "branches have 1 and 2 columns"),
        ("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT n FROM r",
         "recursive"),
    ],
)
def test_errors(sql, msg):
    with pytest.raises(LineageError, match=None) as e:
        q(sql)
    assert msg in str(e.value)
