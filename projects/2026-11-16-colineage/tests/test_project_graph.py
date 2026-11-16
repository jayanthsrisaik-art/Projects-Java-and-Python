import json

from colineage.analyzer import IDENTITY
from colineage.graph import Dep, LineageGraph
from colineage.project import Project, SourceTable, analyze_project, model_order, results_to_json

from conftest import SAMPLES, load


def _project(models, sources=None):
    sources = sources or {"t": {"id": {"type": "INTEGER", "pk": True}, "v": {"type": "REAL"}, "s": {"type": "TEXT"}}}
    return Project({n: SourceTable(n, 10, c) for n, c in sources.items()}, models)


def test_models_are_topologically_ordered(shop):
    _, results, _ = shop
    order = list(results)
    for name, r in results.items():
        for dep in r.depends_on:
            assert order.index(dep) < order.index(name)


def test_lineage_matches_golden_file(shop, saas):
    for name, (_, results, graph) in (("shop", shop), ("saas", saas)):
        golden = json.loads((SAMPLES / name / "expected" / "lineage.json").read_text())
        assert golden["models"] == json.loads(json.dumps(results_to_json(results)))
        e2e = {n: {s: d.to_json() for s, d in sorted(graph.trace(n).items())} for n in graph.model_columns()}
        assert golden["end_to_end"] == e2e


def test_broken_project_statuses():
    _, results, _ = load("broken")
    status = {n: (r.status, r.error) for n, r in results.items()}
    assert status["ok_model"][0] == "ok" and status["downstream"][0] == "ok"
    assert status["ambiguous"] == ("error", "ambiguous column 'id': present in a, b; qualify it")
    assert status["blocked"] == ("blocked", "upstream model(s) failed: ambiguous")
    assert status["unknown_table"] == ("error", "unknown relation(s): c")
    assert "missing" in status["unknown_column"][1]
    assert status["cycle_a"][1].startswith("dependency cycle: cycle_a -> cycle_b -> cycle_a")


def test_value_through_control_becomes_control():
    p = _project({
        "f": "SELECT id, v FROM t WHERE s = 'keep'",
        "g": "SELECT id FROM f WHERE v > 1",
    })
    g = LineageGraph(analyze_project(p), {"t"})
    assert g.trace("f.v") == {"t.v": Dep(IDENTITY, False), "t.s": Dep(None, True)}
    # g.id: value from t.id; t.v and t.s only decide which rows exist
    assert g.trace("g.id") == {"t.id": Dep(IDENTITY, False), "t.v": Dep(None, True), "t.s": Dep(None, True)}


def test_count_star_over_filtered_model_sees_upstream_filter():
    p = _project({"f": "SELECT id FROM t WHERE s = 'keep'", "n": "SELECT count(*) AS n FROM f"})
    g = LineageGraph(analyze_project(p), {"t"})
    assert g.trace("n.n") == {"t.s": Dep(None, True)}


def test_impact_and_value_only(shop):
    _, _, g = shop
    hits = g.impact("orders.discount")
    assert set(hits) == {
        "stg_orders.discount", "order_totals.net", "customer_ltv.ltv", "customer_rank.ltv",
        "customer_rank.country_rank", "vip_customers.customer_id", "vip_customers.name", "vip_customers.ltv",
    }
    assert set(g.impact("orders.discount", value_only=True)) == {
        "stg_orders.discount", "order_totals.net", "customer_ltv.ltv", "customer_rank.ltv",
        "customer_rank.country_rank", "vip_customers.ltv",
    }
    assert g.impact("customers.created_at") == {}


def test_explain_paths(shop):
    _, _, g = shop
    paths = g.explain("vip_customers.name", "orders.discount")
    assert [n for n, _ in paths[0]] == ["customer_ltv.ltv", "order_totals.net", "stg_orders.discount", "orders.discount"]
    assert paths[0][0][1] == "control"
    assert g.explain("stg_items.quantity", "orders.discount") == []


def test_schema_change_breaks_and_reshapes(shop):
    project, _, _ = shop
    after = analyze_project(project.with_source_changes(["orders.discount"]))
    assert after["stg_orders"].status == "error"
    assert after["vip_customers"].status == "blocked"
    after = analyze_project(project.with_source_changes(["refunds.reason"]))
    assert list(after["refunds_enriched"].columns) == ["order_id", "amount", "customer_id"]
    after = analyze_project(project.with_source_changes([], {"customers.name": "full_name"}))
    assert after["customer_ltv"].status == "error"


def test_model_order_handles_cycles():
    p = _project({"x": "SELECT id FROM y", "y": "SELECT id FROM x", "z": "SELECT id FROM t"})
    order, deps, errors = model_order(p)
    assert order[0] == "z" and set(errors) == {"x", "y"}
