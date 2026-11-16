import json
import sqlite3

import pytest

from colineage.evaluate import colineage_pairs, evaluate, name_match_pairs, prf, soundness, table_level_pairs
from colineage.oracle import _perturb, build_db, generate_rows, run_oracle, snapshot
from colineage.project import analyze_project

from conftest import ROOT, SAMPLES


@pytest.fixture(scope="module")
def oracles(shop, saas):
    return {"shop": run_oracle(*shop[:2], seeds=(0,)), "saas": run_oracle(*saas[:2], seeds=(0,))}


def test_generated_data_respects_keys(shop):
    project = shop[0]
    data = generate_rows(project, seed=3)
    ids = [r["id"] for r in data["customers"]]
    assert len(set(ids)) == len(ids) == project.sources["customers"].rows
    assert {r["customer_id"] for r in data["orders"]} <= set(ids)
    assert generate_rows(project, seed=3) == data  # seeded


def test_views_run_and_sqlite_schema_matches_inferred(shop, saas):
    for project, results, _ in (shop, saas):
        conn = build_db(project, generate_rows(project, 0), results)
        snap = snapshot(conn, results)  # raises if SQLite's columns differ from the inferred ones
        assert set(snap) == {f"{m}.{c}" for m, r in results.items() for c in r.columns}


def test_static_lineage_is_sound_on_both_samples(shop, saas, oracles):
    for name, (_, _, graph) in (("shop", shop), ("saas", saas)):
        s = soundness(colineage_pairs(graph), oracles[name])
        assert s["missed"] == {}, s["missed"]
        assert s["observed_pairs"] > 100


def test_oracle_catches_a_lineage_bug(shop, oracles):
    """Mutation check: dropping control edges must be detected (the oracle is not vacuous)."""
    _, _, graph = shop
    s = soundness(colineage_pairs(graph, value_only=True), oracles["shop"])
    assert s["missed"]
    assert "orders.status" in s["missed"]["order_totals.gross"]


def test_truth_labels_contain_everything_the_oracle_observes(oracles):
    from colineage.evaluate import load_truth

    for name in ("shop", "saas"):
        truth, _ = load_truth(SAMPLES / name / "truth.json")
        assert soundness(truth, oracles[name])["missed"] == {}


def test_pk_perturbations_keep_keys_unique(shop):
    project, results, _ = shop
    conn = build_db(project, generate_rows(project, 0), results)
    import random

    for how in ("shuffle", "resample", "shift"):
        c = sqlite3.connect(":memory:")
        conn.backup(c)
        assert _perturb(c, "customers", "id", project.sources["customers"].columns["id"], how, random.Random(1))
        n, d = c.execute("SELECT count(*), count(DISTINCT id) FROM customers").fetchone()
        assert n == d
    assert not _perturb(conn, "customers", "id", {"type": "INTEGER", "pk": True}, "nullify", random.Random(1))


def test_prf_math():
    m = prf({"a": {"x", "y"}, "b": set()}, {"a": {"x"}, "b": {"z"}})
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["exact_columns"] == 0.0


def test_baselines_behave_as_described(shop):
    project, results, graph = shop
    tl = table_level_pairs(project, results)
    nm = name_match_pairs(project, results)
    cl = colineage_pairs(graph)
    for k in cl:  # table-level lineage is a superset of column lineage
        assert cl[k] <= tl[k]
    assert nm["stg_orders.order_id"] == set()  # `id AS order_id` defeats name matching
    assert nm["stg_orders.customer_id"] == {"orders.customer_id"}


def test_committed_bench_results_reproduce(shop, saas):
    """results/*.json were produced by `python -m colineage bench` with seeds 0 1 2."""
    for name, (project, results, graph) in (("shop", shop), ("saas", saas)):
        committed = json.loads((ROOT / "results" / f"{name}.json").read_text())
        oracle = run_oracle(project, results, seeds=(0, 1, 2))
        rep = json.loads(json.dumps(evaluate(project, results, graph, SAMPLES / name / "truth.json", oracle), sort_keys=True))
        assert rep == committed
