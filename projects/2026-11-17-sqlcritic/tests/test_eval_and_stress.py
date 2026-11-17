import json

from sqlcritic import evaluate, golden, stress
from sqlcritic.contract import Contract
from sqlcritic.golden import Example

TOPK = Example("t", "x", "dev", "q",
               "SELECT c.name, COUNT(*) AS n FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
               "GROUP BY c.customer_id, c.name ORDER BY n DESC LIMIT 10", Contract("topk", k=10))


def test_topk_match_tolerates_ties_and_column_order(con):
    full = con.execute(TOPK.gold_sql.replace(" LIMIT 10", "")).fetchall()
    tenth = full[9][1]
    tied = [r for r in full if r[1] == tenth]
    assert len(tied) > 1, "fixture should have a tie at the cut-off"
    # take the other tied customer at position 10, and swap column order
    alt = full[:9] + [tied[-1]] if full[9] != tied[-1] else full[:9] + [tied[0]]
    assert evaluate.results_match(TOPK, [(v, n) for n, v in alt], 2, con)
    assert not evaluate.results_match(TOPK, full[:9] + [full[30]], 2, con)
    assert not evaluate.results_match(TOPK, full[:10], 3, con)


def test_float_rounding_and_multiset(con):
    ex = next(e for e in golden.generate() if e.family == "revenue_by_region_year")
    gold = con.execute(ex.gold_sql).fetchall()
    jittered = [(r[0], r[1] + 1e-7) for r in reversed(gold)]
    assert evaluate.results_match(ex, jittered, 2, con)
    assert not evaluate.results_match(ex, gold[:-1], 2, con)


def test_eval_pipeline_and_baseline_gate(con):
    from sqlcritic.writers import HeuristicWriter
    exs = [e for e in golden.generate() if e.family in ("orders_in_year", "revenue_by_region_year")]
    rep = evaluate.run_eval(exs, HeuristicWriter, con, modes=("single", "critic"))
    assert rep["modes"]["critic"]["dev"]["accuracy"] == 1.0
    assert evaluate.check_baseline(rep, rep) == []
    worse = json.loads(json.dumps(rep))
    worse["modes"]["critic"]["dev"]["accuracy"] = 0.5
    assert evaluate.check_baseline(worse, rep)          # regression is reported
    assert "| dev | critic |" in evaluate.to_markdown(rep, "t")


def test_stress_structural_caught_semantic_not(con):
    exs = [e for e in golden.generate() if e.split == "dev"]
    rep = stress.run_stress(exs, con)
    assert rep["false_reject_rate"]["gold_contract/dev"] == 0.0
    for name, r in rep["mutations"].items():
        if r["kind"] in ("structural", "safety"):
            assert r["recall_gold_contract"] == 1.0, name
        else:
            assert r["recall_gold_contract"] == 0.0, name
