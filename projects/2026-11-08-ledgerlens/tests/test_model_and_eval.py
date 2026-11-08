import numpy as np

from ledgerlens.evaluate import dedupe, evaluate, recurring_report
from ledgerlens.model import (KeywordRuleBaseline, MerchantLookupBaseline, TxnClassifier,
                              pick_threshold)
from ledgerlens.synth import generate


def test_generator_is_reproducible():
    a, _ = generate(seed=5, n_accounts=4)
    b, _ = generate(seed=5, n_accounts=4)
    assert [t.__dict__ for t in a] == [t.__dict__ for t in b]


def test_pick_threshold_maximises_coverage_at_target():
    conf = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6]
    correct = [1, 1, 1, 1, 0, 0]
    assert pick_threshold(conf, correct, 0.95) == 0.8
    assert pick_threshold(conf, correct, 0.8) == 0.7   # 4/5 = 0.8 still meets target
    assert pick_threshold([0.9, 0.8], [0, 0], 0.95) == 1.0  # nothing qualifies -> abstain all


def test_keyword_rules_specific_phrases_beat_generic_words():
    kb = KeywordRuleBaseline().fit(["x"], [1.0], ["dining"])
    assert kb.predict_one("RECURRING PMT LAKESHORE GAS & ELECTRIC REF1") == "utilities"
    assert kb.predict_one("POS DEBIT 0101 STAR GAS #1 DOVER DE") == "fuel"
    assert kb.predict_one("ACH CREDIT HOOLI INC PAYROLL 1234") == "income"
    assert kb.predict_one("SQ *ZENTORA DOVER DE") is None


def test_merchant_lookup_memorises_seen_merchants_only():
    lk = MerchantLookupBaseline().fit(["TST* BLUE CAFE 1", "TST* BLUE CAFE 2", "POS DEBIT 0101 RED MART #1"],
                                      [-5, -6, -7], ["dining", "dining", "shopping"])
    assert lk.predict(["TST* BLUE CAFE 999", "SQ *NEVER SEEN"], [-1, -1]) == ["dining", "dining"]


def test_model_save_load_roundtrip(tmp_path, small_rows):
    rows = dedupe(small_rows)[:3000]
    m = TxnClassifier().fit([r["descriptor"] for r in rows], [r["amount"] for r in rows],
                            [r["category"] for r in rows])
    m.threshold = 0.5
    m.save(tmp_path / "m.pkl")
    m2 = TxnClassifier.load(tmp_path / "m.pkl")
    d, a = [r["descriptor"] for r in rows[:50]], [r["amount"] for r in rows[:50]]
    assert np.allclose(m.predict_proba(d, a), m2.predict_proba(d, a)) and m2.threshold == 0.5


def test_grouped_evaluation_ranks_models_and_exposes_leakage(small_rows):
    rep = evaluate(small_rows, seed=1, folds=3)
    g = rep["categorization"]["grouped_kfold"]["models"]
    leaky = rep["categorization"]["random_kfold_LEAKY"]["models"]
    f1 = lambda block, m: block[m]["vs_true_labels"]["macro_f1"]["mean"]
    # learned model beats rules, rules beat majority, on merchants never seen in training
    assert f1(g, "logreg") > f1(g, "keyword_rules") > f1(g, "majority")
    # memorisation is useless on unseen merchants but looks near-perfect on a leaky split
    assert f1(g, "merchant_lookup") < 0.1
    assert f1(leaky, "merchant_lookup") > 0.95
    assert f1(leaky, "logreg") - f1(g, "logreg") > 0.1
    sel = rep["categorization"]["grouped_kfold"]["selective"]
    # auto-accepted rows are more accurate than the overall model, rejected rows less
    overall = g["logreg"]["vs_true_labels"]["accuracy"]["mean"]
    assert sel["accepted_acc_true"]["mean"] > overall > sel["rejected_acc_true"]["mean"]
    assert 0.2 < sel["coverage"]["mean"] < 1.0


def test_recurring_detection_beats_raw_descriptor_grouping(small_rows):
    rep = recurring_report(dedupe(small_rows))
    norm, raw = rep["normalized_key"], rep["raw_descriptor_baseline"]
    assert norm["recall"] > 0.95 and norm["precision"] > 0.95
    assert raw["recall"] < 0.6
    assert rep["weekly_fixed_day_variable_amount_series"] > 0
    assert rep["of_which_flagged_recurring"] == 0
