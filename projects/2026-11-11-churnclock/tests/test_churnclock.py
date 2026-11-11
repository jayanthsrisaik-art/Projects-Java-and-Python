import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from churnclock import features as F
from churnclock import pipeline, synth
from churnclock.cli import main, predict_frame
from churnclock.metrics import (
    brier_ipcw,
    calibration_table,
    censoring_km,
    concordance_index,
    integrated_brier,
    uno_c_index,
)
from churnclock.survival import CoxPH, KaplanMeier, breslow_loglik

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture(scope="module")
def tables():
    return synth.generate(synth.SynthConfig(n_customers=6000, seed=7))


@pytest.fixture(scope="module")
def result(tables):
    return pipeline.run(tables)


# ---------------------------------------------------------------- Kaplan-Meier
def test_km_hand_computed():
    # times 1,2,2,3,4(c),5 ; events at 1,2,2,3,5
    km = KaplanMeier().fit([1, 2, 2, 3, 4, 5], [1, 1, 1, 1, 0, 1])
    expected = np.cumprod([1 - 1 / 6, 1 - 2 / 5, 1 - 1 / 3, 1 - 1 / 1])
    assert np.allclose(km.surv, expected)
    assert km.predict([0.5, 1, 1.5, 2, 4.9])[0] == 1.0
    assert km.predict([2])[0] == pytest.approx(expected[1])
    assert km.predict_left([2])[0] == pytest.approx(expected[0])


def test_km_matches_scipy():
    rng = np.random.default_rng(0)
    t = np.ceil(rng.exponential(10, 400))
    c = np.ceil(rng.exponential(15, 400))
    d, e = np.minimum(t, c), (t <= c).astype(int)
    km = KaplanMeier().fit(d, e)
    ref = stats.ecdf(stats.CensoredData(uncensored=d[e == 1], right=d[e == 0])).sf
    grid = np.arange(0, 40, 0.5)
    assert np.allclose(km.predict(grid), ref.evaluate(grid), atol=1e-12)


# ---------------------------------------------------------------- Cox PH
def _brute_breslow(beta, d, e, X):
    ll = 0.0
    for i in range(len(d)):
        if e[i]:
            risk = [j for j in range(len(d)) if d[j] >= d[i]]
            ll += X[i] @ beta - np.log(sum(np.exp(X[j] @ beta) for j in risk))
    return ll


def test_breslow_loglik_matches_brute_force_with_ties():
    rng = np.random.default_rng(1)
    d = rng.integers(1, 6, 25).astype(float)  # many ties
    e = rng.integers(0, 2, 25)
    X = rng.normal(size=(25, 3))
    beta = np.array([0.3, -0.7, 0.2])
    assert breslow_loglik(beta, d, e, X) == pytest.approx(_brute_breslow(beta, d, e, X), rel=1e-10)


def test_cox_fit_is_stationary_point_and_beats_neighbours():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 3))
    d = np.ceil(rng.exponential(np.exp(-X @ [0.5, -0.3, 0.0])) * 10)
    e = (rng.random(300) < 0.8).astype(int)
    m = CoxPH(l2=0.0).fit(X, d, e)
    Xs = (X - m.mean_) / m.scale_
    ll = breslow_loglik(m.beta, d, e, Xs)
    assert ll == pytest.approx(m.loglik_, rel=1e-9)
    for k, s in itertools.product(range(3), (-1e-3, 1e-3)):
        b = m.beta.copy()
        b[k] += s
        assert breslow_loglik(b, d, e, Xs) <= ll + 1e-9
    # numerical gradient ~ 0
    g = [(breslow_loglik(m.beta + h, d, e, Xs) - breslow_loglik(m.beta - h, d, e, Xs)) / 2e-6
         for h in np.eye(3) * 1e-6]
    assert np.max(np.abs(g)) < 1e-4


def test_cox_recovers_true_coefficients():
    rng = np.random.default_rng(3)
    n = 20000
    X = np.column_stack([rng.normal(size=n), rng.integers(0, 2, n), rng.normal(size=n)])
    true = np.array([0.6, -0.8, 0.0])
    t = rng.weibull(1.3, n) / np.exp(X @ true) ** (1 / 1.3) * 100
    c = rng.exponential(150, n)
    m = CoxPH().fit(X, np.minimum(t, c), (t <= c).astype(int))
    assert np.allclose(m.coef_, true, atol=0.05)


def test_cox_baseline_hazard_without_covariate_effect_equals_nelson_aalen():
    rng = np.random.default_rng(4)
    d = np.ceil(rng.exponential(20, 200))
    e = (rng.random(200) < 0.7).astype(int)
    X = rng.normal(size=(200, 1))
    m = CoxPH(l2=1e9).fit(X, d, e)  # huge ridge penalty forces beta ~ 0
    times = np.unique(d[e == 1])
    na = np.cumsum([e[d == t].sum() / (d >= t).sum() for t in times])
    assert np.allclose(m.base_cumhaz, na, rtol=1e-6)


def test_cox_serialisation_roundtrip(result, tables):
    cox = result["cox"]
    again = CoxPH.from_dict(json.loads(json.dumps(cox.to_dict())))
    X = F.build_dataset(tables["customers"], tables["usage"])[F.feature_columns()].to_numpy(float)[:50]
    assert np.allclose(cox.survival(X, [90, 365]), again.survival(X, [90, 365]))


# ---------------------------------------------------------------- metrics
def _brute_c(t, e, r):
    num = den = 0.0
    for i in range(len(t)):
        for j in range(len(t)):
            if e[i] and (t[i] < t[j] or (t[i] == t[j] and not e[j])):
                den += 1
                num += 1.0 if r[i] > r[j] else 0.5 if r[i] == r[j] else 0.0
    return num / den


def test_concordance_matches_brute_force():
    rng = np.random.default_rng(5)
    t = rng.integers(1, 15, 120).astype(float)
    e = rng.integers(0, 2, 120)
    r = rng.integers(0, 5, 120).astype(float)  # risk ties too
    assert concordance_index(t, e, r, chunk=7) == pytest.approx(_brute_c(t, e, r))
    tc = rng.permutation(120).astype(float) + 1  # distinct times: perfect ranking gives C = 1
    assert concordance_index(tc, e, -tc) == pytest.approx(1.0)
    assert concordance_index(tc, e, tc) == pytest.approx(0.0)


def test_ipcw_brier_without_censoring_is_plain_brier():
    rng = np.random.default_rng(6)
    t = rng.exponential(10, 500)
    e = np.ones(500, int)
    S = rng.random(500)
    G = censoring_km(t, e)
    plain = np.mean(((t > 8).astype(float) - S) ** 2)
    assert brier_ipcw(t, e, S, 8.0, G) == pytest.approx(plain)


def test_ipcw_brier_is_approximately_unbiased_under_censoring():
    rng = np.random.default_rng(7)
    n = 40000
    t = rng.exponential(10, n)
    c = rng.exponential(12, n)
    S = np.full(n, np.exp(-8 / 10))  # true marginal survival at 8
    true_bs = np.mean(((t > 8) - S) ** 2)
    d, e = np.minimum(t, c), (t <= c).astype(int)
    est = brier_ipcw(d, e, S, 8.0, censoring_km(d, e))
    assert est == pytest.approx(true_bs, abs=0.01)
    # ignoring censored rows altogether is biased low/high; IPCW is closer
    obs = (d > 8) | (e == 1)
    naive = np.mean((((d > 8)[obs]).astype(float) - S[obs]) ** 2)
    assert abs(est - true_bs) < abs(naive - true_bs)


def test_integrated_brier_constant_scores():
    t = np.array([5.0, 10, 15, 20])
    e = np.ones(4, int)
    grid = np.array([1.0, 2, 3])
    S = np.ones((4, 3))  # everyone survives past the grid: Brier = 0 everywhere
    assert integrated_brier(t, e, S, grid, censoring_km(t, e)) == 0.0


def test_calibration_table_bins_cover_everyone():
    rng = np.random.default_rng(8)
    t = rng.exponential(10, 1000)
    e = np.ones(1000, int)
    p = rng.random(1000)
    rows = calibration_table(t, e, p, 5.0, n_bins=5)
    assert sum(r["n"] for r in rows) == 1000 and len(rows) == 5


# ---------------------------------------------------------------- data + leakage
def test_generator_is_deterministic_and_consistent(tables):
    again = synth.generate(synth.SynthConfig(n_customers=6000, seed=7))
    for k in tables:
        pd.testing.assert_frame_equal(tables[k], again[k])
    c = tables["customers"]
    assert set(c["status"]) == {"active", "cancelled", "closed_other"}
    assert (c.loc[c.status == "active", "end_date"] == "").all()
    assert (c.loc[c.status != "active", "end_date"] != "").all()
    d = F.durations(c)
    assert (d["days_observed"] <= 720).all() and (d["days_observed"] >= 0).all()


def test_landmark_features_ignore_post_landmark_usage(tables):
    c, u = tables["customers"], tables["usage"]
    base = F.landmark_features(c, u)
    u2 = u.copy()
    later = u2["period"] > 0
    u2.loc[later, ["logins", "tickets", "payment_failed"]] = 999
    pd.testing.assert_frame_equal(base, F.landmark_features(c, u2))
    # the leaky aggregates DO change: that is exactly why they are leaky
    assert not F.leaky_features(c, u).equals(F.leaky_features(c, u2))


def test_landmark_cohort_excludes_early_churn_and_short_follow_up(tables):
    c = tables["customers"]
    cohort = F.landmark_cohort(c)
    d = F.durations(c).set_index("customer_id")
    assert (d.loc[cohort["customer_id"], "days_observed"] > 30).all()
    assert (cohort["duration"] >= 1).all()
    excluded = set(c["customer_id"]) - set(cohort["customer_id"])
    assert (d.loc[list(excluded), "days_observed"] <= 30).all()


def test_split_is_reproducible_disjoint_and_order_independent():
    ids = [f"C{i:05d}" for i in range(4000)]
    a = pipeline.split_ids(ids)
    b = pipeline.split_ids(ids[::-1])[::-1]
    assert (a == b).all()
    assert 0.22 < a.mean() < 0.28
    assert not (a == pipeline.split_ids(ids, seed=1)).all()


def test_unknown_category_rejected(tables):
    c = tables["customers"].head(3).copy()
    c.loc[0, "plan"] = "enterprise"
    with pytest.raises(ValueError, match="unknown plan"):
        F.landmark_features(c, tables["usage"])


# ---------------------------------------------------------------- end-to-end claims
def test_report_matches_committed_expected_metrics(result):
    expected = json.loads((SAMPLES / "expected_metrics.json").read_text())
    got = json.loads(json.dumps(result["report"]))
    assert got.keys() == expected.keys()
    for name, m in expected["models"].items():
        for k, v in m.items():
            if isinstance(v, float):
                assert got["models"][name][k] == pytest.approx(v, abs=2e-3), (name, k)


def test_headline_claims(result):
    m = result["report"]["models"]
    km180 = m["cox"]["km_test_churn_180"]
    # Cox beats the population KM and both logistic shortcuts on integrated Brier score
    assert m["cox"]["ibs"] < min(m["kaplan_meier"]["ibs"], m["logistic_censored_as_retained"]["ibs"],
                                 m["logistic_drop_censored"]["ibs"])
    assert m["cox"]["c_index"] > 0.65 and m["kaplan_meier"]["c_index"] == 0.5
    # the oracle (true hazard) bounds what any model can reach
    assert m["oracle"]["c_index"] >= m["cox"]["c_index"] and m["oracle"]["ibs"] <= m["cox"]["ibs"]
    # censoring shortcuts are biased in opposite directions; Cox is close to the KM estimate
    assert m["logistic_censored_as_retained"]["mean_pred_churn_180"] < km180 - 0.03
    assert m["logistic_drop_censored"]["mean_pred_churn_180"] > km180 + 0.03
    assert abs(m["cox"]["mean_pred_churn_180"] - km180) < 0.02
    # lifetime aggregates leak the outcome and inflate offline ranking metrics
    assert m["cox_leaky"]["c_index"] > 0.95


def test_cox_coefficients_close_to_generator_truth(result):
    co = result["report"]["cox_coefficients"]
    for name in ("tickets_30d", "payment_failed_30d", "discount", "channel_paid_social"):
        assert co[name]["estimated"] == pytest.approx(co[name]["true"], abs=0.15), name


# ---------------------------------------------------------------- CLI + samples
def test_predict_sample_matches_expected():
    model = CoxPH.from_dict(json.loads((SAMPLES / "cox.model.json").read_text()))
    read = lambda p: pd.read_csv(p, dtype={"customer_id": str}, keep_default_na=False)
    got = predict_frame(model, read(SAMPLES / "new_customers.csv"), read(SAMPLES / "new_usage.csv"), [90, 180, 365])
    exp = read(SAMPLES / "expected_predictions.csv")
    got = pd.read_csv(pd.io.common.StringIO(got.to_csv(index=False)), dtype={"customer_id": str}, keep_default_na=False)
    pd.testing.assert_frame_equal(got, exp)
    # monotone in horizon; customer without period-0 usage is flagged
    assert (got["churn_by_90d"] <= got["churn_by_180d"]).all() and (got["churn_by_180d"] <= got["churn_by_365d"]).all()
    assert got.set_index("customer_id").loc["N006", "has_period0_usage"] == 0


def test_predict_ignores_post_landmark_rows():
    model = CoxPH.from_dict(json.loads((SAMPLES / "cox.model.json").read_text()))
    read = lambda p: pd.read_csv(p, dtype={"customer_id": str}, keep_default_na=False)
    c, u = read(SAMPLES / "new_customers.csv"), read(SAMPLES / "new_usage.csv")
    a = predict_frame(model, c, u, [180])
    b = predict_frame(model, c, u[u["period"] == 0], [180])
    pd.testing.assert_frame_equal(a, b)


def test_cli_generate_run_predict(tmp_path, capsys):
    assert main(["generate", "--out", str(tmp_path / "d"), "--n", "1500", "--seed", "11"]) == 0
    assert main(["run", "--data", str(tmp_path / "d"), "--out", str(tmp_path / "r")]) == 0
    rep = json.loads((tmp_path / "r" / "metrics.json").read_text())
    assert rep["n_customers"] == 1500 and "cox" in rep["models"]
    assert main(["predict", "--model", str(tmp_path / "r" / "cox.model.json"),
                 "--customers", str(SAMPLES / "new_customers.csv"),
                 "--usage", str(SAMPLES / "new_usage.csv"), "--out", str(tmp_path / "p.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "p.csv")) == 6
    assert main(["run", "--data", str(tmp_path / "missing"), "--out", str(tmp_path / "x")]) == 2


# ---------------------------------------------------------------- follow-up changes
def test_uno_c_equals_harrell_without_censoring_and_matches_brute_force():
    rng = np.random.default_rng(9)
    t = rng.permutation(150).astype(float) + 1
    r = rng.normal(size=150)
    e = np.ones(150, int)
    G = censoring_km(t, e)  # no censoring: G == 1 everywhere
    assert uno_c_index(t, e, r, G) == pytest.approx(concordance_index(t, e, r))
    # with censoring and truncation: compare with an O(n^2) weighted loop
    e = rng.integers(0, 2, 150)
    G = censoring_km(t, e)
    w = 1 / G.predict_left(t) ** 2
    num = den = 0.0
    for i in range(150):
        if e[i] and t[i] < 100:
            for j in range(150):
                if t[i] < t[j]:
                    den += w[i]
                    num += w[i] * (1.0 if r[i] > r[j] else 0.5 if r[i] == r[j] else 0.0)
    assert uno_c_index(t, e, r, G, tau=100, chunk=11) == pytest.approx(num / den)


def test_report_includes_uno_c(result):
    m = result["report"]["models"]
    assert m["kaplan_meier"]["uno_c_365"] == 0.5
    assert m["oracle"]["uno_c_365"] >= m["cox"]["uno_c_365"] > 0.65


@pytest.mark.parametrize(
    "X, d, e, msg",
    [
        ([[np.nan], [1.0]], [1, 2], [1, 0], "finite"),
        ([[0.0], [1.0]], [-1, 2], [1, 0], "non-negative"),
        ([[0.0], [1.0]], [1, 2], [2, 0], "0 \\(censored\\) or 1"),
        ([[0.0], [1.0]], [1, 2], [0, 0], "no events"),
    ],
)
def test_cox_rejects_bad_inputs(X, d, e, msg):
    with pytest.raises(ValueError, match=msg):
        CoxPH().fit(np.array(X), np.array(d, float), np.array(e))

