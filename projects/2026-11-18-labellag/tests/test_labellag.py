import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labellag.evaluate import BacktestConfig, metrics, run_cutoff, summarise
from labellag.labels import TRUTH_COLUMNS, estimate_delay_curve, observe, posterior_unreported_fraud
from labellag.model import STRATEGIES, ModelConfig, _ipw_weights, _soft_dataset, predict, train
from labellag.simulate import FEATURES, REPORT_HORIZON_DAYS, Campaign, SimConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
SMALL = SimConfig(days=300, txns_per_day=400)
FAST = ModelConfig(max_iter=40, min_samples_leaf=100)


@pytest.fixture(scope="module")
def world():
    df, campaigns = simulate(SMALL, seed=7)
    return df, campaigns


# --- simulator -------------------------------------------------------------------------

def test_simulate_is_deterministic_and_well_formed(world):
    df, _ = world
    again, _ = simulate(SMALL, seed=7)
    pd.testing.assert_frame_equal(df, again)
    assert df["day"].is_monotonic_increasing
    fraud = df["is_fraud"] == 1
    assert df.loc[~fraud, "report_day"].isna().all()
    delay = df.loc[fraud, "report_day"] - df.loc[fraud, "day"]
    assert (delay > 0).all() and (delay <= REPORT_HORIZON_DAYS + 1e-9).all()
    assert 0.002 < df["is_fraud"].mean() < 0.05


def test_campaign_intensity_trapezoid():
    c = Campaign(start=10, duration=100, mcc=0, channel=0, amount_lo=1, amount_hi=2, peak_logit=4.0)
    v = c.intensity(np.array([0.0, 10.0, 17.5, 25.0, 60.0, 102.5, 110.0, 200.0]))
    np.testing.assert_allclose(v, [0, 0, 2, 4, 4, 2, 0, 0])


def test_small_amount_frauds_report_later(world):
    df, _ = world
    f = df.loc[df["is_fraud"] == 1]
    d = f["report_day"] - f["day"]
    assert d[f["amount"] < 40].median() > 1.5 * d[f["amount"] >= 40].median()


# --- the observation gate (leakage) ----------------------------------------------------

def test_observe_hides_future_rows_unreported_labels_and_truth(world):
    df, _ = world
    cut = 200.0
    obs = observe(df, cut)
    assert obs["day"].max() <= cut
    assert not set(TRUTH_COLUMNS) & set(obs.columns)
    merged = obs.merge(df[["txn_id", "is_fraud", "report_day"]], on="txn_id")
    # a label is 1 iff the chargeback was reported by the cutoff
    expected = (merged["report_day"] <= cut).astype(int)
    assert (merged["label"] == expected).all()
    # and some frauds are still hidden at the cutoff: the problem this project is about
    hidden = (merged["is_fraud"] == 1) & (merged["label"] == 0)
    assert hidden.sum() > 0
    assert (merged.loc[merged["label"] == 1, "delay"] <= merged.loc[merged["label"] == 1, "age"] + 1e-9).all()


def test_training_is_invariant_to_information_after_cutoff(world):
    """Changing anything after the cutoff (future rows, late chargebacks) must not change a model."""
    df, _ = world
    cut = 220.0
    tampered = df.copy()
    late = tampered["report_day"] > cut
    tampered.loc[late, "report_day"] = cut + 1.0  # every late chargeback moved, still after cutoff
    future = tampered["day"] > cut
    tampered.loc[future, "amount"] = 1.0
    tampered.loc[future, "is_fraud"] = 1
    for strategy in ("naive", "ipw"):
        a = train(observe(df, cut), cut, strategy, cfg=FAST)
        b = train(observe(tampered, cut), cut, strategy, cfg=FAST)
        probe = df.loc[df["day"] <= cut].head(500)
        np.testing.assert_array_equal(predict(a.model, probe), predict(b.model, probe))


# --- delay curve -----------------------------------------------------------------------

def test_delay_curve_uses_only_matured_rows_and_matches_truth(world):
    df, _ = world
    cut = 280.0
    obs = observe(df, cut)
    curve = estimate_delay_curve(obs, cut, n_strata=1)
    assert curve.cdf.shape[0] == 1
    assert np.all(np.diff(curve.cdf[0]) >= 0) and curve.cdf[0, -1] == 1.0
    # compare with the realised delay distribution of *all* frauds (truth)
    f = df.loc[df["is_fraud"] == 1]
    true_F30 = ((f["report_day"] - f["day"]) <= 30).mean()
    est_F30 = curve.reported_by(np.array([30.0]), np.array([50.0]))[0]
    assert abs(est_F30 - true_F30) < 0.08


def test_delay_curve_strata_separate_slow_small_amounts(world):
    df, _ = world
    obs = observe(df, 280.0)
    curve = estimate_delay_curve(obs, 280.0, n_strata=3, min_frauds=20)
    assert curve.cdf.shape[0] >= 2
    F = curve.reported_by(np.array([20.0, 20.0]), np.array([1.0, 10_000.0]))
    assert F[0] < F[1]  # small amounts are reported later
    assert curve.reported_by(np.array([REPORT_HORIZON_DAYS]), np.array([1.0]))[0] == 1.0


def test_delay_curve_refuses_without_matured_history(world):
    df, _ = world
    with pytest.raises(ValueError, match="matured frauds"):
        estimate_delay_curve(observe(df, 100.0), 100.0)


# --- correction maths ------------------------------------------------------------------

def test_posterior_unreported_fraud():
    p = np.array([0.0, 0.5, 0.5, 0.2, 1.0])
    F = np.array([0.5, 0.0, 1.0, 0.75, 0.0])
    q = posterior_unreported_fraud(p, F)
    # p(1-F) / (p(1-F) + 1-p)
    np.testing.assert_allclose(q, [0.0, 0.5, 0.0, 0.05 / 0.85, 1.0])


def test_ipw_weights_clip_and_leave_negatives_alone():
    w = _ipw_weights(np.array([True, True, False]), np.array([0.5, 0.01, 0.2]), min_F=0.1)
    np.testing.assert_allclose(w, [2.0, 10.0, 1.0])


def test_soft_dataset_preserves_total_weight():
    win = pd.DataFrame({c: np.arange(4.0) for c in FEATURES})
    unl = np.array([False, True, True, True])
    q = np.array([1.0, 0.3, 0.0, 0.9])
    d, y, w = _soft_dataset(win, q, unl)
    assert len(d) == 6  # rows 1 and 3 get a positive copy, row 2 (q=0) does not
    assert w.sum() == pytest.approx(4.0)
    assert (w * y).sum() == pytest.approx(1.0 + 0.3 + 0.9)


# --- strategies and evaluation ---------------------------------------------------------

def test_every_strategy_trains_and_scores(world):
    df, _ = world
    cut = 260.0
    obs = observe(df, cut)
    test = df.loc[(df["day"] > cut) & (df["day"] <= cut + 21)]
    for s in STRATEGIES:
        r = train(obs, cut, s, cfg=FAST)
        p = predict(r.model, test)
        assert p.shape == (len(test),) and np.all((p >= 0) & (p <= 1))
    mo = train(obs, cut, "mature_only", cfg=FAST)
    assert (obs.loc[obs["age"] < 180, "age"] >= mo.info["mature_age_days"]).sum() == mo.n_rows
    sl = train(obs, cut, "soft_label", cfg=FAST)
    assert sl.info["expected_hidden_frauds"] > 0
    with pytest.raises(ValueError):
        train(obs, cut, "bogus")


def test_metrics_known_values():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = metrics(y, s, budget=0.2)
    assert m["recall_at_budget"] == 0.5 and m["precision_at_budget"] == 0.5
    assert m["avg_precision"] == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert m["pred_rate_over_true_rate"] == pytest.approx(s.mean() / 0.2)


def test_run_cutoff_reports_every_method_on_identical_test_rows(world):
    df, _ = world
    bc = BacktestConfig(seeds=(7,), cutoffs=(260.0,), model=FAST, sim=SMALL)
    rows = pd.DataFrame(run_cutoff(df, 260.0, bc, seed=7))
    assert list(rows["method"]) == ["rules", *STRATEGIES, "oracle"]
    assert rows["n"].nunique() == 1 and rows["frauds"].nunique() == 1
    assert 0 < rows["last30_label_completeness"].iloc[0] < rows["train_label_completeness"].iloc[0] < 1
    s = summarise(rows)
    assert s.loc["naive", "ap_beats_naive"] == "-"
    assert s.loc["naive", "ap_minus_naive"] == 0.0


# --- CLI and samples -------------------------------------------------------------------

def _cli(*args, cwd):
    return subprocess.run([sys.executable, "-m", "labellag", *args], cwd=cwd, capture_output=True,
                          text=True, check=True, env={"PYTHONPATH": str(ROOT), "PATH": ""})


def test_sample_maturity_report_matches_expected(tmp_path):
    out = _cli("maturity", "--transactions", str(SAMPLES / "transactions.csv"),
               "--chargebacks", str(SAMPLES / "chargebacks.csv"), "--cutoff", "250", cwd=tmp_path)
    assert json.loads(out.stdout) == json.loads((SAMPLES / "expected_maturity.json").read_text())


def test_sample_train_and_score_match_expected(tmp_path):
    _cli("train", "--transactions", str(SAMPLES / "transactions.csv"), "--chargebacks",
         str(SAMPLES / "chargebacks.csv"), "--cutoff", "250", "--strategy", "ipw",
         "--out", str(tmp_path / "m.pkl"), cwd=tmp_path)
    _cli("score", "--model", str(tmp_path / "m.pkl"), "--input", str(SAMPLES / "new_transactions.csv"),
         "--budget", "0.01", "--out", str(tmp_path / "scores.csv"), cwd=tmp_path)
    got = pd.read_csv(tmp_path / "scores.csv")
    exp = pd.read_csv(SAMPLES / "expected_scores.csv")
    assert list(got["txn_id"]) == list(exp["txn_id"])
    np.testing.assert_allclose(got["fraud_probability"], exp["fraud_probability"], atol=1e-5)
    assert got["review"].sum() == exp["review"].sum() == int(np.ceil(0.01 * len(got)))


def test_load_history_rejects_bad_chargebacks(tmp_path):
    tx = pd.read_csv(SAMPLES / "transactions.csv").head(50)
    tx.to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame({"txn_id": [tx["txn_id"][3]], "report_day": [tx["day"][3] - 1]}).to_csv(tmp_path / "c.csv", index=False)
    r = subprocess.run([sys.executable, "-m", "labellag", "maturity", "--transactions", str(tmp_path / "t.csv"),
                        "--chargebacks", str(tmp_path / "c.csv"), "--cutoff", "10"],
                       capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "PATH": ""})
    assert r.returncode != 0 and "before their transaction" in r.stderr
