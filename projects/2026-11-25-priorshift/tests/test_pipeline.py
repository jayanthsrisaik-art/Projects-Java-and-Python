import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from priorshift import data, evaluate
from priorshift.model import ShiftModel, fit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def digits():
    return data.load(factor=2)


def test_splits_are_disjoint_stratified_and_seeded(digits):
    X, y = digits
    s = data.make_splits(X, y, seed=3)
    assert len(s.y_train) + len(s.y_val) + len(s.y_pool) == len(y)
    # (row-level disjointness is checked by index in test_split_indices_never_overlap)
    for part in (s.y_train, s.y_val, s.y_pool):
        share = np.bincount(part, minlength=10) / len(part)
        assert np.abs(share - 0.1).max() < 0.02
    s2 = data.make_splits(X, y, seed=3)
    np.testing.assert_array_equal(s.y_pool, s2.y_pool)


def test_split_indices_never_overlap():
    idx = np.arange(1797)
    y = data.load(1)[1]
    s = data.make_splits(idx[:, None].astype(float), y, seed=0)
    parts = [set(s.X_train[:, 0]), set(s.X_val[:, 0]), set(s.X_pool[:, 0])]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])


def test_sample_batch_follows_priors():
    rng = np.random.default_rng(0)
    y_pool = np.repeat(np.arange(10), 45)
    q = np.array([0.5, 0.5] + [0.0] * 8)
    idx = data.sample_batch(y_pool, q, 1000, rng)
    counts = np.bincount(y_pool[idx], minlength=10)
    assert counts[2:].sum() == 0 and abs(counts[0] - 500) < 60


def test_downsample():
    X = np.arange(64, dtype=float)[None, :]
    D = data.downsample(X, 2)
    assert D.shape == (1, 16)
    assert D.mean() == pytest.approx(X.mean())
    assert D[0, 0] == pytest.approx(np.mean([0, 1, 8, 9]))


def test_fit_never_uses_the_pool(digits):
    """Scrambling the held-out pool must not change anything that was fitted."""
    X, y = digits
    s = data.make_splits(X, y, seed=0)
    m1 = fit(s, 10)
    rng = np.random.default_rng(0)
    s.X_pool = rng.normal(size=s.X_pool.shape)
    s.y_pool = rng.permutation(s.y_pool)
    m2 = fit(s, 10)
    np.testing.assert_allclose(m1.C_soft, m2.C_soft)
    np.testing.assert_allclose(m1.calibrator.bias, m2.calibrator.bias)
    assert m1.calibrator.temperature == m2.calibrator.temperature


def test_model_roundtrip_and_validation(tmp_path, digits):
    X, y = digits
    s = data.make_splits(X, y, seed=1)
    m = fit(s, 10, factor=2)
    m.save(tmp_path / "m.joblib")
    m2 = ShiftModel.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(m.proba(s.X_pool[:20]), m2.proba(s.X_pool[:20]))
    assert m.diagnostics["val_accuracy"] > 0.85
    assert m.diagnostics["val_nll_calibrated"] <= m.diagnostics["val_nll_raw"] + 1e-9
    with pytest.raises(ValueError):
        m.proba(np.zeros((2, 64)))  # pixels, not features
    with pytest.raises(ValueError):
        m.features(np.zeros((2, 16)))


def test_quick_benchmark_end_to_end():
    cfg = evaluate.Config(seeds=(0,), batch_sizes=(300,), repeats=4)
    rep = evaluate.run(cfg)
    o = rep["overall_shifted"]
    assert rep["config"]["batches"] == 5 * 4
    assert o["oracle"]["nll"]["mean"] < o["none"]["nll"]["mean"]
    assert o["em+smooth"]["nll"]["mean"] < o["none"]["nll"]["mean"]
    assert o["em"]["tv"]["mean"] < o["cc"]["tv"]["mean"] < o["none"]["tv"]["mean"]
    md = evaluate.to_markdown(rep)
    assert "| em+smooth |" in md and "Shift detection" in md


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "priorshift", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def test_cli_reproduces_sample_expected_outputs(tmp_path):
    expected = json.loads((ROOT / "samples" / "expected_outputs.json").read_text())
    model = tmp_path / "model.joblib"
    run_cli("train", "--out", str(model), "--seed", "0", "--factor", "2")
    for name in ("batch_form_b", "batch_balanced"):
        got = json.loads(run_cli("estimate", "--model", str(model),
                                 "--batch", str(ROOT / "samples" / f"{name}.csv"),
                                 "--labels", str(ROOT / "samples" / f"{name}_labels.csv")))
        exp = expected[name]
        assert got["adapted"] == exp["adapted"]
        assert got["predicted_counts_after"] == exp["predicted_counts_after"]
        np.testing.assert_allclose(got["estimated_priors"], exp["estimated_priors"], atol=2e-3)
        g, e = got["check_with_labels"], exp["check_with_labels"]
        assert g["true_priors"] == e["true_priors"]  # same rows, so exactly the same mix
        for k in ("prior_tv_error", "accuracy_before", "accuracy_after"):
            assert g[k] == pytest.approx(e[k], abs=2e-3)
    assert expected["batch_form_b"]["adapted"] is True
    assert expected["batch_balanced"]["adapted"] is False


def test_cli_writes_predictions_and_rejects_bad_input(tmp_path):
    model = tmp_path / "model.joblib"
    run_cli("train", "--out", str(model))
    out = tmp_path / "preds.csv"
    run_cli("estimate", "--model", str(model), "--batch", str(ROOT / "samples" / "batch_form_b.csv"),
            "--method", "pacc", "--out", str(out))
    lines = out.read_text().splitlines()
    assert lines[0] == "row,pred_before,pred_after,confidence_after" and len(lines) == 301
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    r = subprocess.run([sys.executable, "-m", "priorshift", "estimate", "--model", str(model), "--batch", str(bad)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "missing columns" in r.stderr
