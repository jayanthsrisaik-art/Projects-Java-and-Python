import json
import sys
from pathlib import Path

import numpy as np

from meterfill.evaluate import evaluate_seed
from meterfill.simulate import SimConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_samples import SAMPLE_CONFIG, expected_outputs  # noqa: E402


def test_sample_outputs_match_expected():
    expected = json.loads((ROOT / "samples" / "expected_outputs.json").read_text())
    got = json.loads(json.dumps(expected_outputs(ROOT / "samples")))
    assert got["ingest"] == expected["ingest"]
    assert got["build"] == expected["build"]
    assert got["issues"] == expected["issues"]
    for m, days in expected["daily_usage"].items():
        for a, b in zip(days, got["daily_usage"][m]):
            assert a["start"] == b["start"] and a["hours"] == b["hours"]
            assert abs(a["kwh"] - b["kwh"]) < 1e-3


def test_sample_faults_are_all_detected():
    exp = json.loads((ROOT / "samples" / "expected_outputs.json").read_text())
    import csv

    truth = {(r["meter_id"], r["ts"], r["fault"]) for r in csv.DictReader(open(ROOT / "samples" / "truth_faults.csv"))}
    found = {(m, i["start"], i["kind"]) for m, iss in exp["issues"].items() for i in iss if i["kind"] in ("misread", "rollover", "reset")}
    assert found == truth


def test_simulation_is_deterministic():
    a = simulate(SimConfig(households=3, days=3, seed=9))
    b = simulate(SimConfig(households=3, days=3, seed=9))
    assert a.readings == b.readings and np.array_equal(a.truth, b.truth)
    assert simulate(SAMPLE_CONFIG).readings[:5] == simulate(SAMPLE_CONFIG).readings[:5]


def test_evaluation_runs_and_conserves_gap_totals():
    r = evaluate_seed(SimConfig(households=30, days=21, seed=3), n_windows=3)
    alloc = r["gap_allocation_test"]["allocated"]
    for m in ("uniform", "profile", "model"):
        assert alloc[m]["hours"] > 100
        assert alloc[m]["gap_total_error"] < 0.01  # register delta pins the total
    assert alloc["model_raw"]["gap_total_error"] > alloc["model"]["gap_total_error"]
    assert r["span_totals_test"]["meterfill"]["meters_over_1pct"] <= r["span_totals_test"]["naive"]["meters_over_1pct"]
    for k, d in r["detection"].items():
        if d["true"]:
            assert d["recall"] >= 0.9
