"""Command line: generate | run | predict."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from churnclock import features as F
from churnclock import pipeline, synth
from churnclock.survival import CoxPH


REQUIRED_CUSTOMER_COLS = ("customer_id", "plan", "channel", "discount", "monthly_price")
REQUIRED_USAGE_COLS = ("customer_id", "period", "logins", "tickets", "payment_failed")


def _require(df: pd.DataFrame, cols, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing required columns: {', '.join(missing)}")


def predict_frame(model: CoxPH, customers: pd.DataFrame, usage: pd.DataFrame, horizons) -> pd.DataFrame:
    _require(customers, REQUIRED_CUSTOMER_COLS, "customers file")
    _require(usage, REQUIRED_USAGE_COLS, "usage file")
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("horizons must be positive day counts, e.g. 90,180,365")
    feats = F.landmark_features(customers, usage)
    cols = model.feature_names
    X = feats[cols].to_numpy(float)
    S = model.survival(X, horizons)
    med = model.median_survival(X)
    out = pd.DataFrame({"customer_id": feats["customer_id"]})
    out["has_period0_usage"] = feats["customer_id"].isin(set(usage.loc[usage["period"] == 0, "customer_id"])).astype(int)
    out["risk_score"] = np.round(model.risk_score(X), 4)
    for k, h in enumerate(horizons):
        out[f"churn_by_{h}d"] = np.round(1 - S[:, k], 4)
    out["median_days_after_landmark"] = [
        "never_below_0.5" if not np.isfinite(m) else int(m) for m in med
    ]
    return out.sort_values("risk_score", ascending=False, kind="stable").reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="churnclock", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write synthetic customers/usage/truth CSVs")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--n", type=int, default=6000)
    g.add_argument("--seed", type=int, default=7)

    r = sub.add_parser("run", help="fit all models on train, evaluate on test, write report + Cox model")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--split-seed", type=int, default=2026)
    r.add_argument("--test-frac", type=float, default=0.25)

    sd = sub.add_parser("seeds", help="repeat generate+run over several data seeds; print key metrics")
    sd.add_argument("--seeds", default="1,2,3,4,5")
    sd.add_argument("--n", type=int, default=6000)

    pr = sub.add_parser("predict", help="score customers at their day-30 landmark with a saved Cox model")
    pr.add_argument("--model", type=Path, required=True)
    pr.add_argument("--customers", type=Path, required=True)
    pr.add_argument("--usage", type=Path, required=True)
    pr.add_argument("--horizons", default="90,180,365")
    pr.add_argument("--out", type=Path)

    a = p.parse_args(argv)
    if a.cmd == "generate":
        tables = synth.generate(synth.SynthConfig(n_customers=a.n, seed=a.seed))
        synth.write(tables, a.out)
        c = tables["customers"]
        print(f"wrote {len(c)} customers, {len(tables['usage'])} usage rows to {a.out}")
        print(c["status"].value_counts().to_string())
        return 0
    if a.cmd == "run":
        tables = synth.read(a.data)
        if "customers" not in tables or "usage" not in tables:
            print(f"error: {a.data} needs customers.csv and usage.csv", file=sys.stderr)
            return 2
        result = pipeline.run(tables, seed=a.split_seed, test_frac=a.test_frac)
        pipeline.write_report(result, a.out)
        print(pipeline.format_report(result["report"]))
        print(f"\nwrote {a.out}/metrics.json, metrics.txt, cox.model.json")
        return 0
    if a.cmd == "seeds":
        keys = ("c_index", "ibs", "brier_180", "mean_pred_churn_180", "km_test_churn_180", "calibration_error_180")
        print(f"{'seed':>4s} {'model':32s} " + " ".join(f"{k:>21s}" for k in keys))
        for seed in [int(x) for x in a.seeds.split(",")]:
            rep = pipeline.run(synth.generate(synth.SynthConfig(n_customers=a.n, seed=seed)))["report"]
            for name, m in rep["models"].items():
                vals = " ".join(f"{'-' if m[k] is None else format(m[k], '.4f'):>21s}" for k in keys)
                print(f"{seed:4d} {name:32s} {vals}")
        return 0
    if a.cmd == "predict":
        model = CoxPH.from_dict(json.loads(a.model.read_text()))
        read = lambda path: pd.read_csv(path, dtype={"customer_id": str}, keep_default_na=False)
        try:
            horizons = [int(h) for h in a.horizons.split(",")]
            out = predict_frame(model, read(a.customers), read(a.usage), horizons)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if a.out:
            out.to_csv(a.out, index=False)
        print(out.to_string(index=False))
        return 0
    return 1
