"""Command line: simulate data, inspect label maturity, train, score, backtest."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from labellag.evaluate import BacktestConfig, backtest, summarise
from labellag.labels import estimate_delay_curve, observe
from labellag.model import STRATEGIES, predict, train
from labellag.simulate import FEATURES, SimConfig, simulate


def load_history(transactions: str, chargebacks: str) -> pd.DataFrame:
    """Join a transactions CSV (txn_id, day, features) with a chargebacks CSV (txn_id, report_day)."""
    tx = pd.read_csv(transactions)
    cb = pd.read_csv(chargebacks)
    missing = [c for c in ("txn_id", "day", *FEATURES) if c not in tx.columns]
    if missing:
        raise SystemExit(f"transactions file is missing columns: {missing}")
    if not {"txn_id", "report_day"} <= set(cb.columns):
        raise SystemExit("chargebacks file needs columns txn_id, report_day")
    if cb["txn_id"].duplicated().any():
        raise SystemExit("chargebacks file has duplicate txn_id values")
    df = tx.merge(cb[["txn_id", "report_day"]], on="txn_id", how="left", validate="one_to_one")
    bad = df["report_day"] < df["day"]
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} chargebacks are dated before their transaction")
    return df.sort_values("day", kind="stable").reset_index(drop=True)


def cmd_simulate(a) -> int:
    cfg = SimConfig(days=a.days, txns_per_day=a.txns_per_day)
    df, campaigns = simulate(cfg, seed=a.seed)
    df["day"] = df["day"].round(4)
    df["report_day"] = df["report_day"].round(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df[["txn_id", "day", *FEATURES]].to_csv(out / "transactions.csv", index=False)
    cb = df.loc[df["is_fraud"] == 1, ["txn_id", "report_day"]]
    cb.to_csv(out / "chargebacks.csv", index=False)
    df[["txn_id", "is_fraud", "true_p"]].to_csv(out / "truth.csv", index=False)
    with open(out / "campaigns.json", "w") as f:
        json.dump([c.__dict__ for c in campaigns], f, indent=2)
    print(f"wrote {len(df)} transactions, {len(cb)} chargebacks to {out}/")
    return 0


def cmd_maturity(a) -> int:
    df = load_history(a.transactions, a.chargebacks)
    obs = observe(df, a.cutoff)
    curve = estimate_delay_curve(obs, a.cutoff)
    recent = obs.loc[obs["age"] < 30]
    F = curve.reported_by(recent["age"].to_numpy(), recent["amount"].to_numpy())
    report = {
        "cutoff": a.cutoff,
        "delay_curve": curve.to_dict(),
        "q90_days": curve.quantile(0.90),
        "last30": {
            "rows": int(len(recent)),
            "reported_frauds": int(recent["label"].sum()),
            "mean_F": round(float(F.mean()), 4),
            "frauds_if_completed_ipw": round(float((recent["label"] / np.maximum(F, 0.1)).sum()), 1),
        },
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_train(a) -> int:
    df = load_history(a.transactions, a.chargebacks)
    obs = observe(df, a.cutoff)
    r = train(obs, a.cutoff, a.strategy, window_days=a.window_days)
    with open(a.out, "wb") as f:
        pickle.dump({"strategy": r.strategy, "cutoff": r.cutoff, "model": r.model, "features": FEATURES}, f)
    print(json.dumps({"strategy": r.strategy, "cutoff": r.cutoff, "train_rows": r.n_rows,
                      "effective_positives": round(r.n_positive, 1), **r.info}, indent=2))
    return 0


def cmd_score(a) -> int:
    with open(a.model, "rb") as f:  # only load model files you produced yourself
        bundle = pickle.load(f)
    df = pd.read_csv(a.input)
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise SystemExit(f"input is missing columns: {missing}")
    out = pd.DataFrame({"txn_id": df["txn_id"], "fraud_probability": np.round(predict(bundle["model"], df), 6)})
    k = int(np.ceil(a.budget * len(out)))
    order = np.argsort(-out["fraud_probability"].to_numpy(), kind="stable")
    out["review"] = 0
    out.loc[order[:k], "review"] = 1
    out.to_csv(a.out, index=False) if a.out else out.to_csv(sys.stdout, index=False)
    return 0


def cmd_backtest(a) -> int:
    bc = BacktestConfig(
        seeds=tuple(a.seeds),
        cutoffs=tuple(a.cutoffs),
        sim=SimConfig(txns_per_day=a.txns_per_day),
    )
    res = backtest(bc, progress=lambda m: print(m, file=sys.stderr))
    summ = summarise(res)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "backtest_runs.csv", index=False)
    summ.to_csv(out / "backtest_summary.csv")
    ctx = res.drop_duplicates(["seed", "cutoff"])
    with open(out / "backtest_summary.md", "w") as f:
        f.write(f"seeds={list(bc.seeds)} cutoffs={list(bc.cutoffs)} test_days={bc.test_days} "
                f"window_days={bc.window_days} budget={bc.budget} txns_per_day={bc.sim.txns_per_day}\n\n")
        f.write(summ.round(4).to_markdown() if _has_tabulate() else summ.round(4).to_string())
        f.write("\n\nlabel completeness (mean over runs): training window "
                f"{ctx['train_label_completeness'].mean():.3f}, last 30 days {ctx['last30_label_completeness'].mean():.3f}\n")
        f.write(f"delay curve F(14d): estimated {ctx['F14_estimated'].mean():.3f}, realised {ctx['F14_realised'].mean():.3f}\n")
    print((out / "backtest_summary.md").read_text())
    return 0


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="labellag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="write a synthetic transactions/chargebacks dataset")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--days", type=int, default=360)
    s.add_argument("--txns-per-day", type=int, default=1200)
    s.set_defaults(fn=cmd_simulate)

    for name, fn, hlp in (("maturity", cmd_maturity, "estimate the report-delay curve at a cutoff"),
                          ("train", cmd_train, "train a model with what was known at a cutoff")):
        t = sub.add_parser(name, help=hlp)
        t.add_argument("--transactions", required=True)
        t.add_argument("--chargebacks", required=True)
        t.add_argument("--cutoff", type=float, required=True)
        if name == "train":
            t.add_argument("--strategy", choices=STRATEGIES, default="ipw")
            t.add_argument("--window-days", type=float, default=180.0)
            t.add_argument("--out", required=True)
        t.set_defaults(fn=fn)

    sc = sub.add_parser("score", help="score transactions and flag the top --budget share for review")
    sc.add_argument("--model", required=True)
    sc.add_argument("--input", required=True)
    sc.add_argument("--budget", type=float, default=0.005)
    sc.add_argument("--out")
    sc.set_defaults(fn=cmd_score)

    b = sub.add_parser("backtest", help="rolling-origin backtest of every strategy")
    b.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    b.add_argument("--cutoffs", type=float, nargs="+", default=[240.0, 270.0, 300.0, 330.0])
    b.add_argument("--txns-per-day", type=int, default=1200)
    b.add_argument("--out", default="results")
    b.set_defaults(fn=cmd_backtest)

    a = p.parse_args(argv)
    return a.fn(a)
