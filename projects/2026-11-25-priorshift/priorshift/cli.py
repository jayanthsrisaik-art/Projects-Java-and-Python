"""Command line interface: python -m priorshift <command>."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import data
from . import estimators as est
from .detect import detect_shift
from .evaluate import Config, run, to_markdown, write
from .model import ShiftModel, fit

PIXELS = [f"p{i}" for i in range(64)]


def read_pixels(path: Path) -> np.ndarray:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in PIXELS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing[:5]}{'...' if len(missing) > 5 else ''}")
        rows = [[float(r[c]) for c in PIXELS] for r in reader]
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return np.asarray(rows)


def _round(v, nd=4):
    return [round(float(x), nd) for x in v]


def cmd_train(a) -> int:
    X, y = data.load(a.factor)
    m = fit(data.make_splits(X, y, a.seed), data.N_CLASSES, factor=a.factor)
    m.save(a.out)
    print(json.dumps({"model": str(a.out), "seed": a.seed, "factor": a.factor,
                      "train_priors": _round(m.train_priors),
                      **{k: round(v, 4) for k, v in m.diagnostics.items()}}, indent=2))
    return 0


def estimate_batch(m: ShiftModel, X_pixels: np.ndarray, method: str, smooth: bool, alpha: float, min_tv: float):
    P = m.proba(m.features(X_pixels))
    n = len(P)
    test = detect_shift(P, m.train_priors, alpha=alpha, min_tv=min_tv)
    q = m.estimate_priors(P, method)
    q_used = est.smooth_priors(q, n) if smooth else q
    adj = est.adjust_posteriors(P, m.train_priors, q_used) if test.shifted else P
    summary = {
        "n": n,
        "method": method,
        "smoothed": smooth,
        "train_priors": _round(m.train_priors),
        "estimated_priors": _round(q),
        "shift_test": {
            "statistic": round(test.statistic, 2), "df": test.df,
            "p_value": float(f"{test.p_value:.3g}"), "tv_distance": round(test.tv_distance, 4),
            "shifted": test.shifted,
        },
        "adapted": test.shifted,
        "predicted_counts_before": np.bincount(P.argmax(1), minlength=m.n_classes).tolist(),
        "predicted_counts_after": np.bincount(adj.argmax(1), minlength=m.n_classes).tolist(),
    }
    return summary, P, adj


def cmd_estimate(a) -> int:
    m = ShiftModel.load(a.model)
    summary, P, adj = estimate_batch(m, read_pixels(a.batch), a.method, not a.no_smooth, a.alpha, a.min_tv)
    if a.labels:
        y = np.loadtxt(a.labels, delimiter=",", skiprows=1, dtype=int, ndmin=1)
        if len(y) != summary["n"]:
            raise SystemExit("labels file length does not match batch")
        q_true = np.bincount(y, minlength=m.n_classes) / len(y)
        summary["check_with_labels"] = {
            "true_priors": _round(q_true),
            "prior_tv_error": round(0.5 * float(np.abs(np.asarray(summary["estimated_priors"]) - q_true).sum()), 4),
            "accuracy_before": round(float((P.argmax(1) == y).mean()), 4),
            "accuracy_after": round(float((adj.argmax(1) == y).mean()), 4),
        }
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "pred_before", "pred_after", "confidence_after"])
            for i, (p0, p1) in enumerate(zip(P, adj)):
                w.writerow([i, int(p0.argmax()), int(p1.argmax()), round(float(p1.max()), 4)])
    print(json.dumps(summary, indent=2))
    return 0


def cmd_make_batch(a) -> int:
    """Write a label-shifted batch from the held-out pool of split `seed` (raw pixels + labels)."""
    priors = np.asarray([float(x) for x in a.priors.split(",")])
    if priors.size != data.N_CLASSES or np.any(priors < 0) or priors.sum() <= 0:
        raise SystemExit("--priors needs 10 non-negative comma-separated weights")
    priors = priors / priors.sum()
    X_raw, y = data.load(factor=1)
    s = data.make_splits(X_raw, y, a.seed)  # same split as `train --seed`: pool rows were never fitted on
    idx = data.sample_batch(s.y_pool, priors, a.n, np.random.default_rng(a.batch_seed))
    out = Path(a.out)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PIXELS)
        for row in s.X_pool[idx]:
            w.writerow([int(v) for v in row])
    if a.labels:
        with open(a.labels, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label"])
            for v in s.y_pool[idx]:
                w.writerow([int(v)])
    print(f"wrote {len(idx)} rows to {out}")
    return 0


def cmd_benchmark(a) -> int:
    cfg = Config(factor=a.factor)
    if a.quick:
        cfg = Config(seeds=(0, 1), batch_sizes=(100, 500), repeats=5, factor=a.factor)
    rep = run(cfg)
    write(rep, a.out)
    print(to_markdown(rep).split("## By scenario")[0])
    print(f"full report: {Path(a.out) / 'REPORT.md'}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="priorshift", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fit classifier + calibration + confusion matrices")
    t.add_argument("--out", type=Path, default=Path("model.joblib"))
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--factor", type=int, default=2, choices=(1, 2, 4))
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("estimate", help="estimate the class mix of an unlabeled batch and adapt predictions")
    e.add_argument("--model", type=Path, default=Path("model.joblib"))
    e.add_argument("--batch", type=Path, required=True, help="CSV with columns p0..p63")
    e.add_argument("--method", default="em", choices=[m for m in est.METHODS if m != "none"])
    e.add_argument("--no-smooth", action="store_true", help="disable add-one smoothing of the estimate")
    e.add_argument("--alpha", type=float, default=0.01)
    e.add_argument("--min-tv", type=float, default=0.05)
    e.add_argument("--labels", type=Path, help="optional label CSV, only used to report errors")
    e.add_argument("--out", type=Path, help="write per-row predictions CSV")
    e.set_defaults(func=cmd_estimate)

    mb = sub.add_parser("make-batch", help="write a label-shifted batch from the held-out pool")
    mb.add_argument("--priors", required=True, help="10 comma-separated class weights")
    mb.add_argument("--n", type=int, default=300)
    mb.add_argument("--seed", type=int, default=0, help="split seed (must match train --seed)")
    mb.add_argument("--batch-seed", type=int, default=7)
    mb.add_argument("--out", type=Path, required=True)
    mb.add_argument("--labels", type=Path)
    mb.set_defaults(func=cmd_make_batch)

    b = sub.add_parser("benchmark", help="run the simulated label-shift benchmark")
    b.add_argument("--out", type=Path, default=Path("results"))
    b.add_argument("--factor", type=int, default=2, choices=(1, 2, 4))
    b.add_argument("--quick", action="store_true")
    b.set_defaults(func=cmd_benchmark)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
