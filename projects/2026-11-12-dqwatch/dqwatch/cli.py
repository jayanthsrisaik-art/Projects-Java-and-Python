"""Command line: generate | profile | check | series | evaluate."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import date

from .config import DEFAULT_SPECS, load_config
from .detect import ALERT_STATUSES, detect_all, detect_series, save_results
from .generate import DEMO_CONFIG, generate, write_truth
from .profile import profile


def _seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cmd_generate(a) -> int:
    truth = generate(a.db, days=a.days, seed=a.seed)
    if a.truth:
        write_truth(truth, a.truth)
    if a.config_out:
        with open(a.config_out, "w") as f:
            json.dump(DEMO_CONFIG, f, indent=2)
    print(f"wrote {a.db}: {a.days} days, {len(truth['incidents'])} injected incidents")
    return 0


def cmd_profile(a) -> int:
    cfg = load_config(a.config)
    conn = sqlite3.connect(a.db)
    n = profile(conn, cfg, date.fromisoformat(a.start) if a.start else None,
                date.fromisoformat(a.end) if a.end else None)
    print(f"profiled {n} metric rows")
    return 0


def _fmt(v):
    return "-" if v is None else f"{v:.4g}"


def cmd_check(a) -> int:
    cfg = load_config(a.config)
    conn = sqlite3.connect(a.db)
    results = detect_all(conn, cfg)
    save_results(conn, results)
    if not results:
        print("no profiled metrics; run `profile` first", file=sys.stderr)
        return 2
    last = max(r.date for r in results)
    lo = a.since or a.date or last
    hi = a.date or last
    shown = sorted((r for r in results if lo <= r.date <= hi and r.status in ALERT_STATUSES),
                   key=lambda r: (r.date, r.table, r.metric))
    if a.json:
        print(json.dumps([r.as_dict() for r in shown], indent=2))
    else:
        print(f"checked {lo}..{hi}: {len(shown)} alert(s)")
        for r in shown:
            band = f"[{_fmt(r.lower)}, {_fmt(r.upper)}]"
            print(f"  {r.date} {r.table:<10} {r.metric:<24} {r.status:<11} value={_fmt(r.value)} "
                  f"expected={_fmt(r.expected)} band={band} {r.reason}")
    return 1 if (shown and a.fail_on_alert) else 0


def cmd_series(a) -> int:
    """Score a single metric series from CSV (date,value[,n]) -- handy for trying parameters."""
    cfg = load_config(a.config) if a.config else None
    params = cfg.detector if cfg else None
    if params is None:
        from .config import DetectorParams
        params = DetectorParams()
    spec = DEFAULT_SPECS[a.metric.split(":", 1)[0]]
    dates, values, ns = [], [], []
    with open(a.csv, newline="") as f:
        for row in csv.DictReader(f):
            dates.append(row["date"])
            values.append(float(row["value"]) if row["value"] != "" else None)
            ns.append(int(row.get("n") or 0))
    res = detect_series("series", a.metric, dates, values, ns, spec, params)
    rows = [r.as_dict() for r in res if a.all or r.status in ALERT_STATUSES]
    print(json.dumps(rows, indent=2))
    return 0


def cmd_evaluate(a) -> int:
    from .evaluate import evaluate, to_markdown, write_results

    res = evaluate(_seeds(a.seeds), days=a.days)
    if a.out:
        write_results(res, a.out, a.name)
    print(to_markdown(res))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dqwatch", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create a synthetic warehouse with known incidents")
    g.add_argument("--db", required=True)
    g.add_argument("--days", type=int, default=365)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--truth", help="write the injected incidents as JSON")
    g.add_argument("--config-out", help="write a matching monitor config")
    g.set_defaults(fn=cmd_generate)

    p = sub.add_parser("profile", help="profile partitions into dq_metrics")
    p.add_argument("--db", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--start")
    p.add_argument("--end")
    p.set_defaults(fn=cmd_profile)

    c = sub.add_parser("check", help="detect anomalies; print alerts for a date (default: latest)")
    c.add_argument("--db", required=True)
    c.add_argument("--config", required=True)
    c.add_argument("--date")
    c.add_argument("--since", help="show alerts from this date up to --date / latest")
    c.add_argument("--json", action="store_true")
    c.add_argument("--fail-on-alert", action="store_true", help="exit 1 when an alert is shown")
    c.set_defaults(fn=cmd_check)

    s = sub.add_parser("series", help="score one metric series from a CSV")
    s.add_argument("--csv", required=True)
    s.add_argument("--metric", required=True, help="e.g. row_count, null_rate:email, freshness_lag_h")
    s.add_argument("--config")
    s.add_argument("--all", action="store_true", help="print every day, not only alerts")
    s.set_defaults(fn=cmd_series)

    e = sub.add_parser("evaluate", help="benchmark against baselines on synthetic warehouses")
    e.add_argument("--seeds", default="11-15")
    e.add_argument("--days", type=int, default=365)
    e.add_argument("--out")
    e.add_argument("--name", default="eval")
    e.set_defaults(fn=cmd_evaluate)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, KeyError, FileNotFoundError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
