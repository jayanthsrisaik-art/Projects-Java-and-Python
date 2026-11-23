"""Measure gtfsflat against its oracle and against common loader shortcuts.

For each seed and window:
  1. generate a synthetic feed, build it with the SQL engine, diff against the
     row-at-a-time oracle (must be 0 differences)
  2. diff each shortcut (and all of them together) against the correct events
  3. answer random departure-board queries ("next 3 departures from stop X in
     the next 60 minutes") and count wrong answers, split by night/day

Usage: python scripts/bench.py [--seeds 5] [--queries 2000] [--out results]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gtfsflat.board import departures  # noqa: E402
from gtfsflat.expand import build, fetch_events  # noqa: E402
from gtfsflat.feed import load_feed  # noqa: E402
from gtfsflat.oracle import Shortcuts, compare, expand_python  # noqa: E402
from gtfsflat.synth import generate  # noqa: E402

WINDOWS = {
    "spring": (date(2026, 3, 1), date(2026, 3, 14)),   # DST starts 2026-03-08, event service 03-07
    "fall": (date(2026, 10, 25), date(2026, 12, 5)),   # DST ends 2026-11-01, Thanksgiving, event 11-28
}
VARIANTS = {
    "ignore_exceptions": Shortcuts(ignore_exceptions=True),
    "wrap_24h": Shortcuts(wrap_24h=True),
    "wallclock": Shortcuts(wallclock=True),
    "ignore_frequencies": Shortcuts(ignore_frequencies=True),
    "naive_all": Shortcuts(True, True, True, True),
}
DST_DATES = {date(2026, 3, 8), date(2026, 11, 1)}


class ListBoard:
    """Departure board over an in-memory event list (for baseline event sets)."""

    def __init__(self, events):
        last = defaultdict(int)
        for iid, seq, *_ in events:
            last[iid] = max(last[iid], seq)
        self.by_stop = defaultdict(list)
        for iid, seq, stop, _arr, dep in events:
            if seq != last[iid]:
                self.by_stop[stop].append((dep, iid))
        for v in self.by_stop.values():
            v.sort()

    def next3(self, stop, lo, hi):
        return [(d, i.split("@")[0]) for d, i in self.by_stop.get(stop, []) if lo <= d < hi][:3]


def run_one(seed: int, window: str, n_queries: int, tmp: str) -> dict:
    start, end = WINDOWS[window]
    feed_dir = os.path.join(tmp, f"feed_{seed}")
    generate(feed_dir, seed)
    feed = load_feed(feed_dir)
    assert not feed.issues, feed.issues
    db = os.path.join(tmp, f"{window}_{seed}.db")
    if os.path.exists(db):
        os.remove(db)
    t0 = time.perf_counter()
    conn, counts = build(feed, db, start, end)
    t_sql = time.perf_counter() - t0
    sql_events = fetch_events(conn)
    t0 = time.perf_counter()
    truth = expand_python(feed, start, end)
    t_py = time.perf_counter() - t0
    res = {"seed": seed, "window": window, **counts, "sql_build_s": round(t_sql, 3),
           "oracle_s": round(t_py, 3), "sql_vs_oracle": compare(truth, sql_events), "variants": {}}

    variant_events = {name: expand_python(feed, start, end, sc) for name, sc in VARIANTS.items()}
    for name, ev in variant_events.items():
        diff = compare(truth, ev)
        if name == "wallclock":
            dst = {f"@{d.isoformat()}" for d in DST_DATES}
            on_dst = [r for r in truth if any(s in r[0] for s in dst)]
            diff["wrong_time_on_dst_days"] = compare(on_dst, [r for r in ev if any(s in r[0] for s in dst)])["wrong_time"]
        res["variants"][name] = diff

    # departure-board queries; skip the first 2 days so prior service dates are built
    rng = random.Random(1000 + seed)
    stops = sorted({r[2] for r in truth})
    q_lo = int(datetime.combine(start + timedelta(days=2), datetime.min.time(), feed.tz).timestamp())
    q_hi = int(datetime.combine(end, datetime.min.time(), feed.tz).timestamp())
    truth_board = ListBoard(truth)
    boards = {name: ListBoard(ev) for name, ev in variant_events.items()}
    wrong = defaultdict(lambda: {"night": 0, "day": 0})
    totals = {"night": 0, "day": 0}
    sql_mismatch = 0
    for _ in range(n_queries):
        stop = rng.choice(stops)
        lo = rng.randrange(q_lo, q_hi)
        hi = lo + 3600
        at = datetime.fromtimestamp(lo, tz=timezone.utc).astimezone(feed.tz)
        bucket = "night" if (at.hour >= 22 or at.hour < 4) else "day"
        totals[bucket] += 1
        expected = truth_board.next3(stop, lo, hi)
        got = [(r["departure_epoch"], r["trip_id"]) for r in departures(conn, stop, at, 60)][:3]
        sql_mismatch += got != expected
        same_date = [(r["departure_epoch"], r["trip_id"])
                     for r in departures(conn, stop, at, 60, same_date_only=True)][:3]
        wrong["board_same_service_date_only"][bucket] += same_date != expected
        for name, b in boards.items():
            wrong[name][bucket] += b.next3(stop, lo, hi) != expected
    res["board"] = {"queries": totals, "sql_board_vs_truth_mismatches": sql_mismatch,
                    "wrong": {k: dict(v) for k, v in wrong.items()}}
    conn.close()
    return res


def summarise(results: list[dict]) -> str:
    lines = ["# gtfsflat benchmark", "",
             f"Seeds: {sorted({r['seed'] for r in results})}; windows: "
             + ", ".join(f"{k} {a}..{b}" for k, (a, b) in WINDOWS.items()), ""]
    lines += ["## SQL engine vs row-at-a-time oracle", "",
              "| window | seed | service dates | trip instances | stop events | exact | wrong | missing | extra | SQL build s | oracle s |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["sql_vs_oracle"]
        lines.append(f"| {r['window']} | {r['seed']} | {r['service_dates']} | {r['trip_instances']} | "
                     f"{r['stop_events']} | {c['exact']} | {c['wrong_time']} | {c['missing']} | {c['extra']} | "
                     f"{r['sql_build_s']} | {r['oracle_s']} |")
    lines += ["", "## Shortcut loaders vs correct events (summed over seeds)", "",
              "| window | variant | expected | exact | wrong time | missing | extra | % events not exact |",
              "|---|---|---|---|---|---|---|---|"]
    for w in WINDOWS:
        rs = [r for r in results if r["window"] == w]
        for name in VARIANTS:
            agg = defaultdict(int)
            for r in rs:
                for k, v in r["variants"][name].items():
                    agg[k] += v
            bad = 100 * (1 - agg["exact"] / agg["expected"])
            extra = f" (on DST days: {agg['wrong_time_on_dst_days']})" if name == "wallclock" else ""
            lines.append(f"| {w} | {name} | {agg['expected']} | {agg['exact']} | {agg['wrong_time']}{extra} | "
                         f"{agg['missing']} | {agg['extra']} | {bad:.2f}% |")
    lines += ["", "## Departure board: next 3 departures within 60 min (summed over seeds)", "",
              "| window | variant | night queries wrong (22:00-04:00) | day queries wrong |", "|---|---|---|---|"]
    for w in WINDOWS:
        rs = [r for r in results if r["window"] == w]
        tn = sum(r["board"]["queries"]["night"] for r in rs)
        td = sum(r["board"]["queries"]["day"] for r in rs)
        sqlm = sum(r["board"]["sql_board_vs_truth_mismatches"] for r in rs)
        lines.append(f"| {w} | gtfsflat SQL board (any mismatch vs oracle board, all queries) | "
                     f"{sqlm} / {tn + td} | |")
        for name in ["board_same_service_date_only", *VARIANTS]:
            n = sum(r["board"]["wrong"][name]["night"] for r in rs)
            d = sum(r["board"]["wrong"][name]["day"] for r in rs)
            lines.append(f"| {w} | {name} | {n} / {tn} ({100 * n / tn:.1f}%) | {d} / {td} ({100 * d / td:.1f}%) |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(a.seeds):
            for w in WINDOWS:
                r = run_one(seed, w, a.queries, tmp)
                print(f"{w} seed {seed}: {r['stop_events']} events, sql_vs_oracle={r['sql_vs_oracle']}", flush=True)
                results.append(r)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "bench.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    md = summarise(results)
    with open(os.path.join(a.out, "bench.md"), "w") as fh:
        fh.write(md)
    print(md)
    bad = [r for r in results if r["sql_vs_oracle"]["exact"] != r["sql_vs_oracle"]["expected"]
           or r["sql_vs_oracle"]["extra"] or r["board"]["sql_board_vs_truth_mismatches"]]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
