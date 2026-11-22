"""End-to-end evaluation against simulator ground truth.

Per seed:
1. simulate a world (faults included) and split meters 70/30 into train/test
   by a seeded permutation - no meter contributes to both sides;
2. train the shape model on train meters only (targets: their actual hours);
3. score fault detection and whole-pipeline totals vs a naive pipeline;
4. on test meters, delete readings in extra random windows and rebuild with
   each allocation method; score every ``allocated`` hour (natural and
   injected gaps) and every ``estimated`` hour against true kWh.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .allocate import coverage, fill, fill_intervals_uniform
from .clean import clean_meter, naive_intervals
from .config import ALLOCATED, ESTIMATED, HOUR
from .model import choose_windows, drop_windows, predict_hourly, train_model
from .simulate import SimConfig, World, simulate
from .timeutil import overlaps

BUCKETS = [("2-6h", 0, 6), ("6-24h", 6, 24), ("24-80h", 24, 1e9)]


def group_rows(readings) -> dict[str, list[tuple[int, float]]]:
    """First value received per (meter, ts) wins - same rule as the SQLite ingest."""
    seen: dict[str, dict[int, float]] = defaultdict(dict)
    for m, t, v in readings:
        seen[m].setdefault(int(t), float(v))
    return {m: sorted(d.items()) for m, d in seen.items()}


def split_meters(meter_ids: list[str], seed: int, test_frac: float = 0.3) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed + 99)
    perm = list(rng.permutation(sorted(meter_ids)))
    n_test = int(round(len(perm) * test_frac))
    return sorted(perm[n_test:]), sorted(perm[:n_test])


def _cum_truth(truth_row: np.ndarray, h0: int, ts: int) -> float:
    x = (ts - h0) / HOUR
    i = min(int(x), len(truth_row) - 1)
    return float(truth_row[:i].sum() + truth_row[i] * (x - i))


def _detection(world: World, rows: dict) -> dict:
    out = {}
    counts = {k: [0, 0, 0] for k in ("misread", "rollover", "reset")}  # tp, fp, fn
    for m in world.meter_ids:
        c = clean_meter(rows.get(m, []))
        for k in counts:
            pred = {t for t, _, s in c.flags if s == k}
            true = world.labels[m][k]
            counts[k][0] += len(pred & true)
            counts[k][1] += len(pred - true)
            counts[k][2] += len(true - pred)
    for k, (tp, fp, fn) in counts.items():
        out[k] = {
            "true": tp + fn,
            "flagged": tp + fp,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        }
    return out


def _span_totals(world: World, rows: dict, meters: list[str], model) -> dict:
    """Energy over each meter's metered span [first reading, last reading]: naive vs meterfill."""
    idx = world.meter_index()
    naive_err, mf_err, naive_max_hour, mf_max_hour = [], [], [], []
    for m in meters:
        r = rows.get(m, [])
        if len(r) < 2:
            continue
        truth = world.truth[idx[m]]
        t_first, t_last = r[0][0], r[-1][0]
        true_total = _cum_truth(truth, world.h0, t_last) - _cum_truth(truth, world.h0, t_first)
        naive = naive_intervals(r)
        naive_err.append(abs(sum(iv.kwh for iv in naive) - true_total) / true_total)
        c = clean_meter(r)
        cov = coverage(c, world.h0, world.nh)
        pred = predict_hourly("model", cov, world.temps, model)
        est = 0.0
        for (a, b) in zip(c.segments, c.segments[1:]):
            ii, sec = overlaps(a[-1][0], b[0][0], world.h0, world.nh)
            est += float((pred[ii] * sec / HOUR).sum())
        mf_total = sum(iv.kwh for iv in c.intervals) + est
        mf_err.append(abs(mf_total - true_total) / true_total)
        # hourly view inside the metered span
        lo, hi = (t_first - world.h0) // HOUR + 1, (t_last - world.h0) // HOUR
        nh_series = fill_intervals_uniform(naive, world.h0, world.nh)
        mf_series = fill(cov, pred).kwh
        naive_max_hour.append(float(np.abs(nh_series[lo:hi] - truth[lo:hi]).max()))
        mf_max_hour.append(float(np.abs(mf_series[lo:hi] - truth[lo:hi]).max()))
    naive_err, mf_err = np.array(naive_err), np.array(mf_err)
    return {
        "meters": int(len(mf_err)),
        "naive": {
            "median_total_error_pct": round(100 * float(np.median(naive_err)), 3),
            "meters_over_1pct": int((naive_err > 0.01).sum()),
            "max_total_error_pct": round(100 * float(naive_err.max()), 1),
            "median_worst_hour_kwh": round(float(np.median(naive_max_hour)), 3),
            "max_worst_hour_kwh": round(float(np.max(naive_max_hour)), 1),
        },
        "meterfill": {
            "median_total_error_pct": round(100 * float(np.median(mf_err)), 3),
            "meters_over_1pct": int((mf_err > 0.01).sum()),
            "max_total_error_pct": round(100 * float(mf_err.max()), 1),
            "median_worst_hour_kwh": round(float(np.median(mf_max_hour)), 3),
            "max_worst_hour_kwh": round(float(np.max(mf_max_hour)), 1),
        },
    }


class _Acc:
    def __init__(self):
        self.abs_err = 0.0
        self.truth = 0.0
        self.n = 0
        self.gap_abs = 0.0
        self.gap_truth = 0.0
        self.peak_hit = 0
        self.peak_n = 0

    def add(self, pred, truth):
        self.abs_err += float(np.abs(pred - truth).sum())
        self.truth += float(truth.sum())
        self.n += int(pred.size)

    def result(self) -> dict:
        out = {
            "hours": self.n,
            "nmae": round(self.abs_err / self.truth, 4) if self.truth else None,
            "mae_kwh": round(self.abs_err / self.n, 4) if self.n else None,
        }
        if self.gap_truth:
            out["gap_total_error"] = round(self.gap_abs / self.gap_truth, 4)
        if self.peak_n:
            out["peak_hour_hit_rate"] = round(self.peak_hit / self.peak_n, 4)
            out["peak_days"] = self.peak_n
        return out


def _gap_eval(world: World, rows: dict, test: list[str], model, seed: int, n_windows: int) -> dict:
    idx = world.meter_index()
    rng = np.random.default_rng(seed + 1000)
    methods = ("uniform", "profile", "model", "model_raw")
    acc = {m: _Acc() for m in methods}
    bucket = {(m, b[0]): _Acc() for m in methods for b in BUCKETS}
    est = {m: _Acc() for m in ("uniform", "profile", "model")}
    n_gaps = 0
    for mid in test:
        truth = world.truth[idx[mid]]
        r = rows.get(mid, [])
        full_cov = coverage(clean_meter(r), world.h0, world.nh)
        windows = choose_windows(full_cov, rng, n_windows)
        cov = coverage(clean_meter(drop_windows(r, windows)), world.h0, world.nh)
        n_gaps += len(cov.gaps)
        preds = {"uniform": predict_hourly("uniform", cov, world.temps),
                 "profile": predict_hourly("profile", cov, world.temps)}
        preds["model"] = preds["model_raw"] = predict_hourly("model", cov, world.temps, model)
        for meth in methods:
            f = fill(cov, preds[meth], scale=meth != "model_raw")
            mask = f.quality == ALLOCATED
            acc[meth].add(f.kwh[mask], truth[mask])
            for gi, g in enumerate(cov.gaps):
                gm = f.gap_id == gi
                if not gm.any():
                    continue
                hrs = (g.t1 - g.t0) / HOUR
                for name, lo, hi in BUCKETS:
                    if lo < hrs <= hi:
                        bucket[(meth, name)].add(f.kwh[gm & mask], truth[gm & mask])
                acc[meth].gap_abs += abs(float(f.kwh[gm].sum() - truth[gm].sum()))
                acc[meth].gap_truth += float(truth[gm].sum())
                # peak hour on calendar days lying fully inside the gap
                first_day = -(-(g.t0 - world.h0) // (24 * HOUR))
                last_day = (g.t1 - world.h0) // (24 * HOUR)
                for d in range(first_day, last_day):
                    sl = slice(d * 24, d * 24 + 24)
                    acc[meth].peak_n += 1
                    acc[meth].peak_hit += int(abs(int(np.argmax(f.kwh[sl])) - int(np.argmax(truth[sl]))) <= 1)
            if meth in est:
                em = f.quality == ESTIMATED
                est[meth].add(f.kwh[em], truth[em])
    return {
        "test_meters": len(test),
        "gaps_scored": n_gaps,
        "allocated": {m: acc[m].result() for m in methods},
        "by_gap_length": {m: {b[0]: bucket[(m, b[0])].result() for b in BUCKETS} for m in methods},
        "estimated": {m: est[m].result() for m in est},
    }


def evaluate_seed(cfg: SimConfig, n_windows: int = 4) -> dict:
    t0 = time.time()
    world = simulate(cfg)
    rows = group_rows(world.readings)
    train, test = split_meters(world.meter_ids, cfg.seed)
    assert not set(train) & set(test)
    model = train_model({m: rows[m] for m in train if m in rows}, world.temps, world.h0, world.nh, seed=cfg.seed)
    out = {
        "seed": cfg.seed,
        "readings_received": len(world.readings),
        "unique_readings": sum(len(v) for v in rows.values()),
        "train_meters": len(train),
        "train_rows": model.meta["train_rows"],
        "detection": _detection(world, rows),
        "span_totals_test": _span_totals(world, rows, test, model),
        "gap_allocation_test": _gap_eval(world, rows, test, model, cfg.seed, n_windows),
    }
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(runs: list[dict]) -> dict:
    """Mean and range over seeds for the headline numbers."""
    def stat(vals):
        vals = [v for v in vals if v is not None]
        return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4), "max": round(float(np.max(vals)), 4)}

    s: dict = {"allocated_nmae": {}, "peak_hour_hit_rate": {}, "gap_total_error": {}, "estimated_nmae": {}, "nmae_by_gap_length": {}}
    methods = runs[0]["gap_allocation_test"]["allocated"].keys()
    for m in methods:
        s["allocated_nmae"][m] = stat([r["gap_allocation_test"]["allocated"][m]["nmae"] for r in runs])
        s["peak_hour_hit_rate"][m] = stat([r["gap_allocation_test"]["allocated"][m].get("peak_hour_hit_rate") for r in runs])
        s["gap_total_error"][m] = stat([r["gap_allocation_test"]["allocated"][m].get("gap_total_error") for r in runs])
        s["nmae_by_gap_length"][m] = {
            b[0]: stat([r["gap_allocation_test"]["by_gap_length"][m][b[0]]["nmae"] for r in runs]) for b in BUCKETS
        }
    for m in runs[0]["gap_allocation_test"]["estimated"]:
        s["estimated_nmae"][m] = stat([r["gap_allocation_test"]["estimated"][m]["nmae"] for r in runs])
    return s


def run(seeds: list[int], households: int, days: int, out_dir: str | Path | None) -> dict:
    runs = []
    for seed in seeds:
        cfg = SimConfig(households=households, days=days, seed=seed)
        r = evaluate_seed(cfg)
        print(f"seed {seed}: done in {r['seconds']} s", flush=True)
        runs.append(r)
    report = {"config": {**asdict(SimConfig(households=households, days=days)), "seeds": seeds}, "summary": summarize(runs), "runs": runs}
    report["config"].pop("seed")
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=2))
        (out / "REPORT.md").write_text(render_markdown(report))
    return report


def render_markdown(report: dict) -> str:
    s = report["summary"]
    cfg = report["config"]
    lines = [
        "# meterfill evaluation report",
        "",
        f"Seeds {cfg['seeds']}, {cfg['households']} households x {cfg['days']} days each, 30% of meters held out "
        "for testing. Values are mean [min, max] over seeds. Generated by `python -m meterfill evaluate`.",
        "",
        "## Gap allocation on held-out meters (every `allocated` hour vs true kWh)",
        "",
        "| method | NMAE | peak hour within ±1 h | gap total error |",
        "|---|---|---|---|",
    ]
    fmt = lambda d: f"{d['mean']:.3f} [{d['min']:.3f}, {d['max']:.3f}]"  # noqa: E731
    for m in s["allocated_nmae"]:
        lines.append(f"| {m} | {fmt(s['allocated_nmae'][m])} | {fmt(s['peak_hour_hit_rate'][m])} | {fmt(s['gap_total_error'][m])} |")
    lines += ["", "NMAE by gap length:", "", "| method | " + " | ".join(b[0] for b in BUCKETS) + " |", "|---" * (len(BUCKETS) + 1) + "|"]
    for m, d in s["nmae_by_gap_length"].items():
        lines.append(f"| {m} | " + " | ".join(fmt(d[b[0]]) for b in BUCKETS) + " |")
    lines += ["", "## `estimated` hours (no closing reading) on held-out meters", "", "| method | NMAE |", "|---|---|"]
    for m, d in s["estimated_nmae"].items():
        lines.append(f"| {m} | {fmt(d)} |")
    lines += ["", "## Per seed", ""]
    for r in report["runs"]:
        g = r["gap_allocation_test"]
        st = r["span_totals_test"]
        lines.append(
            f"### seed {r['seed']}  ({r['readings_received']} readings received, {r['unique_readings']} unique, "
            f"{r['train_rows']} training rows, {g['gaps_scored']} gaps scored, "
            f"{g['allocated']['model']['hours']} allocated / {g['estimated']['model']['hours']} estimated hours)"
        )
        lines.append("")
        lines.append("| fault | true | flagged | precision | recall |")
        lines.append("|---|---|---|---|---|")
        for k, d in r["detection"].items():
            lines.append(f"| {k} | {d['true']} | {d['flagged']} | {d['precision']} | {d['recall']} |")
        lines.append("")
        lines.append("| pipeline (test meters) | median span-total error % | meters > 1% | max error % | median worst hour kWh | max worst hour kWh |")
        lines.append("|---|---|---|---|---|---|")
        for k in ("naive", "meterfill"):
            d = st[k]
            lines.append(
                f"| {k} | {d['median_total_error_pct']} | {d['meters_over_1pct']} / {st['meters']} | {d['max_total_error_pct']} | "
                f"{d['median_worst_hour_kwh']} | {d['max_worst_hour_kwh']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
