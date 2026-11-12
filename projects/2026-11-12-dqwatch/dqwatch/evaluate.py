"""Measure detectors against injected incidents on synthetic warehouses.

An alert (table, date, metric) is *explained* when an incident active on that table and date
lists the metric among the ones it moves; everything else is a false alert. An incident is
*caught* when at least one alert on any of its days explains it. Only dates on/after the warm-up count, for
every method alike.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from dataclasses import replace

from .baselines import static_alerts, zscore_alerts
from .config import Config, config_from_dict
from .detect import alerts, detect_table
from .generate import DEMO_CONFIG, INCIDENT_TYPES, generate
from .profile import load_series, profile

METHODS = ["dqwatch", "dqwatch_exclude_flagged", "dqwatch_no_sampling", "dqwatch_no_seasonal",
           "zscore_28d", "static_rules"]


def run_methods(conn: sqlite3.Connection, cfg: Config) -> dict[str, set[tuple[str, str, str]]]:
    out: dict[str, set] = {m: set() for m in METHODS}
    p = cfg.detector
    variants = {
        "dqwatch": p,
        "dqwatch_exclude_flagged": replace(p, exclude_anomalies=True),
        "dqwatch_no_sampling": replace(p, use_sampling=False),
        "dqwatch_no_seasonal": replace(p, use_seasonal=False),
    }
    for t in cfg.tables:
        s = load_series(conn, t)
        for name, params in variants.items():
            out[name] |= {(r.table, r.date, r.metric) for r in alerts(detect_table(s, t, params))}
        out["zscore_28d"] |= set(zscore_alerts(s, t))
        out["static_rules"] |= set(static_alerts(s, t))
    return out


def score(alert_set: set, truth: dict) -> dict:
    start = truth["eval_start"]
    al = {a for a in alert_set if a[1] >= start}
    explains = {}
    for inc in truth["incidents"]:
        for d in inc["dates"]:
            for m in inc["metrics"]:
                explains[(inc["table"], d, m)] = inc
    explained = [a for a in al if a in explains]
    caught = {id(explains[a]) for a in explained}
    by_type = {k: [0, 0] for k in INCIDENT_TYPES}
    for inc in truth["incidents"]:
        by_type[inc["type"]][1] += 1
        if id(inc) in caught:
            by_type[inc["type"]][0] += 1
    shift = next(e for e in truth["benign_events"] if e["type"] == "level_shift")
    return {
        "alerts": len(al),
        "explained_alerts": len(explained),
        "false_alerts": len(al) - len(explained),
        "incidents": len(truth["incidents"]),
        "caught": len(caught),
        "by_type": by_type,
        "missed": [{"table": i["table"], "dates": i["dates"], "type": i["type"], "params": i["params"]}
                   for i in truth["incidents"] if id(i) not in caught],
        "benign_shift_alerts": sum(1 for a in al if a[0] == shift["table"] and a[2] == "row_count"
                                   and a[1] >= shift["date"] and a not in explains),
    }


def evaluate(seeds: list[int], days: int = 365, cfg: Config | None = None, workdir: str | None = None) -> dict:
    cfg = cfg or config_from_dict(DEMO_CONFIG)
    per_seed = {}
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        for seed in seeds:
            db = os.path.join(tmp, f"wh_{seed}.db")
            t0 = time.perf_counter()
            truth = generate(db, days=days, seed=seed)
            conn = sqlite3.connect(db)
            t1 = time.perf_counter()
            profile(conn, cfg)
            t2 = time.perf_counter()
            sets = run_methods(conn, cfg)
            t3 = time.perf_counter()
            table_days = len(cfg.tables) * _eval_days(truth)
            conn.close()
            per_seed[seed] = {
                "table_days": table_days,
                "timing_s": {"generate": round(t1 - t0, 2), "profile": round(t2 - t1, 2),
                             "detect_all_methods": round(t3 - t2, 2)},
                "methods": {m: score(s, truth) for m, s in sets.items()},
            }
    return {"seeds": seeds, "days": days, "per_seed": per_seed, "summary": summarize(per_seed)}


def _eval_days(truth: dict) -> int:
    from datetime import date
    return truth["days"] - (date.fromisoformat(truth["eval_start"]) - date.fromisoformat(truth["start"])).days


def summarize(per_seed: dict) -> dict:
    summary = {}
    table_days = sum(v["table_days"] for v in per_seed.values())
    for m in METHODS:
        tot = {"alerts": 0, "explained_alerts": 0, "false_alerts": 0, "incidents": 0, "caught": 0,
               "benign_shift_alerts": 0}
        by_type = {k: [0, 0] for k in INCIDENT_TYPES}
        recalls = []
        for v in per_seed.values():
            s = v["methods"][m]
            for k in tot:
                tot[k] += s[k]
            for k, (c, n) in s["by_type"].items():
                by_type[k][0] += c
                by_type[k][1] += n
            recalls.append(s["caught"] / s["incidents"] if s["incidents"] else 0.0)
        summary[m] = {
            **tot,
            "recall": round(tot["caught"] / tot["incidents"], 4) if tot["incidents"] else None,
            "precision": round(tot["explained_alerts"] / tot["alerts"], 4) if tot["alerts"] else None,
            "false_alerts_per_100_table_days": round(100 * tot["false_alerts"] / table_days, 2),
            "recall_min_seed": round(min(recalls), 4),
            "recall_max_seed": round(max(recalls), 4),
            "by_type": by_type,
        }
    return summary


def to_markdown(res: dict) -> str:
    s = res["summary"]
    td = sum(v["table_days"] for v in res["per_seed"].values())
    lines = [f"Seeds {res['seeds']}, {res['days']} days each, {td} evaluated table-days, "
             f"{s['dqwatch']['incidents']} injected incidents.", "",
             "| method | incidents caught | recall (min-max per seed) | alerts | precision "
             "| false alerts / 100 table-days | alerts after benign +35% shift |",
             "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        r = s[m]
        prec = "n/a" if r["precision"] is None else f"{r['precision']:.3f}"
        lines.append(f"| {m} | {r['caught']}/{r['incidents']} | {r['recall']:.3f} "
                     f"({r['recall_min_seed']:.2f}-{r['recall_max_seed']:.2f}) | {r['alerts']} | {prec} | "
                     f"{r['false_alerts_per_100_table_days']:.2f} | {r['benign_shift_alerts']} |")
    lines += ["", "Recall by incident type (caught/injected):", "",
              "| method | " + " | ".join(INCIDENT_TYPES) + " |",
              "|---|" + "---|" * len(INCIDENT_TYPES)]
    for m in METHODS:
        bt = s[m]["by_type"]
        lines.append(f"| {m} | " + " | ".join(f"{c}/{n}" for c, n in (bt[k] for k in INCIDENT_TYPES)) + " |")
    return "\n".join(lines) + "\n"


def write_results(res: dict, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out_dir, f"{name}.md"), "w") as f:
        f.write(to_markdown(res))
