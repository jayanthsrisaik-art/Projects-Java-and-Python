"""Synthetic warehouse with known incidents (the ground truth for evaluation).

Two daily-partitioned tables:
  orders   ~1,500 rows/day, weekly seasonality, ~34%/year growth, a permanent +35% benign
           level shift ("new market launch") in the second half, a slow benign payment-mix drift
  refunds  ~60 rows/day: a small table where sampling noise dominates every rate

Incidents last 1-3 days (a broken load usually repeats until someone fixes it) and start on
random days after the warm-up, at most one per table per 9 days, with random magnitudes --
some are deliberately subtle, so no detector can reach 100% recall.
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
from datetime import date, datetime, timedelta

START = date(2025, 1, 6)  # a Monday
WARMUP_DAYS = 42  # no incidents before this; evaluation starts here for every method

WEEKDAY_ORDERS = [1.00, 1.02, 1.03, 1.05, 1.15, 0.75, 0.65]
WEEKDAY_REFUNDS = [1.25, 1.10, 1.00, 1.00, 0.95, 0.80, 0.70]
PAYMENT = ["card", "paypal", "bank_transfer", "gift_card"]
CHANNEL = ["web", "app", "support"]

INCIDENT_TYPES = ["volume_drop", "missing_partition", "duplicate_load", "null_spike",
                  "unit_change", "mix_shift", "late_load"]

# Which metrics an incident type is expected to move. Used to decide whether an alert is
# explained by an incident (precision) and whether an incident was caught (recall).
INCIDENT_METRICS = {
    "volume_drop": ["row_count"],
    "missing_partition": ["row_count"],
    "duplicate_load": ["dup_rate", "row_count"],
    "null_spike": ["null_rate:{null_col}"],
    "unit_change": ["mean:amount"],
    "mix_shift": ["psi:{cat_col}"],
    "late_load": ["freshness_lag_h"],
}

TABLES = {
    "orders": {"null_col": "customer_id", "cat_col": "payment_method"},
    "refunds": {"null_col": "reason", "cat_col": "channel"},
}

DDL = """
CREATE TABLE orders (
    order_id INTEGER, partition_date TEXT, customer_id INTEGER, amount REAL,
    payment_method TEXT, loaded_at TEXT);
CREATE TABLE refunds (
    refund_id INTEGER, partition_date TEXT, reason TEXT, amount REAL,
    channel TEXT, loaded_at TEXT);
CREATE INDEX ix_orders_pd ON orders(partition_date);
CREATE INDEX ix_refunds_pd ON refunds(partition_date);
"""


def _poisson(rng: random.Random, lam: float) -> int:
    if lam > 50:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= l:
            return k
        k += 1


def plan_incidents(rng: random.Random, days: int, shift_day: int) -> list[dict]:
    incidents = []
    for table in TABLES:
        day = WARMUP_DAYS + rng.randint(0, 8)
        while day < days:
            # keep incidents away from the benign shift so its alerts are attributable
            if not (table == "orders" and shift_day - 3 <= day <= shift_day + 10):
                kind = rng.choice(INCIDENT_TYPES)
                duration = rng.choices([1, 2, 3], [0.6, 0.25, 0.15])[0]
                incidents.append({"table": table, "day": day, "duration": duration, "type": kind,
                                  "params": _incident_params(rng, table, kind)})
            day += rng.randint(9, 20)
    return incidents


def _incident_params(rng: random.Random, table: str, kind: str) -> dict:
    if kind == "volume_drop":
        return {"keep": round(rng.uniform(0.25, 0.75), 3)}
    if kind == "duplicate_load":
        return {"dup_frac": round(rng.uniform(0.02, 1.0), 3)}
    if kind == "null_spike":
        lo, hi = (0.01, 0.15) if table == "orders" else (0.25, 0.6)
        return {"null_p": round(rng.uniform(lo, hi), 3)}
    if kind == "unit_change":
        return {"factor": rng.choice([100.0, 0.01])}
    if kind == "mix_shift":
        cats = PAYMENT[:3] if table == "orders" else CHANNEL
        src, dst = rng.sample(cats, 2)
        return {"from": src, "to": dst, "frac": round(rng.uniform(0.5, 1.0), 3)}
    if kind == "late_load":
        return {"extra_h": round(rng.uniform(5.0, 18.0), 2)}
    return {}


def generate(db_path: str, days: int = 365, seed: int = 1) -> dict:
    """Create the warehouse at db_path and return the ground truth dict."""
    rng = random.Random(seed)
    shift_day = rng.randint(days // 2, max(days // 2, days - 60))
    incidents = plan_incidents(rng, days, shift_day)
    by_key = {(i["table"], i["day"] + k): i for i in incidents for k in range(i["duration"])}

    conn = sqlite3.connect(db_path)
    conn.executescript("".join(f"DROP TABLE IF EXISTS {t};" for t in (
        "orders", "refunds", "dq_metrics", "dq_category_counts", "dq_results")) + DDL)
    next_id = {"orders": 1, "refunds": 1}

    for t in range(days):
        day = START + timedelta(days=t)
        wd = day.weekday()
        for table in TABLES:
            inc = by_key.get((table, t))
            kind = inc["type"] if inc else None
            p = inc["params"] if inc else {}
            if table == "orders":
                level = 1500 * math.exp(0.0008 * t) * (1.35 if t >= shift_day else 1.0)
                n = _poisson(rng, level * WEEKDAY_ORDERS[wd] * math.exp(rng.gauss(0, 0.04)))
                null_p = 0.004
                drift = 0.07 * t / days  # paypal slowly takes share from card (benign)
                weights = [0.60 - drift, 0.25 + drift, 0.10, 0.05]
                cats = PAYMENT
            else:
                n = _poisson(rng, 60 * WEEKDAY_REFUNDS[wd])
                null_p = 0.08
                weights = [0.5, 0.35, 0.15]
                cats = CHANNEL
            if kind == "missing_partition":
                n = 0
            if kind == "volume_drop":
                n = int(n * p["keep"])
            if kind == "null_spike":
                null_p = p["null_p"]
            lag = 1.5 + rng.expovariate(1 / 0.6) + (p["extra_h"] if kind == "late_load" else 0.0)
            loaded_at = (datetime.combine(day + timedelta(days=1), datetime.min.time())
                         + timedelta(hours=lag)).isoformat(sep=" ", timespec="seconds")
            rows = []
            for _ in range(n):
                rid = next_id[table]
                next_id[table] += 1
                amount = rng.lognormvariate(3.6, 0.6) if table == "orders" else rng.lognormvariate(3.0, 0.7)
                if kind == "unit_change":
                    amount *= p["factor"]
                cat = rng.choices(cats, weights)[0]
                if kind == "mix_shift" and cat == p["from"] and rng.random() < p["frac"]:
                    cat = p["to"]
                if table == "orders":
                    who = None if rng.random() < null_p else rng.randint(1, 60000)
                else:
                    who = None if rng.random() < null_p else rng.choice(
                        ["damaged", "late", "wrong_item", "changed_mind", "other"])
                rows.append((rid, day.isoformat(), who, round(amount, 2), cat, loaded_at))
            if kind == "duplicate_load" and rows:
                k = max(1, int(len(rows) * p["dup_frac"]))
                rows += rng.sample(rows, k)
            conn.executemany(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    truth = {
        "seed": seed,
        "days": days,
        "start": START.isoformat(),
        "eval_start": (START + timedelta(days=WARMUP_DAYS)).isoformat(),
        "benign_events": [{"table": "orders", "date": (START + timedelta(days=shift_day)).isoformat(),
                           "type": "level_shift", "note": "+35% volume, permanent (new market)"},
                          {"table": "orders", "date": START.isoformat(), "type": "mix_drift",
                           "note": "paypal share grows 25% -> 32% over the period"}],
        "incidents": [],
    }
    for inc in sorted(incidents, key=lambda i: (i["day"], i["table"])):
        cols = TABLES[inc["table"]]
        truth["incidents"].append({
            "table": inc["table"],
            "dates": [(START + timedelta(days=inc["day"] + k)).isoformat() for k in range(inc["duration"])],
            "type": inc["type"],
            "params": inc["params"],
            "metrics": [m.format(**cols) for m in INCIDENT_METRICS[inc["type"]]],
        })
    return truth


DEMO_CONFIG = {
    "tables": [
        {"name": "orders", "partition_column": "partition_date", "key": "order_id",
         "loaded_at": "loaded_at", "null_checks": ["customer_id"], "mean_checks": ["amount"],
         "category_checks": ["payment_method"],
         "static_rules": {"null_rate_max": 0.01}},
        {"name": "refunds", "partition_column": "partition_date", "key": "refund_id",
         "loaded_at": "loaded_at", "null_checks": ["reason"], "mean_checks": ["amount"],
         "category_checks": ["channel"],
         "static_rules": {"null_rate_max": 0.20}},
    ],
}


def write_truth(truth: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(truth, f, indent=2)
