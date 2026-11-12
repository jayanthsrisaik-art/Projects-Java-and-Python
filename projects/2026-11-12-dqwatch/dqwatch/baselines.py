"""Two common alternatives the detector is measured against.

static   hand-written thresholds (week-over-week volume, max null rate, any duplicate, ...)
zscore   |x - mean| / std > 3 over the trailing 28 days of raw values: no seasonality, no
         exclusion of past anomalies, no awareness of sample size
Both see exactly the same profiled metrics as the detector and never look ahead.
"""
from __future__ import annotations

import math

from .config import TableConfig
from .detect import psi
from .profile import Series

Alert = tuple[str, str, str]  # (table, date, metric)


def _psi_series(cats: list[dict[str, int]], window: int, prev_day_only: bool) -> list[float | None]:
    out: list[float | None] = []
    for i, today in enumerate(cats):
        if not today:
            out.append(None)
            continue
        ref: dict[str, int] = {}
        lo = i - 1 if prev_day_only else i - window
        for j in range(max(0, lo), i):
            for c, n in cats[j].items():
                ref[c] = ref.get(c, 0) + n
        out.append(psi(today, ref) if ref else None)
    return out


def static_alerts(s: Series, t: TableConfig) -> list[Alert]:
    r = t.static_rules
    out: list[Alert] = []
    for m in t.metrics():
        if m.startswith("psi:"):
            vals = _psi_series(s.categories[m.split(":", 1)[1]], 1, prev_day_only=True)
        else:
            vals = s.values[m]
        for i, v in enumerate(vals):
            if m == "row_count" and v is not None and v == 0:
                out.append((t.name, s.dates[i], m))
                continue
            if v is None:
                continue
            prev = vals[i - 7] if i >= 7 else None
            fire = False
            if m == "row_count":
                fire = bool(prev) and abs(v / prev - 1) > r.row_count_wow
            elif m == "dup_rate":
                fire = v > r.dup_rate_max
            elif m.startswith("null_rate:"):
                fire = v > r.null_rate_max
            elif m.startswith("mean:"):
                fire = bool(prev) and abs(v / prev - 1) > r.mean_wow
            elif m.startswith("psi:"):
                fire = v > r.psi_max
            elif m == "freshness_lag_h":
                fire = v > r.freshness_max_h
            if fire:
                out.append((t.name, s.dates[i], m))
    return out


def zscore_alerts(s: Series, t: TableConfig, window: int = 28, z: float = 3.0, min_points: int = 7) -> list[Alert]:
    out: list[Alert] = []
    for m in t.metrics():
        if m.startswith("psi:"):
            vals = _psi_series(s.categories[m.split(":", 1)[1]], window, prev_day_only=False)
        else:
            vals = s.values[m]
        for i, v in enumerate(vals):
            if v is None:
                continue
            hist = [x for x in vals[max(0, i - window):i] if x is not None]
            if len(hist) < min_points:
                continue
            mu = sum(hist) / len(hist)
            sd = math.sqrt(sum((x - mu) ** 2 for x in hist) / len(hist))
            if abs(v - mu) > z * max(sd, 1e-9):
                out.append((t.name, s.dates[i], m))
    return out
