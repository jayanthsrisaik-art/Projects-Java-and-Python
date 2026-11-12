"""Partition profiler: one GROUP BY pass per table writes daily metrics into dq_metrics.

Re-profiling a date range replaces those rows in a single transaction, so running the
profiler twice (or after a partition is reloaded) is idempotent.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .config import Config, TableConfig

SCHEMA = """
CREATE TABLE IF NOT EXISTS dq_metrics (
    table_name     TEXT NOT NULL,
    partition_date TEXT NOT NULL,
    metric         TEXT NOT NULL,
    value          REAL,            -- NULL when the metric is undefined (e.g. empty partition)
    n              INTEGER NOT NULL, -- sample size the metric was computed over
    sd             REAL,             -- standard deviation of the column (mean metrics only)
    PRIMARY KEY (table_name, partition_date, metric)
);
CREATE TABLE IF NOT EXISTS dq_category_counts (
    table_name     TEXT NOT NULL,
    partition_date TEXT NOT NULL,
    column_name    TEXT NOT NULL,
    category       TEXT NOT NULL,
    n              INTEGER NOT NULL,
    PRIMARY KEY (table_name, partition_date, column_name, category)
);
"""

NULL_CATEGORY = "__null__"


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def _dates(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def partition_range(conn: sqlite3.Connection, t: TableConfig) -> tuple[date, date] | None:
    row = conn.execute(f"SELECT MIN({t.partition_column}), MAX({t.partition_column}) FROM {t.name}").fetchone()
    if row[0] is None:
        return None
    return date.fromisoformat(row[0]), date.fromisoformat(row[1])


def _lag_hours(max_loaded_at: str | None, day: date) -> float | None:
    if max_loaded_at is None:
        return None
    loaded = datetime.fromisoformat(max_loaded_at)
    partition_end = datetime.combine(day + timedelta(days=1), datetime.min.time())
    return round((loaded - partition_end).total_seconds() / 3600.0, 4)


def profile_table(conn: sqlite3.Connection, t: TableConfig, start: date, end: date) -> int:
    """Profile partitions start..end (inclusive). Dates with no rows get row_count 0.

    Returns the number of metric rows written.
    """
    ensure_schema(conn)
    exprs = ["COUNT(*)"]
    if t.key:
        exprs.append(f"COUNT(DISTINCT {t.key})")
    exprs += [f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)" for c in t.null_checks]
    for c in t.mean_checks:
        exprs += [f"AVG({c})", f"AVG(CAST({c} AS REAL) * {c})", f"COUNT({c})"]
    if t.loaded_at:
        exprs.append(f"MAX({t.loaded_at})")
    sql = (f"SELECT {t.partition_column}, {', '.join(exprs)} FROM {t.name} "
           f"WHERE {t.partition_column} BETWEEN ? AND ? GROUP BY {t.partition_column}")
    by_day = {r[0]: r[1:] for r in conn.execute(sql, (start.isoformat(), end.isoformat()))}

    cats: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for c in t.category_checks:
        q = (f"SELECT {t.partition_column}, COALESCE(CAST({c} AS TEXT), ?), COUNT(*) FROM {t.name} "
             f"WHERE {t.partition_column} BETWEEN ? AND ? GROUP BY 1, 2")
        for d, cat, n in conn.execute(q, (NULL_CATEGORY, start.isoformat(), end.isoformat())):
            cats.setdefault((d, c), []).append((cat, n))

    metric_rows = []
    for day in _dates(start, end):
        ds = day.isoformat()
        r = by_day.get(ds)
        n = r[0] if r else 0
        # metric -> (value, sample size, sd)
        vals: dict[str, tuple] = {"row_count": (float(n), n, None)}
        i = 1
        if t.key:
            vals["dup_rate"] = ((n - r[i]) / n if n else None, n, None)
            i += 1
        for c in t.null_checks:
            vals[f"null_rate:{c}"] = (r[i] / n if n else None, n, None)
            i += 1
        for c in t.mean_checks:
            mean, meansq, cnt = (r[i], r[i + 1], r[i + 2]) if n else (None, None, 0)
            sd = math.sqrt(max(meansq - mean * mean, 0.0)) if cnt else None
            vals[f"mean:{c}"] = (mean, cnt, sd)
            i += 3
        if t.loaded_at:
            vals["freshness_lag_h"] = (_lag_hours(r[i], day) if n else None, n, None)
        for c in t.category_checks:
            # PSI needs a reference distribution, so the detector computes it; the profiler
            # stores the raw counts and a placeholder row so every metric exists every day.
            vals[f"psi:{c}"] = (None, n, None)
        for m, (v, cnt, sd) in vals.items():
            metric_rows.append((t.name, ds, m, v, cnt, sd))

    with conn:
        conn.execute("DELETE FROM dq_metrics WHERE table_name=? AND partition_date BETWEEN ? AND ?",
                     (t.name, start.isoformat(), end.isoformat()))
        conn.execute("DELETE FROM dq_category_counts WHERE table_name=? AND partition_date BETWEEN ? AND ?",
                     (t.name, start.isoformat(), end.isoformat()))
        conn.executemany("INSERT INTO dq_metrics VALUES (?,?,?,?,?,?)", metric_rows)
        conn.executemany("INSERT INTO dq_category_counts VALUES (?,?,?,?,?)",
                         [(t.name, d, c, cat, n) for (d, c), lst in cats.items() for cat, n in lst])
    return len(metric_rows)


def profile(conn: sqlite3.Connection, cfg: Config, start: date | None = None, end: date | None = None) -> int:
    total = 0
    for t in cfg.tables:
        rng = partition_range(conn, t)
        if rng is None and (start is None or end is None):
            continue
        s = start or rng[0]
        e = end or rng[1]
        total += profile_table(conn, t, s, e)
    return total


@dataclass
class Series:
    """All profiled metrics of one table on a contiguous daily calendar."""

    table: str
    dates: list[str]
    values: dict[str, list[float | None]]
    n: dict[str, list[int]]  # per-metric sample size
    sd: dict[str, list[float | None]]  # per-metric column std dev (mean metrics)
    categories: dict[str, list[dict[str, int]]]  # column -> per-day {category: count}


def load_series(conn: sqlite3.Connection, t: TableConfig, until: str | None = None) -> Series:
    """Read a table's metrics; `until` truncates (inclusive) so callers can prove no look-ahead."""
    q = "SELECT partition_date, metric, value, n, sd FROM dq_metrics WHERE table_name=?"
    args: list = [t.name]
    if until:
        q += " AND partition_date <= ?"
        args.append(until)
    rows = conn.execute(q, args).fetchall()
    if not rows:
        ms = t.metrics()
        return Series(t.name, [], {m: [] for m in ms}, {m: [] for m in ms}, {m: [] for m in ms},
                      {c: [] for c in t.category_checks})
    first = date.fromisoformat(min(r[0] for r in rows))
    last = date.fromisoformat(max(r[0] for r in rows))
    dates = [d.isoformat() for d in _dates(first, last)]
    idx = {d: i for i, d in enumerate(dates)}
    values = {m: [None] * len(dates) for m in t.metrics()}
    n = {m: [0] * len(dates) for m in t.metrics()}
    sds = {m: [None] * len(dates) for m in t.metrics()}
    for d, m, v, cnt, sd in rows:
        if m in values:
            values[m][idx[d]] = v
            n[m][idx[d]] = cnt
            sds[m][idx[d]] = sd
    cats = {c: [dict() for _ in dates] for c in t.category_checks}
    q = "SELECT partition_date, column_name, category, n FROM dq_category_counts WHERE table_name=?"
    args = [t.name]
    if until:
        q += " AND partition_date <= ?"
        args.append(until)
    for d, c, cat, cnt in conn.execute(q, args):
        if c in cats:
            cats[c][idx[d]][cat] = cnt
    return Series(t.name, dates, values, n, sds, cats)
