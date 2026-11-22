"""SQLite persistence: idempotent ingest, rebuilds and queries."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np

from .allocate import coverage, fill
from .clean import clean_meter
from .config import HOUR, QUALITY_NAMES, REGISTER_MAX
from .model import ShapeModel, predict_hourly
from .timeutil import ceil_hour, floor_hour, fmt_ts, parse_ts

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    meter_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (meter_id, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS ingest_conflicts (
    meter_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    kept_value REAL NOT NULL,
    rejected_value REAL NOT NULL,
    received_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS weather (hour_ts INTEGER PRIMARY KEY, temp_c REAL NOT NULL);
CREATE TABLE IF NOT EXISTS reading_flags (
    meter_id TEXT NOT NULL, ts INTEGER NOT NULL, status TEXT NOT NULL,
    PRIMARY KEY (meter_id, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS hourly (
    meter_id TEXT NOT NULL, hour_ts INTEGER NOT NULL, kwh REAL NOT NULL, quality INTEGER NOT NULL,
    PRIMARY KEY (meter_id, hour_ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS issues (
    meter_id TEXT NOT NULL, kind TEXT NOT NULL, start_ts INTEGER NOT NULL, end_ts INTEGER NOT NULL, value REAL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class ValidationError(ValueError):
    pass


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self):
        try:
            self.conn.execute("BEGIN")
            yield self.conn
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    # ---- meta -------------------------------------------------------------
    def _meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def _set_meta(self, conn, key: str, value) -> None:
        conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))

    # ---- ingest -----------------------------------------------------------
    def ingest_readings(self, rows: Iterable[tuple[str, str | int, float | str]]) -> dict:
        """Insert readings. Re-sending the same reading is a no-op; a different value for an
        existing (meter, ts) keeps the first value and records the conflict."""
        received = inserted = dups = conflicts = 0
        now = time.time()
        with self.tx() as c:
            for meter_id, ts, value in rows:
                received += 1
                meter_id = str(meter_id).strip()
                try:
                    t = parse_ts(ts)
                    v = float(value)
                except (TypeError, ValueError) as e:
                    raise ValidationError(f"row {received}: {e}") from None
                if not meter_id or not (0 <= v < REGISTER_MAX):
                    raise ValidationError(f"row {received}: bad meter_id or register value {value!r}")
                cur = c.execute("INSERT OR IGNORE INTO readings(meter_id, ts, value) VALUES (?, ?, ?)", (meter_id, t, v))
                if cur.rowcount:
                    inserted += 1
                    continue
                kept = c.execute("SELECT value FROM readings WHERE meter_id = ? AND ts = ?", (meter_id, t)).fetchone()[0]
                if abs(kept - v) < 0.005:
                    dups += 1
                else:
                    conflicts += 1
                    c.execute("INSERT INTO ingest_conflicts VALUES (?, ?, ?, ?, ?)", (meter_id, t, kept, v, now))
            if inserted:
                self._set_meta(c, "ingest_seq", int(self._meta("ingest_seq", "0")) + 1)
        return {"received": received, "inserted": inserted, "duplicates": dups, "conflicts": conflicts}

    def ingest_weather(self, rows: Iterable[tuple[str | int, float | str]]) -> int:
        n = 0
        with self.tx() as c:
            for ts, temp in rows:
                t = parse_ts(ts)
                if t % HOUR:
                    raise ValidationError(f"weather timestamp not on the hour: {ts}")
                c.execute(
                    "INSERT INTO weather(hour_ts, temp_c) VALUES (?, ?) ON CONFLICT(hour_ts) DO UPDATE SET temp_c = excluded.temp_c",
                    (t, float(temp)),
                )
                n += 1
            self._set_meta(c, "ingest_seq", int(self._meta("ingest_seq", "0")) + 1)
        return n

    # ---- reads ------------------------------------------------------------
    def meter_ids(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT meter_id FROM readings ORDER BY meter_id")]

    def rows(self, meter_id: str) -> list[tuple[int, float]]:
        return list(self.conn.execute("SELECT ts, value FROM readings WHERE meter_id = ? ORDER BY ts", (meter_id,)))

    def all_rows(self) -> dict[str, list[tuple[int, float]]]:
        out: dict[str, list[tuple[int, float]]] = {}
        for m, t, v in self.conn.execute("SELECT meter_id, ts, value FROM readings ORDER BY meter_id, ts"):
            out.setdefault(m, []).append((t, v))
        return out

    def window(self) -> tuple[int, int]:
        """Hour grid: the weather range if loaded, otherwise the readings range."""
        w = self.conn.execute("SELECT MIN(hour_ts), MAX(hour_ts) FROM weather").fetchone()
        if w[0] is not None:
            return w[0], (w[1] - w[0]) // HOUR + 1
        r = self.conn.execute("SELECT MIN(ts), MAX(ts) FROM readings").fetchone()
        if r[0] is None:
            raise ValidationError("no readings loaded")
        h0 = floor_hour(r[0])
        return h0, max((ceil_hour(r[1]) - h0) // HOUR, 1)

    def temps(self, h0: int, nh: int) -> tuple[np.ndarray, int]:
        """Hourly temperatures for the window; gaps filled by linear interpolation (count returned)."""
        temps = np.full(nh, np.nan)
        for t, c in self.conn.execute("SELECT hour_ts, temp_c FROM weather WHERE hour_ts >= ? AND hour_ts < ?", (h0, h0 + nh * HOUR)):
            temps[(t - h0) // HOUR] = c
        missing = int(np.isnan(temps).sum())
        if missing == nh:
            return np.full(nh, 10.0), missing
        if missing:
            ok = ~np.isnan(temps)
            temps = np.interp(np.arange(nh), np.flatnonzero(ok), temps[ok])
        return temps, missing

    # ---- build ------------------------------------------------------------
    def rebuild(self, method: str = "model", model: ShapeModel | None = None) -> dict:
        if method == "model_raw":
            raise ValidationError("model_raw is an evaluation ablation, not a build method")
        if method == "model" and model is None:
            raise ValidationError("method 'model' needs a trained model (run `meterfill train`) - or use 'profile'")
        h0, nh = self.window()
        temps, missing_weather = self.temps(h0, nh)
        seq = self._meta("ingest_seq", "0")
        totals = {"meters": 0, "hours": 0, "actual": 0, "allocated": 0, "estimated": 0, "misread": 0, "rollover": 0, "reset": 0, "gap": 0}
        with self.tx() as c:
            c.execute("DELETE FROM hourly")
            c.execute("DELETE FROM reading_flags")
            c.execute("DELETE FROM issues")
            for mid, rows in self.all_rows().items():
                cr = clean_meter(rows)
                cov = coverage(cr, h0, nh)
                f = fill(cov, predict_hourly(method, cov, temps, model))
                c.executemany(
                    "INSERT INTO hourly VALUES (?, ?, ?, ?)",
                    ((mid, h0 + i * HOUR, round(float(f.kwh[i]), 4), int(f.quality[i])) for i in range(nh)),
                )
                c.executemany("INSERT INTO reading_flags VALUES (?, ?, ?)", ((mid, t, s) for t, _, s in cr.flags))
                issues = cr.issues()
                c.executemany(
                    "INSERT INTO issues VALUES (?, ?, ?, ?, ?)",
                    ((mid, d["kind"], d["start_ts"], d["end_ts"], d["value"]) for d in issues),
                )
                totals["meters"] += 1
                totals["hours"] += nh
                for code, name in QUALITY_NAMES.items():
                    totals[name] += int((f.quality == code).sum())
                for d in issues:
                    if d["kind"] in totals:
                        totals[d["kind"]] += 1
            self._set_meta(c, "built_seq", seq)
            self._set_meta(c, "built_method", method)
            self._set_meta(c, "built_at", fmt_ts(int(time.time())))
        return {"method": method, "window_start": fmt_ts(h0), "window_hours": nh, "weather_hours_interpolated": missing_weather, **totals}

    def build_state(self) -> dict:
        built = self._meta("built_seq")
        return {
            "method": self._meta("built_method"),
            "built_at": self._meta("built_at"),
            "stale": built is None or built != self._meta("ingest_seq", "0"),
        }

    def usage(self, meter_id: str, start: str | None, end: str | None, granularity: str = "hour") -> dict | None:
        if granularity not in ("hour", "day"):
            raise ValidationError("granularity must be 'hour' or 'day'")
        if not self.conn.execute("SELECT 1 FROM hourly WHERE meter_id = ? LIMIT 1", (meter_id,)).fetchone():
            return None
        lo = parse_ts(start) if start else -(2**62)
        hi = parse_ts(end) if end else 2**62
        rows = self.conn.execute(
            "SELECT hour_ts, kwh, quality FROM hourly WHERE meter_id = ? AND hour_ts >= ? AND hour_ts < ? ORDER BY hour_ts",
            (meter_id, lo, hi),
        ).fetchall()
        out = []
        if granularity == "hour":
            out = [{"start": fmt_ts(t), "kwh": k, "quality": QUALITY_NAMES[q]} for t, k, q in rows]
        else:
            days: dict[int, dict] = {}
            for t, k, q in rows:
                d = days.setdefault(t // 86400 * 86400, {"kwh": 0.0, "hours": {n: 0 for n in QUALITY_NAMES.values()}})
                d["kwh"] += k
                d["hours"][QUALITY_NAMES[q]] += 1
            out = [{"start": fmt_ts(t), "kwh": round(d["kwh"], 3), "hours": d["hours"]} for t, d in sorted(days.items())]
        return {
            "meter_id": meter_id,
            "granularity": granularity,
            "total_kwh": round(sum(k for _, k, _ in rows), 3),
            "build": self.build_state(),
            "rows": out,
        }

    def issues(self, meter_id: str) -> list[dict]:
        return [
            {"kind": k, "start": fmt_ts(s), "end": fmt_ts(e), "value": v}
            for k, s, e, v in self.conn.execute(
                "SELECT kind, start_ts, end_ts, value FROM issues WHERE meter_id = ? ORDER BY start_ts, kind", (meter_id,)
            )
        ]

    def meters_summary(self) -> list[dict]:
        q = """
        SELECT r.meter_id, r.n, r.first_ts, r.last_ts,
               COALESCE(h.actual, 0), COALESCE(h.allocated, 0), COALESCE(h.estimated, 0), COALESCE(i.n_issues, 0)
        FROM (SELECT meter_id, COUNT(*) n, MIN(ts) first_ts, MAX(ts) last_ts FROM readings GROUP BY meter_id) r
        LEFT JOIN (SELECT meter_id, SUM(quality = 0) actual, SUM(quality = 1) allocated, SUM(quality = 2) estimated
                   FROM hourly GROUP BY meter_id) h USING (meter_id)
        LEFT JOIN (SELECT meter_id, COUNT(*) n_issues FROM issues WHERE kind != 'gap' GROUP BY meter_id) i USING (meter_id)
        ORDER BY r.meter_id
        """
        return [
            {
                "meter_id": m,
                "readings": n,
                "first_reading": fmt_ts(a),
                "last_reading": fmt_ts(b),
                "hours": {"actual": ha, "allocated": hl, "estimated": he},
                "faults": ni,
            }
            for m, n, a, b, ha, hl, he, ni in self.conn.execute(q)
        ]
