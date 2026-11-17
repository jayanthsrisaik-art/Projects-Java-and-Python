"""Sandboxed execution: read-only connection, VM-step budget, row cap."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

MAX_ROWS = 1000
STEP_BUDGET = 20_000_000   # SQLite VM instructions (progress handler fires every 1000)


@dataclass
class ExecResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False


def run_sql(con: sqlite3.Connection, sql: str, max_rows: int = MAX_ROWS,
            step_budget: int = STEP_BUDGET) -> ExecResult:
    ticks = {"n": 0}

    def guard() -> int:
        ticks["n"] += 1000
        return 1 if ticks["n"] > step_budget else 0   # non-zero aborts the query

    con.set_progress_handler(guard, 1000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        cols = [d[0] for d in cur.description or []]
        return ExecResult(True, cols, rows[:max_rows], truncated=len(rows) > max_rows)
    except sqlite3.OperationalError as exc:
        msg = "query exceeded execution budget" if "interrupted" in str(exc) else str(exc)
        return ExecResult(False, error=msg)
    except sqlite3.Error as exc:
        return ExecResult(False, error=f"{type(exc).__name__}: {exc}")
    finally:
        con.set_progress_handler(None, 0)
