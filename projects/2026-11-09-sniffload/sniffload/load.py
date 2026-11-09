"""Apply an inferred (or locked) schema to every record and load SQLite.

* Rows whose field count differs from the schema, or that contain a cell which does
  not parse as its column type, go to ``<table>__rejects`` with the source line and
  a reason. With ``on_bad="null"`` bad cells become NULL and the row is kept.
* The load is one transaction that drops and re-creates both tables, so re-running
  it gives the same result and a failure leaves the previous load in place.
* A locked schema (from an earlier ``infer``) is checked against the file header
  first. Added, missing or renamed columns fail the load with ``SchemaDriftError``.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from . import columns as C
from .encoding import EncodingGuess, decode
from .infer import Inference, Options, infer_text, read_records

SQL_TYPES = {"boolean": "INTEGER", "integer": "INTEGER", "decimal": "REAL",
             "date": "TEXT", "datetime": "TEXT", "string": "TEXT"}


class SchemaDriftError(Exception):
    pass


@dataclass
class Reject:
    line: int
    raw: str
    reasons: list[str]


@dataclass
class LoadResult:
    inference: Inference
    rows: list[tuple[int, list[Any]]]  # (source line, typed values)
    rejects: list[Reject]
    nulled_cells: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "encoding": self.inference.encoding,
            "delimiter": self.inference.delimiter,
            "header": self.inference.header,
            "decimal_sep": self.inference.decimal_sep,
            "date_order": self.inference.date_order,
            "rows_loaded": len(self.rows),
            "rows_rejected": len(self.rejects),
            "cells_nulled": self.nulled_cells,
            "columns": {c.name: c.type for c in self.inference.columns},
        }


def _raw_line(fields: list[str], delimiter: str) -> str:
    buf = io.StringIO()
    csv.writer(buf, delimiter=delimiter, lineterminator="").writerow(fields)
    return buf.getvalue()


def check_drift(locked: Inference, header_row: list[str]) -> None:
    expected = [c.source_name for c in locked.columns]
    got = [v.strip() for v in header_row]
    if expected == got:
        return
    missing = [c for c in expected if c not in got]
    added = [c for c in got if c not in expected]
    msg = []
    if missing:
        msg.append(f"missing columns: {missing}")
    if added:
        msg.append(f"new columns: {added}")
    if not missing and not added:
        msg.append(f"column order changed: expected {expected}, got {got}")
    raise SchemaDriftError("; ".join(msg))


def parse(data: bytes, opt: Options | None = None, schema: Inference | None = None,
          on_bad: str = "reject") -> LoadResult:
    if on_bad not in ("reject", "null"):
        raise ValueError("on_bad must be 'reject' or 'null'")
    if schema is not None:
        text, _ = decode(data, EncodingGuess(schema.encoding, False, "locked schema"))
        inf = schema
    else:
        text, enc = decode(data)
        inf = infer_text(text, enc, opt)
    records = read_records(text, inf.delimiter)
    if inf.header and records:
        if schema is not None:
            check_drift(schema, records[0][1])
        records = records[1:]
    width = len(inf.columns)
    rows: list[tuple[int, list[Any]]] = []
    rejects: list[Reject] = []
    nulled = 0
    for line, rec in records:
        if len(rec) != width:
            rejects.append(Reject(line, _raw_line(rec, inf.delimiter),
                                  [f"expected {width} fields, got {len(rec)}"]))
            continue
        out: list[Any] = []
        reasons: list[str] = []
        for cell, col in zip(rec, inf.columns):
            ok, val = C.convert(cell, col)
            if not ok:
                reasons.append(f"{col.name}: {cell!r} is not a valid {col.type}")
            out.append(val)
        if reasons and on_bad == "reject":
            rejects.append(Reject(line, _raw_line(rec, inf.delimiter), reasons))
            continue
        nulled += len(reasons)
        rows.append((line, out))
    return LoadResult(inf, rows, rejects, nulled)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_value(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, Decimal):
        return float(v)
    return C.to_jsonable(v)


def write_sqlite(result: LoadResult, db_path: str, table: str) -> None:
    cols = result.inference.columns
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute(f"DROP TABLE IF EXISTS {_q(table)}")
            con.execute(f"DROP TABLE IF EXISTS {_q(table + '__rejects')}")
            defs = ", ".join(f"{_q(c.name)} {SQL_TYPES[c.type]}" for c in cols)
            con.execute(f"CREATE TABLE {_q(table)} (_line INTEGER NOT NULL, {defs})")
            con.execute(f"CREATE TABLE {_q(table + '__rejects')} "
                        "(_line INTEGER NOT NULL, raw TEXT NOT NULL, reasons TEXT NOT NULL)")
            ph = ", ".join("?" * (len(cols) + 1))
            con.executemany(f"INSERT INTO {_q(table)} VALUES ({ph})",
                            ([line] + [_sql_value(v) for v in vals] for line, vals in result.rows))
            con.executemany(f"INSERT INTO {_q(table + '__rejects')} VALUES (?, ?, ?)",
                            ((r.line, r.raw, json.dumps(r.reasons)) for r in result.rejects))
    finally:
        con.close()


def write_clean_csv(result: LoadResult, path: str) -> None:
    """UTF-8, comma-delimited, ISO dates, '.' decimals, empty string for NULL."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([c.name for c in result.inference.columns])
        for _, vals in result.rows:
            w.writerow(["" if v is None else ("true" if v is True else "false" if v is False
                                               else C.to_jsonable(v)) for v in vals])
