"""Column type inference.

Inference runs in two steps so that file-level evidence can settle questions a
single column cannot answer:

1. ``analyze_column`` looks at one column and returns its candidate types. When
   the column itself proves the decimal separator ("1,234.56") or the date order
   ("31/01/2024"), the decision is marked ``data``. When it cannot ("1,234" may be
   1234 or 1.234; "03/04/2024" may be March or April), the column stays ambiguous.
2. ``finalize_column`` turns the analysis into a ``ColumnSchema``. It uses the
   file-level decimal separator and date order that ``infer.py`` derived from the
   decisive columns, and records in ``resolved_by`` how each decision was made.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from . import values as V

TYPES = ("boolean", "integer", "decimal", "date", "datetime", "string")


@dataclass
class ColumnSchema:
    name: str
    source_name: str
    type: str
    nullable: bool = True
    decimal_sep: str | None = None
    scale: int | None = None
    date_format: str | None = None
    date_order: str | None = None
    resolved_by: str | None = None  # data | file | delimiter-prior | locale-prior | default
    null_count: int = 0
    bad_count: int = 0
    non_null_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if v is not None and v != []}
        d.setdefault("warnings", [])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ColumnSchema":
        return cls(**d)


@dataclass
class ColumnAnalysis:
    values: list[str]
    non_null: list[str]
    null_count: int
    need: int  # number of non-null values a type must parse to be accepted
    kind: str  # boolean | number | date | datetime | string | empty
    bool_ok: int = 0
    num_ok: dict[str, int] = field(default_factory=dict)
    num_decisive: str | None = None  # '.' or ',' when the column decides by itself
    num_neutral: bool = False  # separator irrelevant: no value changes meaning
    date_ok: dict[str, int] = field(default_factory=dict)  # fmt -> ok count
    date_decisive: str | None = None  # order (YMD/DMY/MDY) when the column decides
    date_orders: tuple[str, ...] = ()


def _best_formats(candidates: list[tuple[str, str]], non_null: list[str], parser) -> dict[str, int]:
    ok: dict[str, int] = {}
    for fmt, _order in candidates:
        n = 0
        for v in non_null:
            if parser(v, fmt) is not None:
                n += 1
        if n:
            ok[fmt] = n
    return ok


def analyze_column(raw: list[str], min_ratio: float = 0.95) -> ColumnAnalysis:
    non_null = [v for v in raw if not V.is_null(v)]
    null_count = len(raw) - len(non_null)
    n = len(non_null)
    need = max(1, math.ceil(min_ratio * n))
    a = ColumnAnalysis(raw, non_null, null_count, need, kind="string")
    if n == 0:
        a.kind = "empty"
        return a

    a.bool_ok = sum(V.parse_bool(v) is not None for v in non_null)
    if a.bool_ok >= need:
        a.kind = "boolean"
        return a

    parsed = {d: [V.parse_number(v, d) for v in non_null] for d in (".", ",")}
    a.num_ok = {d: sum(p is not None for p in ps) for d, ps in parsed.items()}
    passing = [d for d in (".", ",") if a.num_ok[d] >= need]
    if passing:
        a.kind = "number"
        if len(passing) == 1 or a.num_ok["."] != a.num_ok[","]:
            a.num_decisive = max(passing, key=lambda d: a.num_ok[d])
        else:
            same = all(
                (p is None and q is None)
                or (p is not None and q is not None and p.value == q.value and p.has_fraction == q.has_fraction)
                for p, q in zip(parsed["."], parsed[","])
            )
            a.num_neutral = same
        return a

    for kind, formats, parser in (
        ("date", V.DATE_FORMATS, V.parse_date),
        ("datetime", V.DATETIME_FORMATS, V.parse_datetime),
    ):
        ok = _best_formats(formats, non_null, parser)
        passing_fmts = {f: c for f, c in ok.items() if c >= need}
        if not passing_fmts:
            continue
        a.kind = kind
        a.date_ok = passing_fmts
        order_of = dict(formats)
        best_by_order: dict[str, int] = {}
        for f, c in passing_fmts.items():
            o = order_of[f]
            best_by_order[o] = max(best_by_order.get(o, 0), c)
        top = max(best_by_order.values())
        winners = tuple(o for o, c in best_by_order.items() if c == top)
        a.date_orders = winners
        if len(winners) == 1:
            a.date_decisive = winners[0]
        return a
    return a


def finalize_column(
    a: ColumnAnalysis,
    name: str,
    source_name: str,
    file_dec: tuple[str, str],
    file_order: tuple[str, str],
) -> ColumnSchema:
    """``file_dec``/``file_order`` are (value, how-it-was-decided) pairs."""
    col = ColumnSchema(name=name, source_name=source_name, type="string",
                       null_count=a.null_count, non_null_count=len(a.non_null))
    if a.kind == "empty":
        col.warnings.append("column is entirely null; typed as string")
    elif a.kind == "boolean":
        col.type = "boolean"
    elif a.kind == "number":
        if a.num_decisive:
            dec, how = a.num_decisive, "data"
        else:
            dec, how = file_dec
            if not a.num_neutral:
                col.warnings.append(
                    f"decimal separator ambiguous in this column (e.g. {_ambiguous_example(a)!r}); "
                    f"used '{dec}' ({how})"
                )
            else:
                how = "irrelevant"
        nums = [V.parse_number(v, dec) for v in a.non_null]
        good = [p for p in nums if p is not None]
        if any(p.leading_zero and not p.has_fraction for p in good):
            col.type = "string"
            col.warnings.append("numeric-looking values with leading zeros kept as string identifiers")
        elif any(p.has_fraction for p in good):
            col.type = "decimal"
            col.decimal_sep, col.resolved_by = dec, how
            col.scale = max((_scale(p.value) for p in good), default=0)
        else:
            col.type = "integer"
            col.decimal_sep, col.resolved_by = dec, how
    elif a.kind in ("date", "datetime"):
        formats = dict(V.DATE_FORMATS if a.kind == "date" else V.DATETIME_FORMATS)
        if a.date_decisive:
            order, how = a.date_decisive, "data"
        else:
            pref, how = file_order
            order = pref if pref in a.date_orders else sorted(a.date_orders)[0]
            if order != pref:
                how = "default"
            col.warnings.append(
                f"day/month order ambiguous ({'/'.join(sorted(a.date_orders))}); used {order} ({how})"
            )
        fmt = max((f for f in a.date_ok if formats[f] == order), key=lambda f: a.date_ok[f])
        col.type, col.date_format, col.date_order, col.resolved_by = a.kind, fmt, order, how

    if col.type != "string" or a.kind == "string":
        col.bad_count = sum(1 for v in a.non_null if not convert(v, col)[0])
    col.nullable = a.null_count > 0
    return col


def _scale(d: Decimal) -> int:
    exp = d.as_tuple().exponent
    return max(0, -exp) if isinstance(exp, int) else 0


def _ambiguous_example(a: ColumnAnalysis) -> str:
    for v in a.non_null:
        p, q = V.parse_number(v, "."), V.parse_number(v, ",")
        if p and q and (p.value != q.value):
            return v
    return a.non_null[0]


def convert(text: str, col: ColumnSchema) -> tuple[bool, Any]:
    """Convert one cell. Returns (ok, value); value is None for nulls."""
    if V.is_null(text):
        return True, None
    t = col.type
    if t == "string":
        return True, text
    if t == "boolean":
        b = V.parse_bool(text)
        return (b is not None), b
    if t in ("integer", "decimal"):
        p = V.parse_number(text, col.decimal_sep or ".")
        if p is None:
            return False, None
        if t == "integer":
            if p.has_fraction or p.leading_zero:
                return False, None
            return True, int(p.value)
        return True, p.value
    if t == "date":
        d = V.parse_date(text, col.date_format or "%Y-%m-%d")
        return (d is not None), d
    if t == "datetime":
        dt = V.parse_datetime(text, col.date_format or "iso")
        return (dt is not None), dt
    raise ValueError(f"unknown type {t}")


def to_jsonable(v: Any) -> Any:
    if isinstance(v, Decimal):
        return format(v, "f")
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v
