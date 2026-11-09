"""File-level inference: encoding -> dialect -> header -> column types.

The decimal separator and the day/month order are settled per file before any
column is finalised. Columns that prove one locale ("1.234,56", "31/01/2024")
vote. Columns that cannot decide on their own ("1.234", "03/04/2024") then take
the file's answer instead of a hard-coded default.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from . import columns as C
from .encoding import EncodingGuess, decode

DELIMITERS = [",", ";", "\t", "|"]


@dataclass
class Options:
    min_ratio: float = 0.95  # share of non-null values a type must parse
    default_date_order: str = "MDY"  # used when nothing in the file decides
    default_decimal: str = "."
    sample_rows: int = 5000  # rows used for inference (all rows are loaded)
    file_level: bool = True  # resolve ambiguous columns from other columns (ablation switch)


@dataclass
class Inference:
    encoding: str
    encoding_reason: str
    delimiter: str
    header: bool
    columns: list[C.ColumnSchema]
    decimal_sep: str
    decimal_resolved_by: str
    date_order: str
    date_order_resolved_by: str
    rows_sampled: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["columns"] = [c.to_dict() for c in self.columns]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Inference":
        d = dict(d)
        d["columns"] = [C.ColumnSchema.from_dict(c) for c in d["columns"]]
        return cls(**d)


def read_records(text: str, delimiter: str) -> list[tuple[int, list[str]]]:
    """Parse CSV text into (start_line, fields) records, skipping blank lines."""
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, quotechar='"', doublequote=True)
    out: list[tuple[int, list[str]]] = []
    prev = 0
    try:
        for rec in reader:
            start = prev + 1
            prev = reader.line_num
            if not rec or (len(rec) == 1 and rec[0].strip() == ""):
                continue
            out.append((start, rec))
    except csv.Error:
        # Unterminated quote at EOF and similar: keep what was read so far; the
        # loader will report a field-count error for the broken tail.
        pass
    return out


def _sample_text(text: str, max_chars: int = 200_000) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    return text[: cut if cut > 0 else max_chars]


def detect_delimiter(text: str, min_ratio: float = 0.95) -> tuple[str, dict[str, tuple]]:
    """Score each candidate delimiter, compared as a tuple in this order:

    field-count consistency, share of typed columns, minus the share of text columns
    that keep containing another candidate delimiter, then column count.
    """
    sample = _sample_text(text)
    scores: dict[str, tuple] = {}
    for d in DELIMITERS:
        recs = [r for _, r in read_records(sample, d)][:500]
        if not recs:
            continue
        counts = Counter(len(r) for r in recs)
        width, freq = counts.most_common(1)[0]
        if width < 2:
            continue
        consistency = round(freq / len(recs), 2)
        body = [r for r in recs[1:] if len(r) == width] or [r for r in recs if len(r) == width]
        typed = leaky = 0
        others = [o for o in DELIMITERS if o != d]
        for i in range(width):
            col = [r[i] for r in body]
            kind = C.analyze_column(col, min_ratio).kind
            if kind not in ("string", "empty"):
                typed += 1
            elif sum(any(o in v for o in others) for v in col) * 2 >= len(col):
                # Text that keeps containing another candidate delimiter suggests a wrong split.
                leaky += 1
        scores[d] = (consistency, round(typed / width, 3), -round(leaky / width, 3), width)
    if not scores:
        return ",", scores
    best = max(scores, key=lambda d: scores[d])
    return best, scores


def _header_votes(row0: list[str], cols: list[C.ColumnSchema]) -> tuple[int, int]:
    header, data = 0, 0
    for v, col in zip(row0, cols):
        if col.type == "string":
            continue
        ok, _ = C.convert(v, col)
        if ok and v.strip():
            data += 1
        else:
            header += 1
    return header, data


_TRANSLIT = str.maketrans({"ß": "ss", "æ": "ae", "Æ": "AE", "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d"})


def normalise_names(raw: list[str]) -> list[str]:
    out: list[str] = []
    seen: Counter[str] = Counter()
    for i, name in enumerate(raw, 1):
        s = name.translate(_TRANSLIT)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower() or f"col_{i}"
        if s[0].isdigit():
            s = f"c_{s}"
        seen[s] += 1
        out.append(s if seen[s] == 1 else f"{s}_{seen[s]}")
    # a later plain name may collide with an earlier suffixed one; resolve deterministically
    final, used = [], set()
    for s in out:
        base, k = s, 2
        while s in used:
            s, k = f"{base}_{k}", k + 1
        used.add(s)
        final.append(s)
    return final


def _resolve_file_locale(analyses: list[C.ColumnAnalysis], delimiter: str, opt: Options):
    if not opt.file_level:
        return (opt.default_decimal, "default"), (opt.default_date_order, "default")
    dec_votes = Counter(a.num_decisive for a in analyses if a.kind == "number" and a.num_decisive)
    if dec_votes:
        dec = (dec_votes.most_common(1)[0][0], "file")
    elif delimiter == ";":
        # Excel writes ';'-separated CSV exactly in the locales that use ',' as decimal separator.
        dec = (",", "delimiter-prior")
    else:
        dec = (opt.default_decimal, "default")

    order_votes = Counter(
        a.date_decisive for a in analyses
        if a.kind in ("date", "datetime") and a.date_decisive in ("DMY", "MDY")
    )
    if order_votes:
        order = (order_votes.most_common(1)[0][0], "file")
    elif dec[0] == "," and dec[1] != "default":
        order = ("DMY", "locale-prior")  # comma-decimal locales write day first
    else:
        order = (opt.default_date_order, "default")
    return dec, order


def _infer_columns(body: list[list[str]], width: int, names: list[str], source: list[str],
                   delimiter: str, opt: Options):
    analyses = [C.analyze_column([r[i] for r in body], opt.min_ratio) for i in range(width)]
    dec, order = _resolve_file_locale(analyses, delimiter, opt)
    cols = [C.finalize_column(a, names[i], source[i], dec, order) for i, a in enumerate(analyses)]
    return cols, dec, order


def infer_text(text: str, enc: EncodingGuess, opt: Options | None = None,
               delimiter: str | None = None) -> Inference:
    opt = opt or Options()
    warnings: list[str] = []
    if delimiter is None:
        delimiter, _ = detect_delimiter(text, opt.min_ratio)
    records = read_records(text, delimiter)
    if not records:
        raise ValueError("file has no records")
    width = Counter(len(r) for _, r in records).most_common(1)[0][0]
    good = [r for _, r in records if len(r) == width]
    ragged = len(records) - len(good)
    if ragged:
        warnings.append(f"{ragged} record(s) do not have {width} fields")
    sample = good[: opt.sample_rows + 1]

    first_is_full = len(records[0][1]) == width
    row0 = records[0][1] if first_is_full else [""] * width
    body = sample[1:] if first_is_full else sample
    placeholder = [f"col_{i}" for i in range(1, width + 1)]
    cols, dec, order = _infer_columns(body, width, placeholder, placeholder, delimiter, opt)
    h, d = _header_votes(row0, cols)
    if h or d:
        header = h > d
    else:
        # Every column is text: call it a header when row 0 is non-empty, unique and
        # none of its values repeats further down its own column.
        header = (
            all(v.strip() for v in row0)
            and len(set(row0)) == len(row0)
            and all(v not in {r[i] for r in body} for i, v in enumerate(row0))
        )
        if header:
            warnings.append("all columns are text; header detected by uniqueness heuristic")
    if header:
        source = [v.strip() for v in row0]
        names = normalise_names(source)
    else:
        source = placeholder
        names = placeholder
        body = sample
    cols, dec, order = _infer_columns(body, width, names, source, delimiter, opt)
    for c in cols:
        warnings.extend(f"{c.name}: {w}" for w in c.warnings)
    return Inference(
        encoding=enc.encoding, encoding_reason=enc.reason, delimiter=delimiter, header=header,
        columns=cols, decimal_sep=dec[0], decimal_resolved_by=dec[1],
        date_order=order[0], date_order_resolved_by=order[1],
        rows_sampled=len(body), warnings=warnings,
    )


def infer_bytes(data: bytes, opt: Options | None = None) -> tuple[Inference, str]:
    text, enc = decode(data)
    return infer_text(text, enc, opt), text


def infer_file(path: str, opt: Options | None = None) -> Inference:
    with open(path, "rb") as f:
        return infer_bytes(f.read(), opt)[0]
