"""Benchmark sniffload against stdlib and pandas baselines on synthetic ground truth.

Systems compared
----------------
* ``sniffload``: full pipeline, nothing given in advance.
* ``sniffload-no-file-level``: ablation. Ambiguous columns use fixed defaults
  ('.' decimals, MDY) instead of evidence from other columns in the file.
* ``csv.Sniffer``: stdlib dialect and header detection only.
* ``pandas-default``: ``pd.read_csv`` given the *true* encoding, delimiter and
  header row (oracle dialect), otherwise defaults.
* ``pandas-oracle-locale``: as above, also given the true ``decimal=`` and
  ``thousands=`` arguments. pandas still parses no dates unless told which columns.

Metrics
-------
* dialect: share of files with the right delimiter, header flag and decoded text.
* column type accuracy: inferred logical type == true type.
* good-cell accuracy: share of non-corrupt truth cells recovered with the exact
  value (nulls included). Cells in wrongly rejected rows count as wrong.
* bad-cell catch rate: share of injected corrupt cells that were rejected or nulled
  rather than loaded as a value.
"""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from . import synth
from .infer import Options
from .load import parse


def _eq(pred: Any, truth: Any) -> bool:
    if truth is None:
        return pred is None or (isinstance(pred, float) and math.isnan(pred))
    if pred is None or (isinstance(pred, float) and math.isnan(pred)):
        return False
    if isinstance(truth, bool) or isinstance(pred, bool):
        return type(pred) is bool and pred == truth
    if isinstance(truth, (int, Decimal)) and not isinstance(truth, bool):
        if isinstance(pred, str) or isinstance(pred, (date, datetime)):
            return False
        try:
            return abs(Decimal(str(pred)) - Decimal(truth)) <= Decimal("1e-9")
        except Exception:
            return False
    if isinstance(truth, datetime):
        return isinstance(pred, datetime) and pred.replace(tzinfo=None) == truth
    if isinstance(truth, date):
        return isinstance(pred, date) and not isinstance(pred, datetime) and pred == truth
    return isinstance(pred, str) and pred == truth


@dataclass
class Tally:
    files: int = 0
    delim_ok: int = 0
    header_ok: int = 0
    text_ok: int = 0
    cols: int = 0
    cols_ok: int = 0
    good_cells: int = 0
    good_ok: int = 0
    bad_cells: int = 0
    bad_caught: int = 0
    good_rows_rejected: int = 0
    errors: int = 0

    def row(self) -> dict[str, Any]:
        r = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
        return {
            "files": self.files,
            "delimiter_acc": r(self.delim_ok, self.files),
            "header_acc": r(self.header_ok, self.files),
            "decode_acc": r(self.text_ok, self.files),
            "column_type_acc": r(self.cols_ok, self.cols),
            "good_cell_acc": r(self.good_ok, self.good_cells),
            "bad_cell_catch": r(self.bad_caught, self.bad_cells),
            "good_rows_rejected": self.good_rows_rejected,
            "errors": self.errors,
        }


def _score_sniffload(case: synth.Case, t: Tally, opt: Options) -> None:
    res = parse(case.data, opt)
    inf = res.inference
    t.delim_ok += inf.delimiter == case.delimiter
    t.header_ok += inf.header == case.header
    t.text_ok += case.data.decode(inf.encoding).lstrip("﻿") == case.text
    truth_types = [ty for _, ty in case.columns]
    if len(inf.columns) == len(truth_types):
        t.cols_ok += sum(c.type == ty for c, ty in zip(inf.columns, truth_types))
    # map the i-th data record to the i-th truth row via source line order
    loaded = {line: vals for line, vals in res.rows}
    lines = sorted(list(loaded) + [r.line for r in res.rejects])
    rejected = {r.line for r in res.rejects}
    aligned = len(lines) == len(case.cells) and len(inf.columns) == len(truth_types)
    for i, trow in enumerate(case.cells):
        line = lines[i] if aligned else None
        vals = loaded.get(line) if line is not None else None
        row_has_bad = any(v is synth.BAD or v == synth.BAD for v in trow)
        if line in rejected and not row_has_bad:
            t.good_rows_rejected += 1
        for j, tv in enumerate(trow):
            if tv == synth.BAD:
                # caught = row rejected, or the cell was nulled
                t.bad_caught += (line in rejected) or (vals is not None and vals[j] is None)
            else:
                t.good_ok += vals is not None and _eq(vals[j], tv)


def _score_sniffer(case: synth.Case, t: Tally) -> None:
    text = case.data.decode(case.encoding).lstrip("﻿")
    sample = text[:20000]
    t.text_ok += 1  # given the true encoding
    try:
        d = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        t.delim_ok += d.delimiter == case.delimiter
    except csv.Error:
        t.errors += 1
    try:
        t.header_ok += csv.Sniffer().has_header(sample) == case.header
    except csv.Error:
        t.errors += 1


def _pandas_type(series) -> str:
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        nn = series.dropna()
        return "integer" if len(nn) and (nn == nn.round()).all() else "decimal"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "string"


def _score_pandas(case: synth.Case, t: Tally, oracle_locale: bool) -> None:
    import pandas as pd

    loc = synth.LOCALES[case.locale]
    kw: dict[str, Any] = dict(sep=case.delimiter, encoding=case.encoding,
                              header=0 if case.header else None, engine="python")
    if oracle_locale:
        kw.update(decimal=loc["dec"], thousands=loc["thou"] or None)
    t.delim_ok += 1
    t.header_ok += 1
    t.text_ok += 1
    try:
        df = pd.read_csv(io.BytesIO(case.data), **kw)
    except Exception:
        t.errors += 1
        return
    truth_types = [ty for _, ty in case.columns]
    ptypes = [_pandas_type(df[c]) for c in df.columns]
    if len(ptypes) == len(truth_types):
        for pty, tty in zip(ptypes, truth_types):
            t.cols_ok += pty == tty or (pty == "datetime" and tty == "date")
    if len(df) != len(case.cells) or len(ptypes) != len(truth_types):
        return
    for i, trow in enumerate(case.cells):
        for j, tv in enumerate(trow):
            v = df.iat[i, j]
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, float) and ptypes[j] == "integer" and not math.isnan(v):
                v = int(v)
            if tv == synth.BAD:
                t.bad_caught += v is None or (isinstance(v, float) and math.isnan(v))
            else:
                t.good_ok += _eq(v, tv)


def run(n_files: int = 200, seed: int = 0, include_pandas: bool = True) -> dict[str, Any]:
    systems = ["sniffload", "sniffload-no-file-level", "csv.Sniffer"]
    try:
        import pandas  # noqa: F401
        has_pandas = include_pandas
    except ImportError:
        has_pandas = False
    if has_pandas:
        systems += ["pandas-default (oracle dialect)", "pandas-oracle-locale"]
    tallies = {s: Tally() for s in systems}
    by_locale: dict[str, dict[str, Tally]] = {}
    for k in range(n_files):
        case = synth.make_case(seed * 100_000 + k)
        n_cols = len(case.columns)
        n_good = sum(v != synth.BAD for row in case.cells for v in row)
        n_bad = sum(v == synth.BAD for row in case.cells for v in row)
        loc_t = by_locale.setdefault(case.locale, {s: Tally() for s in ("sniffload", "sniffload-no-file-level")})
        for s in systems:
            targets = [tallies[s]] + ([loc_t[s]] if s in loc_t else [])
            for t in targets:
                t.files += 1
                t.cols += n_cols
                t.good_cells += n_good
                t.bad_cells += n_bad
                if s == "sniffload":
                    _score_sniffload(case, t, Options())
                elif s == "sniffload-no-file-level":
                    _score_sniffload(case, t, Options(file_level=False))
                elif s == "csv.Sniffer":
                    _score_sniffer(case, t)
                elif s.startswith("pandas-default"):
                    _score_pandas(case, t, oracle_locale=False)
                else:
                    _score_pandas(case, t, oracle_locale=True)
    out = {"n_files": n_files, "seed": seed, "systems": {s: t.row() for s, t in tallies.items()}}
    out["sniffload_by_locale"] = {
        loc: {s: t.row() for s, t in d.items()} for loc, d in sorted(by_locale.items())
    }
    for s in ("csv.Sniffer",):
        for key in ("column_type_acc", "good_cell_acc", "bad_cell_catch"):
            out["systems"][s][key] = None  # not applicable: dialect detection only
    return out


def format_table(res: dict[str, Any]) -> str:
    cols = ["delimiter_acc", "header_acc", "decode_acc", "column_type_acc", "good_cell_acc",
            "bad_cell_catch", "good_rows_rejected", "errors"]
    lines = ["| system | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for s, row in res["systems"].items():
        lines.append(f"| {s} | " + " | ".join("n/a" if row[c] is None else str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    r = run()
    print(format_table(r))
    print(json.dumps(r["sniffload_by_locale"], indent=1))
