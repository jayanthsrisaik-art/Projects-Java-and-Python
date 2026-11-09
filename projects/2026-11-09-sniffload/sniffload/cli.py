"""Command line: ``python -m sniffload {infer,load,bench,synth}``."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import bench, synth
from .infer import Inference, Options, infer_bytes
from .load import SchemaDriftError, parse, write_clean_csv, write_sqlite


def _opts(a: argparse.Namespace) -> Options:
    return Options(min_ratio=a.min_ratio, default_date_order=a.date_order, default_decimal=a.decimal)


def _describe(inf: Inference) -> str:
    delim = {"\t": "TAB", ",": ",", ";": ";", "|": "|"}.get(inf.delimiter, inf.delimiter)
    out = [
        f"encoding : {inf.encoding} ({inf.encoding_reason})",
        f"delimiter: {delim}   header: {inf.header}",
        f"decimal  : '{inf.decimal_sep}' ({inf.decimal_resolved_by})   date order: {inf.date_order} ({inf.date_order_resolved_by})",
        "columns  :",
    ]
    for c in inf.columns:
        detail = []
        if c.decimal_sep:
            detail.append(f"dec='{c.decimal_sep}'")
        if c.scale is not None:
            detail.append(f"scale={c.scale}")
        if c.date_format:
            detail.append(f"format={c.date_format}")
        if c.resolved_by:
            detail.append(f"by={c.resolved_by}")
        detail.append(f"nulls={c.null_count}")
        if c.bad_count:
            detail.append(f"bad={c.bad_count}")
        out.append(f"  {c.name:<20} {c.type:<9} {' '.join(detail)}   <- {c.source_name!r}")
    for w in inf.warnings:
        out.append(f"warning  : {w}")
    return "\n".join(out)


def cmd_infer(a: argparse.Namespace) -> int:
    with open(a.file, "rb") as f:
        inf, _ = infer_bytes(f.read(), _opts(a))
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump(inf.to_dict(), f, indent=2, ensure_ascii=False)
    print(json.dumps(inf.to_dict(), indent=2, ensure_ascii=False) if a.json else _describe(inf))
    return 0


def cmd_load(a: argparse.Namespace) -> int:
    schema = None
    if a.schema:
        with open(a.schema, encoding="utf-8") as f:
            schema = Inference.from_dict(json.load(f))
    with open(a.file, "rb") as f:
        data = f.read()
    try:
        res = parse(data, _opts(a), schema=schema, on_bad=a.on_bad)
    except SchemaDriftError as e:
        print(f"schema drift: {e}", file=sys.stderr)
        return 2
    total = len(res.rows) + len(res.rejects)
    ratio = len(res.rejects) / total if total else 0.0
    if ratio > a.max_reject_ratio:
        print(f"reject ratio {ratio:.3f} exceeds --max-reject-ratio {a.max_reject_ratio}; nothing written",
              file=sys.stderr)
        return 3
    table = a.table or os.path.splitext(os.path.basename(a.file))[0].replace("-", "_")
    write_sqlite(res, a.db, table)
    if a.clean_csv:
        write_clean_csv(res, a.clean_csv)
    summary = res.summary() | {"table": table, "db": a.db}
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(summary | {"rejects": [r.__dict__ for r in res.rejects],
                                 "warnings": res.inference.warnings}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    res = bench.run(a.files, a.seed, include_pandas=not a.no_pandas)
    print(bench.format_table(res))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    return 0


def cmd_synth(a: argparse.Namespace) -> int:
    os.makedirs(a.out, exist_ok=True)
    for k in range(a.n):
        c = synth.make_case(a.seed * 100_000 + k)
        with open(os.path.join(a.out, f"{c.name}_{c.locale}.csv"), "wb") as f:
            f.write(c.data)
    print(f"wrote {a.n} files to {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sniffload", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--min-ratio", type=float, default=0.95,
                        help="share of non-null values a type must parse (default 0.95)")
        sp.add_argument("--date-order", choices=["MDY", "DMY", "YMD"], default="MDY",
                        help="fallback day/month order when nothing in the file decides")
        sp.add_argument("--decimal", choices=[".", ","], default=".",
                        help="fallback decimal separator when nothing in the file decides")

    sp = sub.add_parser("infer", help="print the inferred schema")
    sp.add_argument("file")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("-o", "--output", help="write schema JSON (reusable with load --schema)")
    common(sp)
    sp.set_defaults(fn=cmd_infer)

    sp = sub.add_parser("load", help="infer (or apply a locked schema) and load into SQLite")
    sp.add_argument("file")
    sp.add_argument("--db", required=True)
    sp.add_argument("--table")
    sp.add_argument("--schema", help="locked schema JSON from 'infer -o'; header drift fails the load")
    sp.add_argument("--on-bad", choices=["reject", "null"], default="reject")
    sp.add_argument("--max-reject-ratio", type=float, default=1.0)
    sp.add_argument("--clean-csv", help="also write a normalised UTF-8 CSV")
    sp.add_argument("--report", help="write a JSON load report including rejects")
    common(sp)
    sp.set_defaults(fn=cmd_load)

    sp = sub.add_parser("bench", help="benchmark vs csv.Sniffer and pandas on synthetic ground truth")
    sp.add_argument("--files", type=int, default=300)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--no-pandas", action="store_true")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_bench)

    sp = sub.add_parser("synth", help="write synthetic messy CSV files")
    sp.add_argument("--out", required=True)
    sp.add_argument("--n", type=int, default=10)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_synth)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
