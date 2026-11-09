"""Synthetic messy-CSV generator with exact ground truth.

Each case is a CSV file (bytes) plus a record of everything a loader should
recover: encoding, delimiter, header, every column's logical type and every
cell's canonical value (``None`` for null, ``BAD`` for an injected corrupt cell).

Locales follow real spreadsheet exports: US/UK use '.' decimals and ',' thousands,
DE uses ',' decimals with '.' thousands and usually ';' delimiters, FR uses ','
decimals with (no-break) space thousands, ISO uses '.' decimals and YYYY-MM-DD.
"""
from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

BAD = "<BAD>"

LOCALES = {
    "us": {"dec": ".", "thou": ",", "date": "%m/%d/%Y", "dt": "%m/%d/%Y %H:%M", "delims": [",", ",", "\t", "|"]},
    "uk": {"dec": ".", "thou": ",", "date": "%d/%m/%Y", "dt": "%d/%m/%Y %H:%M", "delims": [",", ","]},
    "de": {"dec": ",", "thou": ".", "date": "%d.%m.%Y", "dt": "%d.%m.%Y %H:%M", "delims": [";", ";", ";", "\t", ","]},
    "fr": {"dec": ",", "thou": " ", "date": "%d/%m/%Y", "dt": "%d/%m/%Y %H:%M", "delims": [";", ";"]},
    "iso": {"dec": ".", "thou": "", "date": "%Y-%m-%d", "dt": "%Y-%m-%dT%H:%M:%S", "delims": [",", "|", "\t"]},
}
ENCODINGS = ["utf-8"] * 5 + ["utf-8-sig"] * 2 + ["cp1252"] * 2 + ["utf-16"]
CITIES = ["Zürich", "São Paulo", "Málaga", "Reykjavík", "Köln", "Montréal", "Lyon", "Austin",
          "Leeds", "Porto", "Oslo", "Gdańsk".replace("ń", "n"), "Århus", "Nîmes"]
NOTES = ["ok", "Delivered, signed by \"Bob\"", "late; customer called", "fragile | handle with care",
         "split\nshipment", "", "left at door", "re-sent", "tab\tinside"]
BOOLS = [("yes", "no"), ("true", "false"), ("Y", "N"), ("TRUE", "FALSE")]
NULLS = ["", "NA", "null", "N/A", "-"]
BAD_TOKENS = ["#REF!", "12..5", "n.v.", "TBD", "1O0"]


@dataclass
class Case:
    name: str
    data: bytes
    text: str
    encoding: str
    delimiter: str
    header: bool
    locale: str
    columns: list[tuple[str, str]]  # (header name, logical type)
    cells: list[list[Any]]  # truth values per data row
    meta: dict[str, Any] = field(default_factory=dict)


def _fmt_int(v: int, loc: dict, rng: random.Random) -> str:
    if loc["thou"] and abs(v) >= 1000 and rng.random() < 0.7:
        s = f"{abs(v):,}".replace(",", loc["thou"])
        return ("-" if v < 0 else "") + s
    return str(v)


def _fmt_dec(v: Decimal, loc: dict, rng: random.Random, places: int) -> str:
    q = f"{abs(v):,.{places}f}" if (loc["thou"] and rng.random() < 0.7) else f"{abs(v):.{places}f}"
    q = q.replace(",", "\x00").replace(".", loc["dec"]).replace("\x00", loc["thou"])
    return ("-" if v < 0 else "") + q


def _column_kinds(rng: random.Random) -> list[str]:
    optional = ["qty", "amount", "rate", "zip", "date", "ts", "flag", "city", "note", "code"]
    picked = rng.sample(optional, rng.randint(3, 7))
    if not any(k in picked for k in ("city", "note", "code")):
        picked.append("city")
    return ["id"] + picked


def make_case(seed: int) -> Case:
    rng = random.Random(seed)
    locale = rng.choice(list(LOCALES))
    loc = LOCALES[locale]
    delimiter = rng.choice(loc["delims"])
    encoding = rng.choice(ENCODINGS)
    header = rng.random() < 0.85
    small = rng.random() < 0.15
    n = rng.randint(6, 12) if small else rng.randint(30, 300)
    kinds = _column_kinds(rng)
    bools = rng.choice(BOOLS)
    base = date(2024, 1, 1)
    types = {"id": "integer", "qty": "integer", "amount": "decimal", "rate": "decimal", "zip": "string",
             "date": "date", "ts": "datetime", "flag": "boolean", "city": "string", "note": "string",
             "code": "string"}
    headers = {"id": "ID", "qty": "Quantity", "amount": "Amount", "rate": "Discount Rate", "zip": "Postal Code",
               "date": "Order Date", "ts": "Updated At", "flag": "Paid?", "city": "City", "note": "Note",
               "code": "SKU"}

    text_rows: list[list[str]] = []
    truth: list[list[Any]] = []
    for i in range(n):
        tr, tx = [], []
        for k in kinds:
            nullable = k not in ("id",)
            if nullable and rng.random() < 0.05:
                tr.append(None)
                tx.append(rng.choice(NULLS))
                continue
            if k == "id":
                v = i + 1
                tr.append(v); tx.append(str(v))
            elif k == "qty":
                v = rng.choice([rng.randint(0, 999), rng.randint(1000, 250000)])
                tr.append(v); tx.append(_fmt_int(v, loc, rng))
            elif k == "amount":
                v = Decimal(rng.randint(-50000, 2_500_000)) / 100
                tr.append(v); tx.append(_fmt_dec(v, loc, rng, 2))
            elif k == "rate":
                v = Decimal(rng.randint(0, 999)) / 1000
                tr.append(v); tx.append(_fmt_dec(v, loc, rng, 3))
            elif k == "zip":
                v = f"{rng.randint(0, 99999):05d}"
                tr.append(v); tx.append(v)
            elif k == "date":
                v = base + timedelta(days=rng.randint(0, 700))
                tr.append(v); tx.append(v.strftime(loc["date"]))
            elif k == "ts":
                v = datetime(2024, 1, 1) + timedelta(minutes=rng.randint(0, 900_000))
                tr.append(v); tx.append(v.strftime(loc["dt"]))
            elif k == "flag":
                v = rng.random() < 0.5
                tr.append(v); tx.append(bools[0] if v else bools[1])
            elif k == "city":
                v = rng.choice(CITIES)
                tr.append(v); tx.append(v)
            elif k == "note":
                v = rng.choice(NOTES)
                if v == "":
                    tr.append(None)
                else:
                    tr.append(v)
                tx.append(v)
            elif k == "code":
                v = f"{rng.choice('ABCX')}-{rng.randint(1, 999)}"
                tr.append(v); tx.append(v)
        truth.append(tr)
        text_rows.append(tx)

    # Corrupt a couple of typed cells in larger files (stays within the 95% tolerance).
    typed = [j for j, k in enumerate(kinds) if types[k] not in ("string",) and k != "id"]
    if not small and typed and rng.random() < 0.5:
        for _ in range(rng.randint(1, 2)):
            i, j = rng.randrange(n), rng.choice(typed)
            truth[i][j] = BAD
            text_rows[i][j] = rng.choice(BAD_TOKENS)

    # Dates with time part: truth for 'ts' keeps seconds only for ISO.
    buf = io.StringIO(newline="")
    lt = rng.choice(["\n", "\r\n"])
    w = csv.writer(buf, delimiter=delimiter, quotechar='"', lineterminator=lt, quoting=csv.QUOTE_MINIMAL)
    if header:
        w.writerow([headers[k] for k in kinds])
    w.writerows(text_rows)
    text = buf.getvalue()
    data = text.encode(encoding)
    return Case(
        name=f"case_{seed:04d}", data=data, text=text, encoding=encoding, delimiter=delimiter,
        header=header, locale=locale, columns=[(headers[k], types[k]) for k in kinds], cells=truth,
        meta={"kinds": kinds, "rows": n, "small": small},
    )
