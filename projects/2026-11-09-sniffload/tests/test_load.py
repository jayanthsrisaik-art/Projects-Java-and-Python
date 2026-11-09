import json
import os
import sqlite3
from decimal import Decimal

import pytest

from sniffload import Inference, SchemaDriftError, infer_bytes, parse, write_clean_csv, write_sqlite
from sniffload.cli import _describe, main

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")


def read(name):
    with open(os.path.join(SAMPLES, name), "rb") as f:
        return f.read()


@pytest.mark.parametrize("name", ["de_orders.csv", "us_payments.csv", "readings_noheader.psv",
                                  "ambiguous_dates.csv"])
def test_samples_match_expected(name, tmp_path):
    res = parse(read(name))
    stem = os.path.splitext(name)[0]
    exp = os.path.join(SAMPLES, "expected")
    with open(os.path.join(exp, f"{stem}.schema.txt"), encoding="utf-8") as f:
        assert _describe(res.inference) + "\n" == f.read()
    out = tmp_path / "clean.csv"
    write_clean_csv(res, str(out))
    with open(os.path.join(exp, f"{stem}.clean.csv"), "rb") as f:
        assert out.read_bytes() == f.read()
    with open(os.path.join(exp, f"{stem}.rejects.json"), encoding="utf-8") as f:
        assert [r.__dict__ for r in res.rejects] == json.load(f)


def test_rejects_carry_line_and_reason():
    res = parse(read("de_orders.csv"))
    by_line = {r.line: r.reasons for r in res.rejects}
    assert by_line[10] == ["menge: '#REF!' is not a valid integer"]
    assert by_line[25] == ["expected 6 fields, got 2"]
    assert len(res.rows) == 22


def test_on_bad_null_keeps_row():
    res = parse(read("de_orders.csv"), on_bad="null")
    line10 = dict(res.rows)[10]
    assert line10[2] is None and res.nulled_cells == 1
    assert [r.line for r in res.rejects] == [25]  # ragged rows are always rejected


def test_multiline_quoted_field_line_numbers():
    res = parse(read("us_payments.csv"))
    rows = dict(res.rows)
    assert sorted(rows) == [2, 3, 4, 6, 7]  # record starting at line 4 spans two lines
    assert rows[3][1] == Decimal("-45.10")
    assert rows[4][5] == "multi\nline memo"


def test_sqlite_load_is_idempotent(tmp_path):
    db = str(tmp_path / "t.db")
    res = parse(read("de_orders.csv"))
    write_sqlite(res, db, "orders")
    write_sqlite(res, db, "orders")
    con = sqlite3.connect(db)
    assert con.execute("select count(*) from orders").fetchone()[0] == 22
    assert con.execute("select count(*) from orders__rejects").fetchone()[0] == 2
    total = con.execute("select round(sum(betrag), 2) from orders").fetchone()[0]
    expected = sum(float(v[1]) for _, v in res.rows if v[1] is not None)
    assert total == round(expected, 2)
    assert con.execute("select typeof(menge), typeof(bestelldatum) from orders limit 1").fetchone() == \
        ("integer", "text")


def test_locked_schema_detects_drift_and_reuses_encoding():
    schema = infer_bytes(read("de_orders.csv"))[0]
    with pytest.raises(SchemaDriftError, match="Kanal"):
        parse(read("de_orders_v2.csv"), schema=schema)
    # Same header, ASCII-only body: locked schema still applies DMY + ',' decimals.
    body = "Kunde;Betrag;Menge;Bestelldatum;Bezahlt;Notiz\r\nNeu AG;10,00;1.000;01.06.2024;ja;\r\n"
    res = parse(body.encode("cp1252"), schema=Inference.from_dict(schema.to_dict()))
    vals = res.rows[0][1]
    assert vals[1] == Decimal("10.00") and vals[2] == 1000 and str(vals[3]) == "2024-06-01"


def test_cli_infer_load_and_exit_codes(tmp_path, capsys):
    schema = tmp_path / "s.json"
    db = tmp_path / "out.db"
    assert main(["infer", os.path.join(SAMPLES, "de_orders.csv"), "-o", str(schema)]) == 0
    assert main(["load", os.path.join(SAMPLES, "de_orders.csv"), "--db", str(db),
                 "--schema", str(schema), "--report", str(tmp_path / "r.json")]) == 0
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["rows_loaded"] == 22 and report["rows_rejected"] == 2
    assert main(["load", os.path.join(SAMPLES, "de_orders_v2.csv"), "--db", str(db),
                 "--schema", str(schema)]) == 2
    assert main(["load", os.path.join(SAMPLES, "de_orders.csv"), "--db", str(tmp_path / "x.db"),
                 "--max-reject-ratio", "0.05"]) == 3
    assert not (tmp_path / "x.db").exists()
