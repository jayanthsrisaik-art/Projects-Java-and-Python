from sniffload import bench, parse, synth
from sniffload.infer import Options


def test_synth_is_deterministic():
    a, b = synth.make_case(42), synth.make_case(42)
    assert a.data == b.data and a.cells == b.cells


def test_synth_truth_matches_text():
    c = synth.make_case(3)
    assert c.data.decode(c.encoding).lstrip("﻿") == c.text
    assert all(len(r) == len(c.columns) for r in c.cells)


def test_dialect_recovered_on_synthetic_files():
    for k in range(40):
        c = synth.make_case(k)
        inf = parse(c.data).inference
        assert inf.delimiter == c.delimiter, k
        assert inf.header == c.header, k


def test_benchmark_small_run_beats_baselines():
    r = bench.run(n_files=30, seed=1)
    s = r["systems"]
    assert s["sniffload"]["column_type_acc"] >= 0.95
    assert s["sniffload"]["good_cell_acc"] >= 0.95
    assert s["sniffload"]["good_rows_rejected"] == 0
    assert s["sniffload"]["column_type_acc"] >= s["sniffload-no-file-level"]["column_type_acc"]
    if "pandas-oracle-locale" in s:
        assert s["sniffload"]["good_cell_acc"] > s["pandas-oracle-locale"]["good_cell_acc"]


def test_eq_semantics():
    from datetime import date, datetime
    from decimal import Decimal
    assert bench._eq(Decimal("1.50"), Decimal("1.5"))
    assert bench._eq(1234.5, Decimal("1234.50"))
    assert not bench._eq("1234.5", Decimal("1234.5"))
    assert not bench._eq(datetime(2024, 1, 1), date(2024, 1, 1))
    assert not bench._eq(1, True)
    assert bench._eq(float("nan"), None)
    _ = Options  # imported for API surface check
