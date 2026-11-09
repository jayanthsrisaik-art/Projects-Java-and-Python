from sniffload import Options, infer_bytes
from sniffload.encoding import detect_encoding
from sniffload.infer import detect_delimiter, normalise_names


def types(inf):
    return {c.name: c.type for c in inf.columns}


def test_encoding_detection():
    assert detect_encoding("a,b\n".encode("utf-8")).encoding == "utf-8"
    assert detect_encoding("a,b\n".encode("utf-8-sig")).encoding == "utf-8-sig"
    assert detect_encoding("a,b\n".encode("utf-16")).encoding == "utf-16"
    assert detect_encoding("a,b\nZürich,1\n".encode("utf-16-le")).encoding == "utf-16-le"
    assert detect_encoding("Köln,1\n".encode("cp1252")).encoding == "cp1252"


def test_semicolon_wins_even_when_every_row_has_one_comma():
    # Under ',' every data row also has 2 fields; the header row and typed columns break the tie.
    text = "name;price\nA;1,5\nB;2,5\nC;3,25\n"
    d, scores = detect_delimiter(text)
    assert d == ";", scores


def test_headerless_comma_decimals_with_semicolons():
    text = "a;1,5\nb;2,5\nc;3,25\n"
    assert detect_delimiter(text)[0] == ";"


def test_quoted_delimiters_do_not_confuse_detection():
    text = 'id,note\n1,"x; y; z"\n2,"a; b"\n3,plain\n'
    assert detect_delimiter(text)[0] == ","


def test_file_level_locale_resolves_ambiguous_column():
    # qty alone is ambiguous ("1.200" = 1200 or 1.2); amount proves ',' is the decimal mark.
    data = "amount;qty\n1.234,56;1.200\n12,50;7\n3,00;2.500\n".encode()
    inf, _ = infer_bytes(data)
    qty = inf.columns[1]
    assert inf.decimal_sep == "," and inf.decimal_resolved_by == "file"
    assert qty.type == "integer" and qty.resolved_by == "file"


def test_without_file_level_the_ablation_gets_it_wrong():
    data = "amount,qty\n\"1.234,56\",1.200\n\"12,50\",7\n\"3,00\",2.500\n".encode()
    good, _ = infer_bytes(data)
    ablated, _ = infer_bytes(data, Options(file_level=False))
    assert good.columns[1].type == "integer"
    assert ablated.columns[1].type == "decimal"  # read as 1.2 and 2.5


def test_date_order_from_other_column():
    data = b"start,end\n01/02/2024,25/02/2024\n03/04/2024,13/04/2024\n"
    inf, _ = infer_bytes(data)
    assert [c.date_order for c in inf.columns] == ["DMY", "DMY"]
    assert inf.columns[0].resolved_by == "file"


def test_ambiguous_dates_fall_back_with_warning():
    data = b"d\n01/02/2024\n03/04/2024\n"
    inf, _ = infer_bytes(data, Options(default_date_order="DMY"))
    assert inf.columns[0].date_order == "DMY" and inf.columns[0].resolved_by == "default"
    assert any("ambiguous" in w for w in inf.warnings)


def test_semicolon_prior_without_any_numeric_evidence():
    data = b"name;qty\nx;1.500\ny;2.000\n"
    inf, _ = infer_bytes(data)
    assert inf.decimal_resolved_by == "delimiter-prior"
    assert inf.columns[1].type == "integer"


def test_header_detection():
    inf, _ = infer_bytes(b"id,amount\n1,2.5\n2,3.5\n")
    assert inf.header
    inf, _ = infer_bytes(b"1,2.5\n2,3.5\n3,1.0\n")
    assert not inf.header and [c.name for c in inf.columns] == ["col_1", "col_2"]


def test_type_precedence_and_identifiers():
    data = (b"zip,flag,n,x,when,at\n"
            b"02134,Y,1,1.5,2024-01-02,2024-01-02 10:00:00\n"
            b"10001,N,2,2,2024-01-03,2024-01-03T11:00:00\n")
    inf, _ = infer_bytes(data)
    assert types(inf) == {"zip": "string", "flag": "boolean", "n": "integer", "x": "decimal",
                          "when": "date", "at": "datetime"}


def test_tolerance_keeps_type_and_counts_bad_values():
    rows = "\n".join(str(i) for i in range(40))
    inf, _ = infer_bytes(f"n\n{rows}\n#REF!\n".encode())
    assert inf.columns[0].type == "integer" and inf.columns[0].bad_count == 1
    strict, _ = infer_bytes(f"n\n{rows}\n#REF!\n".encode(), Options(min_ratio=1.0))
    assert strict.columns[0].type == "string"


def test_name_normalisation():
    assert normalise_names(["Order Date", "Paid?", "Größe", "", "order date", "1st"]) == \
        ["order_date", "paid", "grosse", "col_4", "order_date_2", "c_1st"]
