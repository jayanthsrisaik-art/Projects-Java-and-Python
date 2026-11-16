import json

from colineage.cli import main

from conftest import SAMPLES

SHOP = str(SAMPLES / "shop")


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_analyze_writes_json(capsys, tmp_path):
    out = tmp_path / "l.json"
    code, _, err = run(capsys, "analyze", SHOP, "--out", str(out))
    assert code == 0 and "10/10" in err
    doc = json.loads(out.read_text())
    assert doc["end_to_end"]["stg_items.line_total"] == {
        "order_items.quantity": {"value": "transform", "control": False},
        "order_items.unit_price": {"value": "transform", "control": False},
    }


def test_trace_and_impact_match_expected_files(capsys):
    code, out, _ = run(capsys, "impact", SHOP, "orders.status")
    assert code == 0 and out == (SAMPLES / "shop" / "expected" / "impact_orders_status.txt").read_text()
    code, out, _ = run(capsys, "trace", SHOP, "customer_rank.country_rank")
    assert code == 0 and out == (SAMPLES / "shop" / "expected" / "trace_country_rank.txt").read_text()


def test_check_exit_codes(capsys):
    code, out, _ = run(capsys, "check", SHOP, "--drop", "orders.discount", "--rename", "products.supplier=vendor")
    assert code == 1 and out == (SAMPLES / "shop" / "expected" / "check_drop_discount.txt").read_text()
    code, out, _ = run(capsys, "check", SHOP, "--drop", "customers.created_at")
    assert code == 0 and "breaks 0 model(s)" in out
    code, _, err = run(capsys, "check", SHOP, "--drop", "customers.nope")
    assert code == 2 and "no such source column" in err


def test_explain_and_unknowns(capsys):
    code, out, _ = run(capsys, "explain", str(SAMPLES / "saas"), "top_accounts.account_id", "events.duration")
    assert code == 0 and out == (SAMPLES / "saas" / "expected" / "explain_top_accounts.txt").read_text()
    assert run(capsys, "trace", SHOP, "nope.col")[0] == 2
    assert run(capsys, "impact", SHOP, "orders.nope")[0] == 2
    assert run(capsys, "explain", SHOP, "stg_items.quantity", "orders.discount")[0] == 1


def test_broken_project_exit_code(capsys):
    code, _, err = run(capsys, "analyze", str(SAMPLES / "broken"))
    assert code == 2
    assert err == (SAMPLES / "broken" / "expected_errors.txt").read_text()


def test_verify(capsys):
    code, out, _ = run(capsys, "verify", SHOP, "--seeds", "0")
    assert code == 0 and "sound on this sample" in out
