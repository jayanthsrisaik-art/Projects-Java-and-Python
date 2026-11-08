import json

import pytest
from fastapi.testclient import TestClient

from conftest import ROOT
from ledgerlens.api import create_app
from ledgerlens.cli import run_demo
from ledgerlens.synth import read_csv


@pytest.fixture()
def client(tmp_path, model):
    return TestClient(create_app(tmp_path / "ledger.db", model=model))


def _txn(i, desc, amount, date="2026-01-05", acct="a1"):
    return {"txn_id": f"x{i}", "account_id": acct, "posted_date": date, "descriptor": desc, "amount": amount}


def test_ingest_is_idempotent_and_categorises(client):
    batch = [_txn(1, "ACH CREDIT GLOBEX CORP PAYROLL 11112222", 3100.0),
             _txn(2, "TST* LUCKY PIZZERIA 4411", -23.5),
             _txn(2, "TST* LUCKY PIZZERIA 4411", -23.5)]          # duplicate inside the batch
    r = client.post("/transactions", json={"transactions": batch}).json()
    assert (r["received"], r["inserted"], r["duplicates"]) == (3, 2, 1)
    by_id = {t["txn_id"]: t for t in r["transactions"]}
    assert by_id["x1"]["suggested"] == "income" and by_id["x1"]["merchant_key"] == "GLOBEXCORP"
    assert by_id["x2"]["suggested"] == "dining"
    again = client.post("/transactions", json={"transactions": batch}).json()
    assert again["inserted"] == 0
    assert len(client.get("/transactions").json()) == 2


def test_low_confidence_goes_to_review_and_correction_creates_rule(client, model):
    # an unseen brand-only name via a processor shared by many categories
    batch = [_txn(i, f"DEBIT CARD PURCHASE - QUORVATEN DOVER DE CARD {1000 + i}", -40.0 - i,
                  date=f"2026-01-{10 + i:02d}") for i in range(4)]
    r = client.post("/transactions", json={"transactions": batch}).json()
    assert all(t["confidence"] < model.threshold for t in r["transactions"])
    assert {t["source"] for t in r["transactions"]} == {"review"}
    assert all(t["category"] is None for t in r["transactions"])
    assert len(client.get("/transactions", params={"source": "review"}).json()) == 4

    c = client.post("/corrections", json={"txn_id": "x0", "category": "health"}).json()
    assert c["merchant_key"] == "QUORVATEN" and c["other_transactions_updated"] == 3
    rows = {t["txn_id"]: t for t in client.get("/transactions").json()}
    assert rows["x0"]["source"] == "user"
    assert {rows[f"x{i}"]["source"] for i in (1, 2, 3)} == {"rule"}
    assert client.get("/rules").json()[0]["category"] == "health"

    # future transactions from that merchant follow the rule, whatever the model says
    new = client.post("/transactions", json={"transactions": [
        _txn(9, "POS DEBIT 0201 QUORVATEN #77 SALEM OR", -12.0, date="2026-02-01")]}).json()
    assert new["transactions"][0]["source"] == "rule" and new["transactions"][0]["category"] == "health"


def test_rule_never_overrides_an_explicit_user_choice(client):
    batch = [_txn(i, f"SQ *VELOMIRA BRISTOL CT", -20.0, date=f"2026-01-{10 + i:02d}") for i in range(3)]
    client.post("/transactions", json={"transactions": batch})
    client.post("/corrections", json={"txn_id": "x0", "category": "shopping", "apply_to_merchant": False})
    c = client.post("/corrections", json={"txn_id": "x1", "category": "dining"}).json()
    assert c["other_transactions_updated"] == 1   # only x2; x0 keeps the user's 'shopping'
    rows = {t["txn_id"]: t for t in client.get("/transactions").json()}
    assert rows["x0"]["category"] == "shopping" and rows["x0"]["source"] == "user"
    assert rows["x2"]["category"] == "dining" and rows["x2"]["source"] == "rule"


def test_validation_errors(client):
    assert client.post("/corrections", json={"txn_id": "nope", "category": "dining"}).status_code == 404
    client.post("/transactions", json={"transactions": [_txn(1, "TST* BLUE CAFE 1", -5.0)]})
    assert client.post("/corrections", json={"txn_id": "x1", "category": "beer"}).status_code == 422
    bad = _txn(2, "TST* BLUE CAFE 1", -5.0, date="05/01/2026")
    assert client.post("/transactions", json={"transactions": [bad]}).status_code == 422
    assert client.post("/transactions", json={"transactions": []}).status_code == 422
    assert client.get("/summary", params={"month": "2026-1"}).status_code == 422


def test_summary_and_subscriptions_endpoints(client):
    rows = read_csv(ROOT / "samples" / "transactions_sample.csv")
    client.post("/transactions", json={"transactions": rows})
    subs = client.get("/subscriptions", params={"account_id": "acct001"}).json()
    assert {s["merchant_key"] for s in subs} >= {"CONTOSOLLC", "MIGUTORZENVO"}
    summ = client.get("/summary", params={"month": "2026-01"}).json()["by_category"]
    expected_total = round(sum(-r["amount"] for r in rows if r["amount"] < 0 and r["posted_date"].startswith("2026-01")), 2)
    assert abs(sum(c["spend"] for c in summ) - expected_total) < 0.01   # nothing lost, incl. review rows
    assert client.get("/health").json()["status"] == "ok"


def test_demo_reproduces_expected_sample_output(model):
    expected = json.loads((ROOT / "samples" / "expected_demo_output.json").read_text())
    got = run_demo(model, str(ROOT / "samples" / "transactions_sample.csv"))
    assert got["first_ingest"] == expected["first_ingest"]
    assert got["redelivery"] == expected["redelivery"]
    assert got["subscriptions"] == expected["subscriptions"]
    exp_tx = {t["txn_id"]: t for t in expected["transactions"]}
    # labels and routing must match; tolerate float wobble only for rows sitting on the threshold
    mismatches = [t for t in got["transactions"]
                  if (t["suggested"], t["source"]) != (exp_tx[t["txn_id"]]["suggested"], exp_tx[t["txn_id"]]["source"])
                  and abs(exp_tx[t["txn_id"]]["confidence"] - expected["threshold"]) > 0.01]
    assert not mismatches


def test_sample_auto_accepted_rows_are_mostly_right(model):
    labeled = {r["txn_id"]: r for r in read_csv(ROOT / "samples" / "transactions_sample_labeled.csv")}
    got = run_demo(model, str(ROOT / "samples" / "transactions_sample.csv"))
    auto = [t for t in got["transactions"] if t["source"] == "model"]
    acc = sum(t["category"] == labeled[t["txn_id"]]["true_category"] for t in auto) / len(auto)
    assert len(auto) > 50 and acc >= 0.9
