import pytest
from fastapi.testclient import TestClient

from conftest import ROOT
from stockhold.api import create_app
from stockhold.bench import SimClock
from stockhold.store import SequentialIds, Store


@pytest.fixture
def env(tmp_path):
    clock = SimClock(5_000.0)
    store = Store(str(tmp_path / "api.db"), hold_ttl=120, clock=clock, ids=SequentialIds())
    app = create_app(store, catalog_path=ROOT / "samples" / "catalog.json")
    return TestClient(app), clock, store


def H(customer="ana", key=None):
    h = {"X-Customer-Id": customer}
    if key:
        h["Idempotency-Key"] = key
    return h


def test_catalog_seeded_and_listed(env):
    client, clock, _ = env
    r = client.get("/api/products")
    assert r.status_code == 200 and r.json()["server_time"] == clock.t
    vinyl = next(p for p in r.json()["products"] if p["sku"] == "VNYL-07")
    assert (vinyl["total"], vinyl["available"], vinyl["per_customer_limit"]) == (3, 3, 1)
    assert client.get("/api/products/NOPE").status_code == 404


def test_full_checkout_flow(env):
    client, clock, _ = env
    r = client.post("/api/holds", json={"sku": "SNKR-01", "qty": 2}, headers=H(key="intent-aaaa-1"))
    assert r.status_code == 201
    hold = r.json()
    assert hold["expires_at"] - hold["server_time"] == 120
    assert client.get("/api/products/SNKR-01").json()["held"] == 2
    clock.t += 30
    assert client.get(f"/api/holds/{hold['id']}", headers=H()).json()["status"] == "active"
    r = client.post(f"/api/holds/{hold['id']}/confirm", json={"payment_token": "tok_ok"}, headers=H())
    assert r.status_code == 201 and r.json()["amount_cents"] == 2 * 12900
    r2 = client.post(f"/api/holds/{hold['id']}/confirm", json={"payment_token": "tok_ok"}, headers=H())
    assert r2.status_code == 200 and r2.headers["Idempotent-Replayed"] == "true"
    p = client.get("/api/products/SNKR-01").json()
    assert (p["sold"], p["held"], p["available"]) == (2, 0, 23)
    assert client.get("/api/invariants").json() == {"ok": True, "problems": []}


def test_retry_replays_with_header(env):
    client, _, store = env
    body = {"sku": "VNYL-07", "qty": 1}
    a = client.post("/api/holds", json=body, headers=H(key="intent-bbbb-1"))
    b = client.post("/api/holds", json=body, headers=H(key="intent-bbbb-1"))
    assert a.status_code == b.status_code == 201
    assert "Idempotent-Replayed" not in a.headers and b.headers["Idempotent-Replayed"] == "true"
    assert a.json()["id"] == b.json()["id"] and store.product("VNYL-07")["held"] == 1
    c = client.post("/api/holds", json={"sku": "VNYL-07", "qty": 2}, headers=H(key="intent-bbbb-1"))
    assert c.status_code == 422 and c.json()["error"] == "idempotency_key_reused"


def test_headers_are_required_and_validated(env):
    client, _, _ = env
    body = {"sku": "VNYL-07", "qty": 1}
    assert client.post("/api/holds", json=body, headers={"X-Customer-Id": "ana"}).status_code == 422
    assert client.post("/api/holds", json=body, headers={"Idempotency-Key": "intent-cccc-1"}).status_code == 422
    assert client.post("/api/holds", json=body, headers=H(key="short")).status_code == 422
    assert client.post("/api/holds", json=body, headers=H("bad id!", "intent-cccc-1")).status_code == 422
    assert client.post("/api/holds", json={"sku": "VNYL-07", "qty": 0}, headers=H(key="intent-cccc-2")).status_code == 422


def test_expiry_and_release_over_http(env):
    client, clock, _ = env
    h1 = client.post("/api/holds", json={"sku": "VNYL-07", "qty": 1}, headers=H("ana", "intent-dddd-1")).json()
    h2 = client.post("/api/holds", json={"sku": "VNYL-07", "qty": 1}, headers=H("ben", "intent-dddd-2")).json()
    r = client.post(f"/api/holds/{h2['id']}/release", headers=H("ben"))
    assert r.json()["status"] == "released"
    clock.t += 121
    r = client.post(f"/api/holds/{h1['id']}/confirm", json={"payment_token": "tok_ok"}, headers=H("ana"))
    assert r.status_code == 410
    assert client.get("/api/products/VNYL-07").json()["available"] == 3


def test_hold_scoped_to_customer(env):
    client, _, _ = env
    h = client.post("/api/holds", json={"sku": "KBD-65", "qty": 1}, headers=H("ana", "intent-eeee-1")).json()
    assert client.get(f"/api/holds/{h['id']}", headers=H("ben")).status_code == 404
    r = client.post(f"/api/holds/{h['id']}/confirm", json={"payment_token": "tok_ok"}, headers=H("ben"))
    assert r.status_code == 404


def test_declined_then_paid(env):
    client, _, _ = env
    h = client.post("/api/holds", json={"sku": "KBD-65", "qty": 1}, headers=H("ana", "intent-ffff-1")).json()
    r = client.post(f"/api/holds/{h['id']}/confirm", json={"payment_token": "tok_decline"}, headers=H())
    assert r.status_code == 402 and r.json()["message"] == "card_declined"
    r = client.post(f"/api/holds/{h['id']}/confirm", json={"payment_token": "tok_ok"}, headers=H())
    assert r.status_code == 201


def test_serves_ui_shell(env):
    client, _, _ = env
    r = client.get("/")
    assert r.status_code == 200 and "stockhold" in r.text
    assert client.get("/styles.css").status_code == 200
