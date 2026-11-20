import pytest

from stockhold.store import Store, mock_charge


def avail(store, sku="A"):
    return store.product(sku)["available"]


def test_reserve_reduces_availability_and_reports_hold(store):
    out = store.reserve("u1", "A", 2, "key-00001")
    assert out.status == 201 and out.body["status"] == "active"
    assert out.body["expires_at"] == out.body["created_at"] + 60
    assert avail(store) == 1 and store.product("A")["held"] == 2


def test_sold_out_reports_remaining_stock(store):
    store.reserve("u1", "A", 2, "key-00001")
    out = store.reserve("u2", "A", 2, "key-00002")
    assert out.status == 409
    assert (out.body["error"], out.body["available"]) == ("sold_out", 1)


def test_per_customer_limit_counts_holds_and_orders(store):
    h = store.reserve("u1", "A", 1, "key-00001").body["id"]
    assert store.confirm("u1", h, "tok_ok").status == 201
    assert store.reserve("u1", "A", 1, "key-00002").status == 201
    out = store.reserve("u1", "A", 1, "key-00003")
    assert out.status == 409 and out.body["error"] == "limit_exceeded"
    assert out.body["already"] == 2


def test_unknown_product_and_bad_qty(store):
    assert store.reserve("u1", "NOPE", 1, "key-00001").status == 404
    assert store.reserve("u1", "A", 0, "key-00002").status == 422


def test_idempotent_replay_returns_same_hold_without_new_effect(store, clock):
    first = store.reserve("u1", "A", 1, "key-00001")
    clock.t += 5
    again = store.reserve("u1", "A", 1, "key-00001")
    assert again.replayed and again.status == 201
    assert again.body["id"] == first.body["id"]
    assert again.body["server_time"] == clock.t  # fresh clock for the client countdown
    assert store.product("A")["held"] == 1


def test_idempotency_key_is_scoped_per_customer(store):
    a = store.reserve("u1", "A", 1, "shared-key-1")
    b = store.reserve("u2", "A", 1, "shared-key-1")
    assert not b.replayed and a.body["id"] != b.body["id"]


def test_key_reuse_with_different_payload_is_rejected(store):
    store.reserve("u1", "A", 1, "key-00001")
    out = store.reserve("u1", "A", 2, "key-00001")
    assert out.status == 422 and out.body["error"] == "idempotency_key_reused"
    assert store.product("A")["held"] == 1


def test_failed_outcomes_are_replayed_too(store):
    store.reserve("u1", "A", 2, "key-00001")
    assert store.reserve("u2", "A", 2, "key-00002").status == 409
    # stock comes back, but the *same intent* replays its original answer
    store.release("u1", "h_0001")
    out = store.reserve("u2", "A", 2, "key-00002")
    assert out.replayed and out.status == 409
    assert store.reserve("u2", "A", 2, "key-00003").status == 201


def test_hold_expiry_returns_stock_and_blocks_late_payment(store, clock):
    h = store.reserve("u1", "A", 2, "key-00001").body["id"]
    clock.t += 59.9
    assert avail(store) == 1
    clock.t += 0.1  # exactly at expires_at the hold is gone
    assert avail(store) == 3
    assert store.hold("u1", h).body["status"] == "expired"
    out = store.confirm("u1", h, "tok_ok")
    assert out.status == 410 and out.body["error"] == "hold_expired"
    assert store.product("A")["sold"] == 0


def test_expired_stock_can_be_resold_before_late_confirm(store, clock):
    h1 = store.reserve("u1", "A", 2, "key-00001").body["id"]
    clock.t += 61
    store.reserve("u2", "A", 2, "key-00002")
    store.reserve("u3", "A", 1, "key-00003")
    assert store.confirm("u1", h1, "tok_ok").status == 410
    assert store.check_invariants() == []


def test_confirm_is_idempotent_and_records_amount(store):
    h = store.reserve("u1", "A", 2, "key-00001").body["id"]
    first = store.confirm("u1", h, "tok_ok")
    second = store.confirm("u1", h, "tok_ok")
    assert first.status == 201 and second.status == 200 and second.replayed
    assert first.body == second.body and first.body["amount_cents"] == 2000
    p = store.product("A")
    assert (p["sold"], p["held"], p["available"]) == (2, 0, 1)


def test_declined_payment_keeps_hold_then_succeeds(store):
    h = store.reserve("u1", "A", 1, "key-00001").body["id"]
    out = store.confirm("u1", h, "tok_decline")
    assert out.status == 402 and store.hold("u1", h).body["status"] == "active"
    assert store.confirm("u1", h, "tok_visa").status == 201


def test_other_customers_cannot_see_or_touch_a_hold(store):
    h = store.reserve("u1", "A", 1, "key-00001").body["id"]
    assert store.hold("u2", h).status == 404
    assert store.confirm("u2", h, "tok_ok").status == 404
    assert store.release("u2", h).status == 404
    assert store.hold("u1", h).body["status"] == "active"


def test_release_is_idempotent_and_cannot_undo_a_purchase(store):
    h = store.reserve("u1", "A", 1, "key-00001").body["id"]
    assert store.release("u1", h).body["status"] == "released"
    assert store.release("u1", h).status == 200
    assert store.confirm("u1", h, "tok_ok").status == 409
    h2 = store.reserve("u1", "A", 1, "key-00002").body["id"]
    store.confirm("u1", h2, "tok_ok")
    assert store.release("u1", h2).status == 409


def test_invariant_checker_detects_tampering(store):
    store.reserve("u1", "A", 1, "key-00001")
    assert store.check_invariants() == []
    store.conn().execute("UPDATE products SET sold = 1 WHERE sku='A'")
    problems = store.check_invariants()
    assert any("orders sum 0 != sold counter 1" in p for p in problems)


def test_schema_refuses_sold_above_total(store):
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        store.conn().execute("UPDATE products SET sold = 4 WHERE sku='A'")


def test_memory_db_rejected(tmp_path):
    with pytest.raises(ValueError):
        Store(":memory:")


def test_mock_charge():
    assert mock_charge("tok_ok", 1)[0] and not mock_charge("tok_decline", 1)[0]
    assert mock_charge("card", 1) == (False, "invalid_token")
