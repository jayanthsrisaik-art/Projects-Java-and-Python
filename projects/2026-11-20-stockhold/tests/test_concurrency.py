"""Real threads, real SQLite locking."""
import time

import pytest

from stockhold.bench import run_abandonment, run_retries, run_rush


@pytest.mark.parametrize("delay", [0.0, 0.001])
def test_immediate_never_oversells_under_contention(tmp_path, delay):
    for seed in range(3):
        r = run_rush("immediate", str(tmp_path), total=20, customers=80, threads=16,
                     delay_s=delay, seed=seed)
        assert r["granted"] == 20 and r["oversold"] == 0
        assert r["errors"] == 0 and r["violations"] == 0


def test_check_then_act_baseline_does_oversell(tmp_path):
    # The race is timing dependent; with a 2 ms window it shows up reliably.
    worst = max(run_rush("check-then-act", str(tmp_path), total=20, customers=80,
                         threads=16, delay_s=0.002, seed=s)["oversold"] for s in range(3))
    assert worst > 0


def test_concurrent_confirm_double_click(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from stockhold.store import Store

    s = Store(str(tmp_path / "c.db"), between_check_and_write=lambda: time.sleep(0.001))
    s.upsert_product("A", "Thing", 500, total=1, per_customer_limit=1)
    h = s.reserve("u1", "A", 1, "key-00001").body["id"]
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda _: s.confirm("u1", h, "tok_ok"), range(8)))
    assert sorted(o.status for o in outs) == [200] * 7 + [201]
    assert len({o.body["id"] for o in outs}) == 1
    assert s.product("A")["sold"] == 1 and s.check_invariants() == []


def test_retries_with_keys_create_no_duplicates(tmp_path):
    keyed = run_retries(True, str(tmp_path), seed=1)
    unkeyed = run_retries(False, str(tmp_path), seed=1)
    assert keyed["duplicate_holds"] == 0 and keyed["served"] == 60
    assert unkeyed["duplicate_holds"] > 0 and unkeyed["served"] < keyed["served"]
    assert keyed["violations"] == unkeyed["violations"] == 0


def test_ttl_returns_abandoned_stock(tmp_path):
    none = run_abandonment(None, str(tmp_path), seed=2)
    ttl = run_abandonment(600.0, str(tmp_path), seed=2)
    assert ttl["sold"] > none["sold"]
    assert none["expired_before_pay"] == 0
