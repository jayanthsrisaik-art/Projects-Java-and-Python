"""Measured comparison of stockhold against baselines.

Three experiments, each isolating one failure mode of a flash-sale checkout:

A. rush       - N threads race for the last units. ``immediate`` (one BEGIN
                IMMEDIATE transaction) vs ``check-then-act`` (autocommit
                statements). Counts holds granted beyond stock. Real threads and
                real SQLite locking, so numbers vary run to run.
B. retries    - responses are lost after the server committed; clients retry.
                With Idempotency-Key vs without. Deterministic (seeded).
C. abandonment- shoppers reserve and walk away. No TTL vs 600 s vs 120 s TTL,
                with payers who take 5-180 s. Deterministic (seeded, simulated clock).

Usage:  python -m stockhold.bench [--seeds 5] [--out samples/bench_expected.md]
"""
from __future__ import annotations

import argparse
import heapq
import os
import random
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .naive import STRATEGIES
from .store import SequentialIds, Store

SKU = "DROP"


class SimClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _fresh(cls: type[Store], workdir: str, name: str, **kw: Any) -> Store:
    path = os.path.join(workdir, f"{name}.db")
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
    return cls(path, **kw)


# --------------------------------------------------------------------------- A
def run_rush(strategy: str, workdir: str, *, total: int = 50, customers: int = 200,
             threads: int = 32, delay_s: float = 0.0, seed: int = 0) -> dict[str, Any]:
    hook = (lambda: time.sleep(delay_s)) if delay_s > 0 else None
    store = _fresh(STRATEGIES[strategy], workdir, f"rush-{strategy}-{seed}",
                   between_check_and_write=hook)
    store.upsert_product(SKU, "Limited sneaker", 12000, total, per_customer_limit=customers)
    barrier = threading.Barrier(threads)
    order = list(range(customers))
    random.Random(seed).shuffle(order)
    chunks = [order[i::threads] for i in range(threads)]
    errors = 0
    granted = 0
    lock = threading.Lock()

    def worker(ids: list[int]) -> None:
        nonlocal errors, granted
        barrier.wait()
        for i in ids:
            try:
                out = store.reserve(f"c{i}", SKU, 1, idempotency_key=f"k{i}")
            except Exception:  # noqa: BLE001 - counted, not hidden
                with lock:
                    errors += 1
                continue
            if out.status == 201:
                with lock:
                    granted += 1

    t0 = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(worker, chunks))
    elapsed = time.perf_counter() - t0
    view = store.product(SKU)
    assert view is not None
    return {
        "strategy": strategy, "granted": granted, "oversold": max(0, granted - total),
        "errors": errors, "violations": len(store.check_invariants()),
        "held": view["held"], "seconds": elapsed,
    }


# --------------------------------------------------------------------------- B
def run_retries(use_key: bool, workdir: str, *, total: int = 60, customers: int = 80,
                loss: float = 0.3, max_attempts: int = 4, seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    clock = SimClock()
    store = _fresh(Store, workdir, f"retry-{use_key}-{seed}", clock=clock,
                   ids=SequentialIds(), hold_ttl=600)
    store.upsert_product(SKU, "Limited sneaker", 12000, total, per_customer_limit=2)
    served = refused = duplicate_holds = gave_up = 0
    for i in range(customers):
        cust = f"c{i}"
        key = f"intent-{i}" if use_key else None
        seen = None
        created = 0
        for _ in range(max_attempts):
            out = store.reserve(cust, SKU, 1, idempotency_key=key)
            if out.status == 201 and not out.replayed:
                created += 1
            if rng.random() < loss:  # response lost on the way back; client retries
                clock.t += 1
                continue
            seen = out
            break
        duplicate_holds += max(0, created - 1)
        clock.t += 1
        if seen is None:
            gave_up += 1
        elif seen.ok:
            if store.confirm(cust, seen.body["id"], "tok_ok").ok:
                served += 1
        else:
            refused += 1
    view = store.product(SKU)
    assert view is not None
    return {"idempotency_key": use_key, "served": served, "refused": refused,
            "gave_up": gave_up, "duplicate_holds": duplicate_holds,
            "stranded_units": view["held"], "sold": view["sold"],
            "violations": len(store.check_invariants())}


# --------------------------------------------------------------------------- C
def run_abandonment(ttl: float | None, workdir: str, *, total: int = 100,
                    customers: int = 250, window_s: float = 1800.0,
                    abandon: float = 0.4, pay_min: float = 5.0, pay_max: float = 180.0,
                    seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    clock = SimClock(0.0)
    store = _fresh(Store, workdir, f"abandon-{ttl}-{seed}", clock=clock,
                   ids=SequentialIds(), hold_ttl=ttl if ttl is not None else 1e12)
    store.upsert_product(SKU, "Limited sneaker", 12000, total, per_customer_limit=1)
    events: list[tuple[float, int, str, str]] = []
    for i in range(customers):
        heapq.heappush(events, (rng.uniform(0, window_s), i, "arrive", f"c{i}"))
    plans = {f"c{i}": (rng.random() < abandon, rng.uniform(pay_min, pay_max))
             for i in range(customers)}
    holds: dict[str, str] = {}
    refused = expired_before_pay = 0
    seq = customers
    while events:
        t, _, kind, cust = heapq.heappop(events)
        clock.t = t
        if kind == "arrive":
            out = store.reserve(cust, SKU, 1, idempotency_key=f"{cust}-1")
            if not out.ok:
                refused += 1
                continue
            holds[cust] = out.body["id"]
            walks_away, pay_after = plans[cust]
            if not walks_away:
                seq += 1
                heapq.heappush(events, (t + pay_after, seq, "pay", cust))
        else:
            out = store.confirm(cust, holds[cust], "tok_ok")
            if out.status == 410:
                expired_before_pay += 1
    clock.t = window_s + 1e13  # long after the sale: count what really sold
    view = store.product(SKU)
    assert view is not None
    return {"ttl": ttl, "sold": view["sold"], "refused": refused,
            "expired_before_pay": expired_before_pay,
            "violations": len(store.check_invariants())}


# --------------------------------------------------------------------------- report
def _mean(xs: list[float]) -> str:
    return f"{statistics.mean(xs):.1f}"


def report(seeds: int = 5, workdir: str | None = None) -> str:
    own = workdir is None
    tmp = tempfile.TemporaryDirectory() if own else None
    wd = tmp.name if tmp else workdir  # type: ignore[assignment]
    lines: list[str] = ["# stockhold benchmark", ""]
    try:
        lines += ["## A. Rush: 200 shoppers, 32 threads, 50 units "
                  f"(mean over {seeds} runs; timing-dependent)", "",
                  "| strategy | extra check-to-write delay | holds granted | oversold (max) "
                  "| errors | invariant violations | wall s |",
                  "|---|---|---|---|---|---|---|"]
        for delay in (0.0, 0.001):
            for strat in STRATEGIES:
                rs = [run_rush(strat, wd, delay_s=delay, seed=s) for s in range(seeds)]
                lines.append(
                    f"| {strat} | {delay * 1000:.0f} ms | {_mean([r['granted'] for r in rs])} "
                    f"| {_mean([r['oversold'] for r in rs])} ({max(r['oversold'] for r in rs)}) "
                    f"| {sum(r['errors'] for r in rs)} | {sum(r['violations'] for r in rs)} "
                    f"| {_mean([r['seconds'] for r in rs])} |")
        lines += ["", f"## B. Lost responses: 80 shoppers, 60 units, limit 2, 30% of responses "
                  f"lost, up to 4 attempts (mean over {seeds} seeds)", "",
                  "| client | served | refused | gave up | duplicate holds | stranded units "
                  "| violations |", "|---|---|---|---|---|---|---|"]
        for use_key in (True, False):
            rs = [run_retries(use_key, wd, seed=s) for s in range(seeds)]
            name = "retry with Idempotency-Key" if use_key else "retry without key"
            lines.append(
                f"| {name} | {_mean([r['served'] for r in rs])} | {_mean([r['refused'] for r in rs])} "
                f"| {_mean([r['gave_up'] for r in rs])} | {_mean([r['duplicate_holds'] for r in rs])} "
                f"| {_mean([r['stranded_units'] for r in rs])} | {sum(r['violations'] for r in rs)} |")
        lines += ["", f"## C. Abandonment: 250 shoppers over 30 min, 100 units, 40% abandon, "
                  f"payers take 5-180 s (mean over {seeds} seeds)", "",
                  "| hold TTL | units sold | shoppers refused | payers whose hold expired "
                  "| violations |", "|---|---|---|---|---|"]
        for ttl in (None, 600.0, 120.0):
            rs = [run_abandonment(ttl, wd, seed=s) for s in range(seeds)]
            name = "none" if ttl is None else f"{ttl:.0f} s"
            lines.append(
                f"| {name} | {_mean([r['sold'] for r in rs])} | {_mean([r['refused'] for r in rs])} "
                f"| {_mean([r['expired_before_pay'] for r in rs])} "
                f"| {sum(r['violations'] for r in rs)} |")
    finally:
        if tmp:
            tmp.cleanup()
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", help="also write the markdown report here")
    a = ap.parse_args(argv)
    text = report(a.seeds)
    print(text, end="")
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
