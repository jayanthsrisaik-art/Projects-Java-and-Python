"""Fire concurrent reservations at a *running* server and audit the result.

Usage (server started with uvicorn, see README):
    python -m stockhold.loadcheck --url http://127.0.0.1:8000 --sku KBD-65 --shoppers 200

Each shopper sends one POST /api/holds with its own Idempotency-Key from a
thread pool; 20% of shoppers also send an immediate duplicate with the same key
(a double-click / retry). Prints how many holds were granted vs. the stock that
was available, and the server's own invariant check.
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx


def run(url: str, sku: str, shoppers: int, threads: int, seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    tag = uuid.uuid4().hex[:6]
    jobs = []
    for i in range(shoppers):
        key = f"load-{tag}-{i:05d}"
        jobs.append((f"load-{tag}-{i}", key))
        if rng.random() < 0.2:
            jobs.append((f"load-{tag}-{i}", key))
    rng.shuffle(jobs)
    with httpx.Client(base_url=url, timeout=30) as c:
        before = c.get(f"/api/products/{sku}").json()

        def go(job: tuple[str, str]) -> tuple[str, int, bool, str | None]:
            cust, key = job
            r = c.post("/api/holds", json={"sku": sku, "qty": 1},
                       headers={"X-Customer-Id": cust, "Idempotency-Key": key})
            body = r.json()
            return cust, r.status_code, r.headers.get("Idempotent-Replayed") == "true", body.get("id")

        with ThreadPoolExecutor(threads) as ex:
            results = list(ex.map(go, jobs))
        after = c.get(f"/api/products/{sku}").json()
        inv = c.get("/api/invariants").json()
    new_holds = {hid for _, status, replayed, hid in results if status == 201 and not replayed}
    statuses: dict[str, int] = {}
    for _, s, _, _ in results:
        statuses[str(s)] = statuses.get(str(s), 0) + 1
    return {
        "requests": len(jobs), "available_before": before["available"],
        "holds_created": len(new_holds),
        "replays": sum(1 for _, s, rep, _ in results if rep),
        "status_counts": statuses, "available_after": after["available"],
        "oversold": max(0, len(new_holds) - before["available"]),
        "invariants_ok": inv["ok"],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="concurrent reservation check against a live server")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--sku", default="KBD-65")
    ap.add_argument("--shoppers", type=int, default=200)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.url, a.sku, a.shoppers, a.threads), indent=2))


if __name__ == "__main__":
    main()
