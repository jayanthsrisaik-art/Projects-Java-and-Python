"""FastAPI search service that serves a trained ranker and logs events in the ingest schema.

The service closes the loop: its impression/click log is exactly what ``clickrank ingest``
consumes, and an optional exploration rate applies the same rank-1 <-> rank-k swaps that the
propensity estimator needs.
"""
from __future__ import annotations

import json
import random
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .catalog import load_products
from .features import SearchIndex, rank
from .text import normalize_query
from .train import load_model


class ClickIn(BaseModel):
    request_id: str
    product_id: str
    user_id: str | None = None


def create_app(artifacts: Path, events_path: Path, *, explore_rate: float = 0.0, seed: int = 0,
               max_tracked_requests: int = 50000) -> FastAPI:
    products, vocab = load_products(artifacts)
    index = SearchIndex(products, vocab)
    weights, model_meta = load_model(artifacts / "model.json")
    by_id = {p.id: p for p in products}
    rng = random.Random(seed)
    lock = threading.Lock()
    recent: OrderedDict[str, dict] = OrderedDict()
    events_path.parent.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="clickrank", version="1.0")

    def log(event: dict) -> None:
        with lock, events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, separators=(",", ":")) + "\n")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "products": len(products), "model": model_meta["method"]}

    @app.get("/model")
    def model() -> dict:
        return {"method": model_meta["method"], "weights": dict(zip(model_meta["features"], model_meta["weights"])),
                "explore_rate": explore_rate}

    @app.get("/search")
    def search(q: str = Query(..., min_length=1, max_length=200), k: int = Query(10, ge=1, le=20),
               user_id: str = "anonymous") -> dict:
        query = normalize_query(q)
        if not query:
            raise HTTPException(422, "query has no searchable tokens")
        cs = index.candidates(query)
        ordered = rank(cs, weights)[:k]
        intervention = None
        with lock:
            explore = len(ordered) >= 2 and rng.random() < explore_rate
            swap_k = rng.randint(2, len(ordered)) if explore else None
        if swap_k is not None:
            ordered[0], ordered[swap_k - 1] = ordered[swap_k - 1], ordered[0]
            intervention = {"kind": "swap", "k": swap_k}
        rid = uuid.uuid4().hex
        ts = time.time()
        if ordered:
            log({"type": "impression", "event_id": uuid.uuid4().hex, "request_id": rid,
                 "session_id": user_id, "user_id": user_id, "ts": round(ts, 3), "query": q,
                 "results": ordered, "intervention": intervention, "policy": model_meta["method"]})
            with lock:
                recent[rid] = {"results": ordered, "user_id": user_id}
                while len(recent) > max_tracked_requests:
                    recent.popitem(last=False)
        return {"request_id": rid, "query": query,
                "results": [{"position": i, "product_id": pid, "title": by_id[pid].title,
                             "brand": by_id[pid].brand, "price": by_id[pid].price, "rating": by_id[pid].rating}
                            for i, pid in enumerate(ordered, start=1)]}

    @app.post("/click")
    def click(body: ClickIn) -> dict:
        with lock:
            req = recent.get(body.request_id)
        if req is None:
            raise HTTPException(404, "unknown or expired request_id")
        if body.product_id not in req["results"]:
            raise HTTPException(422, "product was not shown for this request")
        position = req["results"].index(body.product_id) + 1
        user = body.user_id or req["user_id"]
        log({"type": "click", "event_id": uuid.uuid4().hex, "request_id": body.request_id,
             "session_id": user, "user_id": user, "ts": round(time.time(), 3),
             "product_id": body.product_id, "position": position})
        return {"logged": True, "position": position}

    @app.get("/products/{product_id}")
    def product(product_id: str) -> dict:
        p = by_id.get(product_id)
        if p is None:
            raise HTTPException(404, "no such product")
        return p.__dict__

    return app
