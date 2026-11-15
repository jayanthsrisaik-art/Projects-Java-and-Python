import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from clickrank.features import FEATURES
from clickrank.ingest import ingest_file
from clickrank.service import create_app
from clickrank.train import save_model


@pytest.fixture()
def artifacts(world, tmp_path):
    world.catalog.save(tmp_path)
    w = np.zeros(len(FEATURES))
    w[FEATURES.index("bm25")] = 1.0
    w[FEATURES.index("category_match")] = 2.0
    save_model(tmp_path / "model.json", w, "test")
    return tmp_path


def test_search_click_and_log_roundtrip(artifacts, tmp_path):
    events = tmp_path / "events.jsonl"
    client = TestClient(create_app(artifacts, events))
    assert client.get("/health").json()["status"] == "ok"
    q = next(q for q in json.loads((artifacts / "truth.json").read_text())["queries"] if " " in q["text"])
    r = client.get("/search", params={"q": q["text"].upper(), "k": 5, "user_id": "alice"}).json()
    assert r["query"] == q["text"] and 1 <= len(r["results"]) <= 5
    assert [x["position"] for x in r["results"]] == list(range(1, len(r["results"]) + 1))
    third = r["results"][min(2, len(r["results"]) - 1)]
    c = client.post("/click", json={"request_id": r["request_id"], "product_id": third["product_id"]})
    assert c.status_code == 200 and c.json()["position"] == third["position"]

    assert client.post("/click", json={"request_id": "nope", "product_id": "P0001"}).status_code == 404
    shown = {x["product_id"] for x in r["results"]}
    missing = next(f"P{i:04d}" for i in range(1, 999) if f"P{i:04d}" not in shown)
    assert client.post("/click", json={"request_id": r["request_id"], "product_id": missing}).status_code == 422

    report = ingest_file(events, tmp_path / "svc.db")
    assert report["impressions_loaded"] == 1 and report["clicks_loaded"] == 1
    assert report.get("click_positions_corrected", 0) == 0


def test_validation_and_unknown_product(artifacts, tmp_path):
    client = TestClient(create_app(artifacts, tmp_path / "e.jsonl"))
    assert client.get("/search", params={"q": "!!!"}).status_code == 422
    assert client.get("/search", params={"q": "tent", "k": 0}).status_code == 422
    assert client.get("/search", params={"q": "zzzunknownzzz"}).json()["results"] == []
    assert client.get("/products/P9999").status_code == 404
    assert client.get("/products/P0001").json()["id"] == "P0001"
    assert client.get("/model").json()["weights"]["category_match"] == 2.0


def test_exploration_swaps_rank_one_and_logs_it(artifacts, tmp_path):
    events = tmp_path / "e.jsonl"
    plain = TestClient(create_app(artifacts, tmp_path / "p.jsonl"))
    explore = TestClient(create_app(artifacts, events, explore_rate=1.0, seed=3))
    base = [x["product_id"] for x in plain.get("/search", params={"q": "waterproof tent"}).json()["results"]]
    got = [x["product_id"] for x in explore.get("/search", params={"q": "waterproof tent"}).json()["results"]]
    logged = json.loads(events.read_text().splitlines()[0])
    k = logged["intervention"]["k"]
    assert logged["intervention"]["kind"] == "swap" and 2 <= k <= len(base)
    expected = base[:]
    expected[0], expected[k - 1] = expected[k - 1], expected[0]
    assert got == expected == logged["results"]
