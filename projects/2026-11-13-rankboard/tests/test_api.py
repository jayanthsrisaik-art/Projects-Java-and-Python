import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rankboard import bench, demo
from rankboard.api import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "api.db")))


@pytest.fixture
def board(client):
    snap = client.post("/api/boards", json={"name": "Sprint", "columns": ["Todo", "Done"]}).json()
    return snap["board"]["id"], [c["id"] for c in snap["columns"]]


def test_board_lifecycle(client, board):
    bid, (todo, done) = board
    r = client.post(f"/api/boards/{bid}/cards", json={"column_id": todo, "title": "A", "id": "a"})
    assert r.status_code == 201 and r.headers["etag"] == '"1"'
    assert r.json()["card"]["rank"] == "a0"
    r = client.post(f"/api/boards/{bid}/cards", json={"column_id": todo, "title": "A", "id": "a"})
    assert r.status_code == 200 and r.json()["placement"] == "replay"
    r = client.post(f"/api/boards/{bid}/cards", json={"column_id": todo, "title": "B", "id": "a"})
    assert r.status_code == 409
    r = client.post("/api/cards/a/move", json={"column_id": done})
    assert r.status_code == 200 and r.json()["card"]["column_id"] == done
    snap = client.get(f"/api/boards/{bid}").json()
    assert [c["id"] for c in snap["cards"]] == ["a"] and snap["seq"] > 0
    assert client.get("/api/boards").json()[0]["id"] == bid


def test_edit_needs_if_match(client, board):
    bid, (todo, _done) = board
    client.post(f"/api/boards/{bid}/cards", json={"column_id": todo, "title": "A", "id": "a"})
    assert client.patch("/api/cards/a", json={"title": "x"}).status_code == 428
    assert client.patch("/api/cards/a", json={"title": "x"},
                        headers={"If-Match": "abc"}).status_code == 400
    r = client.patch("/api/cards/a", json={"title": "x"}, headers={"If-Match": '"1"'})
    assert r.status_code == 200 and r.headers["etag"] == '"2"'
    r = client.patch("/api/cards/a", json={"title": "y"}, headers={"If-Match": 'W/"1"'})
    assert r.status_code == 412
    assert r.json()["current"]["title"] == "x" and r.headers["etag"] == '"2"'
    assert client.delete("/api/cards/a").status_code == 428
    assert client.delete("/api/cards/a", headers={"If-Match": '"1"'}).status_code == 412
    assert client.delete("/api/cards/a", headers={"If-Match": '"2"'}).status_code == 204
    assert client.get("/api/cards/a").status_code == 404


def test_validation_errors(client, board):
    bid, (todo, _done) = board
    assert client.post("/api/boards", json={"name": "", "columns": ["x"]}).status_code == 400
    assert client.post("/api/boards", json={"name": "x", "columns": []}).status_code == 400
    assert client.get("/api/boards/nope").status_code == 404
    r = client.post(f"/api/boards/{bid}/cards", json={"column_id": "nope", "title": "A"})
    assert r.status_code == 400
    r = client.post(f"/api/boards/{bid}/cards", json={"column_id": todo, "title": "A", "x": 1})
    assert r.status_code == 422  # unknown fields are rejected, not ignored
    assert client.get(f"/api/boards/{bid}/changes", params={"since": -1}).status_code == 422


def test_changes_and_gone(client, board):
    bid, (todo, _done) = board
    for i in range(5):
        client.post(f"/api/boards/{bid}/cards", json={"column_id": todo, "title": f"c{i}"})
    page = client.get(f"/api/boards/{bid}/changes", params={"since": 0, "limit": 2}).json()
    assert len(page["changes"]) == 2 and page["has_more"]
    rest = client.get(f"/api/boards/{bid}/changes", params={"since": page["cursor"]}).json()
    assert len(rest["changes"]) == 3 and not rest["has_more"]
    client.app.state.store.compact(bid, keep_last=1)
    r = client.get(f"/api/boards/{bid}/changes", params={"since": 0})
    assert r.status_code == 410 and r.json()["min_seq"] > 0


def test_ui_index_reports_unbuilt_or_serves(client):
    r = client.get("/")
    assert r.status_code in (200, 503)


def test_scenario_transcript_matches_sample():
    scenario = json.loads((SAMPLES / "scenario.json").read_text())
    expected = (SAMPLES / "scenario_expected.txt").read_text()
    assert "\n".join(demo.run(scenario)) + "\n" == expected


def test_bench_matches_sample_and_is_ordered():
    rows = bench.run(7)
    assert bench.to_markdown(rows, 7) == (SAMPLES / "bench_expected.md").read_text()
    by = {(r["workload"], r["strategy"]): r for r in rows}
    # the claims the README makes, asserted on the measured numbers
    assert by[("hotspot", "float64 midpoint")]["breaks_at"] is not None
    assert not by[("hotspot", "float64 midpoint")]["ordered"]
    assert by[("append", "rankboard keys")]["max_len"] <= 3
    assert by[("append", "fraction-only keys")]["max_len"] > 100
    assert by[("prepend", "dense integer index")]["rows_per_op"] > 400
    assert by[("hotspot", "rankboard keys + rebalance@48")]["max_len"] <= 48
    for r in rows:
        if r["strategy"] != "float64 midpoint":
            assert r["ordered"], r
