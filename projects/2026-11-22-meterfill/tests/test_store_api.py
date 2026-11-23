import pytest
from fastapi.testclient import TestClient

from meterfill.api import create_app
from meterfill.store import Store, ValidationError

READINGS = [
    ("M1", "2026-01-05T00:10:00Z", 100.0),
    ("M1", "2026-01-05T00:40:00Z", 100.6),
    ("M1", "2026-01-05T01:10:00Z", 101.0),
    ("M1", "2026-01-05T07:10:00Z", 109.0),  # 6 h gap carrying 8 kWh
    ("M1", "2026-01-05T07:40:00Z", 109.5),
    ("M1", "2026-01-05T08:10:00Z", 110.0),
]
WEATHER = [(f"2026-01-05T{h:02d}:00:00Z", 2.0 + h * 0.1) for h in range(10)]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    s.ingest_weather(WEATHER)
    s.ingest_readings(READINGS)
    yield s
    s.close()


def test_ingest_is_idempotent_and_records_conflicts(store):
    again = store.ingest_readings(READINGS)
    assert again == {"received": 6, "inserted": 0, "duplicates": 6, "conflicts": 0}
    c = store.ingest_readings([("M1", "2026-01-05T00:40:00Z", 555.0)])
    assert c["conflicts"] == 1
    assert dict(store.rows("M1"))[1767573600] == 100.6  # first value kept (00:40Z)
    assert store.conn.execute("SELECT COUNT(*) FROM ingest_conflicts").fetchone()[0] == 1


def test_ingest_rejects_bad_rows_atomically(store):
    with pytest.raises(ValidationError):
        store.ingest_readings([("M2", "2026-01-05T00:00:00Z", 1.0), ("M2", "2026-01-05T00:30:00", 2.0)])
    assert store.rows("M2") == []  # first row rolled back too
    with pytest.raises(ValidationError):
        store.ingest_readings([("M2", "2026-01-05T00:00:00Z", 100000.0)])


def test_rebuild_conserves_energy_and_marks_quality(store):
    summary = store.rebuild("uniform")
    assert summary["meters"] == 1 and summary["gap"] == 1
    u = store.usage("M1", None, None, "hour")
    # 10 kWh metered between 00:10 and 08:10, plus 2 h with no reading (00:00-00:10, 08:10-10:00)
    # estimated at the meter's mean rate of 10 kWh / 8 h
    assert abs(u["total_kwh"] - (10.0 + 2 * 1.25)) < 1e-3
    q = {r["start"][11:13]: r["quality"] for r in u["rows"]}
    assert q["00"] == "estimated" and q["03"] == "allocated" and q["09"] == "estimated"
    d = store.usage("M1", None, None, "day")
    assert abs(d["rows"][0]["kwh"] - u["total_kwh"]) < 1e-3
    assert sum(d["rows"][0]["hours"].values()) == 10


def test_stale_flag_follows_ingest(store):
    store.rebuild("profile")
    assert store.build_state()["stale"] is False
    store.ingest_readings([("M1", "2026-01-05T08:40:00Z", 110.4)])
    assert store.build_state()["stale"] is True


def test_model_method_without_model_is_rejected(store):
    with pytest.raises(ValidationError):
        store.rebuild("model", None)


def test_api_end_to_end(tmp_path):
    client = TestClient(create_app(tmp_path / "api.db"))
    assert client.get("/health").json()["model_loaded"] is False
    r = client.post("/weather", json={"hours": [{"hour_ts": t, "temp_c": c} for t, c in WEATHER]})
    assert r.status_code == 200 and r.json()["hours"] == 10
    body = {"readings": [{"meter_id": m, "ts": t, "register_kwh": v} for m, t, v in READINGS]}
    assert client.post("/readings", json=body).json()["inserted"] == 6
    assert client.post("/readings", json=body).json()["duplicates"] == 6
    assert client.get("/meters/M1/usage").status_code == 404  # not built yet
    r = client.post("/rebuild")
    assert r.status_code == 200 and r.json()["method"] == "profile"  # no model -> profile
    assert client.post("/rebuild?method=model").status_code == 422
    u = client.get("/meters/M1/usage", params={"start": "2026-01-05T02:00:00Z", "end": "2026-01-05T07:00:00Z"}).json()
    assert [r["quality"] for r in u["rows"]] == ["allocated"] * 5
    assert u["build"]["stale"] is False
    iss = client.get("/meters/M1/issues").json()["issues"]
    assert [i["kind"] for i in iss] == ["gap"] and iss[0]["value"] == 8.0
    assert client.get("/meters/NOPE/issues").status_code == 404
    assert client.get("/meters/M1/usage?granularity=week").status_code == 422
    bad = {"readings": [{"meter_id": "M1", "ts": "2026-01-05T00:00:00Z", "register_kwh": -1}]}
    assert client.post("/readings", json=bad).status_code == 422
    m = client.get("/meters").json()["meters"][0]
    assert m["meter_id"] == "M1" and m["readings"] == 6 and sum(m["hours"].values()) == 10
