import json
import random
import sqlite3

from clickrank.ingest import ingest

TABLES = ["impressions", "shown", "clicks", "bot_users", "ingest_report"]


def dump(db) -> dict:
    conn = sqlite3.connect(db)
    try:
        return {t: conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall() for t in TABLES}
    finally:
        conn.close()


def test_sample_matches_expected_report_and_tables(samples, tmp_path):
    lines = (samples / "raw_events.jsonl").read_text().splitlines()
    report = ingest(lines, tmp_path / "s.db")
    assert report == json.loads((samples / "expected_ingest_report.json").read_text())
    expected = json.loads((samples / "expected_tables.json").read_text())
    got = dump(tmp_path / "s.db")
    for t, rows in expected.items():
        assert [list(r) for r in got[t]] == rows, t


def test_sample_specific_cleaning_decisions(samples, tmp_path):
    ingest((samples / "raw_events.jsonl").read_text().splitlines(), tmp_path / "s.db")
    conn = sqlite3.connect(tmp_path / "s.db")
    # query text normalised; swap intervention kept
    assert conn.execute("SELECT query, swap_k FROM impressions WHERE request_id='r2'").fetchone() == ("wireless headphones", 3)
    # double click collapsed to the earliest; wrong reported position corrected from the shown list
    assert conn.execute("SELECT position, product_id, ts FROM clicks WHERE request_id='r1' ORDER BY position").fetchall() == [
        (2, "P0002", 104.0), (4, "P0004", 109.0)]
    # click delivered before its impression still joins
    assert conn.execute("SELECT COUNT(*) FROM clicks WHERE request_id='r3'").fetchone() == (1,)
    # bot fully removed
    assert conn.execute("SELECT COUNT(*) FROM impressions WHERE user_id='bot1'").fetchone() == (0,)


def test_order_independent_and_idempotent(world, tmp_path):
    lines = world.lines[:20000]
    shuffled = lines[:]
    random.Random(1).shuffle(shuffled)
    ingest(lines, tmp_path / "a.db")
    ingest(shuffled, tmp_path / "b.db")
    ingest(shuffled, tmp_path / "b.db")      # re-run over an existing DB
    assert dump(tmp_path / "a.db") == dump(tmp_path / "b.db")


def test_conflicting_duplicate_event_ids_resolve_deterministically(tmp_path):
    base = {"type": "impression", "event_id": "e1", "request_id": "r1", "user_id": "u", "ts": 1.0,
            "query": "tent", "results": ["P1", "P2"], "intervention": None}
    other = {**base, "results": ["P2", "P1"]}
    a = ingest([json.dumps(base), json.dumps(other)], tmp_path / "a.db")
    ingest([json.dumps(other), json.dumps(base)], tmp_path / "b.db")
    assert a["conflicting_duplicates"] == 1
    assert dump(tmp_path / "a.db") == dump(tmp_path / "b.db")


def test_simulated_dirt_is_cleaned_and_bots_found(world):
    r = world.report
    for key in ("duplicate_events", "malformed_lines", "orphan_clicks", "repeat_clicks_collapsed",
                "click_positions_corrected"):
        assert r[key] > 0, key
    conn = world.conn()
    flagged = {u for (u,) in conn.execute("SELECT user_id FROM bot_users")}
    assert flagged == set(world.truth["bots"])
    # integrity: every click points at the product shown at that position of a loaded impression
    bad = conn.execute("""SELECT COUNT(*) FROM clicks c LEFT JOIN shown s
                          ON s.request_id = c.request_id AND s.position = c.position AND s.product_id = c.product_id
                          WHERE s.request_id IS NULL""").fetchone()[0]
    assert bad == 0
    assert conn.execute("SELECT COUNT(*) FROM (SELECT request_id, product_id FROM clicks "
                        "GROUP BY 1, 2 HAVING COUNT(*) > 1)").fetchone()[0] == 0
