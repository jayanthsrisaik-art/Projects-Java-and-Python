"""Raw JSONL search events -> clean, deduplicated SQLite tables.

The load is a deterministic full rebuild: the same set of lines in any order produces the same
tables (every "keep one of several" choice is by (ts, event_id), never by file position).
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .text import normalize_query

SCHEMA = """
DROP TABLE IF EXISTS impressions;
DROP TABLE IF EXISTS shown;
DROP TABLE IF EXISTS clicks;
DROP TABLE IF EXISTS bot_users;
DROP TABLE IF EXISTS ingest_report;
CREATE TABLE impressions(
    request_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT, ts REAL NOT NULL,
    query TEXT NOT NULL, policy TEXT, swap_k INTEGER, n_shown INTEGER NOT NULL);
CREATE TABLE shown(
    request_id TEXT NOT NULL REFERENCES impressions(request_id), position INTEGER NOT NULL,
    product_id TEXT NOT NULL, PRIMARY KEY (request_id, position));
CREATE TABLE clicks(
    request_id TEXT NOT NULL REFERENCES impressions(request_id), position INTEGER NOT NULL,
    product_id TEXT NOT NULL, ts REAL NOT NULL, PRIMARY KEY (request_id, product_id));
CREATE TABLE bot_users(user_id TEXT PRIMARY KEY, impressions INTEGER, clicks INTEGER, reason TEXT);
CREATE TABLE ingest_report(metric TEXT PRIMARY KEY, value INTEGER);
CREATE INDEX idx_impressions_query ON impressions(query);
"""


def _valid_impression(e: dict) -> bool:
    res = e.get("results")
    return (isinstance(e.get("request_id"), str) and isinstance(e.get("user_id"), str)
            and isinstance(e.get("ts"), (int, float)) and isinstance(e.get("query"), str)
            and normalize_query(e["query"]) != ""
            and isinstance(res, list) and 0 < len(res) and all(isinstance(r, str) for r in res)
            and len(set(res)) == len(res))


def _valid_click(e: dict) -> bool:
    return (isinstance(e.get("request_id"), str) and isinstance(e.get("product_id"), str)
            and isinstance(e.get("ts"), (int, float)))


def _swap_k(e: dict) -> int | None:
    iv = e.get("intervention")
    if isinstance(iv, dict) and iv.get("kind") == "swap" and isinstance(iv.get("k"), int):
        return iv["k"]
    return None


def ingest(lines: Iterable[str], db_path: str | Path, *, bot_min_impressions: int = 5,
           bot_clicks_per_impression: float = 4.0, bot_fast_gap_s: float = 0.5) -> dict[str, int]:
    rep: Counter = Counter()
    by_event: dict[str, dict] = {}
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        rep["lines"] += 1
        try:
            e = json.loads(raw)
        except json.JSONDecodeError:
            rep["malformed_lines"] += 1
            continue
        if not isinstance(e, dict) or not isinstance(e.get("event_id"), str):
            rep["invalid_events"] += 1
            continue
        prev = by_event.get(e["event_id"])
        if prev is not None:
            rep["duplicate_events"] += 1
            if prev != e:      # same id, different payload: keep a deterministic winner
                rep["conflicting_duplicates"] += 1
                if json.dumps(e, sort_keys=True) < json.dumps(prev, sort_keys=True):
                    by_event[e["event_id"]] = e
            continue
        by_event[e["event_id"]] = e

    impressions: dict[str, dict] = {}
    raw_clicks: list[dict] = []
    for eid in sorted(by_event):
        e = by_event[eid]
        kind = e.get("type")
        if kind == "impression":
            if not _valid_impression(e):
                rep["invalid_events"] += 1
                continue
            rid = e["request_id"]
            prev = impressions.get(rid)
            if prev is not None:
                rep["duplicate_request_ids"] += 1
                if (prev["ts"], prev["event_id"]) <= (e["ts"], e["event_id"]):
                    continue
            impressions[rid] = e
        elif kind == "click":
            if not _valid_click(e):
                rep["invalid_events"] += 1
                continue
            raw_clicks.append(e)
        else:
            rep["invalid_events"] += 1

    clicks: dict[tuple[str, str], dict] = {}
    for c in sorted(raw_clicks, key=lambda c: (c["ts"], c["event_id"])):
        imp = impressions.get(c["request_id"])
        if imp is None:
            rep["orphan_clicks"] += 1
            continue
        if c["product_id"] not in imp["results"]:
            rep["clicks_not_in_results"] += 1
            continue
        key = (c["request_id"], c["product_id"])
        if key in clicks:
            rep["repeat_clicks_collapsed"] += 1
            continue
        actual = imp["results"].index(c["product_id"]) + 1
        if c.get("position") != actual:
            rep["click_positions_corrected"] += 1
        clicks[key] = {**c, "position": actual}

    # ---- bot filter -----------------------------------------------------------------
    n_imp: Counter = Counter(imp["user_id"] for imp in impressions.values())
    n_clk: Counter = Counter()
    gaps: dict[str, list[float]] = defaultdict(list)
    per_req: dict[str, list[float]] = defaultdict(list)
    for (rid, _), c in clicks.items():
        n_clk[impressions[rid]["user_id"]] += 1
        per_req[rid].append(c["ts"])
    for rid, tss in per_req.items():
        tss.sort()
        gaps[impressions[rid]["user_id"]].extend(b - a for a, b in zip(tss, tss[1:]))
    bots: dict[str, str] = {}
    for user, n in n_imp.items():
        if n >= bot_min_impressions and n_clk[user] / n >= bot_clicks_per_impression:
            bots[user] = "clicks_per_impression"
        elif len(gaps[user]) >= 10 and sum(g < bot_fast_gap_s for g in gaps[user]) / len(gaps[user]) >= 0.8:
            bots[user] = "sub_second_click_bursts"
    bot_rids = {rid for rid, imp in impressions.items() if imp["user_id"] in bots}
    rep["bot_users"] = len(bots)
    rep["bot_impressions_removed"] = len(bot_rids)
    rep["bot_clicks_removed"] = sum(1 for (rid, _) in clicks if rid in bot_rids)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO impressions VALUES (?,?,?,?,?,?,?,?)",
            [(rid, i["user_id"], i.get("session_id"), float(i["ts"]), normalize_query(i["query"]),
              i.get("policy"), _swap_k(i), len(i["results"]))
             for rid, i in sorted(impressions.items()) if rid not in bot_rids])
        conn.executemany(
            "INSERT INTO shown VALUES (?,?,?)",
            [(rid, pos, pid) for rid, i in sorted(impressions.items()) if rid not in bot_rids
             for pos, pid in enumerate(i["results"], start=1)])
        conn.executemany(
            "INSERT INTO clicks VALUES (?,?,?,?)",
            [(rid, c["position"], pid, float(c["ts"])) for (rid, pid), c in sorted(clicks.items())
             if rid not in bot_rids])
        conn.executemany("INSERT INTO bot_users VALUES (?,?,?,?)",
                         [(u, n_imp[u], n_clk[u], r) for u, r in sorted(bots.items())])
        rep["impressions_loaded"] = len(impressions) - len(bot_rids)
        rep["clicks_loaded"] = len(clicks) - rep["bot_clicks_removed"]
        conn.executemany("INSERT INTO ingest_report VALUES (?,?)", sorted(rep.items()))
        conn.commit()
    finally:
        conn.close()
    return dict(rep)


def ingest_file(path: str | Path, db_path: str | Path, **kw) -> dict[str, int]:
    with open(path, encoding="utf-8") as fh:
        return ingest(fh, db_path, **kw)
