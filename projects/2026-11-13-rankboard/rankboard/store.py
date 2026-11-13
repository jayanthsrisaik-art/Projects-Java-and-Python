"""SQLite persistence: boards, columns, ranked cards and a per-board change feed.

Every mutation runs in its own ``BEGIN IMMEDIATE`` transaction on a fresh
connection, so the read of the neighbours' ranks, the rank computation, the card
write and the change-feed row are one atomic, serialised step. Two people dropping
a card into the same gap at the same moment therefore get distinct, correctly
ordered ranks; the UNIQUE(column_id, rank) index is a second line of defence.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from . import fracindex

SCHEMA = """
CREATE TABLE IF NOT EXISTS boards (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS columns (
    id       TEXT PRIMARY KEY,
    board_id TEXT NOT NULL REFERENCES boards(id),
    name     TEXT NOT NULL,
    position INTEGER NOT NULL,
    UNIQUE (board_id, position)
);
CREATE TABLE IF NOT EXISTS cards (
    id          TEXT PRIMARY KEY,
    board_id    TEXT NOT NULL REFERENCES boards(id),
    column_id   TEXT NOT NULL REFERENCES columns(id),
    title       TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    rank        TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    UNIQUE (column_id, rank)
);
CREATE TABLE IF NOT EXISTS changes (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    board_id TEXT NOT NULL,
    card_id  TEXT NOT NULL,
    op       TEXT NOT NULL CHECK (op IN ('upsert', 'delete')),
    card     TEXT NOT NULL,
    reason   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS changes_board_seq ON changes (board_id, seq);
CREATE INDEX IF NOT EXISTS changes_card ON changes (card_id);
CREATE TABLE IF NOT EXISTS feed_floor (
    board_id TEXT PRIMARY KEY,
    min_seq  INTEGER NOT NULL      -- changes with seq <= min_seq were compacted away
);
"""

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DEFAULT_MAX_KEY_LEN = 48


class NotFound(Exception):
    pass


class Invalid(Exception):
    pass


class Conflict(Exception):
    """Same client id reused for a different card."""


class VersionMismatch(Exception):
    def __init__(self, current: dict):
        super().__init__("version mismatch")
        self.current = current


class FeedGone(Exception):
    def __init__(self, min_seq: int):
        super().__init__("changes compacted")
        self.min_seq = min_seq


@dataclass
class Placement:
    rank: str
    resolved: str  # "after", "before", "end" or "fallback_end"


def resolve_drop(column: list[tuple[str, str]], after_id: Optional[str],
                 before_id: Optional[str], moving: Optional[str] = None) -> Placement:
    """Turn a drop intent into a rank, against the *current* column contents.

    ``column`` is the column's (card_id, rank) pairs sorted by rank. The client
    says where it dropped the card as *it* saw the column: "after card X" and/or
    "before card Y". Others may have changed the column since, so the intent is
    resolved in this order (the moving card itself is ignored):

    1. X still in this column -> immediately after X (before X's current successor)
    2. else Y still in this column -> immediately before Y
    3. neither anchor given      -> end of the column
    4. anchors given but gone    -> end of the column ("fallback_end")

    The TypeScript client runs the same function (web/src/board.ts) to predict
    the server's answer; ``samples/key_vectors.json`` holds shared test cases.
    """
    if moving is not None and moving in (after_id, before_id):
        raise Invalid("a card cannot be placed relative to itself")
    cards = [(cid, rank) for cid, rank in column if cid != moving]
    pos = {cid: i for i, (cid, _r) in enumerate(cards)}
    if after_id is not None and after_id in pos:
        i = pos[after_id]
        nxt = cards[i + 1][1] if i + 1 < len(cards) else None
        return Placement(fracindex.key_between(cards[i][1], nxt), "after")
    if before_id is not None and before_id in pos:
        i = pos[before_id]
        prv = cards[i - 1][1] if i > 0 else None
        return Placement(fracindex.key_between(prv, cards[i][1]), "before")
    last = cards[-1][1] if cards else None
    resolved = "end" if after_id is None and before_id is None else "fallback_end"
    return Placement(fracindex.key_between(last, None), resolved)


def _card_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "board_id": row["board_id"],
        "column_id": row["column_id"],
        "title": row["title"],
        "description": row["description"],
        "rank": row["rank"],
        "version": row["version"],
    }


class Store:
    def __init__(self, path: str, max_key_len: int = DEFAULT_MAX_KEY_LEN):
        self.path = path
        self.max_key_len = max_key_len
        self._memory_anchor: Optional[sqlite3.Connection] = None
        if path == ":memory:":
            # a shared in-memory database must stay open for its whole lifetime
            self.path = f"file:rankboard-{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._memory_anchor = self._connect()
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    # ------------------------------------------------------------------ infra
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path, isolation_level=None, timeout=10.0, check_same_thread=False,
            uri=self.path.startswith("file:"),
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        if not self.path.startswith("file:"):
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _check_id(value: str, what: str) -> None:
        if not ID_RE.match(value):
            raise Invalid(f"{what} must match {ID_RE.pattern}")

    def _record(self, conn, board_id: str, op: str, card: dict, reason: str) -> int:
        cur = conn.execute(
            "INSERT INTO changes (board_id, card_id, op, card, reason) VALUES (?, ?, ?, ?, ?)",
            (board_id, card["id"], op, json.dumps(card, sort_keys=True), reason),
        )
        return int(cur.lastrowid)

    @staticmethod
    def _board_seq(conn, board_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS s FROM changes WHERE board_id = ?", (board_id,)
        ).fetchone()
        floor = conn.execute(
            "SELECT min_seq FROM feed_floor WHERE board_id = ?", (board_id,)
        ).fetchone()
        return max(int(row["s"]), int(floor["min_seq"]) if floor else 0)

    # ----------------------------------------------------------------- boards
    def create_board(self, name: str, columns: list[str]) -> dict:
        name = name.strip()
        if not name:
            raise Invalid("board name is required")
        if not columns or len(columns) > 20:
            raise Invalid("a board needs 1..20 columns")
        if any(not c.strip() for c in columns):
            raise Invalid("column names must be non-empty")
        board_id = self._new_id()
        with self._tx() as conn:
            conn.execute("INSERT INTO boards (id, name) VALUES (?, ?)", (board_id, name))
            for pos, col in enumerate(columns):
                conn.execute(
                    "INSERT INTO columns (id, board_id, name, position) VALUES (?, ?, ?, ?)",
                    (self._new_id(), board_id, col.strip(), pos),
                )
        return self.snapshot(board_id)

    def snapshot(self, board_id: str) -> dict:
        """Board, columns and ordered cards plus the feed cursor, read atomically."""
        with self._tx() as conn:
            board = conn.execute("SELECT * FROM boards WHERE id = ?", (board_id,)).fetchone()
            if board is None:
                raise NotFound(f"board {board_id} not found")
            cols = conn.execute(
                "SELECT id, name, position FROM columns WHERE board_id = ? ORDER BY position",
                (board_id,),
            ).fetchall()
            cards = conn.execute(
                "SELECT c.* FROM cards c JOIN columns k ON k.id = c.column_id "
                "WHERE c.board_id = ? ORDER BY k.position, c.rank",
                (board_id,),
            ).fetchall()
            seq = self._board_seq(conn, board_id)
        return {
            "board": {"id": board["id"], "name": board["name"]},
            "columns": [dict(c) for c in cols],
            "cards": [_card_row(c) for c in cards],
            "seq": seq,
        }

    def list_boards(self) -> list[dict]:
        with self._tx() as conn:
            rows = conn.execute("SELECT id, name FROM boards ORDER BY rowid").fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- placement
    def _column(self, conn, board_id: str, column_id: str) -> sqlite3.Row:
        col = conn.execute(
            "SELECT * FROM columns WHERE id = ? AND board_id = ?", (column_id, board_id)
        ).fetchone()
        if col is None:
            raise Invalid(f"column {column_id} is not on board {board_id}")
        return col

    def _place(self, conn, column_id: str, after_id: Optional[str], before_id: Optional[str],
               moving: Optional[str]) -> Placement:
        rows = conn.execute(
            "SELECT id, rank FROM cards WHERE column_id = ? ORDER BY rank", (column_id,)
        ).fetchall()
        return resolve_drop([(r["id"], r["rank"]) for r in rows], after_id, before_id, moving)

    def _maybe_rebalance(self, conn, board_id: str, column_id: str, new_rank: str) -> int:
        """Rewrite a column's ranks as short evenly-spaced keys if one grew too long.

        Ranks are a representation, not content, so a rebalance does not bump
        versions (it must not make a concurrent title edit fail with 412). It
        does emit one change per card so every client's copy is re-keyed.
        """
        if len(new_rank) <= self.max_key_len:
            return 0
        rows = conn.execute(
            "SELECT * FROM cards WHERE column_id = ? ORDER BY rank", (column_id,)
        ).fetchall()
        fresh = fracindex.keys_sequence(len(rows))
        # park every rank on a non-key value first so UNIQUE(column_id, rank) never trips
        conn.execute("UPDATE cards SET rank = '~' || id WHERE column_id = ?", (column_id,))
        for row, key in zip(rows, fresh):
            conn.execute("UPDATE cards SET rank = ? WHERE id = ?", (key, row["id"]))
            card = _card_row(row)
            card["rank"] = key
            self._record(conn, board_id, "upsert", card, "rebalance")
        return len(rows)

    # ------------------------------------------------------------------ cards
    def _get_card(self, conn, card_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
        if row is None:
            raise NotFound(f"card {card_id} not found")
        return row

    def get_card(self, card_id: str) -> dict:
        with self._tx() as conn:
            return _card_row(self._get_card(conn, card_id))

    def create_card(self, board_id: str, column_id: str, title: str, description: str = "",
                    after_id: Optional[str] = None, before_id: Optional[str] = None,
                    card_id: Optional[str] = None) -> tuple[dict, dict]:
        """Create a card. A client-chosen ``card_id`` makes retries idempotent.

        Returns ``(card, info)`` where info has ``created`` (False for a replay),
        ``placement`` and ``rebalanced``.
        """
        title = title.strip()
        if not title:
            raise Invalid("title is required")
        if card_id is not None:
            self._check_id(card_id, "card id")
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM boards WHERE id = ?", (board_id,)).fetchone() is None:
                raise NotFound(f"board {board_id} not found")
            self._column(conn, board_id, column_id)
            if card_id is not None:
                existing = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
                if existing is not None:
                    same = (existing["board_id"] == board_id and existing["title"] == title
                            and existing["description"] == description)
                    if not same:
                        raise Conflict(f"card id {card_id} already used for a different card")
                    return _card_row(existing), {"created": False, "placement": "replay",
                                                 "rebalanced": 0}
                if conn.execute("SELECT 1 FROM changes WHERE card_id = ? AND op = 'delete'",
                                (card_id,)).fetchone():
                    raise Conflict(f"card id {card_id} belonged to a deleted card")
            cid = card_id or self._new_id()
            place = self._place(conn, column_id, after_id, before_id, None)
            conn.execute(
                "INSERT INTO cards (id, board_id, column_id, title, description, rank) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cid, board_id, column_id, title, description, place.rank),
            )
            card = _card_row(self._get_card(conn, cid))
            self._record(conn, board_id, "upsert", card, "create")
            rebalanced = self._maybe_rebalance(conn, board_id, column_id, place.rank)
            if rebalanced:
                card = _card_row(self._get_card(conn, cid))
        return card, {"created": True, "placement": place.resolved, "rebalanced": rebalanced}

    def move_card(self, card_id: str, column_id: str, after_id: Optional[str] = None,
                  before_id: Optional[str] = None,
                  if_version: Optional[int] = None) -> tuple[dict, dict]:
        """Move a card (within or across columns). Moves are intent-based, so the
        version check is optional: without it, the last move wins."""
        with self._tx() as conn:
            row = self._get_card(conn, card_id)
            if if_version is not None and row["version"] != if_version:
                raise VersionMismatch(_card_row(row))
            board_id = row["board_id"]
            self._column(conn, board_id, column_id)
            place = self._place(conn, column_id, after_id, before_id, card_id)
            conn.execute(
                "UPDATE cards SET column_id = ?, rank = ?, version = version + 1 WHERE id = ?",
                (column_id, place.rank, card_id),
            )
            card = _card_row(self._get_card(conn, card_id))
            self._record(conn, board_id, "upsert", card, "move")
            rebalanced = self._maybe_rebalance(conn, board_id, column_id, place.rank)
            if rebalanced:
                card = _card_row(self._get_card(conn, card_id))
        return card, {"placement": place.resolved, "rebalanced": rebalanced}

    def edit_card(self, card_id: str, if_version: int, title: Optional[str] = None,
                  description: Optional[str] = None) -> dict:
        """Edit text fields. Requires the version the editor last saw (optimistic
        concurrency): a stale version raises VersionMismatch with the current card."""
        if title is not None:
            title = title.strip()
            if not title:
                raise Invalid("title cannot be empty")
        with self._tx() as conn:
            row = self._get_card(conn, card_id)
            if row["version"] != if_version:
                raise VersionMismatch(_card_row(row))
            conn.execute(
                "UPDATE cards SET title = ?, description = ?, version = version + 1 WHERE id = ?",
                (title if title is not None else row["title"],
                 description if description is not None else row["description"], card_id),
            )
            card = _card_row(self._get_card(conn, card_id))
            self._record(conn, row["board_id"], "upsert", card, "edit")
        return card

    def delete_card(self, card_id: str, if_version: int) -> None:
        with self._tx() as conn:
            row = self._get_card(conn, card_id)
            if row["version"] != if_version:
                raise VersionMismatch(_card_row(row))
            conn.execute("DELETE FROM cards WHERE id = ?", (card_id,))
            self._record(conn, row["board_id"], "delete",
                         {"id": card_id, "column_id": row["column_id"]}, "delete")

    # ------------------------------------------------------------ change feed
    def changes(self, board_id: str, since: int, limit: int = 500) -> dict:
        if since < 0:
            raise Invalid("since must be >= 0")
        limit = max(1, min(limit, 1000))
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM boards WHERE id = ?", (board_id,)).fetchone() is None:
                raise NotFound(f"board {board_id} not found")
            floor = conn.execute(
                "SELECT min_seq FROM feed_floor WHERE board_id = ?", (board_id,)
            ).fetchone()
            if floor is not None and since < floor["min_seq"]:
                raise FeedGone(int(floor["min_seq"]))
            rows = conn.execute(
                "SELECT seq, op, card, reason FROM changes WHERE board_id = ? AND seq > ? "
                "ORDER BY seq LIMIT ?",
                (board_id, since, limit + 1),
            ).fetchall()
            head = self._board_seq(conn, board_id)
        more = len(rows) > limit
        rows = rows[:limit]
        out = [{"seq": r["seq"], "op": r["op"], "card": json.loads(r["card"]),
                "reason": r["reason"]} for r in rows]
        # with more pages pending, resume after the last row returned; otherwise
        # jump to the head (all of this board's rows <= head were returned)
        cursor = out[-1]["seq"] if more else max(since, head)
        return {"changes": out, "cursor": cursor, "has_more": more}

    def compact(self, board_id: str, keep_last: int) -> int:
        """Drop all but the newest ``keep_last`` change rows of a board.

        Clients whose cursor is older than what was dropped get FeedGone (HTTP 410)
        and must reload a snapshot, instead of silently missing changes.
        """
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT seq FROM changes WHERE board_id = ? ORDER BY seq DESC LIMIT 1 OFFSET ?",
                (board_id, max(keep_last, 0)),
            ).fetchone()
            if rows is None:
                return 0
            cut = int(rows["seq"])
            cur = conn.execute(
                "DELETE FROM changes WHERE board_id = ? AND seq <= ?", (board_id, cut)
            )
            conn.execute(
                "INSERT INTO feed_floor (board_id, min_seq) VALUES (?, ?) "
                "ON CONFLICT(board_id) DO UPDATE SET min_seq = MAX(min_seq, excluded.min_seq)",
                (board_id, cut),
            )
            return cur.rowcount

    # ---------------------------------------------------------------- checks
    def check_invariants(self, board_id: str) -> list[str]:
        """Problems found (empty list == healthy). Used by tests and the demo."""
        problems: list[str] = []
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT id, column_id, rank FROM cards WHERE board_id = ? ORDER BY column_id, rank",
                (board_id,),
            ).fetchall()
        seen: dict[tuple[str, str], str] = {}
        for r in rows:
            if not fracindex.is_valid_key(r["rank"]):
                problems.append(f"card {r['id']} has invalid rank {r['rank']!r}")
            key = (r["column_id"], r["rank"])
            if key in seen:
                problems.append(f"cards {seen[key]} and {r['id']} share rank {r['rank']!r}")
            seen[key] = r["id"]
            if len(r["rank"]) > self.max_key_len:
                problems.append(f"card {r['id']} rank longer than {self.max_key_len}")
        return problems


def replay_feed(changes: list[dict[str, Any]], state: Optional[dict[str, dict]] = None
                ) -> dict[str, dict]:
    """Apply feed entries to a {card_id: card} map, the way a client does."""
    state = dict(state or {})
    for ch in changes:
        if ch["op"] == "delete":
            state.pop(ch["card"]["id"], None)
        else:
            state[ch["card"]["id"]] = ch["card"]
    return state
