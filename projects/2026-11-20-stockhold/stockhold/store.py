"""Inventory reservation store: TTL holds, idempotent writes, no oversell.

Every state-changing operation runs inside one ``BEGIN IMMEDIATE`` SQLite
transaction. That takes the database write lock *before* the stock check, so
"check availability" and "create the hold" can never interleave with another
writer. The idempotency record is written in the same transaction as the
effect it describes, so a crash can never leave one without the other.

Availability is computed, not stored:

    available = total - sold - SUM(qty of active holds whose expires_at > now)

so an abandoned hold returns its stock the instant it expires, with no
background sweeper needed for correctness. Expired holds are also marked
``expired`` lazily by the next write that touches the SKU, purely for display.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    sku                TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    price_cents        INTEGER NOT NULL CHECK (price_cents >= 0),
    total              INTEGER NOT NULL CHECK (total >= 0),
    sold               INTEGER NOT NULL DEFAULT 0 CHECK (sold >= 0),
    per_customer_limit INTEGER NOT NULL CHECK (per_customer_limit >= 1),
    CHECK (sold <= total)
);
CREATE TABLE IF NOT EXISTS holds (
    id          TEXT PRIMARY KEY,
    sku         TEXT NOT NULL REFERENCES products(sku),
    customer_id TEXT NOT NULL,
    qty         INTEGER NOT NULL CHECK (qty > 0),
    status      TEXT NOT NULL CHECK (status IN ('active','confirmed','released','expired')),
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    order_id    TEXT
);
CREATE INDEX IF NOT EXISTS holds_sku_active ON holds(sku, status, expires_at);
CREATE INDEX IF NOT EXISTS holds_customer ON holds(customer_id, sku, status);
CREATE TABLE IF NOT EXISTS orders (
    id           TEXT PRIMARY KEY,
    hold_id      TEXT NOT NULL UNIQUE REFERENCES holds(id),
    sku          TEXT NOT NULL REFERENCES products(sku),
    customer_id  TEXT NOT NULL,
    qty          INTEGER NOT NULL CHECK (qty > 0),
    amount_cents INTEGER NOT NULL,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS orders_customer ON orders(customer_id, sku);
CREATE TABLE IF NOT EXISTS idempotency (
    customer_id  TEXT NOT NULL,
    key          TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status_code  INTEGER NOT NULL,
    body         TEXT NOT NULL,
    created_at   REAL NOT NULL,
    PRIMARY KEY (customer_id, key)
);
"""


@dataclass
class Outcome:
    """An HTTP-shaped result: status code + JSON body (+ replay flag)."""

    status: int
    body: dict[str, Any]
    replayed: bool = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


def _err(status: int, code: str, message: str, **extra: Any) -> Outcome:
    return Outcome(status, {"error": code, "message": message, **extra})


def request_fingerprint(op: str, payload: dict[str, Any]) -> str:
    raw = json.dumps({"op": op, **payload}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def mock_charge(token: str, amount_cents: int) -> tuple[bool, str]:
    """Deterministic stand-in for a payment provider (no network, no keys).

    ``tok_decline`` is declined, any other ``tok_*`` token is approved.
    """
    if token == "tok_decline":
        return False, "card_declined"
    if not token.startswith("tok_"):
        return False, "invalid_token"
    return True, "approved"


class SequentialIds:
    """Deterministic ids (h_0001, o_0001 ...) for tests and the sample scenario."""

    def __init__(self) -> None:
        self._n: dict[str, int] = {}
        self._lock = threading.Lock()

    def __call__(self, prefix: str) -> str:
        with self._lock:
            self._n[prefix] = self._n.get(prefix, 0) + 1
            return f"{prefix}_{self._n[prefix]:04d}"


def random_ids(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Store:
    path: str
    hold_ttl: float = 120.0
    clock: Callable[[], float] = time.time
    ids: Callable[[str], str] = random_ids
    busy_timeout: float = 30.0
    # Test/benchmark hook, called after the stock check and before the hold is
    # written. Used to widen the race window identically for every strategy.
    between_check_and_write: Callable[[], None] | None = None
    _local: threading.local = field(default_factory=threading.local, repr=False)

    def __post_init__(self) -> None:
        if self.path == ":memory:":
            raise ValueError("use a file path: each thread opens its own connection")
        conn = self.conn()
        conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ plumbing
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=self.busy_timeout,
                                isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute(f"PRAGMA busy_timeout={int(self.busy_timeout * 1000)}")
            self._local.conn = c
        return c

    @contextlib.contextmanager
    def _write_tx(self) -> Iterator[sqlite3.Connection]:
        c = self.conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
        except BaseException:
            c.execute("ROLLBACK")
            raise
        else:
            c.execute("COMMIT")

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    # ------------------------------------------------------------------ catalog
    def upsert_product(self, sku: str, name: str, price_cents: int, total: int,
                       per_customer_limit: int) -> None:
        with self._write_tx() as c:
            c.execute(
                """INSERT INTO products(sku, name, price_cents, total, per_customer_limit)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(sku) DO UPDATE SET name=excluded.name,
                     price_cents=excluded.price_cents, total=excluded.total,
                     per_customer_limit=excluded.per_customer_limit""",
                (sku, name, price_cents, total, per_customer_limit))

    def load_catalog(self, items: list[dict[str, Any]]) -> None:
        for it in items:
            self.upsert_product(it["sku"], it["name"], it["price_cents"], it["total"],
                                it["per_customer_limit"])

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _held(c: sqlite3.Connection, sku: str, now: float) -> int:
        row = c.execute(
            "SELECT COALESCE(SUM(qty),0) FROM holds WHERE sku=? AND status='active' AND expires_at>?",
            (sku, now)).fetchone()
        return int(row[0])

    def _product_view(self, c: sqlite3.Connection, sku: str, now: float) -> dict[str, Any] | None:
        p = c.execute("SELECT * FROM products WHERE sku=?", (sku,)).fetchone()
        if p is None:
            return None
        held = self._held(c, sku, now)
        return {
            "sku": p["sku"], "name": p["name"], "price_cents": p["price_cents"],
            "total": p["total"], "sold": p["sold"], "held": held,
            "available": p["total"] - p["sold"] - held,
            "per_customer_limit": p["per_customer_limit"],
        }

    def product(self, sku: str) -> dict[str, Any] | None:
        return self._product_view(self.conn(), sku, self.clock())

    def products(self) -> list[dict[str, Any]]:
        c, now = self.conn(), self.clock()
        skus = [r[0] for r in c.execute("SELECT sku FROM products ORDER BY sku")]
        return [v for s in skus if (v := self._product_view(c, s, now)) is not None]

    def _hold_dict(self, h: sqlite3.Row, now: float) -> dict[str, Any]:
        status = h["status"]
        if status == "active" and h["expires_at"] <= now:
            status = "expired"
        return {
            "id": h["id"], "sku": h["sku"], "customer_id": h["customer_id"],
            "qty": h["qty"], "status": status, "created_at": h["created_at"],
            "expires_at": h["expires_at"], "order_id": h["order_id"],
            "server_time": now,
        }

    def hold(self, customer_id: str, hold_id: str) -> Outcome:
        now = self.clock()
        h = self.conn().execute("SELECT * FROM holds WHERE id=? AND customer_id=?",
                                (hold_id, customer_id)).fetchone()
        if h is None:
            return _err(404, "hold_not_found", "no such hold for this customer")
        return Outcome(200, self._hold_dict(h, now))

    # ------------------------------------------------------------------ writes
    @staticmethod
    def _expire_due(c: sqlite3.Connection, sku: str, now: float) -> None:
        c.execute("UPDATE holds SET status='expired' WHERE sku=? AND status='active' AND expires_at<=?",
                  (sku, now))

    def _idem_lookup(self, c: sqlite3.Connection, customer_id: str, key: str,
                     fp: str) -> Outcome | None:
        row = c.execute("SELECT request_hash, status_code, body FROM idempotency "
                        "WHERE customer_id=? AND key=?", (customer_id, key)).fetchone()
        if row is None:
            return None
        if row["request_hash"] != fp:
            return _err(422, "idempotency_key_reused",
                        "this Idempotency-Key was already used for a different request")
        body = json.loads(row["body"])
        if "server_time" in body:
            # The stored hold is replayed as-is, but the clock reading must be
            # fresh or a retrying client would compute a wrong countdown.
            body["server_time"] = self.clock()
        return Outcome(row["status_code"], body, replayed=True)

    def reserve(self, customer_id: str, sku: str, qty: int,
                idempotency_key: str | None = None) -> Outcome:
        """Place a hold on ``qty`` units for ``hold_ttl`` seconds, or explain why not."""
        if qty < 1:
            return _err(422, "invalid_qty", "qty must be >= 1")
        fp = request_fingerprint("reserve", {"sku": sku, "qty": qty})
        with self._write_tx() as c:
            if idempotency_key is not None:
                prior = self._idem_lookup(c, customer_id, idempotency_key, fp)
                if prior is not None:
                    return prior
            now = self.clock()
            out = self._reserve_locked(c, customer_id, sku, qty, now)
            if idempotency_key is not None:
                c.execute("INSERT INTO idempotency VALUES (?,?,?,?,?,?)",
                          (customer_id, idempotency_key, fp, out.status,
                           json.dumps(out.body), now))
            return out

    def _reserve_locked(self, c: sqlite3.Connection, customer_id: str, sku: str,
                        qty: int, now: float) -> Outcome:
        self._expire_due(c, sku, now)
        view = self._product_view(c, sku, now)
        if view is None:
            return _err(404, "product_not_found", f"unknown sku {sku!r}")
        mine = c.execute(
            """SELECT (SELECT COALESCE(SUM(qty),0) FROM holds
                        WHERE sku=? AND customer_id=? AND status='active' AND expires_at>?)
                    + (SELECT COALESCE(SUM(qty),0) FROM orders WHERE sku=? AND customer_id=?)""",
            (sku, customer_id, now, sku, customer_id)).fetchone()[0]
        limit = view["per_customer_limit"]
        if mine + qty > limit:
            return _err(409, "limit_exceeded",
                        f"limit is {limit} per customer; you already hold or bought {mine}",
                        limit=limit, already=mine)
        if qty > view["available"]:
            return _err(409, "sold_out", "not enough stock available",
                        available=view["available"])
        if self.between_check_and_write is not None:
            self.between_check_and_write()
        hid = self.ids("h")
        c.execute("INSERT INTO holds VALUES (?,?,?,?, 'active', ?, ?, NULL)",
                  (hid, sku, customer_id, qty, now, now + self.hold_ttl))
        h = c.execute("SELECT * FROM holds WHERE id=?", (hid,)).fetchone()
        return Outcome(201, self._hold_dict(h, now))

    def confirm(self, customer_id: str, hold_id: str, payment_token: str) -> Outcome:
        """Pay for a hold and turn it into an order. Safe to call repeatedly."""
        with self._write_tx() as c:
            now = self.clock()
            h = c.execute("SELECT * FROM holds WHERE id=? AND customer_id=?",
                          (hold_id, customer_id)).fetchone()
            if h is None:
                return _err(404, "hold_not_found", "no such hold for this customer")
            if h["status"] == "confirmed":
                o = c.execute("SELECT * FROM orders WHERE id=?", (h["order_id"],)).fetchone()
                return Outcome(200, dict(o), replayed=True)
            if h["status"] == "released":
                return _err(409, "hold_released", "this hold was released")
            if h["status"] == "expired" or h["expires_at"] <= now:
                c.execute("UPDATE holds SET status='expired' WHERE id=?", (hold_id,))
                return _err(410, "hold_expired",
                            "the hold expired and its stock went back on sale")
            price = c.execute("SELECT price_cents FROM products WHERE sku=?",
                              (h["sku"],)).fetchone()[0]
            amount = price * h["qty"]
            approved, reason = mock_charge(payment_token, amount)
            if not approved:
                # The hold stays active: the customer may retry with another card
                # until it expires.
                return _err(402, "payment_declined", reason)
            # Defence in depth: the guarded UPDATE and the CHECK constraint both
            # refuse to let sold exceed total even if a bug elsewhere let it try.
            cur = c.execute("UPDATE products SET sold = sold + ? WHERE sku=? AND sold + ? <= total",
                            (h["qty"], h["sku"], h["qty"]))
            if cur.rowcount != 1:  # pragma: no cover - unreachable while holds are correct
                raise RuntimeError("stock invariant violated during confirm")
            oid = self.ids("o")
            c.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
                      (oid, hold_id, h["sku"], customer_id, h["qty"], amount, now))
            c.execute("UPDATE holds SET status='confirmed', order_id=? WHERE id=?", (oid, hold_id))
            o = c.execute("SELECT * FROM orders WHERE id=?", (oid,)).fetchone()
            return Outcome(201, dict(o))

    def release(self, customer_id: str, hold_id: str) -> Outcome:
        """Give a hold's stock back early. Idempotent."""
        with self._write_tx() as c:
            now = self.clock()
            h = c.execute("SELECT * FROM holds WHERE id=? AND customer_id=?",
                          (hold_id, customer_id)).fetchone()
            if h is None:
                return _err(404, "hold_not_found", "no such hold for this customer")
            if h["status"] == "confirmed":
                return _err(409, "hold_confirmed", "this hold was already paid for")
            if h["status"] == "active" and h["expires_at"] > now:
                c.execute("UPDATE holds SET status='released' WHERE id=?", (hold_id,))
            elif h["status"] == "active":
                c.execute("UPDATE holds SET status='expired' WHERE id=?", (hold_id,))
            h = c.execute("SELECT * FROM holds WHERE id=?", (hold_id,)).fetchone()
            return Outcome(200, self._hold_dict(h, now))

    # ------------------------------------------------------------------ audit
    def check_invariants(self) -> list[str]:
        """Return human-readable violations (empty list == healthy)."""
        c, now = self.conn(), self.clock()
        problems: list[str] = []
        for p in c.execute("SELECT * FROM products ORDER BY sku").fetchall():
            sku = p["sku"]
            held = self._held(c, sku, now)
            if p["sold"] + held > p["total"]:
                problems.append(f"{sku}: sold {p['sold']} + held {held} > total {p['total']}")
            ordered = c.execute("SELECT COALESCE(SUM(qty),0) FROM orders WHERE sku=?",
                                (sku,)).fetchone()[0]
            if ordered != p["sold"]:
                problems.append(f"{sku}: orders sum {ordered} != sold counter {p['sold']}")
            for r in c.execute(
                    """SELECT customer_id, SUM(q) AS n FROM (
                         SELECT customer_id, qty AS q FROM orders WHERE sku=?
                         UNION ALL
                         SELECT customer_id, qty FROM holds
                          WHERE sku=? AND status='active' AND expires_at>?)
                       GROUP BY customer_id HAVING n > ?""",
                    (sku, sku, now, p["per_customer_limit"])):
                problems.append(f"{sku}: customer {r['customer_id']} has {r['n']} "
                                f"> limit {p['per_customer_limit']}")
        bad = c.execute(
            """SELECT h.id FROM holds h LEFT JOIN orders o ON o.hold_id = h.id
               WHERE (h.status='confirmed') != (o.id IS NOT NULL)""").fetchall()
        for r in bad:
            problems.append(f"hold {r[0]}: confirmed status and order row disagree")
        return problems
