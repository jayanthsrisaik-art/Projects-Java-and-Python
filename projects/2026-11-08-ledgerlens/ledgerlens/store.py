"""SQLite ledger: idempotent ingest, categorisation with rule > model > review precedence,
user corrections that become merchant rules, and SQL reporting views."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .model import TxnClassifier
from .normalize import normalize
from .recurring import detect
from .synth import CATEGORIES

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    txn_id        TEXT PRIMARY KEY,
    account_id    TEXT NOT NULL,
    posted_date   TEXT NOT NULL,
    descriptor    TEXT NOT NULL,
    amount        REAL NOT NULL,
    merchant_key  TEXT NOT NULL,
    merchant_name TEXT NOT NULL,
    processor     TEXT NOT NULL,
    category      TEXT,              -- NULL while in the review queue
    suggested     TEXT NOT NULL,     -- model's top label, always kept for audit
    confidence    REAL NOT NULL,
    source        TEXT NOT NULL CHECK (source IN ('user', 'rule', 'model', 'review'))
);
CREATE INDEX IF NOT EXISTS ix_txn_key ON transactions(merchant_key);
CREATE INDEX IF NOT EXISTS ix_txn_acct_date ON transactions(account_id, posted_date);
CREATE TABLE IF NOT EXISTS merchant_rules (
    merchant_key TEXT PRIMARY KEY,
    category     TEXT NOT NULL,
    created_from TEXT NOT NULL       -- txn_id of the correction that created the rule
);
CREATE VIEW IF NOT EXISTS monthly_spend AS
    SELECT account_id, substr(posted_date, 1, 7) AS month,
           COALESCE(category, 'needs_review') AS category,
           ROUND(SUM(-amount), 2) AS spend, COUNT(*) AS n
    FROM transactions WHERE amount < 0
    GROUP BY account_id, month, COALESCE(category, 'needs_review');
"""


class Ledger:
    def __init__(self, db_path: str | Path, model: TxnClassifier):
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.model = model

    # ------------------------------------------------------------- ingest
    def ingest(self, rows: list[dict]) -> dict:
        """Insert new transactions; re-delivered txn_ids are ignored (idempotent)."""
        ids = [r["txn_id"] for r in rows]
        existing = {r[0] for r in self.db.execute(
            f"SELECT txn_id FROM transactions WHERE txn_id IN ({','.join('?' * len(ids))})", ids)} if ids else set()
        fresh, seen = [], set()
        for r in rows:
            if r["txn_id"] in existing or r["txn_id"] in seen:
                continue
            seen.add(r["txn_id"])
            fresh.append(r)
        if fresh:
            preds = self.model.predict_with_confidence([r["descriptor"] for r in fresh],
                                                       [float(r["amount"]) for r in fresh])
            rules = dict(self.db.execute("SELECT merchant_key, category FROM merchant_rules").fetchall())
            with self.db:
                for r, (label, conf) in zip(fresh, preds):
                    n = normalize(r["descriptor"])
                    if n.key in rules:
                        category, source = rules[n.key], "rule"
                    elif conf >= self.model.threshold:
                        category, source = label, "model"
                    else:
                        category, source = None, "review"
                    self.db.execute(
                        "INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (r["txn_id"], r["account_id"], r["posted_date"], r["descriptor"], float(r["amount"]),
                         n.key, n.name, n.processor, category, label, round(conf, 4), source))
        return {"received": len(rows), "inserted": len(fresh), "duplicates": len(rows) - len(fresh),
                "transactions": self.get_many(ids)}

    # ------------------------------------------------------------- queries
    def get_many(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        q = f"SELECT * FROM transactions WHERE txn_id IN ({','.join('?' * len(ids))})"
        found = {r["txn_id"]: dict(r) for r in self.db.execute(q, ids)}
        out, seen = [], set()
        for i in ids:
            if i in found and i not in seen:
                seen.add(i)
                out.append(found[i])
        return out

    def list(self, source: str | None = None, account_id: str | None = None, limit: int = 100) -> list[dict]:
        q, args = "SELECT * FROM transactions WHERE 1=1", []
        if source:
            q, args = q + " AND source = ?", args + [source]
        if account_id:
            q, args = q + " AND account_id = ?", args + [account_id]
        q += " ORDER BY posted_date, txn_id LIMIT ?"
        return [dict(r) for r in self.db.execute(q, args + [limit])]

    # ------------------------------------------------------------- corrections
    def correct(self, txn_id: str, category: str, apply_to_merchant: bool = True) -> dict:
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        row = self.db.execute("SELECT merchant_key FROM transactions WHERE txn_id = ?", (txn_id,)).fetchone()
        if row is None:
            raise KeyError(txn_id)
        key = row["merchant_key"]
        with self.db:
            self.db.execute("UPDATE transactions SET category = ?, source = 'user' WHERE txn_id = ?",
                            (category, txn_id))
            updated = 0
            if apply_to_merchant:
                self.db.execute("INSERT INTO merchant_rules VALUES (?,?,?) ON CONFLICT(merchant_key) "
                                "DO UPDATE SET category = excluded.category, created_from = excluded.created_from",
                                (key, category, txn_id))
                # never overwrite another explicit user decision
                updated = self.db.execute(
                    "UPDATE transactions SET category = ?, source = 'rule' "
                    "WHERE merchant_key = ? AND source != 'user'", (category, key)).rowcount
        return {"txn_id": txn_id, "merchant_key": key, "category": category,
                "rule_created": apply_to_merchant, "other_transactions_updated": updated}

    def rules(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM merchant_rules ORDER BY merchant_key")]

    # ------------------------------------------------------------- reporting
    def monthly_summary(self, month: str, account_id: str | None = None) -> list[dict]:
        q = ("SELECT category, ROUND(SUM(spend), 2) AS spend, SUM(n) AS n FROM monthly_spend "
             "WHERE month = ?")
        args: list = [month]
        if account_id:
            q, args = q + " AND account_id = ?", args + [account_id]
        q += " GROUP BY category ORDER BY spend DESC, category"
        return [dict(r) for r in self.db.execute(q, args)]

    def recurring(self, account_id: str | None = None):
        q, args = "SELECT account_id, posted_date, amount, merchant_key FROM transactions", []
        if account_id:
            q, args = q + " WHERE account_id = ?", [account_id]
        return detect([dict(r) for r in self.db.execute(q, args)])
