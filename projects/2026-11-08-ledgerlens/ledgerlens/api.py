"""FastAPI service over the ledger."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .model import TxnClassifier
from .store import Ledger
from .synth import CATEGORIES


class RawTxn(BaseModel):
    txn_id: str = Field(min_length=1, max_length=64)
    account_id: str = Field(min_length=1, max_length=64)
    posted_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    descriptor: str = Field(min_length=1, max_length=200)
    amount: float


class Batch(BaseModel):
    transactions: list[RawTxn] = Field(min_length=1, max_length=5000)


class Correction(BaseModel):
    txn_id: str
    category: str
    apply_to_merchant: bool = True


def create_app(db_path: str | Path, model: TxnClassifier | None = None,
               model_path: str | Path | None = None) -> FastAPI:
    if model is None:
        if model_path is None:
            raise ValueError("pass a model or a model_path")
        model = TxnClassifier.load(model_path)
    ledger = Ledger(db_path, model)
    app = FastAPI(title="ledgerlens", version="1.0.0")
    app.state.ledger = ledger

    @app.get("/health")
    def health():
        return {"status": "ok", "threshold": ledger.model.threshold, "categories": CATEGORIES}

    @app.post("/transactions")
    def ingest(batch: Batch):
        return ledger.ingest([t.model_dump() for t in batch.transactions])

    @app.get("/transactions")
    def list_txns(source: str | None = Query(None, pattern="^(user|rule|model|review)$"),
                  account_id: str | None = None, limit: int = Query(100, ge=1, le=1000)):
        return ledger.list(source, account_id, limit)

    @app.post("/corrections")
    def correct(c: Correction):
        if c.category not in CATEGORIES:
            raise HTTPException(422, f"category must be one of {CATEGORIES}")
        try:
            return ledger.correct(c.txn_id, c.category, c.apply_to_merchant)
        except KeyError:
            raise HTTPException(404, f"unknown txn_id {c.txn_id}")

    @app.get("/rules")
    def rules():
        return ledger.rules()

    @app.get("/summary")
    def summary(month: str = Query(..., pattern=r"^\d{4}-\d{2}$"), account_id: str | None = None):
        return {"month": month, "account_id": account_id, "by_category": ledger.monthly_summary(month, account_id)}

    @app.get("/subscriptions")
    def subscriptions(account_id: str | None = None):
        return [asdict(r) for r in ledger.recurring(account_id)]

    return app
