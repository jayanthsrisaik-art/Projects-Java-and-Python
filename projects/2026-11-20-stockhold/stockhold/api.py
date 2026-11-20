"""HTTP API for stockhold (FastAPI). Also serves the TypeScript UI from ../web.

Customer identity is a demo header (``X-Customer-Id``); a real deployment would
take it from an authenticated session. Holds are scoped to their customer, and
another customer's hold id answers 404 rather than 403 so ids do not leak.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .store import Outcome, Store

ROOT = Path(__file__).resolve().parent.parent
CUSTOMER_PATTERN = r"^[A-Za-z0-9_.@-]{1,64}$"
KEY_PATTERN = r"^[A-Za-z0-9_:.-]{8,128}$"


class ReserveIn(BaseModel):
    sku: str = Field(min_length=1, max_length=64)
    qty: int = Field(ge=1, le=100)


class ConfirmIn(BaseModel):
    payment_token: str = Field(min_length=1, max_length=64)


def _respond(out: Outcome) -> JSONResponse:
    headers = {"Idempotent-Replayed": "true"} if out.replayed else {}
    return JSONResponse(out.body, status_code=out.status, headers=headers)


def create_app(store: Store | None = None, catalog_path: str | Path | None = None,
               web_dir: str | Path | None = None) -> FastAPI:
    if store is None:
        store = Store(os.environ.get("STOCKHOLD_DB", "stockhold.db"),
                      hold_ttl=float(os.environ.get("STOCKHOLD_HOLD_TTL", "120")))
    if catalog_path is None:
        catalog_path = ROOT / "samples" / "catalog.json"
    if not store.products() and Path(catalog_path).exists():
        store.load_catalog(json.loads(Path(catalog_path).read_text()))

    app = FastAPI(title="stockhold", version="1.0.0")
    app.state.store = store

    CustomerId = Header(alias="X-Customer-Id", pattern=CUSTOMER_PATTERN)

    @app.get("/api/products")
    def list_products() -> dict[str, Any]:
        return {"products": store.products(), "server_time": store.clock()}

    @app.get("/api/products/{sku}")
    def get_product(sku: str) -> dict[str, Any]:
        view = store.product(sku)
        if view is None:
            raise HTTPException(404, detail="product_not_found")
        return {**view, "server_time": store.clock()}

    @app.post("/api/holds")
    def reserve(body: ReserveIn, customer_id: str = CustomerId,
                idempotency_key: str = Header(alias="Idempotency-Key",
                                              pattern=KEY_PATTERN)) -> JSONResponse:
        return _respond(store.reserve(customer_id, body.sku, body.qty, idempotency_key))

    @app.get("/api/holds/{hold_id}")
    def get_hold(hold_id: str, customer_id: str = CustomerId) -> JSONResponse:
        return _respond(store.hold(customer_id, hold_id))

    @app.post("/api/holds/{hold_id}/confirm")
    def confirm(hold_id: str, body: ConfirmIn, customer_id: str = CustomerId) -> JSONResponse:
        return _respond(store.confirm(customer_id, hold_id, body.payment_token))

    @app.post("/api/holds/{hold_id}/release")
    def release(hold_id: str, customer_id: str = CustomerId) -> JSONResponse:
        return _respond(store.release(customer_id, hold_id))

    @app.get("/api/invariants")
    def invariants() -> dict[str, Any]:
        problems = store.check_invariants()
        return {"ok": not problems, "problems": problems}

    web = Path(web_dir) if web_dir is not None else ROOT / "web"
    if (web / "index.html").exists():
        if (web / "dist").exists():
            app.mount("/dist", StaticFiles(directory=web / "dist"), name="dist")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(web / "index.html")

        @app.get("/styles.css", include_in_schema=False)
        def styles() -> FileResponse:
            return FileResponse(web / "styles.css", media_type="text/css")

    return app


def app_factory() -> FastAPI:  # for `uvicorn --factory stockhold.api:app_factory`
    return create_app()
