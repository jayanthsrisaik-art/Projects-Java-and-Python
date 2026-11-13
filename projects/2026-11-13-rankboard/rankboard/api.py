"""FastAPI app: JSON API under /api and the compiled TypeScript UI at /."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .store import Conflict, FeedGone, Invalid, NotFound, Store, VersionMismatch

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoardIn(Strict):
    name: str = Field(max_length=200)
    columns: list[str] = Field(default_factory=lambda: ["To do", "Doing", "Done"])


class Drop(Strict):
    after_id: Optional[str] = None
    before_id: Optional[str] = None


class CardIn(Drop):
    id: Optional[str] = None
    column_id: str
    title: str = Field(max_length=500)
    description: str = Field(default="", max_length=10_000)


class MoveIn(Drop):
    column_id: str


class EditIn(Strict):
    title: Optional[str] = Field(default=None, max_length=500)
    description: Optional[str] = Field(default=None, max_length=10_000)


def _etag(card: dict) -> str:
    return f'"{card["version"]}"'


def _parse_if_match(value: Optional[str], required: bool) -> Optional[int]:
    if value is None:
        if required:
            raise HTTPException(428, "If-Match header with the card version is required")
        return None
    raw = value.strip()
    if raw.startswith("W/"):
        raw = raw[2:]
    raw = raw.strip('"')
    if not raw.isdigit():
        raise HTTPException(400, "If-Match must be a card version such as \"3\"")
    return int(raw)


def create_app(db_path: Optional[str] = None, max_key_len: Optional[int] = None) -> FastAPI:
    store = Store(db_path or os.environ.get("RANKBOARD_DB", "rankboard.db"),
                  **({"max_key_len": max_key_len} if max_key_len else {}))
    app = FastAPI(title="rankboard", version="1.0.0")
    app.state.store = store

    def guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotFound as e:
            raise HTTPException(404, str(e))
        except Invalid as e:
            raise HTTPException(400, str(e))
        except Conflict as e:
            raise HTTPException(409, str(e))

    def mismatch(e: VersionMismatch) -> JSONResponse:
        return JSONResponse(status_code=412, headers={"ETag": _etag(e.current)},
                            content={"detail": "card changed since you loaded it",
                                     "current": e.current})

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/boards")
    def boards() -> list[dict]:
        return store.list_boards()

    @app.post("/api/boards", status_code=201)
    def create_board(body: BoardIn) -> dict:
        return guard(store.create_board, body.name, body.columns)

    @app.get("/api/boards/{board_id}")
    def snapshot(board_id: str) -> dict:
        return guard(store.snapshot, board_id)

    @app.get("/api/boards/{board_id}/changes")
    def changes(board_id: str, since: int = Query(0, ge=0),
                limit: int = Query(500, ge=1, le=1000)):
        try:
            return guard(store.changes, board_id, since, limit)
        except FeedGone as e:
            return JSONResponse(status_code=410, content={
                "detail": "changes before this cursor were compacted; reload the board",
                "min_seq": e.min_seq})

    @app.post("/api/boards/{board_id}/cards")
    def create_card(board_id: str, body: CardIn, response: Response) -> dict:
        card, info = guard(store.create_card, board_id, body.column_id, body.title,
                           body.description, body.after_id, body.before_id, body.id)
        response.status_code = 201 if info["created"] else 200
        response.headers["ETag"] = _etag(card)
        return {"card": card, **info}

    @app.get("/api/cards/{card_id}")
    def get_card(card_id: str, response: Response) -> dict:
        card = guard(store.get_card, card_id)
        response.headers["ETag"] = _etag(card)
        return card

    @app.post("/api/cards/{card_id}/move")
    def move_card(card_id: str, body: MoveIn, response: Response,
                  if_match: Optional[str] = Header(default=None)):
        version = _parse_if_match(if_match, required=False)
        try:
            card, info = guard(store.move_card, card_id, body.column_id, body.after_id,
                               body.before_id, version)
        except VersionMismatch as e:
            return mismatch(e)
        response.headers["ETag"] = _etag(card)
        return {"card": card, **info}

    @app.patch("/api/cards/{card_id}")
    def edit_card(card_id: str, body: EditIn, response: Response,
                  if_match: Optional[str] = Header(default=None)):
        version = _parse_if_match(if_match, required=True)
        try:
            card = guard(store.edit_card, card_id, version, body.title, body.description)
        except VersionMismatch as e:
            return mismatch(e)
        response.headers["ETag"] = _etag(card)
        return card

    @app.delete("/api/cards/{card_id}", status_code=204)
    def delete_card(card_id: str, if_match: Optional[str] = Header(default=None)):
        version = _parse_if_match(if_match, required=True)
        try:
            guard(store.delete_card, card_id, version)
        except VersionMismatch as e:
            return mismatch(e)
        return Response(status_code=204)

    dist = WEB_DIR / "dist" / "src"
    if dist.is_dir():
        app.mount("/static", StaticFiles(directory=dist), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        page = WEB_DIR / "index.html"
        if not dist.is_dir():
            return JSONResponse(status_code=503, content={
                "detail": "UI not built: run `npm install && npm run build` in web/"})
        return FileResponse(page)

    @app.get("/styles.css", include_in_schema=False)
    def styles():
        return FileResponse(WEB_DIR / "styles.css")

    return app

