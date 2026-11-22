"""FastAPI service over the SQLite store."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import REGISTER_MAX
from .model import ShapeModel
from .store import Store, ValidationError


class Reading(BaseModel):
    meter_id: str = Field(min_length=1, max_length=64)
    ts: str | int = Field(description="ISO-8601 with timezone, or epoch seconds")
    register_kwh: float = Field(ge=0, lt=REGISTER_MAX)


class ReadingBatch(BaseModel):
    readings: list[Reading] = Field(max_length=100_000)


class WeatherHour(BaseModel):
    hour_ts: str | int
    temp_c: float = Field(ge=-60, le=60)


class WeatherBatch(BaseModel):
    hours: list[WeatherHour]


def create_app(db_path: str | Path, model_path: str | Path | None = None) -> FastAPI:
    store = Store(db_path)
    model = ShapeModel.load(model_path) if model_path and Path(model_path).exists() else None
    app = FastAPI(title="meterfill", version="0.1.0")
    app.state.store = store
    app.state.model = model

    def bad(e: Exception):
        raise HTTPException(status_code=422, detail=str(e))

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": model is not None, "build": store.build_state()}

    @app.post("/readings")
    def post_readings(batch: ReadingBatch):
        try:
            return store.ingest_readings((r.meter_id, r.ts, r.register_kwh) for r in batch.readings)
        except ValidationError as e:
            bad(e)

    @app.post("/weather")
    def post_weather(batch: WeatherBatch):
        try:
            return {"hours": store.ingest_weather((h.hour_ts, h.temp_c) for h in batch.hours)}
        except ValidationError as e:
            bad(e)

    @app.post("/rebuild")
    def rebuild(method: str | None = Query(None, pattern="^(uniform|profile|model)$")):
        chosen = method or ("model" if model is not None else "profile")
        try:
            return store.rebuild(chosen, model)
        except ValidationError as e:
            bad(e)

    @app.get("/meters")
    def meters():
        return {"build": store.build_state(), "meters": store.meters_summary()}

    @app.get("/meters/{meter_id}/usage")
    def usage(meter_id: str, start: str | None = None, end: str | None = None,
              granularity: str = Query("hour", pattern="^(hour|day)$")):
        try:
            out = store.usage(meter_id, start, end, granularity)
        except ValidationError as e:
            bad(e)
        except ValueError as e:
            bad(e)
        if out is None:
            raise HTTPException(status_code=404, detail="unknown meter or not built yet (POST /rebuild)")
        return out

    @app.get("/meters/{meter_id}/issues")
    def issues(meter_id: str):
        if meter_id not in store.meter_ids():
            raise HTTPException(status_code=404, detail="unknown meter")
        return {"meter_id": meter_id, "build": store.build_state(), "issues": store.issues(meter_id)}

    return app
