"""Command line: simulate | ingest | train | build | usage | issues | evaluate | serve."""

from __future__ import annotations

import argparse
import csv
import json
import sys

from .simulate import SimConfig, simulate, write_world
from .store import Store


def _csv_rows(path: str, cols: list[str]):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        missing = set(cols) - set(r.fieldnames or [])
        if missing:
            sys.exit(f"{path}: missing columns {sorted(missing)}")
        for row in r:
            yield tuple(row[c] for c in cols)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="meterfill", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="write a synthetic meter fleet (readings, weather, truth)")
    s.add_argument("--out", required=True)
    s.add_argument("--households", type=int, default=150)
    s.add_argument("--days", type=int, default=56)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--no-truth", action="store_true")

    s = sub.add_parser("ingest", help="load readings (and weather) CSVs into SQLite; idempotent")
    s.add_argument("--db", required=True)
    s.add_argument("--readings")
    s.add_argument("--weather")

    s = sub.add_parser("train", help="train the shape model on the meters in the database")
    s.add_argument("--db", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("build", help="clean all meters and rebuild hourly usage")
    s.add_argument("--db", required=True)
    s.add_argument("--method", choices=["uniform", "profile", "model"], default="model")
    s.add_argument("--model")

    s = sub.add_parser("usage", help="print usage for one meter as JSON")
    s.add_argument("--db", required=True)
    s.add_argument("--meter", required=True)
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--granularity", choices=["hour", "day"], default="day")

    s = sub.add_parser("issues", help="print detected faults and gaps for one meter as JSON")
    s.add_argument("--db", required=True)
    s.add_argument("--meter", required=True)

    s = sub.add_parser("evaluate", help="multi-seed evaluation against simulator ground truth")
    s.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    s.add_argument("--households", type=int, default=150)
    s.add_argument("--days", type=int, default=56)
    s.add_argument("--out", default="results")

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--db", required=True)
    s.add_argument("--model")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)

    if a.cmd == "simulate":
        world = simulate(SimConfig(households=a.households, days=a.days, seed=a.seed))
        print(json.dumps(write_world(world, a.out, include_truth=not a.no_truth), indent=2))
    elif a.cmd == "ingest":
        st = Store(a.db)
        if a.weather:
            print(json.dumps({"weather_hours": st.ingest_weather(_csv_rows(a.weather, ["hour_ts", "temp_c"]))}))
        if a.readings:
            print(json.dumps(st.ingest_readings(_csv_rows(a.readings, ["meter_id", "ts", "register_kwh"]))))
    elif a.cmd == "train":
        from .model import train_model

        st = Store(a.db)
        h0, nh = st.window()
        temps, _ = st.temps(h0, nh)
        m = train_model(st.all_rows(), temps, h0, nh, seed=a.seed)
        m.save(a.model)
        print(json.dumps({"model": a.model, **m.meta}))
    elif a.cmd == "build":
        from .model import ShapeModel

        model = ShapeModel.load(a.model) if a.model else None
        print(json.dumps(Store(a.db).rebuild(a.method, model), indent=2))
    elif a.cmd == "usage":
        out = Store(a.db).usage(a.meter, a.start, a.end, a.granularity)
        if out is None:
            sys.exit("unknown meter or not built yet")
        print(json.dumps(out, indent=2))
    elif a.cmd == "issues":
        print(json.dumps(Store(a.db).issues(a.meter), indent=2))
    elif a.cmd == "evaluate":
        from .evaluate import run

        report = run(a.seeds, a.households, a.days, a.out)
        print(json.dumps(report["summary"]["allocated_nmae"], indent=2))
    elif a.cmd == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(a.db, a.model), host=a.host, port=a.port)
    return 0
