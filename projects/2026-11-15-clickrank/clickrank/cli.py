"""Command line interface: python -m clickrank <command> ..."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .catalog import build_catalog, load_products, vocabulary
from .experiment import run, write_report
from .features import SearchIndex, rank
from .ingest import ingest_file
from .propensity import estimate_propensities
from .simulate import simulate
from .text import normalize_query
from .train import click_pairs, fit_pairwise, load_model, save_model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clickrank", description="Unbiased learning-to-rank from click logs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("experiment", help="full simulate->ingest->train->evaluate run over seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--impressions", type=int, default=60000)
    p.add_argument("--swap-rate", type=float, default=0.1)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--work", type=Path, default=Path("work"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="catalog + IPS model of the first seed")

    p = sub.add_parser("simulate", help="write a catalog and simulated raw event log")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--impressions", type=int, default=20000)
    p.add_argument("--swap-rate", type=float, default=0.1)
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--events", type=Path, default=Path("work/events.jsonl"))

    p = sub.add_parser("ingest", help="clean raw JSONL events into SQLite")
    p.add_argument("events", type=Path)
    p.add_argument("--db", type=Path, default=Path("work/logs.db"))

    p = sub.add_parser("train", help="estimate propensities from a log DB and fit a ranker")
    p.add_argument("--db", type=Path, default=Path("work/logs.db"))
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--method", choices=["ipw", "naive"], default="ipw")

    p = sub.add_parser("search", help="rank products for a query offline")
    p.add_argument("query")
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("-k", type=int, default=10)

    p = sub.add_parser("serve", help="run the FastAPI search service")
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--events", type=Path, default=Path("work/service_events.jsonl"))
    p.add_argument("--explore-rate", type=float, default=0.05)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    a = ap.parse_args(argv)
    if a.cmd == "experiment":
        runs = [run(s, a.work, n_impressions=a.impressions, swap_rate=a.swap_rate, eta=a.eta,
                    artifacts=a.artifacts if i == 0 else None) for i, s in enumerate(a.seeds)]
        doc = write_report(runs, a.out)
        for m, s in doc["summary"].items():
            print(f"{m:15s} NDCG@10 {s['ndcg@10']['mean']:.4f}  (min {s['ndcg@10']['min']:.4f}, max {s['ndcg@10']['max']:.4f})")
        print(f"report: {a.out / 'report.md'}")
    elif a.cmd == "simulate":
        cat = build_catalog(a.seed)
        cat.save(a.artifacts)
        lines, truth = simulate(cat, SearchIndex(cat.products, vocabulary()), a.impressions,
                                swap_rate=a.swap_rate, seed=a.seed)
        a.events.parent.mkdir(parents=True, exist_ok=True)
        a.events.write_text("\n".join(lines) + "\n")
        (a.artifacts / "sim_truth.json").write_text(json.dumps(truth, indent=1))
        print(f"wrote {len(lines)} events to {a.events}; catalog to {a.artifacts}")
    elif a.cmd == "ingest":
        a.db.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(ingest_file(a.events, a.db), indent=1, sort_keys=True))
    elif a.cmd == "train":
        products, vocab = load_products(a.artifacts)
        index = SearchIndex(products, vocab)
        conn = sqlite3.connect(a.db)
        try:
            queries = {q for (q,) in conn.execute("SELECT DISTINCT query FROM impressions")}
            prop = estimate_propensities(conn, queries) if a.method == "ipw" else None
            pairs = click_pairs(conn, index, queries, prop.smoothed if prop else None)
        finally:
            conn.close()
        w = fit_pairwise(pairs)
        extra = {"propensity": prop.smoothed} if prop else {}
        save_model(a.artifacts / "model.json", w, "ipw_estimated" if prop else "naive_clicks", extra)
        print(json.dumps({"pairs": len(pairs.X), "clicks": pairs.n_clicks, **extra}, indent=1))
    elif a.cmd == "search":
        products, vocab = load_products(a.artifacts)
        index = SearchIndex(products, vocab)
        w, _ = load_model(a.artifacts / "model.json")
        by_id = {p.id: p for p in products}
        for i, pid in enumerate(rank(index.candidates(normalize_query(a.query)), w)[: a.k], start=1):
            p = by_id[pid]
            print(f"{i:2d}. {pid}  {p.title:55s} rating {p.rating}  ${p.price}")
    elif a.cmd == "serve":
        import uvicorn

        from .service import create_app
        uvicorn.run(create_app(a.artifacts, a.events, explore_rate=a.explore_rate), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
