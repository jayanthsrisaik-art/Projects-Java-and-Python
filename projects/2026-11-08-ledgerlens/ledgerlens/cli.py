"""Command line entry point: ``python -m ledgerlens <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import dedupe, evaluate
from .model import fit_with_threshold
from .normalize import normalize
from .synth import generate, read_csv, write_csv


def cmd_generate(a) -> None:
    rows, merchants = generate(seed=a.seed, n_accounts=a.accounts)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, a.out, with_labels=not a.raw)
    print(f"wrote {len(rows)} rows ({len(merchants)} merchants) to {a.out}")


def train_model(rows: list[dict]):
    rows = dedupe(rows)
    return fit_with_threshold([r["descriptor"] for r in rows], [r["amount"] for r in rows],
                              [r["category"] for r in rows],
                              [normalize(r["descriptor"]).key for r in rows])


def cmd_train(a) -> None:
    model = train_model(read_csv(a.data))
    model.save(a.out)
    print(f"saved model to {a.out} (abstain threshold {model.threshold:.4f})")


def cmd_evaluate(a) -> None:
    report = evaluate(read_csv(a.data), seed=a.seed, folds=a.folds)
    text = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(text)


def cmd_ingest(a) -> None:
    from .model import TxnClassifier
    from .store import Ledger

    ledger = Ledger(a.db, TxnClassifier.load(a.model))
    res = ledger.ingest(read_csv(a.csv))
    print(json.dumps({k: v for k, v in res.items() if k != "transactions"}))
    if a.show:
        for t in res["transactions"]:
            print(f"{t['txn_id']:>8} {t['amount']:>9.2f} {t['source']:<6} {str(t['category']):<13} "
                  f"({t['suggested']} {t['confidence']:.2f}) {t['descriptor']}")


def run_demo(model, sample_csv: str, db_path: str = ":memory:") -> dict:
    """Ingest the sample, re-deliver it (idempotency), apply one user correction, report."""
    from .store import Ledger

    ledger = Ledger(db_path, model)
    rows = read_csv(sample_csv)
    first = ledger.ingest(rows)
    again = ledger.ingest(rows)
    review = [t for t in first["transactions"] if t["source"] == "review"]
    correction = None
    if review:
        target = review[0]
        # the generator's hidden truth for the sample is not available here, so the demo
        # "user" simply accepts the model's suggestion for the first review item
        correction = ledger.correct(target["txn_id"], target["suggested"], apply_to_merchant=True)
    final = ledger.list(limit=1000)
    months = sorted({r["posted_date"][:7] for r in rows})
    return {
        "first_ingest": {k: v for k, v in first.items() if k != "transactions"},
        "redelivery": {k: v for k, v in again.items() if k != "transactions"},
        "correction": correction,
        "source_counts": {s: sum(t["source"] == s for t in final) for s in ("user", "rule", "model", "review")},
        "transactions": [{k: t[k] for k in ("txn_id", "merchant_key", "processor", "suggested", "confidence",
                                            "category", "source")} for t in final],
        "monthly_summary": {m: ledger.monthly_summary(m) for m in months},
        "subscriptions": [r.__dict__ for r in ledger.recurring()],
        "threshold": round(model.threshold, 4),
    }


def cmd_demo(a) -> None:
    from .model import TxnClassifier

    model = TxnClassifier.load(a.model)
    out = run_demo(model, a.csv)
    text = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(json.dumps({k: out[k] for k in ("first_ingest", "redelivery", "correction", "source_counts",
                                           "subscriptions")}, indent=2))


def cmd_serve(a) -> None:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(a.db, model_path=a.model), host=a.host, port=a.port)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ledgerlens")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write a synthetic transaction CSV")
    g.add_argument("--out", default="data/transactions.csv")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--accounts", type=int, default=60)
    g.add_argument("--raw", action="store_true", help="omit label columns")
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("train", help="train the classifier + abstain threshold")
    t.add_argument("--data", default="data/transactions.csv")
    t.add_argument("--out", default="artifacts/model.pkl")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", help="grouped CV, baselines, leakage comparison, recurring")
    e.add_argument("--data", default="data/transactions.csv")
    e.add_argument("--out", default="")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--folds", type=int, default=5)
    e.set_defaults(fn=cmd_evaluate)

    i = sub.add_parser("ingest", help="ingest a raw CSV into the SQLite ledger")
    i.add_argument("--db", default="ledger.db")
    i.add_argument("--model", default="artifacts/model.pkl")
    i.add_argument("--csv", required=True)
    i.add_argument("--show", action="store_true")
    i.set_defaults(fn=cmd_ingest)

    d = sub.add_parser("demo", help="ingest the sample, correct one review item, print results")
    d.add_argument("--model", default="artifacts/model.pkl")
    d.add_argument("--csv", default="samples/transactions_sample.csv")
    d.add_argument("--out", default="")
    d.set_defaults(fn=cmd_demo)

    s = sub.add_parser("serve", help="run the FastAPI service")
    s.add_argument("--db", default="ledger.db")
    s.add_argument("--model", default="artifacts/model.pkl")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
