"""Command line: python -m sqlcritic {build-db,golden,ask,eval,stress,score}."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import evaluate, golden, stress
from .db import build_db, connect_readonly
from .execute import run_sql
from .graph import MODES, Pipeline
from .writers import WriterError, make_writer

DEFAULT_DB = Path(__file__).resolve().parents[1] / ".cache" / "warehouse.db"


def _con(path: str | None):
    p = Path(path) if path else DEFAULT_DB
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        build_db(p)
    return connect_readonly(p)


def cmd_build_db(a) -> int:
    print(build_db(a.path))
    return 0


def cmd_golden(a) -> int:
    n = golden.export_jsonl(a.out)
    print(f"wrote {n} examples to {a.out}")
    return 0


def cmd_ask(a) -> int:
    pipe = Pipeline(make_writer(a.writer), _con(a.db), a.mode, a.max_attempts)
    print(json.dumps(pipe.ask(a.question).as_dict(), indent=2, default=str))
    return 0


def cmd_eval(a) -> int:
    con = _con(a.db)
    examples = golden.load(a.split)
    report = evaluate.run_eval(examples, lambda: make_writer(a.writer), con, a.modes, a.max_attempts)
    md = evaluate.to_markdown(report, f"sqlcritic eval - writer={a.writer}, split={a.split}")
    if a.out_dir:
        evaluate.save(report, a.out_dir, f"eval_{a.writer}", f"sqlcritic eval - writer={a.writer}, split={a.split}")
    print(md, end="")
    if a.baseline:
        problems = evaluate.check_baseline(report, json.loads(Path(a.baseline).read_text()), a.max_drop)
        for p in problems:
            print(f"REGRESSION: {p}", file=sys.stderr)
        if problems:
            return 1
        print("baseline check passed", file=sys.stderr)
    return 0


def cmd_stress(a) -> int:
    rep = stress.run_stress(golden.load("all"), _con(a.db))
    md = stress.to_markdown(rep)
    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "stress.json").write_text(json.dumps(rep, indent=2) + "\n")
        (out / "stress.md").write_text(md)
    print(md, end="")
    return 0


def cmd_score(a) -> int:
    """Score any system's predictions ({"id", "sql"} per line) against the golden set."""
    con = _con(a.db)
    by_id = {e.id: e for e in golden.generate()}
    rows, correct = [], 0
    for line in Path(a.predictions).read_text().splitlines():
        if not line.strip():
            continue
        pred = json.loads(line)
        ex = by_id.get(pred["id"])
        if ex is None:
            rows.append({"id": pred["id"], "correct": False, "error": "unknown id"})
            continue
        res = run_sql(con, pred["sql"])
        ok = res.ok and evaluate.results_match(ex, res.rows, len(res.columns), con)
        correct += ok
        rows.append({"id": ex.id, "correct": ok, **({"error": res.error} if not res.ok else {})})
    out = {"n": len(rows), "correct": correct,
           "accuracy": round(correct / len(rows), 4) if rows else None, "results": rows}
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sqlcritic", description=__doc__)
    p.add_argument("--db", default=None, help=f"SQLite warehouse (built on first use; default {DEFAULT_DB})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build-db", help="build the synthetic warehouse")
    s.add_argument("path"); s.set_defaults(fn=cmd_build_db)

    s = sub.add_parser("golden", help="export the golden set as JSONL")
    s.add_argument("--out", default="golden.jsonl"); s.set_defaults(fn=cmd_golden)

    def common(sp):
        sp.add_argument("--writer", choices=["heuristic", "claude"], default="heuristic")
        sp.add_argument("--max-attempts", type=int, default=3)

    s = sub.add_parser("ask", help="answer one question through the writer/critic graph")
    s.add_argument("question"); common(s)
    s.add_argument("--mode", choices=MODES, default="critic"); s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("eval", help="run the golden-set evaluation")
    common(s)
    s.add_argument("--split", choices=["dev", "test", "all"], default="all")
    s.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    s.add_argument("--out-dir", default=None)
    s.add_argument("--baseline", default=None, help="eval JSON to gate against (exit 1 on regression)")
    s.add_argument("--max-drop", type=float, default=0.0)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("stress", help="mutation test of the critic on gold SQL")
    s.add_argument("--out-dir", default=None); s.set_defaults(fn=cmd_stress)

    s = sub.add_parser("score", help="score a predictions JSONL ({id, sql}) against the golden set")
    s.add_argument("predictions"); s.set_defaults(fn=cmd_score)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except WriterError as exc:
        print(f"sqlcritic: writer error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
