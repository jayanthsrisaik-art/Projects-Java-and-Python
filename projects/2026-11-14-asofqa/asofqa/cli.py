"""Command line: generate a corpus, ask questions, run the evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .corpus import DEFAULT_REF_DATE, write_corpus
from .evaluate import (DEV_SEEDS, STRATEGIES, TEST_SEEDS, ablation_history_filter, dump_rows,
                       render_report, run_split, summarise)
from .index import Corpus
from .pipeline import Answerer
from .reader import make_reader


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="asofqa", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write a synthetic policy corpus + questions")
    g.add_argument("out", type=Path)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--split", choices=("dev", "test"), default="dev")
    g.add_argument("--ref-date", type=date.fromisoformat, default=DEFAULT_REF_DATE)

    a = sub.add_parser("ask", help="answer questions against a directory of policy documents")
    a.add_argument("docs", type=Path)
    a.add_argument("question", nargs="*", help="questions (or read one per line from stdin)")
    a.add_argument("--ref-date", type=date.fromisoformat, default=DEFAULT_REF_DATE,
                   help="what 'today' means (default %(default)s, fixed for reproducibility)")
    a.add_argument("--strategy", choices=STRATEGIES, default="asof")
    a.add_argument("--reader", choices=("extractive", "claude"), default="extractive")
    a.add_argument("--json", action="store_true")

    e = sub.add_parser("eval", help="evaluate every strategy on generated dev/test corpora")
    e.add_argument("--split", choices=("dev", "test", "both"), default="both")
    e.add_argument("--out", type=Path, default=None, help="directory for report.md, metrics.json, predictions")

    # argparse cannot interleave options with a nargs="*" positional, so collect stray question words.
    args, extra = p.parse_known_args(argv)
    if extra:
        if args.cmd != "ask" or any(x.startswith("-") for x in extra):
            p.error(f"unrecognized arguments: {' '.join(extra)}")
        args.question = list(args.question) + extra
    if args.cmd == "generate":
        nd, nq = write_corpus(args.out, args.seed, args.split, args.ref_date)
        print(f"wrote {nd} documents and {nq} questions to {args.out}")
        return 0
    if args.cmd == "ask":
        corpus = Corpus.from_dir(args.docs)
        ans = Answerer(corpus, args.ref_date, args.strategy, make_reader(args.reader))
        qs = args.question or [ln.strip() for ln in sys.stdin if ln.strip()]
        for q in qs:
            res = ans.answer(q)
            if args.json:
                print(json.dumps({"question": q, **res.to_dict()}))
            else:
                print(f"Q: {q}\nA: {res.text}\n")
        return 0
    # eval
    splits = [("dev", DEV_SEEDS), ("test", TEST_SEEDS)] if args.split == "both" else \
        [(args.split, DEV_SEEDS if args.split == "dev" else TEST_SEEDS)]
    report, metrics = ["# asofqa evaluation", "", f"reference date: {DEFAULT_REF_DATE}", ""], {}
    for split, seeds in splits:
        rows = run_split(seeds, split)
        summ = summarise(rows)
        abl = ablation_history_filter(seeds, split)
        metrics[split] = {"seeds": list(seeds), "strategies": summ, "ablation_history_filter": abl}
        report.append(render_report(summ, f"{split} split (seeds {', '.join(map(str, seeds))})"))
        report.append("Reader ablation (asof accuracy): " + ", ".join(f"{k} {v:.3f}" for k, v in abl.items()) + "\n")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            dump_rows(rows, args.out / f"predictions_{split}.jsonl")
    text = "\n".join(report)
    print(text)
    if args.out:
        (args.out / "report.md").write_text(text)
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
