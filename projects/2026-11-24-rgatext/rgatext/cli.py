"""Command line: simulate scenarios, fuzz convergence, benchmark, merge op logs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

from .doc import CRDTError, Replica
from .sim import ScenarioError, fuzz, run_scenario


def _read_jsonl(path: str) -> List[dict]:
    ops = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                try:
                    ops.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise CRDTError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from None
    return ops


def cmd_simulate(args) -> int:
    with open(args.scenario, encoding="utf-8") as fh:
        scenario = json.load(fh)
    result = run_scenario(scenario)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["crdt"]["converged"] else 1


def cmd_fuzz(args) -> int:
    rows = [fuzz(args.replicas, args.steps, seed) for seed in range(args.seed, args.seed + args.runs)]
    summary = {
        "runs": len(rows),
        "replicas": args.replicas,
        "steps_per_run": args.steps,
        "crdt_converged": sum(r["crdt_converged"] for r in rows),
        "crdt_matches_oracle": sum(r["crdt_matches_oracle"] for r in rows),
        "crdt_stuck_pending_ops": sum(r["crdt_pending"] for r in rows),
        "naive_converged": sum(r["naive_converged"] for r in rows),
    }
    print(json.dumps(summary, indent=2))
    ok = summary["crdt_converged"] == summary["crdt_matches_oracle"] == len(rows)
    return 0 if ok and not summary["crdt_stuck_pending_ops"] else 1


def cmd_bench(args) -> int:
    a, b = Replica("a"), Replica("b")
    t0 = time.perf_counter()
    ops = []
    for i in range(args.chars):
        # append-heavy typing with an occasional mid-document edit
        pos = len(a) if i % 10 else len(a) // 2
        ops.extend(a.insert(pos, "x"))
    t1 = time.perf_counter()
    b.receive(reversed(ops))  # worst case: every op arrives before its parent
    t2 = time.perf_counter()
    assert a.text() == b.text()
    print(json.dumps({
        "chars": args.chars,
        "local_insert_ms": round((t1 - t0) * 1000, 1),
        "remote_reversed_receive_ms": round((t2 - t1) * 1000, 1),
        "converged": a.text() == b.text(),
    }, indent=2))
    return 0


def cmd_merge(args) -> int:
    """Merge op-log files (JSONL) from several replicas and print the text."""
    r = Replica(args.as_replica)
    for path in args.logs:
        r.receive(_read_jsonl(path))
    if r.pending_count():
        print(f"warning: {r.pending_count()} ops still missing dependencies", file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for op in r.snapshot_ops():
                fh.write(json.dumps(op, separators=(",", ":")) + "\n")
    sys.stdout.write(r.text())
    if not r.text().endswith("\n"):
        sys.stdout.write("\n")
    return 0 if not r.pending_count() else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rgatext", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="run a JSON scenario on the CRDT and the naive baseline")
    s.add_argument("scenario")
    s.set_defaults(func=cmd_simulate)
    f = sub.add_parser("fuzz", help="random concurrent edits with lossy/reordered delivery")
    f.add_argument("--replicas", type=int, default=3)
    f.add_argument("--steps", type=int, default=300)
    f.add_argument("--runs", type=int, default=50)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=cmd_fuzz)
    b = sub.add_parser("bench", help="time local inserts and worst-case remote integration")
    b.add_argument("--chars", type=int, default=5000)
    b.set_defaults(func=cmd_bench)
    m = sub.add_parser("merge", help="merge JSONL op logs and print the document")
    m.add_argument("logs", nargs="+")
    m.add_argument("--as-replica", default="merge")
    m.add_argument("--out", help="write the merged op log (JSONL) here")
    m.set_defaults(func=cmd_merge)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CRDTError, ScenarioError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
