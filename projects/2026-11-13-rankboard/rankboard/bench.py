"""Compare ways of storing card order in one column, on deterministic workloads.

Measured per strategy and workload:
* rows written per operation (what a database has to update),
* key size (mean / max characters of the stored order value at the end),
* ``breaks``: operations after which the strategy could no longer produce a
  value strictly between two neighbours without renumbering (float precision).
Every run also checks that the stored values sort in exactly the intended order.

    python -m rankboard.bench            # prints a Markdown table
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import fracindex

# ---------------------------------------------------------------- workloads
# An op is (kind, index): "insert" at index i of the current list, or
# "move" (remove a random card, insert at index i). Indices are drawn here so
# every strategy sees exactly the same sequence.


def make_workload(name: str, n: int, seed: int) -> list[tuple[str, int, int]]:
    rng = random.Random(f"{name}:{seed}")
    ops: list[tuple[str, int, int]] = []
    size = 0
    for _ in range(n):
        if name == "append":
            ops.append(("insert", -1, size)); size += 1
        elif name == "prepend":
            ops.append(("insert", -1, 0)); size += 1
        elif name == "hotspot":  # triage: always "just below the first card"
            ops.append(("insert", -1, min(1, size))); size += 1
        elif name == "mixed":
            r = rng.random()
            if size < 2 or r < 0.55:
                ops.append(("insert", -1, size)); size += 1
            elif r < 0.80:
                ops.append(("insert", -1, rng.randint(0, size))); size += 1
            else:
                src = rng.randrange(size)
                ops.append(("move", src, rng.randint(0, size - 1)))
        else:
            raise ValueError(name)
    return ops


WORKLOADS = {"append": 1000, "prepend": 1000, "hotspot": 1000, "mixed": 2000}

# --------------------------------------------------------------- strategies


@dataclass
class Result:
    rows_written: int = 0
    renumbers: int = 0
    breaks_at: Optional[int] = None
    values: list = field(default_factory=list)


def run_dense(ops) -> Result:
    """position = 0..n-1; every card after the insertion point is rewritten."""
    res = Result()
    order: list[int] = []
    for i, (kind, src, dst) in enumerate(ops):
        before = {c: p for p, c in enumerate(order)}
        if kind == "move":
            card = order.pop(src)
        else:
            card = i
        order.insert(dst, card)
        changed = sum(1 for p, c in enumerate(order) if before.get(c) != p)
        res.rows_written += changed
    res.values = list(range(len(order)))
    return res


def run_gapped(ops, gap: int = 1024) -> Result:
    """Integers spaced ``gap`` apart, midpoint on insert, renumber when no room."""
    res = Result()
    vals: list[int] = []
    for kind, src, dst in ops:
        if kind == "move":
            vals.pop(src)
        lo = vals[dst - 1] if dst > 0 else None
        hi = vals[dst] if dst < len(vals) else None
        if lo is None and hi is None:
            v = gap
        elif hi is None:
            v = lo + gap
        elif lo is None:
            v = hi - gap
        else:
            v = (lo + hi) // 2
            if v == lo:
                vals.insert(dst, None)
                vals = [gap * (k + 1) for k in range(len(vals))]
                res.renumbers += 1
                res.rows_written += len(vals)
                continue
        vals.insert(dst, v)
        res.rows_written += 1
    res.values = vals
    return res


def run_float(ops) -> Result:
    """float64 midpoints, no renumbering: records when precision runs out."""
    res = Result()
    vals: list[float] = []
    for i, (kind, src, dst) in enumerate(ops):
        if kind == "move":
            vals.pop(src)
        lo = vals[dst - 1] if dst > 0 else None
        hi = vals[dst] if dst < len(vals) else None
        if lo is None and hi is None:
            v = 1.0
        elif hi is None:
            v = lo + 1.0
        elif lo is None:
            v = hi - 1.0
        else:
            v = (lo + hi) / 2
            if (v <= lo or v >= hi) and res.breaks_at is None:
                res.breaks_at = i + 1
        vals.insert(dst, v)
        res.rows_written += 1
    res.values = vals
    return res


def _run_keys(ops, between: Callable, max_len: Optional[int]) -> Result:
    res = Result()
    keys: list[str] = []
    for kind, src, dst in ops:
        if kind == "move":
            keys.pop(src)
        lo = keys[dst - 1] if dst > 0 else None
        hi = keys[dst] if dst < len(keys) else None
        k = between(lo, hi)
        keys.insert(dst, k)
        res.rows_written += 1
        if max_len is not None and len(k) > max_len:
            keys = fracindex.keys_sequence(len(keys))
            res.renumbers += 1
            res.rows_written += len(keys)
    res.values = keys
    return res


STRATEGIES: dict[str, Callable] = {
    "dense integer index": run_dense,
    "gapped integers (1024)": run_gapped,
    "float64 midpoint": run_float,
    "fraction-only keys": lambda ops: _run_keys(ops, fracindex.fraction_only_between, None),
    "rankboard keys": lambda ops: _run_keys(ops, fracindex.key_between, None),
    "rankboard keys + rebalance@48": lambda ops: _run_keys(ops, fracindex.key_between, 48),
}


def _size(v) -> int:
    return len(v) if isinstance(v, str) else len(repr(v))


def run(seed: int = 7) -> list[dict]:
    rows = []
    for wname, n in WORKLOADS.items():
        ops = make_workload(wname, n, seed)
        for sname, fn in STRATEGIES.items():
            r = fn(ops)
            ordered = all(r.values[i] < r.values[i + 1] for i in range(len(r.values) - 1))
            sizes = [_size(v) for v in r.values] or [0]
            rows.append({
                "workload": wname, "ops": n, "strategy": sname,
                "rows_per_op": r.rows_written / n,
                "renumbers": r.renumbers,
                "mean_len": sum(sizes) / len(sizes), "max_len": max(sizes),
                "breaks_at": r.breaks_at, "ordered": ordered,
            })
    return rows


def to_markdown(rows: list[dict], seed: int) -> str:
    out = [f"# Ordering strategies, one column (seed {seed})", "",
           "| workload | strategy | rows written / op | renumbers | mean key chars (final) | max key chars (final) | precision lost at op | order correct |",
           "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        if r["strategy"] in ("dense integer index", "gapped integers (1024)"):
            mean, mx = "int", "int"
        elif r["strategy"] == "float64 midpoint":
            mean, mx = "8 bytes", "8 bytes"
        else:
            mean, mx = f"{r['mean_len']:.1f}", str(r["max_len"])
        out.append(
            f"| {r['workload']} ({r['ops']}) | {r['strategy']} | {r['rows_per_op']:.2f} | "
            f"{r['renumbers']} | {mean} | {mx} | {r['breaks_at'] or '-'} | "
            f"{'yes' if r['ordered'] else 'NO'} |")
    return "\n".join(out) + "\n"


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    print(to_markdown(run(args.seed), args.seed), end="")


if __name__ == "__main__":
    main()
