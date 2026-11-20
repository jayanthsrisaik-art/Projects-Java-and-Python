"""Replay a scripted scenario against a fresh store and print one line per step.

Usage: python -m stockhold.demo samples/scenario.json
The output is deterministic (simulated clock, sequential ids) and is checked
into samples/scenario_expected.txt.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any

from .bench import SimClock
from .store import SequentialIds, Store

START = 1_700_000_000.0
SHOW = ("error", "id", "status", "qty", "available", "limit", "already", "message",
        "amount_cents", "hold_id", "sold", "held")


def _brief(body: dict[str, Any]) -> str:
    return " ".join(f"{k}={body[k]}" for k in SHOW if k in body and body[k] is not None)


def run(scenario: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    with tempfile.TemporaryDirectory() as d:
        clock = SimClock(START)
        store = Store(os.path.join(d, "demo.db"), hold_ttl=scenario["hold_ttl"],
                      clock=clock, ids=SequentialIds())
        store.load_catalog(scenario["catalog"])
        for s in scenario["steps"]:
            clock.t = START + s["t"]
            op = s["op"]
            if op == "reserve":
                out = store.reserve(s["customer"], s["sku"], s["qty"], s["key"])
            elif op == "confirm":
                out = store.confirm(s["customer"], s["hold"], s["token"])
            elif op == "release":
                out = store.release(s["customer"], s["hold"])
            elif op == "product":
                v = store.product(s["sku"])
                assert v is not None
                line = f"t={s['t']:>3} product {s['sku']}: total={v['total']} sold={v['sold']} " \
                       f"held={v['held']} available={v['available']}"
                lines.append(line + (f"  # {s['note']}" if "note" in s else ""))
                continue
            elif op == "invariants":
                problems = store.check_invariants()
                lines.append(f"t={s['t']:>3} invariants: " + ("ok" if not problems else "; ".join(problems)))
                continue
            else:
                raise ValueError(f"unknown op {op!r}")
            who = s["customer"]
            flag = " (replayed)" if out.replayed else ""
            line = f"t={s['t']:>3} {op:<7} {who:<4} -> {out.status}{flag} {_brief(out.body)}"
            lines.append(line + (f"  # {s['note']}" if "note" in s else ""))
        store.close()
    return lines


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    path = args[0] if args else os.path.join(os.path.dirname(__file__), "..", "samples", "scenario.json")
    with open(path) as f:
        print("\n".join(run(json.load(f))))


if __name__ == "__main__":
    main()
