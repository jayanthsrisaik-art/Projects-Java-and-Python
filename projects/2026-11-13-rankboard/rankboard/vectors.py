"""Generate ``samples/key_vectors.json``: (a, b) -> key cases the TS client must match.

    python -m rankboard.vectors > samples/key_vectors.json
"""
from __future__ import annotations

import json
import random

from .fracindex import KeyError_, key_between, keys_sequence
from .store import resolve_drop


def build(seed: int = 2026) -> dict:
    rng = random.Random(seed)
    cases: list[list] = []
    seen: set[tuple] = set()

    def add(a, b):
        if (a, b) in seen:
            return
        seen.add((a, b))
        cases.append([a, b, key_between(a, b)])

    for _ in range(8):  # random insertion histories
        keys: list[str] = []
        for _ in range(60):
            i = rng.randint(0, len(keys))
            a = keys[i - 1] if i > 0 else None
            b = keys[i] if i < len(keys) else None
            add(a, b)
            keys.insert(i, key_between(a, b))
    # long append / prepend runs cross integer length classes (a0 -> b00 ..., Zz -> Y00 ...)
    k = None
    for _ in range(4000):
        nxt = key_between(k, None)
        if len(nxt) != len(k or ""):
            add(k, None)
        k = nxt
    k = None
    for _ in range(4000):
        nxt = key_between(None, k)
        if len(nxt) != len(k or ""):
            add(None, k)
        k = nxt
    # deep hotspot: keys with long fractions
    a, b = "a0", "a1"
    for _ in range(40):
        b = key_between(a, b)
        add(a, b)
    for edge in [("Zz", "a0"), ("a0", "a0V"), ("zzzzzzzzzzzzzzzzzzzzzzzzzzy", None),
                 ("az", "b00"), (None, "A00000000000000000000000001")]:
        add(*edge)

    invalid = []
    for a, b in [("a0", "a0"), ("a1", "a0"), ("a00", None), ("b0", None), ("a$", None),
                 ("", None), ("A" + "0" * 26, None)]:
        try:
            key_between(a, b)
        except KeyError_:
            invalid.append([a, b])
        else:  # pragma: no cover - would mean the generator is wrong
            raise AssertionError((a, b))
    placements = []
    for _ in range(60):  # drop intents against random columns (incl. stale anchors)
        keys: list[str] = []
        for _ in range(rng.randint(0, 8)):
            i = rng.randint(0, len(keys))
            keys.insert(i, key_between(keys[i - 1] if i else None,
                                       keys[i] if i < len(keys) else None))
        column = [[f"c{j}", k] for j, k in enumerate(keys)]
        ids = [c[0] for c in column] + ["gone", None]
        moving = rng.choice([c[0] for c in column] + [None, None]) if column else None
        after, before = rng.choice(ids), rng.choice(ids)
        if moving is not None and moving in (after, before):
            continue
        p = resolve_drop([tuple(c) for c in column], after, before, moving)
        placements.append({"column": column, "moving": moving, "after": after,
                           "before": before, "rank": p.rank, "resolved": p.resolved})
    return {"between": cases, "invalid": invalid, "sequence_10": keys_sequence(10),
            "placements": placements}


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
