"""Run edit/sync scenarios against the CRDT and the naive baseline."""

from __future__ import annotations

import random
from typing import Dict, List

from .doc import CRDTError, Replica
from .naive import NaiveReplica
from .oracle import oracle_text

ENGINES = {"crdt": Replica, "naive": NaiveReplica}


class ScenarioError(ValueError):
    pass


def _deliver(src, dst, shuffle_seed=None, duplicate=False, limit=None):
    ops = src.ops_since(dst.version())
    if limit is not None:
        ops = ops[:limit]
    if duplicate:
        ops = ops + ops
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(ops)
    dst.receive(ops)
    return len(ops)


def execute(scenario: dict, engine: str):
    """Run the steps; return ``(replicas_by_name, errors)``."""
    names = scenario.get("replicas")
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ScenarioError("'replicas' must be a non-empty list of unique names")
    reps = {n: ENGINES[engine](n) for n in names}
    errors: List[str] = []

    def get(name):
        if name not in reps:
            raise ScenarioError(f"unknown replica {name!r}")
        return reps[name]

    for i, step in enumerate(scenario.get("steps", [])):
        if "edit" in step:
            r = get(step["edit"])
            try:
                if "insert" in step:
                    pos, text = step["insert"]
                    r.insert(min(pos, len(r)), text)
                elif "delete" in step:
                    pos, length = step["delete"]
                    pos = min(pos, len(r))
                    r.delete(pos, min(length, len(r) - pos))
                else:
                    raise ScenarioError(f"step {i}: edit needs 'insert' or 'delete'")
            except CRDTError as exc:  # pragma: no cover - positions are clamped
                errors.append(f"step {i}: {exc}")
        elif "sync" in step:
            a, b = (get(n) for n in step["sync"])
            _deliver(a, b)
            _deliver(b, a)
        elif "send" in step:
            s = step["send"]
            _deliver(get(s["from"]), get(s["to"]), s.get("shuffle_seed"),
                     s.get("duplicate", False), s.get("limit"))
        else:
            raise ScenarioError(f"step {i}: unknown step {sorted(step)}")
    return reps, errors


def run_engine(scenario: dict, engine: str) -> dict:
    reps, errors = execute(scenario, engine)
    texts = {n: r.text() for n, r in reps.items()}
    return {
        "texts": texts,
        "converged": len(set(texts.values())) == 1,
        "pending": {n: r.pending_count() for n, r in reps.items()},
        "errors": errors,
    }


def run_scenario(scenario: dict) -> Dict[str, dict]:
    return {engine: run_engine(scenario, engine) for engine in ENGINES}


def fuzz(replicas: int = 3, steps: int = 300, seed: int = 0) -> dict:
    """Random concurrent editing with lossy, reordered, duplicated delivery.

    Every replica edits its own copy; partial deliveries move a random prefix
    of the missing ops, shuffled and sometimes duplicated.  At the end all
    replicas fully sync.  The CRDT must converge and match the oracle; the
    naive engine is run on the identical schedule for comparison.
    """
    rng = random.Random(seed)
    names = [f"r{i}" for i in range(replicas)]
    schedule = []
    # Build the schedule against a CRDT run so edit positions are valid there;
    # the naive run clamps them.
    for _ in range(steps):
        roll = rng.random()
        who = rng.choice(names)
        if roll < 0.45:
            schedule.append({"edit": who, "insert": [rng.randint(0, 40),
                             "".join(rng.choice("abcdefgh ") for _ in range(rng.randint(1, 4)))]})
        elif roll < 0.65:
            schedule.append({"edit": who, "delete": [rng.randint(0, 40), rng.randint(1, 3)]})
        else:
            dst = rng.choice([n for n in names if n != who] or names)
            schedule.append({"send": {"from": who, "to": dst, "shuffle_seed": rng.randint(0, 10**6),
                                      "duplicate": rng.random() < 0.3,
                                      "limit": rng.randint(1, 12)}})
    for a in names:
        for b in names:
            if a != b:
                schedule.append({"sync": [a, b]})
    for a in names:  # second round so every op reaches everyone
        for b in names:
            if a != b:
                schedule.append({"sync": [a, b]})
    scenario = {"replicas": names, "steps": schedule}
    result = run_scenario(scenario)

    # Oracle check needs the op log, not only the text.
    reps, _ = execute(scenario, "crdt")
    oracle = oracle_text(reps[names[0]].snapshot_ops())
    return {
        "seed": seed,
        "replicas": replicas,
        "steps": steps,
        "crdt_converged": result["crdt"]["converged"],
        "crdt_matches_oracle": all(t == oracle for t in result["crdt"]["texts"].values()),
        "crdt_pending": sum(result["crdt"]["pending"].values()),
        "naive_converged": result["naive"]["converged"],
        "final_length": len(oracle),
    }
