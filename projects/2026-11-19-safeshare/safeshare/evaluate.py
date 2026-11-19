"""Measure what a release leaks and what it still lets analysts do.

Privacy - a *prosecutor* linkage attack: the attacker holds a public roll
(name, dob, sex, zip) and knows each target is in the extract.  For a target,
the candidate set is every released subject row whose generalised cells
contain the target's (dob, sex, zip).  If the true row is among ``m``
candidates the attacker picks it with probability ``1/m``.

* ``unique_reid``   - share of targets with exactly one candidate, the right one
* ``mean_risk``     - mean of 1/m (0 if the target was suppressed)
* ``risk_gt_0.2``   - share of targets with risk above 1 in 5
* ``attr_disclosure`` - share of targets whose candidates all share the true
  diagnosis (learned with certainty even without picking the row)
* ``false_unique``  - non-patient roll entries that match exactly one row
  (a confident but wrong re-identification)

Utility - average NCP from the release, plus a random count-query workload
over the QIs answered from the generalised cells under a uniform-spread
assumption, reported as relative error against the original data.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from .pseudonym import Pseudonymizer
from .release import read_tables, subject_records


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r)
        return header, [row for row in r if row]


def load_release(release_dir, policy, specs):
    header, rows = _read_csv(Path(release_dir) / f"{policy.subject_table}.csv")
    qcols = [header.index(s.name) for s in specs]
    kcol = header.index(policy.subject_key)
    srule = policy.sensitive_rule()
    scol = header.index(srule.column) if srule else None
    out = []
    for r in rows:
        out.append({"token": r[kcol],
                    "cells": tuple(s.parse(r[c]) for s, c in zip(specs, qcols)),
                    "sens": r[scol] if scol is not None else None})
    return out


class _Index:
    """Candidate lookup: rows bucketed by every zip / sex value they cover."""

    def __init__(self, released, specs):
        self.released = released
        self.specs = specs
        self.anchor = next((d for d, s in enumerate(specs) if s.kind == "zip"), None)
        self.buckets: dict = {}
        if self.anchor is not None:
            zs = specs[self.anchor].zip_values
            for i, r in enumerate(released):
                lo, hi = r["cells"][self.anchor]
                for z in zs:
                    if lo <= z <= hi:
                        self.buckets.setdefault(z, []).append(i)

    def candidates(self, qi):
        pool = (self.buckets.get(qi[self.anchor], []) if self.anchor is not None
                else range(len(self.released)))
        return [i for i in pool
                if all(s.contains(c, v) for s, c, v in
                       zip(self.specs, self.released[i]["cells"], qi))]


def linkage_attack(released, specs, voters_csv, policy, pz: Pseudonymizer,
                   truth_sens: dict) -> dict:
    header, rows = _read_csv(voters_csv)
    pos = {c: header.index(c) for c in header}
    idx = _Index(released, specs)
    targets = uniq = gt = disc = 0
    risk_sum = 0.0
    risk_max = 0.0
    non_patients = false_unique = 0
    for row in rows:
        try:
            qi = tuple(s.encode(row[pos[s.name]]) for s in specs)
        except Exception:
            continue          # roll value outside the encodable domain
        truth = row[pos["_truth_patient_id"]] if "_truth_patient_id" in pos else ""
        cand = idx.candidates(qi)
        if not truth:
            non_patients += 1
            false_unique += len(cand) == 1
            continue
        targets += 1
        tok = pz.token(policy.subject_domain, truth)
        hit = any(released[i]["token"] == tok for i in cand)
        risk = 1 / len(cand) if hit else 0.0
        risk_sum += risk
        risk_max = max(risk_max, risk)
        uniq += hit and len(cand) == 1
        gt += risk > 0.2
        if hit and truth_sens.get(truth) is not None:
            disc += all(released[i]["sens"] == truth_sens[truth] for i in cand)
    t = max(targets, 1)
    return {"targets": targets, "unique_reid": uniq / t, "mean_risk": risk_sum / t,
            "max_risk": risk_max, "risk_gt_0.2": gt / t,
            "attr_disclosure": disc / t,
            "false_unique": false_unique / max(non_patients, 1)}


def _overlap(spec, cell, q) -> float:
    """Fraction of a cell's (uniform) mass inside query predicate q."""
    if spec.kind == "categorical":
        return len(cell & q) / len(cell)
    lo, hi = cell
    if spec.kind == "zip":
        inside = [z for z in spec.zip_values if lo <= z <= hi]
        if not inside:
            return 0.0
        return sum(q[0] <= z <= q[1] for z in inside) / len(inside)
    a, b = max(lo, q[0]), min(hi, q[1])
    return 0.0 if a > b else (b - a + 1) / (hi - lo + 1)


def query_workload(specs, records, n_queries=400, seed=0):
    rng = random.Random(seed)
    queries = []
    for _ in range(n_queries):
        q = []
        for s in specs:
            if s.kind == "categorical":
                vals = sorted(s.domain)
                q.append(frozenset(rng.sample(vals, rng.randint(1, len(vals)))))
            elif s.kind == "zip":
                zs = s.zip_values
                a = rng.randrange(len(zs))
                b = min(len(zs) - 1, a + rng.randint(0, max(1, len(zs) // 4)))
                q.append((zs[a], zs[b]))
            else:
                span = s.hi - s.lo
                width = rng.randint(max(1, span // 20), max(1, span // 3))
                a = rng.randint(s.lo, max(s.lo, s.hi - width))
                q.append((a, a + width))
        queries.append(tuple(q))
    return queries


def query_error(specs, records, released, queries, floor=10) -> dict:
    boxes: dict = {}
    for r in released:
        boxes[r["cells"]] = boxes.get(r["cells"], 0) + 1
    points: dict = {}
    for r in records:
        points[r["qi"]] = points.get(r["qi"], 0) + 1
    errs = []
    for q in queries:
        true = sum(n for qi, n in points.items()
                   if all(s.contains(c, v) for s, c, v in zip(specs, q, qi)))
        est = 0.0
        for cells, n in boxes.items():
            p = float(n)
            for s, c, qc in zip(specs, cells, q):
                p *= _overlap(s, c, qc)
                if p == 0.0:
                    break
            est += p
        errs.append(abs(est - true) / max(true, floor))
    errs.sort()
    return {"queries": len(errs), "median_rel_error": errs[len(errs) // 2],
            "mean_rel_error": sum(errs) / len(errs),
            "p90_rel_error": errs[int(0.9 * (len(errs) - 1))]}


def endpoint_exposure(records, released, specs, pz, policy) -> float:
    """Share of released rows where a *date* QI range is non-degenerate and one
    of its endpoints is a real value held by a member of that row's class (a
    range that starts on someone's exact birth date).  Zip / numeric ranges are
    not counted: with adjacent integer values every cut boundary is a real
    value, and a zip code is already public at that resolution."""
    by_tok = {pz.token(policy.subject_domain, r["key"]): r for r in records}
    members: dict = {}
    for row in released:
        members.setdefault(row["cells"], []).append(by_tok[row["token"]])
    exposed = 0
    for cells, recs in members.items():
        for d, s in enumerate(specs):
            if s.kind != "date":
                continue
            lo, hi = cells[d]
            vals = {r["qi"][d] for r in recs}
            if lo != hi and (lo in vals or hi in vals):
                exposed += len(recs)
                break
    return exposed / max(len(released), 1)


def evaluate(original_dir, release_dir, voters_csv, policy, key: bytes,
             n_queries=400, seed=0) -> dict:
    tables = read_tables(original_dir, policy)
    specs, records = subject_records(tables, policy)
    released = load_release(release_dir, policy, specs)
    pz = Pseudonymizer(key, policy.max_date_shift_days)
    truth_sens = {r["key"]: r["sens"] for r in records}
    attack = linkage_attack(released, specs, voters_csv, policy, pz, truth_sens)
    util = query_error(specs, records, released,
                       query_workload(specs, records, n_queries, seed))
    attack["endpoint_exposure"] = endpoint_exposure(records, released, specs, pz,
                                                    policy)
    return {"attack": attack, "utility": util, "released_rows": len(released)}
