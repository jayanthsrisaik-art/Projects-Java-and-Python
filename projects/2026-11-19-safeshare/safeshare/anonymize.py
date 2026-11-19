"""k-anonymity / distinct l-diversity over the subject table's quasi-identifiers.

Two algorithms share one interface - ``(records, qis, k, l) -> Result``:

* ``mondrian``: strict multidimensional Mondrian (LeFevre et al., 2006).  Top
  down, it cuts the widest (normalised) dimension at the allowable point
  closest to the median, where *allowable* means both halves keep >= k
  records and >= l distinct sensitive values.  Cells are the tight bounding
  box of each final partition, so nothing is suppressed.
* ``full_domain``: the baseline - one generalisation level per column for the
  whole table (e.g. birth *year* + 3-digit zip everywhere), chosen by
  exhaustive search over the hierarchy lattice for minimum information loss,
  suppressing failing classes up to a row budget (Samarati/Incognito style).

Records are dicts ``{"qi": tuple, "sens": str | None}``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .qi import QISpec


class Infeasible(ValueError):
    pass


@dataclass
class Result:
    method: str
    cells: list            # per record: tuple of cells, or None if suppressed
    classes: list = field(default_factory=list)   # list of record-index lists
    level: tuple | None = None                    # full-domain: chosen node

    @property
    def suppressed(self) -> list[int]:
        return [i for i, c in enumerate(self.cells) if c is None]


def _check_root(records, k: int, l: int) -> None:
    if len(records) < k:
        raise Infeasible(f"only {len(records)} subjects; cannot reach k={k}")
    if l > 1:
        distinct = {r["sens"] for r in records}
        if len(distinct) < l:
            raise Infeasible(f"sensitive column has {len(distinct)} distinct values; "
                             f"cannot reach l={l}")


# --------------------------------------------------------------- Mondrian
def _widths(idx, records, qis):
    out = []
    for d, q in enumerate(qis):
        vals = [records[i]["qi"][d] for i in idx]
        if q.kind == "categorical":
            w = 0.0 if len(q.domain) <= 1 else (len(set(vals)) - 1) / (len(q.domain) - 1)
        else:
            span = q.hi - q.lo
            w = 0.0 if span <= 0 else (max(vals) - min(vals)) / span
        out.append(w)
    return out


def _ordered_cut(idx, records, d, k, l):
    """Best allowable threshold cut on ordered dimension d (or None)."""
    by_val: dict = {}
    for i in idx:
        by_val.setdefault(records[i]["qi"][d], []).append(i)
    vals = sorted(by_val)
    if len(vals) < 2:
        return None
    n = len(idx)
    # suffix distinct-sensitive counts for the right-hand side
    suffix = [0] * (len(vals) + 1)
    if l > 1:
        seen: set = set()
        for j in range(len(vals) - 1, -1, -1):
            seen.update(records[i]["sens"] for i in by_val[vals[j]])
            suffix[j] = len(seen)
    best = None
    left_n = 0
    left_sens: set = set()
    for j in range(len(vals) - 1):        # cut between vals[j] and vals[j+1]
        group = by_val[vals[j]]
        left_n += len(group)
        if l > 1:
            left_sens.update(records[i]["sens"] for i in group)
        right_n = n - left_n
        if left_n < k or right_n < k:
            continue
        if l > 1 and (len(left_sens) < l or suffix[j + 1] < l):
            continue
        score = abs(left_n - n / 2)
        if best is None or score < best[0]:
            best = (score, j)
    if best is None:
        return None
    cut = vals[best[1]]
    boundary = (cut + vals[best[1] + 1]) // 2    # between two observed values
    left = [i for i in idx if records[i]["qi"][d] <= cut]
    right = [i for i in idx if records[i]["qi"][d] > cut]
    return left, right, boundary


def _categorical_cut(idx, records, d, k, l):
    groups: dict = {}
    for i in idx:
        groups.setdefault(records[i]["qi"][d], []).append(i)
    if len(groups) < 2:
        return None
    left: list = []
    right: list = []
    for _, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        (left if len(left) <= len(right) else right).extend(members)
    for side in (left, right):
        if len(side) < k:
            return None
        if l > 1 and len({records[i]["sens"] for i in side}) < l:
            return None
    return left, right, None


def mondrian(records, qis: list[QISpec], k: int, l: int = 1,
             tight: bool = False) -> Result:
    """``tight=False`` (default) releases each partition's *cut region*, whose
    inner boundaries fall midway between observed values; ``tight=True``
    releases the min/max bounding box, which is narrower but publishes real
    values (e.g. two people's exact birth dates) as range endpoints."""
    _check_root(records, k, l)
    root = []
    for q in qis:
        root.append(q.domain if q.kind == "categorical" else q.padded_bounds())
    stack = [(list(range(len(records))), tuple(root))]
    final = []
    while stack:
        idx, region = stack.pop()
        widths = _widths(idx, records, qis)
        order = sorted(range(len(qis)), key=lambda d: (-widths[d], d))
        split = None
        for d in order:
            if widths[d] == 0:
                continue
            cut = (_categorical_cut if qis[d].kind == "categorical" else _ordered_cut)
            split = cut(idx, records, d, k, l)
            if split:
                break
        if not split:
            final.append((idx, region))
            continue
        left, right, boundary = split
        for side, is_left in ((left, True), (right, False)):
            reg = list(region)
            if boundary is None:
                reg[d] = frozenset(records[i]["qi"][d] for i in side)
            else:
                lo, hi = region[d]
                reg[d] = (lo, boundary) if is_left else (boundary + 1, hi)
            stack.append((side, tuple(reg)))
    cells: list = [None] * len(records)
    for idx, region in final:
        if tight:
            box = []
            for d, q in enumerate(qis):
                vals = [records[i]["qi"][d] for i in idx]
                box.append(frozenset(vals) if q.kind == "categorical"
                           else (min(vals), max(vals)))
            box = tuple(box)
        else:
            box = tuple(frozenset(records[i]["qi"][d] for i in idx)
                        if q.kind == "categorical" else region[d]
                        for d, q in enumerate(qis))
        for i in idx:
            cells[i] = box
    classes = sorted((idx for idx, _ in final), key=min)
    return Result("mondrian_tight" if tight else "mondrian", cells, classes)


# ------------------------------------------------------------ full domain
def full_domain(records, qis: list[QISpec], k: int, l: int = 1,
                suppression_limit: float = 0.02) -> Result:
    _check_root(records, k, l)
    n = len(records)
    budget = int(suppression_limit * n)
    best = None
    for node in itertools.product(*(range(q.levels()) for q in qis)):
        groups: dict = {}
        for i, r in enumerate(records):
            box = tuple(q.generalise(v, lv) for q, v, lv in zip(qis, r["qi"], node))
            groups.setdefault(box, []).append(i)
        kept, dropped = [], 0
        for box, members in groups.items():
            ok = len(members) >= k and (
                l <= 1 or len({records[i]["sens"] for i in members}) >= l)
            if ok:
                kept.append((box, members))
            else:
                dropped += len(members)
        if dropped > budget:
            continue
        # information loss: average NCP, a suppressed row costing 1 per column
        loss = dropped * len(qis)
        for box, members in kept:
            loss += len(members) * sum(q.ncp(c) for q, c in zip(qis, box))
        loss /= n * len(qis)
        key = (loss, dropped, node)
        if best is None or key < best[0]:
            best = (key, node, kept)
    if best is None:
        raise Infeasible("no full-domain generalisation satisfies the constraints")
    _, node, kept = best
    cells: list = [None] * n
    classes = []
    for box, members in kept:
        for i in members:
            cells[i] = box
        classes.append(sorted(members))
    classes.sort(key=min)
    return Result("full_domain", cells, classes, level=node)


def identity(records, qis: list[QISpec]) -> Result:
    """Baseline: direct identifiers removed, quasi-identifiers released exactly."""
    cells = [tuple(q.point(v) for q, v in zip(qis, r["qi"])) for r in records]
    groups: dict = {}
    for i, c in enumerate(cells):
        groups.setdefault(c, []).append(i)
    return Result("none", cells, sorted(groups.values(), key=min))


def class_stats(result: Result, records) -> dict:
    sizes = [len(c) for c in result.classes]
    sens = [len({records[i]["sens"] for i in c}) for c in result.classes]
    return {
        "classes": len(sizes),
        "min_class_size": min(sizes) if sizes else 0,
        "median_class_size": sorted(sizes)[len(sizes) // 2] if sizes else 0,
        "min_distinct_sensitive": min(sens) if sens else 0,
        "suppressed": len(result.suppressed),
    }
