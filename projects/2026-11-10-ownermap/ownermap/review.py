"""Plan the smallest set of reviewers that satisfies "require review from code owners".

For every changed file the deciding CODEOWNERS rule names owners; one
approval from *any* of them (or any member of an owning team) satisfies that
file.  The PR author's own approval does not count.  Picking the fewest people
that cover every file is a minimum hitting-set problem (NP-hard in general but
tiny in practice), solved here exactly with branch and bound and compared
against the usual greedy heuristic.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .codeowners import CodeOwners
from .registry import Registry


def greedy_hitting_set(reqs: list[frozenset[str]]) -> list[str]:
    """Repeatedly take the person covering the most unsatisfied requirements."""
    remaining = [r for r in reqs if r]
    chosen: list[str] = []
    while remaining:
        counts: dict[str, int] = {}
        for r in remaining:
            for p in r:
                counts[p] = counts.get(p, 0) + 1
        best = min(counts, key=lambda p: (-counts[p], p))
        chosen.append(best)
        remaining = [r for r in remaining if best not in r]
    return sorted(chosen)


def _reduce(reqs: list[frozenset[str]]) -> list[frozenset[str]]:
    """Drop duplicates and supersets: hitting the subset hits the superset."""
    uniq = sorted(set(reqs), key=lambda r: (len(r), sorted(r)))
    kept: list[frozenset[str]] = []
    for r in uniq:
        if not any(k <= r for k in kept):
            kept.append(r)
    return kept


def _disjoint_lower_bound(reqs: list[frozenset[str]]) -> int:
    """Pairwise-disjoint requirements each need a different person."""
    used: set[str] = set()
    lb = 0
    for r in sorted(reqs, key=len):
        if used.isdisjoint(r):
            used |= r
            lb += 1
    return lb


@dataclass
class SolveStats:
    nodes: int = 0
    optimal: bool = True


def exact_hitting_set(reqs: list[frozenset[str]], node_budget: int = 200_000) -> tuple[list[str], SolveStats]:
    """Minimum hitting set by branch and bound; ties broken lexicographically.

    If the node budget runs out, the best solution found so far (never worse
    than greedy) is returned with ``optimal=False``.
    """
    stats = SolveStats()
    base = _reduce([r for r in reqs if r])
    best = greedy_hitting_set(base)
    best_key = (len(best), best)

    def search(open_reqs: list[frozenset[str]], chosen: list[str]) -> None:
        nonlocal best, best_key
        stats.nodes += 1
        if stats.nodes > node_budget:
            stats.optimal = False
            return
        if not open_reqs:
            cand = sorted(chosen)
            key = (len(cand), cand)
            if key < best_key:
                best, best_key = cand, key
            return
        if len(chosen) + _disjoint_lower_bound(open_reqs) > len(best):
            return
        pivot = min(open_reqs, key=lambda r: (len(r), sorted(r)))
        freq: dict[str, int] = {}
        for r in open_reqs:
            for p in r:
                freq[p] = freq.get(p, 0) + 1
        for person in sorted(pivot, key=lambda p: (-freq[p], p)):
            rest = [r for r in open_reqs if person not in r]
            search(rest, chosen + [person])
            if stats.nodes > node_budget:
                return

    search(base, [])
    return best, stats


def brute_force_hitting_set(reqs: list[frozenset[str]]) -> list[str]:
    """Reference oracle for tests: try all subsets by increasing size."""
    reqs = [r for r in reqs if r]
    people = sorted(set().union(*reqs)) if reqs else []
    for k in range(len(people) + 1):
        for combo in itertools.combinations(people, k):
            s = set(combo)
            if all(s & r for r in reqs):
                return list(combo)
    raise AssertionError("unreachable")


@dataclass
class FileRequirement:
    path: str
    rule_line: int | None
    owners: tuple[str, ...]
    approvers: frozenset[str]
    status: str  # required | no-owners | unsatisfiable


@dataclass
class ReviewPlan:
    files: list[FileRequirement]
    reviewers: list[str]
    greedy_reviewers: list[str]
    optimal: bool
    nodes: int
    covers: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "reviewers": self.reviewers,
            "optimal": self.optimal,
            "greedy_reviewers": self.greedy_reviewers,
            "covers": self.covers,
            "files": [
                {"path": f.path, "rule_line": f.rule_line, "owners": list(f.owners),
                 "approvers": sorted(f.approvers), "status": f.status}
                for f in self.files
            ],
            "unsatisfiable": [f.path for f in self.files if f.status == "unsatisfiable"],
        }


def plan_review(co: CodeOwners, changed: list[str], registry: Registry | None = None,
                author: str | None = None, node_budget: int = 200_000) -> ReviewPlan:
    reg = registry or Registry()
    me = author.lower() if author else None
    if me and not me.startswith("@"):
        me = "@" + me
    reqs: list[FileRequirement] = []
    for path in changed:
        rule = co.deciding_rule(path)
        owners = rule.owners if rule else ()
        approvers: set[str] = set()
        for o in owners:
            approvers |= reg.approvers(o)
        if me:
            approvers.discard(me)  # the author's own approval does not count
        if not owners:
            status = "no-owners"
        elif not approvers:
            status = "unsatisfiable"
        else:
            status = "required"
        reqs.append(FileRequirement(path, rule.line if rule else None, owners, frozenset(approvers), status))
    sets = [r.approvers for r in reqs if r.status == "required"]
    best, stats = exact_hitting_set(sets, node_budget)
    greedy = greedy_hitting_set(sets)
    covers: dict[str, list[str]] = {p: [] for p in best}
    for r in reqs:
        if r.status == "required":
            for p in best:
                if p in r.approvers:
                    covers[p].append(r.path)
    return ReviewPlan(reqs, best, greedy, stats.optimal, stats.nodes, covers)


def format_text(plan: ReviewPlan) -> str:
    lines = []
    for f in plan.files:
        rule = f"line {f.rule_line}" if f.rule_line else "no rule"
        own = " ".join(f.owners) if f.owners else "-"
        lines.append(f"{f.path}  [{f.status}]  owners: {own}  ({rule})")
    lines.append("")
    tag = "minimum" if plan.optimal else "best found (search budget exhausted)"
    lines.append(f"Request {len(plan.reviewers)} reviewer(s) [{tag}]: {' '.join(plan.reviewers) or '-'}")
    for p, files in plan.covers.items():
        lines.append(f"  {p} covers {len(files)} file(s): {', '.join(files)}")
    if len(plan.greedy_reviewers) != len(plan.reviewers):
        lines.append(f"(greedy heuristic would request {len(plan.greedy_reviewers)}: {' '.join(plan.greedy_reviewers)})")
    bad = [f.path for f in plan.files if f.status == "unsatisfiable"]
    if bad:
        lines.append("UNSATISFIABLE (only the author owns them): " + ", ".join(bad))
    return "\n".join(lines)
