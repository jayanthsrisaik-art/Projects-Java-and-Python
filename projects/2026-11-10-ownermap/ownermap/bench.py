"""Measured benchmarks: git differential check and reviewer planning (exact vs greedy)."""

from __future__ import annotations

import random
import time

from .review import brute_force_hitting_set, exact_hitting_set, greedy_hitting_set


def synthetic_requirements(rng: random.Random, people: int, teams: int, files: int) -> list[frozenset[str]]:
    """Requirement sets shaped like real CODEOWNERS: teams overlap, some files have 2 owners."""
    names = [f"@p{i:02d}" for i in range(people)]
    team_sets = [frozenset(rng.sample(names, rng.randint(2, 5))) for _ in range(teams)]
    owners = team_sets + [frozenset([n]) for n in names]
    reqs = []
    for _ in range(files):
        k = 1 if rng.random() < 0.6 else 2
        s: frozenset[str] = frozenset()
        for o in rng.sample(owners, k):
            s |= o
        reqs.append(s)
    return reqs


def bench_review(instances: int = 500, seed: int = 11, check_oracle: bool = True) -> dict:
    rng = random.Random(seed)
    worse = extra = 0
    sizes_exact = sizes_greedy = 0
    t_exact = t_greedy = 0.0
    nonoptimal = oracle_checked = oracle_fail = 0
    max_nodes = 0
    for _ in range(instances):
        people = rng.randint(8, 30)
        reqs = synthetic_requirements(rng, people, rng.randint(3, 10), rng.randint(5, 40))
        t0 = time.perf_counter()
        g = greedy_hitting_set(reqs)
        t1 = time.perf_counter()
        e, st = exact_hitting_set(reqs)
        t2 = time.perf_counter()
        t_greedy += t1 - t0
        t_exact += t2 - t1
        max_nodes = max(max_nodes, st.nodes)
        nonoptimal += not st.optimal
        sizes_exact += len(e)
        sizes_greedy += len(g)
        if len(g) > len(e):
            worse += 1
            extra += len(g) - len(e)
        if check_oracle and people <= 14:
            oracle_checked += 1
            if len(brute_force_hitting_set(reqs)) != len(e):
                oracle_fail += 1
    return {
        "instances": instances,
        "seed": seed,
        "greedy_worse_instances": worse,
        "greedy_worse_pct": round(100.0 * worse / instances, 1),
        "greedy_extra_reviewers_total": extra,
        "mean_reviewers_exact": round(sizes_exact / instances, 3),
        "mean_reviewers_greedy": round(sizes_greedy / instances, 3),
        "exact_hit_budget": nonoptimal,
        "max_search_nodes": max_nodes,
        "mean_ms_exact": round(1000 * t_exact / instances, 3),
        "mean_ms_greedy": round(1000 * t_greedy / instances, 3),
        "oracle_checked": oracle_checked,
        "oracle_mismatches": oracle_fail,
    }


def bench_ignore(cases: int = 300, seed: int = 7) -> dict:
    from .gitoracle import git_available, run_differential

    if not git_available():
        return {"skipped": "git not found on PATH"}
    t0 = time.perf_counter()
    s = run_differential(cases, seed)
    return {
        "cases": s.cases,
        "seed": seed,
        "queries": s.queries,
        "ownermap_verdict_mismatches": s.ours_mismatch,
        "ownermap_deciding_line_mismatches": s.ours_line_mismatch,
        "naive_fnmatch_verdict_mismatches": s.naive_mismatch,
        "naive_fnmatch_mismatch_pct": round(100.0 * s.naive_mismatch / s.queries, 2) if s.queries else None,
        "cases_where_naive_wrong": s.cases_naive_wrong,
        "first_mismatch": s.first_mismatch,
        "seconds": round(time.perf_counter() - t0, 2),
    }
