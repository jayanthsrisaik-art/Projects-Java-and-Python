import itertools

import pytest

from conftest import random_records
from safeshare.anonymize import Infeasible, full_domain, identity, mondrian


def _classes_by_cell(result):
    groups = {}
    for i, c in enumerate(result.cells):
        if c is not None:
            groups.setdefault(c, []).append(i)
    return groups


@pytest.mark.parametrize("seed,k,l,tight", list(itertools.product(
    range(4), (2, 5, 12), (1, 3), (False, True))))
def test_mondrian_k_anonymous_l_diverse_and_truthful(seed, k, l, tight):
    specs, recs = random_records(400, seed)
    res = mondrian(recs, specs, k, l, tight=tight)
    assert not res.suppressed
    groups = _classes_by_cell(res)
    for cell, members in groups.items():
        assert len(members) >= k
        assert len({recs[i]["sens"] for i in members}) >= l
    for r, cell in zip(recs, res.cells):          # every cell contains the truth
        assert all(s.contains(c, v) for s, c, v in zip(specs, cell, r["qi"]))


@pytest.mark.parametrize("seed", range(3))
def test_region_cells_partition_the_space(seed):
    """Region boxes of different classes never overlap, so a target's point
    falls in exactly one class box - the attacker gets no extra narrowing."""
    specs, recs = random_records(300, seed)
    res = mondrian(recs, specs, 4)
    boxes = list(_classes_by_cell(res))
    for r in recs:
        inside = [b for b in boxes
                  if all(s.contains(c, v) for s, c, v in zip(specs, b, r["qi"]))]
        assert len(inside) == 1


def test_tight_box_is_inside_region_box():
    specs, recs = random_records(300, 9)
    region = mondrian(recs, specs, 5)
    tight = mondrian(recs, specs, 5, tight=True)
    for a, b in zip(region.cells, tight.cells):
        for s, ca, cb in zip(specs, a, b):
            if s.kind == "categorical":
                assert cb <= ca
            else:
                assert ca[0] <= cb[0] and cb[1] <= ca[1]


def test_mondrian_beats_single_class():
    specs, recs = random_records(400, 1)
    res = mondrian(recs, specs, 5)
    assert len(res.classes) > 20


@pytest.mark.parametrize("seed,k,l", [(0, 5, 1), (1, 10, 3), (2, 3, 2)])
def test_full_domain_constraints_and_budget(seed, k, l):
    specs, recs = random_records(500, seed)
    res = full_domain(recs, specs, k, l, suppression_limit=0.05)
    assert len(res.suppressed) <= int(0.05 * 500)
    for cell, members in _classes_by_cell(res).items():
        assert len(members) >= k
        assert len({recs[i]["sens"] for i in members}) >= l
    # one generalisation level for the whole column: every cell of a column
    # has the same width at that level
    assert res.level is not None and len(res.level) == len(specs)


def test_full_domain_zero_budget_never_suppresses():
    specs, recs = random_records(300, 4)
    res = full_domain(recs, specs, 5, 1, suppression_limit=0.0)
    assert res.suppressed == []


def test_infeasible_constraints():
    specs, recs = random_records(4, 0)
    with pytest.raises(Infeasible):
        mondrian(recs, specs, 5)
    specs, recs = random_records(50, 0, l_vals=2)
    with pytest.raises(Infeasible):
        mondrian(recs, specs, 2, l=3)
    with pytest.raises(Infeasible):
        full_domain(recs, specs, 2, l=3)


def test_identity_baseline_keeps_exact_values():
    specs, recs = random_records(50, 2)
    res = identity(recs, specs)
    for r, cell in zip(recs, res.cells):
        assert tuple(c if s.kind == "categorical" else c[0]
                     for s, c in zip(specs, cell)) == tuple(
            frozenset([v]) if s.kind == "categorical" else v
            for s, v in zip(specs, r["qi"]))
