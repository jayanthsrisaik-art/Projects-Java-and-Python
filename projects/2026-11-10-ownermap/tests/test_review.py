import random

import pytest

from ownermap.bench import synthetic_requirements
from ownermap.codeowners import CodeOwners
from ownermap.registry import Registry
from ownermap.review import brute_force_hitting_set, exact_hitting_set, greedy_hitting_set, plan_review


def fs(*xs):
    return frozenset(xs)


def test_greedy_is_fooled_by_a_hub():
    reqs = [fs("x", "h"), fs("x", "h"), fs("x"), fs("y", "h"), fs("y", "h"), fs("y")]
    assert greedy_hitting_set(reqs) == ["h", "x", "y"]
    best, st = exact_hitting_set(reqs)
    assert best == ["x", "y"] and st.optimal


@pytest.mark.parametrize("seed", range(25))
def test_exact_equals_brute_force_oracle(seed):
    rng = random.Random(seed)
    reqs = synthetic_requirements(rng, people=rng.randint(5, 12), teams=rng.randint(2, 5), files=rng.randint(3, 20))
    best, st = exact_hitting_set(reqs)
    assert st.optimal
    assert best == brute_force_hitting_set(reqs)  # same size AND same lexicographic tie-break
    assert all(set(best) & r for r in reqs)
    assert len(best) <= len(greedy_hitting_set(reqs))


def test_budget_exhaustion_is_reported_not_hidden():
    rng = random.Random(3)
    reqs = synthetic_requirements(rng, 30, 10, 40)
    best, st = exact_hitting_set(reqs, node_budget=1)
    assert not st.optimal and all(set(best) & r for r in reqs)


CO = CodeOwners.parse("* @org/core\n/web/ @org/web @dana\n/pay/ @erin\n/gen/\n")
REG = Registry.from_dict({"teams": {"@org/core": ["@olga"], "@org/web": ["@dana", "@wade"]}})


def test_plan_expands_teams_and_skips_unowned():
    plan = plan_review(CO, ["web/a.ts", "README", "gen/x"], REG)
    assert plan.reviewers == ["@dana", "@olga"]
    assert [f.status for f in plan.files] == ["required", "required", "no-owners"]


def test_author_cannot_approve_own_files():
    plan = plan_review(CO, ["pay/x.py", "web/a.ts"], REG, author="erin")
    assert plan.files[0].status == "unsatisfiable"
    assert plan.reviewers == ["@dana"]


def test_without_registry_teams_are_units():
    plan = plan_review(CO, ["README", "web/a.ts"])
    assert plan.reviewers in (["@org/core", "@dana"], ["@dana", "@org/core"])
