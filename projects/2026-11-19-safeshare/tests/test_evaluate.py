import pytest

from safeshare.evaluate import evaluate
from safeshare.policy import load_policy
from safeshare.release import run_release
from conftest import POLICY

KEY = b"eval-key-0123456789abcdef"


@pytest.fixture(scope="module")
def results(clinic, tmp_path_factory):
    data, voters = clinic
    out = {}
    for method, k, l in [("none", 5, 1), ("mondrian", 5, 1), ("mondrian", 5, 3),
                         ("mondrian_tight", 5, 1), ("full_domain", 5, 1)]:
        pol = load_policy(POLICY, k, l)
        rel = tmp_path_factory.mktemp("r") / "rel"
        rep = run_release(data, pol, rel, KEY, method)
        out[(method, l)] = (rep, evaluate(data, rel, voters, pol, KEY, n_queries=150))
    return out


def test_identifier_drop_alone_is_reidentifiable(results):
    _, ev = results[("none", 1)]
    assert ev["attack"]["targets"] > 300
    assert ev["attack"]["unique_reid"] > 0.9
    assert ev["utility"]["median_rel_error"] == 0.0


@pytest.mark.parametrize("key", [("mondrian", 1), ("mondrian", 3),
                                 ("mondrian_tight", 1), ("full_domain", 1)])
def test_k_anonymity_bounds_prosecutor_risk(results, key):
    _, ev = results[key]
    a = ev["attack"]
    assert a["unique_reid"] == 0.0
    assert a["max_risk"] <= 1 / 5 + 1e-12
    assert a["mean_risk"] > 0


def test_l_diversity_prevents_homogeneity_disclosure(results):
    assert results[("mondrian", 3)][1]["attack"]["attr_disclosure"] == 0.0


def test_tight_boxes_expose_real_birth_dates(results):
    tight = results[("mondrian_tight", 1)][1]["attack"]["endpoint_exposure"]
    region = results[("mondrian", 1)][1]["attack"]["endpoint_exposure"]
    assert tight > 0.9 and region < tight / 3


def test_mondrian_has_less_information_loss_than_full_domain(results):
    m_rep, m_ev = results[("mondrian", 1)]
    f_rep, f_ev = results[("full_domain", 1)]
    assert m_rep["avg_ncp"] < f_rep["avg_ncp"]
    assert m_ev["utility"]["median_rel_error"] < f_ev["utility"]["median_rel_error"]
