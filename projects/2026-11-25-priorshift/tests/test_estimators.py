import numpy as np
import pytest

from priorshift import estimators as est


def tv(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


def confusions(world, n=40_000, seed=1):
    """Validation statistics drawn at the training mix."""
    P, y, _ = world.batch(world.train_priors, n, seed)
    return est.hard_confusion(P, y, 3), est.soft_confusion(P, y, 3)


@pytest.mark.parametrize("q", [[0.1, 0.1, 0.8], [0.6, 0.4, 0.0], [1 / 3, 1 / 3, 1 / 3]])
def test_em_recovers_true_priors_with_calibrated_posteriors(world, q):
    P, _, _ = world.batch(q, 30_000, seed=2)
    res = est.em_priors(P, world.train_priors)
    assert res.converged
    assert tv(res.priors, q) < 0.02
    assert res.loglik_gain >= 0.0


@pytest.mark.parametrize("method", ["acc", "pacc"])
def test_confusion_matrix_methods_are_consistent(world, method):
    C_hard, C_soft = confusions(world)
    q = np.array([0.1, 0.2, 0.7])
    P, _, _ = world.batch(q, 30_000, seed=3)
    q_hat = est.estimate(method, P, world.train_priors, C_hard, C_soft)
    assert tv(q_hat, q) < 0.03


def test_unadjusted_counting_is_biased_toward_training_mix(world):
    q = np.array([0.05, 0.05, 0.9])
    P, _, _ = world.batch(q, 30_000, seed=4)
    err_cc = tv(est.classify_and_count(P), q)
    err_pcc = tv(est.probabilistic_count(P), q)
    err_em = tv(est.em_priors(P, world.train_priors).priors, q)
    assert err_cc > 0.05 and err_pcc > 0.05
    assert err_em < err_cc / 3 and err_em < err_pcc / 3


def test_adjust_posteriors_matches_bayes_optimal_target_posterior(world):
    q = np.array([0.2, 0.2, 0.6])
    _, _, X = world.batch(q, 500, seed=5)
    P_train = world.exact_posteriors(X, world.train_priors)
    P_target = world.exact_posteriors(X, q)
    np.testing.assert_allclose(est.adjust_posteriors(P_train, world.train_priors, q), P_target, atol=1e-10)


def test_adjust_with_zero_prior_keeps_rows_valid():
    P = np.array([[0.0, 1.0], [0.5, 0.5]])
    out = est.adjust_posteriors(P, np.array([0.5, 0.5]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_allclose(out[1], [1.0, 0.0])
    np.testing.assert_allclose(out[0], [0.0, 1.0])  # impossible row falls back to the input


def test_no_shift_em_stays_near_training_priors(world):
    P, _, _ = world.batch(world.train_priors, 20_000, seed=6)
    res = est.em_priors(P, world.train_priors)
    assert tv(res.priors, world.train_priors) < 0.02
    assert res.loglik_gain < 1e-3


def test_project_to_simplex():
    for v in ([0.2, 0.3, 0.5], [2.0, -1.0, 0.0], [-5.0, -5.0, -5.0], [10.0, 0.0, 0.0]):
        p = est.project_to_simplex(np.array(v))
        assert p.min() >= 0 and abs(p.sum() - 1) < 1e-12
    np.testing.assert_allclose(est.project_to_simplex(np.array([0.2, 0.3, 0.5])), [0.2, 0.3, 0.5])
    np.testing.assert_allclose(est.project_to_simplex(np.array([2.0, -1.0, 0.0])), [1.0, 0.0, 0.0])


def test_solve_on_simplex_exact_and_infeasible():
    C = np.array([[0.9, 0.2], [0.1, 0.8]])
    q = np.array([0.3, 0.7])
    np.testing.assert_allclose(est.solve_on_simplex(C, C @ q), q, atol=1e-5)
    # Observed mix more extreme than any valid q can explain: unconstrained inverse goes negative.
    mu = np.array([0.95, 0.05])
    assert np.linalg.solve(C, mu).min() < 0
    out = est.solve_on_simplex(C, mu)
    assert out.min() >= 0 and abs(out.sum() - 1) < 1e-9
    np.testing.assert_allclose(out, [1.0, 0.0], atol=1e-6)


def test_smoothing():
    q = np.array([0.5, 0.5, 0.0])
    s = est.smooth_priors(q, 97)
    assert abs(s.sum() - 1) < 1e-12 and s.min() > 0
    np.testing.assert_allclose(s, [(48.5 + 1) / 100, (48.5 + 1) / 100, 1 / 100])
    assert est.smooth_priors(q, 10**9).min() < 1e-8  # vanishes for large batches


def test_estimate_dispatch_and_validation(world):
    P, _, _ = world.batch(world.train_priors, 100, seed=7)
    np.testing.assert_allclose(est.estimate("none", P, world.train_priors), world.train_priors)
    with pytest.raises(ValueError):
        est.estimate("acc", P, world.train_priors)
    with pytest.raises(ValueError):
        est.estimate("magic", P, world.train_priors)
    with pytest.raises(ValueError):
        est.classify_and_count(np.array([[0.7, 0.7]]))
    with pytest.raises(ValueError):
        est.hard_confusion(P, np.zeros(100, dtype=int), 3)  # classes 1 and 2 missing from validation
