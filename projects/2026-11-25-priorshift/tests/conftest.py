"""Synthetic label-shift world where the true posteriors are known in closed form."""

import numpy as np
import pytest

MEANS = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])


def sample(priors, n, rng):
    y = rng.choice(len(priors), size=n, p=priors)
    X = MEANS[y] + rng.normal(size=(n, 2))
    return X, y


def exact_posteriors(X, priors):
    """p(y|x) ∝ π_y N(x; μ_y, I) — the Bayes-optimal, perfectly calibrated classifier for π."""
    d2 = ((X[:, None, :] - MEANS[None, :, :]) ** 2).sum(-1)
    logp = np.log(priors)[None, :] - 0.5 * d2
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


@pytest.fixture
def world():
    class W:
        means = MEANS
        train_priors = np.array([0.5, 0.3, 0.2])

        @staticmethod
        def batch(priors, n, seed):
            rng = np.random.default_rng(seed)
            X, y = sample(np.asarray(priors), n, rng)
            return exact_posteriors(X, W.train_priors), y, X

        exact_posteriors = staticmethod(exact_posteriors)

    return W
