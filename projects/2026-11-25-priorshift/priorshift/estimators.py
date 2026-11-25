"""Class-prior (prevalence) estimators for an unlabeled target batch under label shift.

Label shift assumption: p(x | y) is the same in training and deployment; only p(y) changes.
Every estimator here receives *only* model outputs on the unlabeled batch plus statistics that
were computed on the labeled validation split. None of them can see target labels.

Notation: K classes, probs is an (n, K) matrix of predicted class probabilities, train_priors is
the class mix the probabilities were calibrated for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

EPS = 1e-12


def _check_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[0] == 0:
        raise ValueError("probs must be a non-empty (n, K) array")
    if np.any(probs < -1e-9) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each row of probs must be a probability distribution")
    return np.clip(probs, 0.0, 1.0)


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {q : q >= 0, sum q = 1} (Duchi et al., 2008)."""
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.arange(1, v.size + 1)
    rho = np.nonzero(u * k > (css - 1.0))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def adjust_posteriors(probs: np.ndarray, train_priors: np.ndarray, new_priors: np.ndarray) -> np.ndarray:
    """Bayes-rule correction: p_new(y|x) ∝ p_train(y|x) * q(y) / π(y).

    Exact when probs are the true training posteriors and only p(y) changed.
    """
    probs = _check_probs(probs)
    w = np.asarray(new_priors, dtype=float) / np.maximum(np.asarray(train_priors, dtype=float), EPS)
    adj = probs * w[None, :]
    total = adj.sum(axis=1, keepdims=True)
    # A row can sum to 0 only if all its mass sits on classes with q = 0; fall back to the input row.
    bad = total[:, 0] <= EPS
    adj[~bad] /= total[~bad]
    adj[bad] = probs[bad]
    return adj


# --------------------------------------------------------------------------------------------
# Confusion-matrix based estimators (need validation statistics)
# --------------------------------------------------------------------------------------------


def hard_confusion(val_probs: np.ndarray, val_y: np.ndarray, n_classes: int) -> np.ndarray:
    """C[i, j] = P(predicted = i | true = j), estimated on the labeled validation split."""
    pred = np.asarray(val_probs).argmax(axis=1)
    C = np.zeros((n_classes, n_classes))
    for j in range(n_classes):
        mask = val_y == j
        if not mask.any():
            raise ValueError(f"class {j} is missing from the validation split")
        C[:, j] = np.bincount(pred[mask], minlength=n_classes) / mask.sum()
    return C


def soft_confusion(val_probs: np.ndarray, val_y: np.ndarray, n_classes: int) -> np.ndarray:
    """C[i, j] = E[p_i(x) | true = j] (the 'soft' matrix used by PACC / BBSE-soft)."""
    val_probs = np.asarray(val_probs, dtype=float)
    C = np.zeros((n_classes, n_classes))
    for j in range(n_classes):
        mask = val_y == j
        if not mask.any():
            raise ValueError(f"class {j} is missing from the validation split")
        C[:, j] = val_probs[mask].mean(axis=0)
    return C


def solve_on_simplex(C: np.ndarray, mu: np.ndarray, weight: float = 1e3) -> np.ndarray:
    """argmin_q ||C q - mu||^2 subject to q in the simplex.

    Solved as non-negative least squares with a heavily weighted sum-to-one row, then
    renormalised. Unconstrained inversion (q = C^-1 mu) returns negative 'prevalences' on small
    batches; this keeps the answer a valid distribution.
    """
    K = C.shape[1]
    A = np.vstack([C, weight * np.ones((1, K))])
    b = np.concatenate([mu, [weight]])
    q, _ = nnls(A, b, maxiter=50 * K)
    s = q.sum()
    return q / s if s > 0 else np.full(K, 1.0 / K)


# --------------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------------


def classify_and_count(probs: np.ndarray) -> np.ndarray:
    """CC: share of the batch predicted as each class. Biased toward the training mix."""
    probs = _check_probs(probs)
    return np.bincount(probs.argmax(axis=1), minlength=probs.shape[1]) / probs.shape[0]


def probabilistic_count(probs: np.ndarray) -> np.ndarray:
    """PCC: mean predicted probability. Also biased toward the training mix."""
    return _check_probs(probs).mean(axis=0)


def adjusted_count(probs: np.ndarray, C_hard: np.ndarray) -> np.ndarray:
    """ACC (a.k.a. BBSE-hard): invert the hard confusion matrix, constrained to the simplex."""
    return solve_on_simplex(C_hard, classify_and_count(probs))


def probabilistic_adjusted_count(probs: np.ndarray, C_soft: np.ndarray) -> np.ndarray:
    """PACC (a.k.a. BBSE-soft): invert the soft confusion matrix, constrained to the simplex."""
    return solve_on_simplex(C_soft, probabilistic_count(probs))


@dataclass
class EMResult:
    priors: np.ndarray
    iterations: int
    converged: bool
    loglik_gain: float  # mean per-sample log-likelihood gain of q over the training priors


def em_priors(
    probs: np.ndarray,
    train_priors: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> EMResult:
    """Maximum-likelihood label shift (Saerens, Latinne & Decaestecker, 2002).

    Alternates between re-weighting posteriors with the current prior estimate (E-step) and
    setting the prior to the mean re-weighted posterior (M-step). This is EM on the mixture
    weights of p(x) = sum_y q(y) p(x|y), with p(x|y) ∝ p_train(y|x) / π(y); the likelihood is
    concave in q, so it converges to the global MLE. It needs *calibrated* probabilities.
    """
    probs = _check_probs(probs)
    pi = np.asarray(train_priors, dtype=float)
    ratio = probs / np.maximum(pi, EPS)[None, :]  # ∝ p(x|y)
    q = pi.copy()
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        joint = ratio * q[None, :]
        post = joint / np.maximum(joint.sum(axis=1, keepdims=True), EPS)
        q_new = post.mean(axis=0)
        if np.abs(q_new - q).max() < tol:
            q = q_new
            converged = True
            break
        q = q_new
    gain = mixture_loglik(ratio, q) - mixture_loglik(ratio, pi)
    return EMResult(priors=q, iterations=it, converged=converged, loglik_gain=float(gain))


def mixture_loglik(ratio: np.ndarray, q: np.ndarray) -> float:
    """Mean of log sum_y q(y) * ratio[:, y]; equals the batch log-likelihood up to a constant."""
    return float(np.mean(np.log(np.maximum(ratio @ q, EPS))))


def smooth_priors(q: np.ndarray, n: int) -> np.ndarray:
    """Add-one (Laplace) smoothing of an estimated mix from a batch of n rows: (n q + 1) / (n + K).

    Estimators can return q_k = 0 for a class that is rare but present. Bayes-correcting with a
    zero prior makes those rows impossible and the log-loss explodes. The pseudo-count is a fixed
    rule (not tuned on test data) and its weight shrinks as the batch grows.
    """
    q = np.asarray(q, dtype=float)
    return (n * q + 1.0) / (n + q.size)


METHODS = ("none", "cc", "pcc", "acc", "pacc", "em")


def estimate(method: str, probs: np.ndarray, train_priors: np.ndarray, C_hard=None, C_soft=None) -> np.ndarray:
    """Dispatch by name. 'none' returns the training priors (the no-adaptation baseline)."""
    if method == "none":
        return np.asarray(train_priors, dtype=float).copy()
    if method == "cc":
        return classify_and_count(probs)
    if method == "pcc":
        return probabilistic_count(probs)
    if method == "acc":
        if C_hard is None:
            raise ValueError("acc needs the validation hard confusion matrix")
        return adjusted_count(probs, C_hard)
    if method == "pacc":
        if C_soft is None:
            raise ValueError("pacc needs the validation soft confusion matrix")
        return probabilistic_adjusted_count(probs, C_soft)
    if method == "em":
        return em_priors(probs, train_priors).priors
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
