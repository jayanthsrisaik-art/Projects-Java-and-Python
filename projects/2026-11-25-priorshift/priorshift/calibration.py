"""Bias-corrected temperature scaling (BCTS).

EM-based prior estimation assumes the classifier's probabilities are calibrated for the training
mix. Raw scikit-learn probabilities usually are not (logistic regression trained with
regularisation is under- or over-confident). Alexandari et al. (2020, "Maximum likelihood with
bias-corrected calibration is hard-to-beat at label shift adaptation") show that a temperature
plus one bias per class, fitted on held-out data, is enough.

    p_cal(y|x) = softmax( log p(y|x) / T + b_y )

Fitted by minimising negative log-likelihood on the validation split only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

EPS = 1e-12


@dataclass
class BCTS:
    temperature: float = 1.0
    bias: np.ndarray | None = None

    def fit(self, probs: np.ndarray, y: np.ndarray, l2: float = 1e-4) -> "BCTS":
        logits = np.log(np.clip(np.asarray(probs, dtype=float), EPS, 1.0))
        n, K = logits.shape
        y = np.asarray(y)
        onehot = np.eye(K)[y]

        def loss_grad(theta):
            log_t, b = theta[0], theta[1:]
            t = np.exp(log_t)
            z = logits / t + b
            lsm = log_softmax(z, axis=1)
            nll = -np.sum(onehot * lsm) / n + l2 * np.sum(b**2)
            p = np.exp(lsm)
            dz = (p - onehot) / n  # d nll / d z
            g_b = dz.sum(axis=0) + 2 * l2 * b
            g_logt = np.sum(dz * (-logits / t))  # dz/dlog_t = -logits/t
            return nll, np.concatenate([[g_logt], g_b])

        theta0 = np.zeros(K + 1)
        res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B",
                       bounds=[(-4.0, 4.0)] + [(None, None)] * K)
        self.temperature = float(np.exp(res.x[0]))
        self.bias = res.x[1:] - res.x[1:].mean()  # softmax is shift-invariant; centre for readability
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self.bias is None:
            raise RuntimeError("BCTS is not fitted")
        logits = np.log(np.clip(np.asarray(probs, dtype=float), EPS, 1.0))
        return np.exp(log_softmax(logits / self.temperature + self.bias, axis=1))


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true labels."""
    p = np.asarray(probs)[np.arange(len(y)), np.asarray(y)]
    return float(-np.mean(np.log(np.clip(p, EPS, 1.0))))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == np.asarray(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
