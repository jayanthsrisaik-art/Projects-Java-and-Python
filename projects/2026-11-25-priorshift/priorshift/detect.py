"""Is the batch's class mix different from training? A likelihood-ratio test on the EM fit.

H0: q = π (training priors).  Statistic: 2 n [ℓ(q̂) − ℓ(π)], where ℓ is the mean per-sample
log-likelihood of the mixture with fixed components p(x|y) ∝ p(y|x)/π(y). Under H0, with
calibrated probabilities and π in the interior of the simplex, it is asymptotically χ² with
K − 1 degrees of freedom.

Statistical significance alone is a poor trigger on large batches: a tiny calibration error
becomes "significant" once n is big enough. So the alarm also requires a minimum effect size,
the total-variation distance between q̂ and π.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .estimators import em_priors


@dataclass
class ShiftTest:
    estimated_priors: np.ndarray
    statistic: float
    df: int
    p_value: float
    tv_distance: float
    shifted: bool


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def detect_shift(probs: np.ndarray, train_priors: np.ndarray, alpha: float = 0.01, min_tv: float = 0.05) -> ShiftTest:
    res = em_priors(probs, train_priors)
    n, K = np.asarray(probs).shape
    stat = max(0.0, 2.0 * n * res.loglik_gain)
    p_value = float(chi2.sf(stat, df=K - 1))
    tv = total_variation(res.priors, train_priors)
    return ShiftTest(
        estimated_priors=res.priors,
        statistic=float(stat),
        df=K - 1,
        p_value=p_value,
        tv_distance=tv,
        shifted=bool(p_value < alpha and tv >= min_tv),
    )
