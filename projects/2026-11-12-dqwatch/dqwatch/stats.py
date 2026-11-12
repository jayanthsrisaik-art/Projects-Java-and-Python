"""Small numeric helpers (stdlib only): chi-square tail probability and its inverse."""
from __future__ import annotations

import math
from statistics import NormalDist


def _gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) (series / continued fraction, Numerical Recipes)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    b = x + 1 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x: float, df: float) -> float:
    return _gamma_q(df / 2.0, x / 2.0)


def chi2_isf(p: float, df: float) -> float:
    """x such that chi2_sf(x, df) == p, by bisection."""
    lo, hi = 0.0, max(1.0, df)
    while chi2_sf(hi, df) > p:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if chi2_sf(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def z_tail(z: float) -> float:
    """One-sided upper tail probability of a standard normal."""
    return 1.0 - NormalDist().cdf(z)
