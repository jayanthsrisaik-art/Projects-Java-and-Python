"""Censoring-aware evaluation metrics.

* Harrell's concordance index on risk scores (higher score = churns sooner).
* Time-dependent Brier score with inverse-probability-of-censoring weights (Graf et al.
  1999), and its integral over a time grid (integrated Brier score, IBS).
* Calibration at a horizon: per predicted-risk bin, mean predicted churn vs the
  Kaplan-Meier churn estimate of the customers in that bin.
"""
from __future__ import annotations

import numpy as np

from churnclock.survival import KaplanMeier


def concordance_index(duration, event, risk, chunk: int = 2048) -> float:
    """Harrell's C. A pair (i, j) is comparable when i has an event and T_i < T_j, or
    T_i == T_j and j is censored. Concordant if risk_i > risk_j; risk ties count 0.5."""
    t = np.asarray(duration, float)
    e = np.asarray(event, int).astype(bool)
    r = np.asarray(risk, float)
    num = 0.0
    den = 0.0
    ev_idx = np.flatnonzero(e)
    for start in range(0, ev_idx.size, chunk):
        i = ev_idx[start : start + chunk]
        ti, ri = t[i][:, None], r[i][:, None]
        comparable = (t[None, :] > ti) | ((t[None, :] == ti) & ~e[None, :])
        conc = (r[None, :] < ri) & comparable
        tie = (r[None, :] == ri) & comparable
        num += conc.sum() + 0.5 * tie.sum()
        den += comparable.sum()
    if den == 0:
        raise ValueError("no comparable pairs")
    return float(num / den)


def censoring_km(duration, event) -> KaplanMeier:
    """KM estimate of the censoring survival G(t) = P(C > t)."""
    return KaplanMeier().fit(duration, 1 - np.asarray(event, int))


def brier_ipcw(duration, event, surv_at_t, t: float, G: KaplanMeier) -> float:
    """IPCW Brier score at horizon t for predicted S(t | x_i).

    Customers churning by t contribute S^2 / G(T_i-); still observed past t contribute
    (1 - S)^2 / G(t); customers censored before t contribute 0 (their weight is carried by
    the inverse-probability weights of the others)."""
    T = np.asarray(duration, float)
    E = np.asarray(event, int)
    S = np.asarray(surv_at_t, float)
    g_ti = np.clip(G.predict_left(T), 1e-12, None)
    g_t = max(float(G.predict(np.array([t]))[0]), 1e-12)
    churned = (T <= t) & (E == 1)
    alive = T > t
    contrib = np.where(churned, S**2 / g_ti, 0.0) + np.where(alive, (1 - S) ** 2 / g_t, 0.0)
    return float(contrib.mean())


def integrated_brier(duration, event, surv_matrix, grid, G: KaplanMeier) -> float:
    """Trapezoidal mean of the IPCW Brier score over ``grid`` (surv_matrix: n x len(grid))."""
    grid = np.asarray(grid, float)
    scores = np.array([brier_ipcw(duration, event, surv_matrix[:, k], grid[k], G) for k in range(grid.size)])
    return float(np.trapezoid(scores, grid) / (grid[-1] - grid[0]))


def calibration_table(duration, event, pred_churn, t: float, n_bins: int = 5) -> list[dict]:
    """Bins by predicted churn-by-t quantile; observed = 1 - KM(t) within the bin."""
    T = np.asarray(duration, float)
    E = np.asarray(event, int)
    p = np.asarray(pred_churn, float)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        km = KaplanMeier().fit(T[m], E[m])
        rows.append(
            {
                "bin": b + 1,
                "n": int(m.sum()),
                "mean_predicted": round(float(p[m].mean()), 4),
                "km_observed": round(float(1 - km.predict(np.array([t]))[0]), 4),
            }
        )
    return rows


def calibration_error(rows: list[dict]) -> float:
    """Size-weighted mean |predicted - observed| over calibration bins."""
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["mean_predicted"] - r["km_observed"]) for r in rows) / n)
