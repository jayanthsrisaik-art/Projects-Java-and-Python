"""Kaplan-Meier and Cox proportional hazards, implemented with NumPy.

Cox PH uses the Breslow approximation for tied event times, an optional L2 penalty,
and damped Newton-Raphson. The baseline cumulative hazard is the Breslow estimator, so
S(t | x) = exp(-H0(t) * exp(x . beta)).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------- Kaplan-Meier
@dataclass
class KaplanMeier:
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))  # distinct event times
    surv: np.ndarray = field(default_factory=lambda: np.zeros(0))  # S(t) just after each time

    def fit(self, duration, event) -> "KaplanMeier":
        duration = np.asarray(duration, float)
        event = np.asarray(event, int)
        if duration.shape != event.shape or duration.size == 0:
            raise ValueError("duration and event must be non-empty and the same shape")
        order = np.argsort(duration, kind="stable")
        d, e = duration[order], event[order]
        uniq, first = np.unique(d, return_index=True)
        at_risk = d.size - first
        deaths = np.add.reduceat(e, first)
        mask = deaths > 0
        factors = 1.0 - deaths[mask] / at_risk[mask]
        self.times = uniq[mask]
        self.surv = np.cumprod(factors)
        return self

    def predict(self, t) -> np.ndarray:
        """S(t), right-continuous: includes events at exactly t."""
        t = np.asarray(t, float)
        idx = np.searchsorted(self.times, t, side="right") - 1
        return np.where(idx >= 0, self.surv[np.clip(idx, 0, None)] if self.surv.size else 1.0, 1.0)

    def predict_left(self, t) -> np.ndarray:
        """S(t-): excludes events at exactly t (needed for IPCW weights G(T_i-))."""
        t = np.asarray(t, float)
        idx = np.searchsorted(self.times, t, side="left") - 1
        return np.where(idx >= 0, self.surv[np.clip(idx, 0, None)] if self.surv.size else 1.0, 1.0)


# ----------------------------------------------------------------------------- Cox PH
def _sorted_desc(duration, event, X):
    order = np.argsort(-np.asarray(duration, float), kind="stable")
    return np.asarray(duration, float)[order], np.asarray(event, int)[order], np.asarray(X, float)[order]


def breslow_loglik(beta, duration, event, X) -> float:
    """Breslow partial log-likelihood (unpenalised). Reference implementation used in tests too."""
    t, e, x = _sorted_desc(duration, event, X)
    eta = x @ beta
    # risk set for time u: all j with t_j >= u. With descending order, that is a prefix whose
    # end is the last index of the tie group of u.
    s0 = np.cumsum(np.exp(eta))
    last = _tie_group_last(t)
    ll = 0.0
    ll += float(np.sum(eta[e == 1]))
    ll -= float(np.sum(np.log(s0[last][e == 1])))
    return ll


def _tie_group_last(t_desc: np.ndarray) -> np.ndarray:
    """For each position in a descending array, the last index holding the same value."""
    n = t_desc.size
    # boundaries where value changes
    change = np.r_[t_desc[1:] != t_desc[:-1], True]
    last_idx = np.flatnonzero(change)
    group = np.r_[0, np.cumsum(change[:-1])]
    return last_idx[group] if n else last_idx


@dataclass
class CoxPH:
    l2: float = 0.0
    max_iter: int = 100
    tol: float = 1e-9
    beta: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    base_times: np.ndarray | None = None
    base_cumhaz: np.ndarray | None = None
    n_iter_: int = 0
    loglik_: float = float("nan")
    feature_names: list[str] | None = None

    # -- internals
    def _standardize(self, X):
        return (np.asarray(X, float) - self.mean_) / self.scale_

    def _objective(self, beta, t, e, x, last):
        eta = x @ beta
        eta_max = eta.max()
        w = np.exp(eta - eta_max)
        s0 = np.cumsum(w)[last]
        s1 = np.cumsum(w[:, None] * x, axis=0)[last]
        s2 = np.cumsum(w[:, None, None] * (x[:, :, None] * x[:, None, :]), axis=0)[last]
        ev = e == 1
        ll = np.sum(eta[ev]) - np.sum(np.log(s0[ev]) + eta_max) - 0.5 * self.l2 * beta @ beta
        mu = s1[ev] / s0[ev, None]
        grad = x[ev].sum(0) - mu.sum(0) - self.l2 * beta
        hess = -(s2[ev] / s0[ev, None, None] - mu[:, :, None] * mu[:, None, :]).sum(0)
        hess -= self.l2 * np.eye(beta.size)
        return ll, grad, hess

    def fit(self, X, duration, event, feature_names=None) -> "CoxPH":
        X = np.asarray(X, float)
        duration = np.asarray(duration, float)
        event = np.asarray(event, int)
        if X.ndim != 2 or X.shape[0] != duration.size:
            raise ValueError("X must be (n, p) with n == len(duration)")
        if event.sum() == 0:
            raise ValueError("no events: Cox model is not identifiable")
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.mean_ = X.mean(0)
        sd = X.std(0)
        self.scale_ = np.where(sd > 0, sd, 1.0)
        t, e, x = _sorted_desc(duration, event, self._standardize(X))
        last = _tie_group_last(t)
        beta = np.zeros(X.shape[1])
        ll, g, H = self._objective(beta, t, e, x, last)
        for it in range(1, self.max_iter + 1):
            step = np.linalg.solve(H, -g)
            lam = 1.0
            while True:  # step halving keeps every iteration an ascent step
                cand = beta + lam * step
                ll_c, g_c, H_c = self._objective(cand, t, e, x, last)
                if ll_c >= ll - 1e-12 or lam < 1e-8:
                    break
                lam *= 0.5
            converged = abs(ll_c - ll) < self.tol * (1 + abs(ll))
            beta, ll, g, H = cand, ll_c, g_c, H_c
            self.n_iter_ = it
            if converged or np.max(np.abs(lam * step)) < 1e-10:
                break
        self.beta = beta
        self.loglik_ = float(ll)
        # Breslow baseline cumulative hazard on the standardized scale
        eta = x @ beta
        w = np.exp(eta)
        s0 = np.cumsum(w)[last]
        ev_times, inv = np.unique(t[e == 1], return_inverse=True)
        d = np.bincount(inv, minlength=ev_times.size).astype(float)
        s0_at = np.zeros(ev_times.size)
        s0_at[inv] = s0[e == 1]  # identical within a tie group
        self.base_times = ev_times
        self.base_cumhaz = np.cumsum(d / s0_at)
        return self

    # -- public API
    @property
    def coef_(self) -> np.ndarray:
        """Log hazard ratios on the ORIGINAL feature scale."""
        return self.beta / self.scale_

    def risk_score(self, X) -> np.ndarray:
        return self._standardize(X) @ self.beta

    def cumulative_hazard(self, X, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        idx = np.searchsorted(self.base_times, t, side="right") - 1
        h0 = np.where(idx >= 0, self.base_cumhaz[np.clip(idx, 0, None)], 0.0)
        return np.exp(self.risk_score(X))[:, None] * h0[None, :]

    def survival(self, X, t) -> np.ndarray:
        """S(t | x), shape (n, len(t))."""
        return np.exp(-self.cumulative_hazard(X, t))

    def median_survival(self, X) -> np.ndarray:
        """First baseline time where S <= 0.5; inf if the curve never gets there."""
        s = self.survival(X, self.base_times)
        below = s <= 0.5
        first = np.argmax(below, axis=1)
        return np.where(below.any(axis=1), self.base_times[first], np.inf)

    def to_dict(self) -> dict:
        return {
            "kind": "coxph-breslow",
            "l2": self.l2,
            "feature_names": self.feature_names,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "beta_standardized": self.beta.tolist(),
            "base_times": self.base_times.tolist(),
            "base_cumhaz": self.base_cumhaz.tolist(),
            "loglik": self.loglik_,
            "n_iter": self.n_iter_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CoxPH":
        m = cls(l2=d["l2"])
        m.feature_names = d["feature_names"]
        m.mean_ = np.array(d["mean"])
        m.scale_ = np.array(d["scale"])
        m.beta = np.array(d["beta_standardized"])
        m.base_times = np.array(d["base_times"])
        m.base_cumhaz = np.array(d["base_cumhaz"])
        m.loglik_ = d.get("loglik", float("nan"))
        m.n_iter_ = d.get("n_iter", 0)
        return m
