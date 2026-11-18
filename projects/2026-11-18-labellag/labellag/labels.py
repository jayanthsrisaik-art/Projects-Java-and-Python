"""What is knowable at a training cutoff, and how complete the labels are.

``observe`` is the single gate between the simulator's ground truth and training code:
it drops future rows, hides chargebacks not yet reported, and removes the truth columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from labellag.simulate import FEATURES, REPORT_HORIZON_DAYS

TRUTH_COLUMNS = ("is_fraud", "true_p", "report_day")


def observe(df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Rows with ``day <= cutoff`` as a model trained at ``cutoff`` would see them.

    ``label`` is 1 only if the chargeback was reported by the cutoff.
    ``delay`` is the observed report delay for those rows, NaN otherwise.
    ``age`` is how long the row has had to mature.
    """
    past = df.loc[df["day"] <= cutoff]
    reported = (past["report_day"] <= cutoff).to_numpy()
    out = past[["txn_id", "day", *FEATURES]].copy()
    out["label"] = reported.astype(int)
    out["delay"] = np.where(reported, past["report_day"].to_numpy() - past["day"].to_numpy(), np.nan)
    out["age"] = cutoff - out["day"].to_numpy()
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class DelayCurve:
    """Empirical CDF of report delay, F(a) = P(reported within a days | fraud), per stratum."""

    edges: np.ndarray  # amount bucket edges (interior), len = n_strata - 1
    grid: np.ndarray  # ages in days at which F is tabulated
    cdf: np.ndarray  # shape (n_strata, len(grid))
    n_frauds: np.ndarray  # frauds used per stratum

    def stratum(self, amount: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, amount, side="right")

    def reported_by(self, age: np.ndarray, amount: np.ndarray) -> np.ndarray:
        """F(age) for each row: probability a fraud of that age would already be labelled."""
        age = np.asarray(age, dtype=float)
        s = self.stratum(np.asarray(amount, dtype=float))
        idx = np.clip(np.searchsorted(self.grid, age, side="right") - 1, 0, len(self.grid) - 1)
        f = self.cdf[s, idx]
        return np.where(age >= REPORT_HORIZON_DAYS, 1.0, f)

    def quantile(self, q: float) -> float:
        """Smallest tabulated age at which every stratum has reached ``q``."""
        ok = np.all(self.cdf >= q, axis=0)
        return float(self.grid[np.argmax(ok)]) if ok.any() else float(REPORT_HORIZON_DAYS)

    def to_dict(self) -> dict:
        ages = [7, 14, 30, 60, 90]
        return {
            "amount_edges": [round(float(e), 2) for e in self.edges],
            "n_frauds": [int(n) for n in self.n_frauds],
            "F_at_days": {
                str(a): [round(float(v), 4) for v in self.reported_by(np.full(len(self.n_frauds), a), self._reps())]
                for a in ages
            },
        }

    def _reps(self) -> np.ndarray:
        # one representative amount per stratum
        lo = np.concatenate([[0.0], self.edges])
        return lo + 1e-6


def estimate_delay_curve(
    obs: pd.DataFrame,
    cutoff: float,
    cohort_days: float = 90.0,
    n_strata: int = 3,
    min_frauds: int = 40,
    step: float = 1.0,
) -> DelayCurve:
    """Estimate F from a fully matured cohort, using only what is visible at ``cutoff``.

    Rows older than the report horizon have every chargeback they will ever get, so the
    delays observed for them are an unbiased sample (no right-truncation). We use the most
    recent ``cohort_days`` of such rows. Strata are amount quantiles of that cohort's
    reported frauds; strata with too few frauds are merged into a single global curve.
    """
    mature = obs.loc[(obs["age"] >= REPORT_HORIZON_DAYS) & (obs["age"] < REPORT_HORIZON_DAYS + cohort_days)]
    frauds = mature.loc[mature["label"] == 1]
    if len(frauds) < min_frauds:
        raise ValueError(
            f"only {len(frauds)} matured frauds before cutoff {cutoff}; need {min_frauds} to estimate delays"
        )
    grid = np.arange(0.0, REPORT_HORIZON_DAYS + step, step)
    amounts = frauds["amount"].to_numpy()
    # report_day - day can exceed the horizon by float rounding; the horizon is a hard cap
    delays = np.minimum(frauds["delay"].to_numpy(), REPORT_HORIZON_DAYS)
    k = n_strata
    while k > 1:
        edges = np.quantile(amounts, np.linspace(0, 1, k + 1)[1:-1])
        s = np.searchsorted(edges, amounts, side="right")
        counts = np.bincount(s, minlength=k)
        if counts.min() >= min_frauds:
            break
        k -= 1
    if k == 1:
        edges = np.array([])
        s = np.zeros(len(amounts), dtype=int)
        counts = np.array([len(amounts)])
    cdf = np.vstack([_ecdf(delays[s == j], grid) for j in range(k)])
    return DelayCurve(edges=np.asarray(edges, dtype=float), grid=grid, cdf=cdf, n_frauds=counts)


def _ecdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.sort(x)
    return np.searchsorted(x, grid, side="right") / len(x)


def posterior_unreported_fraud(p: np.ndarray, reported_by: np.ndarray) -> np.ndarray:
    """P(fraud | no chargeback yet, x, age) by Bayes.

    p = P(fraud | x), F = P(reported by now | fraud, age). A currently-unlabelled row is
    fraud-not-yet-reported with probability p(1-F) / (p(1-F) + (1-p)).
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    miss = p * (1.0 - np.clip(reported_by, 0.0, 1.0))
    denom = miss + (1.0 - p)
    return np.divide(miss, denom, out=np.zeros_like(miss), where=denom > 0)
