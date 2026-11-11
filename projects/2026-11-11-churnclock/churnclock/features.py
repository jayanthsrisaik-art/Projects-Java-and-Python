"""Landmark feature construction.

Prediction is made at the *landmark*: day 30 after signup. Only customers still active
and still under observation at day 30 enter the cohort, and every feature uses only
data from period 0 (days 0-29) plus signup attributes. The target is the time from the
landmark to cancellation, right-censored at the snapshot date or at a non-churn account
closure.

``leaky_features`` builds the tempting lifetime aggregates (total logins/tickets over
the whole observed history). Those totals grow with time survived, so they encode the
outcome; they are built only to measure how much they inflate offline metrics.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from churnclock.synth import LANDMARK_DAYS, SNAPSHOT

CATEGORICAL = {"plan": ("basic", "pro", "team"), "channel": ("organic", "paid_social", "referral")}
NUMERIC = ["log_logins_30d", "tickets_30d", "payment_failed_30d", "discount", "monthly_price"]


def durations(customers: pd.DataFrame, snapshot: dt.date = SNAPSHOT) -> pd.DataFrame:
    """Days observed since signup and whether observation ended in a cancellation."""
    signup = pd.to_datetime(customers["signup_date"])
    end = pd.to_datetime(customers["end_date"].replace("", None))
    end = end.fillna(pd.Timestamp(snapshot))
    out = pd.DataFrame({"customer_id": customers["customer_id"].values})
    out["days_observed"] = (end - signup).dt.days.values
    out["event"] = (customers["status"] == "cancelled").astype(int).values
    if (out["days_observed"] < 0).any():
        raise ValueError("end_date before signup_date")
    return out


def landmark_cohort(customers: pd.DataFrame, snapshot: dt.date = SNAPSHOT) -> pd.DataFrame:
    """Customers active and observed beyond the landmark, with post-landmark duration."""
    d = durations(customers, snapshot)
    keep = d["days_observed"] > LANDMARK_DAYS
    d = d[keep].copy()
    d["duration"] = d["days_observed"] - LANDMARK_DAYS
    return d[["customer_id", "duration", "event"]].reset_index(drop=True)


def landmark_features(customers: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    """Features knowable at day 30. Raises if usage beyond period 0 would be needed."""
    p0 = usage[usage["period"] == 0]
    if (p0["period"] != 0).any():  # defensive: filtered frame must contain period 0 only
        raise AssertionError("post-landmark usage leaked into features")
    agg = p0.groupby("customer_id")[["logins", "tickets", "payment_failed"]].sum()
    f = customers[["customer_id", "plan", "channel", "discount", "monthly_price"]].merge(
        agg, how="left", left_on="customer_id", right_index=True
    )
    f[["logins", "tickets", "payment_failed"]] = f[["logins", "tickets", "payment_failed"]].fillna(0)
    out = pd.DataFrame({"customer_id": f["customer_id"]})
    out["log_logins_30d"] = np.log1p(f["logins"].astype(float))
    out["tickets_30d"] = f["tickets"].astype(float)
    out["payment_failed_30d"] = (f["payment_failed"] > 0).astype(float)
    out["discount"] = f["discount"].astype(float)
    out["monthly_price"] = f["monthly_price"].astype(float)
    for col, levels in CATEGORICAL.items():
        unknown = set(f[col]) - set(levels)
        if unknown:
            raise ValueError(f"unknown {col} values: {sorted(unknown)}")
        for lvl in levels[1:]:  # first level is the reference
            out[f"{col}_{lvl}"] = (f[col] == lvl).astype(float)
    return out.reset_index(drop=True)


def feature_columns() -> list[str]:
    cols = list(NUMERIC)
    for col, levels in CATEGORICAL.items():
        cols += [f"{col}_{lvl}" for lvl in levels[1:]]
    return cols


def leaky_features(customers: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    """Landmark features plus lifetime totals over the full observed history (LEAKY)."""
    base = landmark_features(customers, usage)
    life = usage.groupby("customer_id").agg(
        lifetime_logins=("logins", "sum"), lifetime_tickets=("tickets", "sum"), periods_seen=("period", "count")
    )
    out = base.merge(life, how="left", left_on="customer_id", right_index=True).fillna(0)
    out["lifetime_logins"] = np.log1p(out["lifetime_logins"].astype(float))
    return out


LEAKY_EXTRA = ["lifetime_logins", "lifetime_tickets", "periods_seen"]


def build_dataset(customers: pd.DataFrame, usage: pd.DataFrame, leaky: bool = False) -> pd.DataFrame:
    cohort = landmark_cohort(customers)
    feats = (leaky_features if leaky else landmark_features)(customers, usage)
    return cohort.merge(feats, on="customer_id", how="left")
