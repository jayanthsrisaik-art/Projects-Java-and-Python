"""Hourly features for the shape model.

Household features are computed only from the meter's own *fully actual*
hours in the build being processed. Hours inside a gap are never used to
describe the household that the gap is being filled for, so training (on
artificially masked gaps) and inference see exactly the same information.
Everything household-specific is divided by the household's mean rate, so one
global model works for small flats and large houses.
"""

from __future__ import annotations

import numpy as np

from .allocate import Coverage
from .timeutil import day_of_week, hour_of_day

HEAT_BASE_C = 16.0
FEATURES = [
    "hod",
    "dow",
    "weekend",
    "temp_c",
    "hdd",
    "prof_hd_rel",
    "prof_how_rel",
    "p10_rel",
    "heat_slope_rel",
    "heat_expect_rel",
    "evening_share",
    "night_share",
]


def household_profile(cov: Coverage, temps: np.ndarray) -> dict:
    hod = hour_of_day(cov.h0, cov.nh)
    dow = day_of_week(cov.h0, cov.nh)
    wknd = (dow >= 5).astype(int)
    full = cov.full_actual
    y = cov.actual_kwh
    mean = max(cov.mean_rate(), 1e-3)

    if full.sum() == 0:
        flat = np.full(cov.nh, mean)
        return {"mean": mean, "prof_hd": flat, "prof_how": flat, "p10": mean, "heat_slope": 0.0,
                "evening_share": 0.0, "night_share": 0.0}

    overall = y[full].mean()
    hour_mean = np.full(24, overall)
    for h in range(24):
        m = full & (hod == h)
        if m.sum() >= 2:
            hour_mean[h] = y[m].mean()
    hd = np.tile(hour_mean[:, None], (1, 2))  # (hour, weekend)
    for h in range(24):
        for w in (0, 1):
            m = full & (hod == h) & (wknd == w)
            if m.sum() >= 2:
                hd[h, w] = y[m].mean()
    how = hd[:, [0] * 5 + [1] * 2].copy()  # (hour, dow) fallback from weekday/weekend
    for h in range(24):
        for d in range(7):
            m = full & (hod == h) & (dow == d)
            if m.sum() >= 2:
                how[h, d] = y[m].mean()
    prof_hd = hd[hod, wknd]
    prof_how = how[hod, dow]

    hdd = np.clip(HEAT_BASE_C - temps, 0, None)
    resid = y[full] - prof_hd[full]
    x = hdd[full] - hdd[full].mean()
    var = float((x * x).sum())
    slope = float((x * resid).sum() / var) if var > 1e-9 and full.sum() >= 48 else 0.0
    slope = max(slope, 0.0)
    total = hour_mean.sum()
    return {
        "mean": mean,
        "prof_hd": prof_hd,
        "prof_how": prof_how,
        "p10": float(np.percentile(y[full], 10)),
        "heat_slope": slope,
        "evening_share": float(hour_mean[17:23].sum() / total) if total > 0 else 0.0,
        "night_share": float((hour_mean[:5].sum() + hour_mean[23]) / total) if total > 0 else 0.0,
    }


def feature_matrix(cov: Coverage, temps: np.ndarray, prof: dict | None = None) -> tuple[np.ndarray, dict]:
    prof = prof or household_profile(cov, temps)
    hod = hour_of_day(cov.h0, cov.nh)
    dow = day_of_week(cov.h0, cov.nh)
    hdd = np.clip(HEAT_BASE_C - temps, 0, None)
    m = prof["mean"]
    X = np.column_stack(
        [
            hod,
            dow,
            (dow >= 5).astype(float),
            temps,
            hdd,
            prof["prof_hd"] / m,
            prof["prof_how"] / m,
            np.full(cov.nh, prof["p10"] / m),
            np.full(cov.nh, prof["heat_slope"] / m),
            prof["heat_slope"] * (hdd - hdd.mean()) / m,
            np.full(cov.nh, prof["evening_share"]),
            np.full(cov.nh, prof["night_share"]),
        ]
    ).astype(float)
    return X, prof
