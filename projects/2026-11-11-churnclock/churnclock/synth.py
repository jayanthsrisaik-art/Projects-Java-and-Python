"""Synthetic subscription business with a known churn mechanism.

The generator writes three raw tables, shaped like an analytics export:

* ``customers.csv`` - one row per customer as of the snapshot date: signup date, plan,
  acquisition channel, discount flag, price, status (active / cancelled / closed_other)
  and the end date for non-active customers.
* ``usage.csv`` - one row per customer per 30-day *period since signup* the customer was
  active and observed in: logins, support tickets, failed payments. Partial periods are
  included with proportionally fewer events.
* ``truth.csv`` - generator internals (latent engagement, true post-landmark linear
  predictor). Only the oracle reference reads this file; no model ever sees it.

Ground truth: churn after the 30-day landmark follows a Weibull proportional-hazards
model, so S(t | x) = exp(-(t / scale) ** shape * exp(eta)). Observation ends at the
snapshot date (administrative censoring - staggered signups mean recent customers are
heavily censored) or when an account is closed for a non-churn reason (random censoring).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LANDMARK_DAYS = 30
PERIOD_DAYS = 30
SNAPSHOT = dt.date(2026, 10, 31)

PLANS = ("basic", "pro", "team")
PLAN_P = (0.5, 0.35, 0.15)
PLAN_PRICE = {"basic": 12.0, "pro": 29.0, "team": 79.0}
CHANNELS = ("organic", "paid_social", "referral")
CHANNEL_P = (0.45, 0.35, 0.20)

# true post-landmark hazard effects (log hazard ratios)
WEIBULL_SHAPE = 1.2
WEIBULL_SCALE = 520.0
EFFECTS = {
    "engagement_z": -0.55,
    "tickets_0": 0.22,
    "payment_failed_0": 0.85,
    "plan_pro": -0.30,
    "plan_team": -0.60,
    "channel_paid_social": 0.40,
    "channel_referral": -0.30,
    "discount": 0.45,
}


@dataclass(frozen=True)
class SynthConfig:
    n_customers: int = 6000
    seed: int = 7
    signup_window_days: int = 720
    other_close_mean_days: float = 2000.0


def true_post_landmark_survival(eta: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Oracle S(t | survived to landmark) for t days after the landmark. Shape (n, len(t))."""
    eta = np.asarray(eta, float)[:, None]
    t = np.asarray(t, float)[None, :]
    return np.exp(-((t / WEIBULL_SCALE) ** WEIBULL_SHAPE) * np.exp(eta))


def generate(cfg: SynthConfig = SynthConfig()) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_customers
    ids = np.array([f"C{i:05d}" for i in range(n)])

    offset = rng.integers(0, cfg.signup_window_days, n)
    signup = np.array([SNAPSHOT - dt.timedelta(days=int(o)) for o in offset])
    follow_up = offset.astype(float)  # days from signup to snapshot

    z = rng.normal(0.0, 1.0, n)
    plan = rng.choice(PLANS, n, p=PLAN_P)
    channel = rng.choice(CHANNELS, n, p=CHANNEL_P)
    discount = rng.random(n) < np.where(channel == "paid_social", 0.25, 0.10)
    price = np.array([PLAN_PRICE[p] for p in plan]) * np.where(discount, 0.8, 1.0)

    login_rate = np.exp(2.3 + 0.5 * z)  # logins per full 30-day period
    ticket_rate = np.exp(-0.7 - 0.35 * z)
    fail_p = np.where(plan == "basic", 0.12, 0.06)

    # period-0 activity (drawn up front because it drives the hazard)
    logins0 = rng.poisson(login_rate)
    tickets0 = rng.poisson(ticket_rate)
    fail0 = (rng.random(n) < fail_p).astype(int)

    eta = (
        EFFECTS["engagement_z"] * z
        + EFFECTS["tickets_0"] * tickets0
        + EFFECTS["payment_failed_0"] * fail0
        + EFFECTS["plan_pro"] * (plan == "pro")
        + EFFECTS["plan_team"] * (plan == "team")
        + EFFECTS["channel_paid_social"] * (channel == "paid_social")
        + EFFECTS["channel_referral"] * (channel == "referral")
        + EFFECTS["discount"] * discount
    )

    early_p = 1.0 / (1.0 + np.exp(-(-2.6 - 0.6 * z + 0.8 * fail0)))
    early = rng.random(n) < early_p
    u = rng.random(n)
    residual = WEIBULL_SCALE * (-np.log(u) / np.exp(eta)) ** (1.0 / WEIBULL_SHAPE)
    churn_day = np.where(early, rng.integers(1, LANDMARK_DAYS, n), LANDMARK_DAYS + np.ceil(residual))
    other_close = np.ceil(rng.exponential(cfg.other_close_mean_days, n))

    end = np.minimum.reduce([churn_day, follow_up, other_close])
    cancelled = churn_day <= np.minimum(follow_up, other_close)
    closed_other = (~cancelled) & (other_close <= follow_up)
    status = np.where(cancelled, "cancelled", np.where(closed_other, "closed_other", "active"))

    end_date = [
        "" if s == "active" else (signup[i] + dt.timedelta(days=int(end[i]))).isoformat()
        for i, s in enumerate(status)
    ]
    customers = pd.DataFrame(
        {
            "customer_id": ids,
            "signup_date": [d.isoformat() for d in signup],
            "plan": plan,
            "channel": channel,
            "discount": discount.astype(int),
            "monthly_price": np.round(price, 2),
            "status": status,
            "end_date": end_date,
        }
    )

    # usage: one row per observed (possibly partial) 30-day period
    rows = []
    for i in range(n):
        days_active = end[i]
        if days_active <= 0:
            continue
        n_periods = int(np.ceil(days_active / PERIOD_DAYS))
        for p in range(n_periods):
            frac = min(PERIOD_DAYS, days_active - p * PERIOD_DAYS) / PERIOD_DAYS
            if p == 0:
                if days_active >= PERIOD_DAYS:
                    lg, tk, pf = logins0[i], tickets0[i], fail0[i]
                else:  # churned/censored inside period 0: thin out the period-0 draw
                    lg = rng.binomial(logins0[i], frac)
                    tk = rng.binomial(tickets0[i], frac)
                    pf = fail0[i]
            else:
                lg = rng.poisson(login_rate[i] * frac)
                tk = rng.poisson(ticket_rate[i] * frac)
                pf = int(rng.random() < fail_p[i] * frac)
            rows.append(
                (
                    ids[i],
                    p,
                    (signup[i] + dt.timedelta(days=p * PERIOD_DAYS)).isoformat(),
                    int(lg),
                    int(tk),
                    int(pf),
                )
            )
    usage = pd.DataFrame(
        rows, columns=["customer_id", "period", "period_start", "logins", "tickets", "payment_failed"]
    )
    truth = pd.DataFrame(
        {"customer_id": ids, "engagement_z": np.round(z, 6), "eta": np.round(eta, 6), "early_churn": early.astype(int)}
    )
    return {"customers": customers, "usage": usage, "truth": truth}


def write(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


def read(data_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("customers", "usage", "truth"):
        path = data_dir / f"{name}.csv"
        if path.exists():
            out[name] = pd.read_csv(path, dtype={"customer_id": str, "end_date": str}, keep_default_na=False)
    return out
