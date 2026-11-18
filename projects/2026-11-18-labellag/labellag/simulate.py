"""Synthetic card-transaction stream with drifting fraud campaigns and delayed chargebacks.

Every transaction gets a ground-truth ``is_fraud`` flag and, for frauds, the day the
chargeback is *reported* (``report_day``). Models only ever see a label once
``report_day <= cutoff``; the simulator is the only place where the truth lives.

Time is measured in fractional days from the start of the stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURES = [
    "amount",
    "mcc",
    "channel",
    "hour",
    "card_age_days",
    "dist_km",
    "velocity_24h",
    "foreign",
    "new_device",
]
CATEGORICAL = ["mcc", "channel"]
N_MCC = 10
N_CHANNEL = 3  # 0 = card present, 1 = online, 2 = ATM
REPORT_HORIZON_DAYS = 120.0  # chargebacks later than this are not accepted


@dataclass(frozen=True)
class Campaign:
    """A fraud pattern that is active for a while and targets one slice of traffic."""

    start: float
    duration: float
    mcc: int
    channel: int
    amount_lo: float
    amount_hi: float
    peak_logit: float

    def intensity(self, day: np.ndarray) -> np.ndarray:
        """Trapezoid: ramp up over 15 days, hold, ramp down over the last 15 days."""
        ramp = 15.0
        up = np.clip((day - self.start) / ramp, 0.0, 1.0)
        down = np.clip((self.start + self.duration - day) / ramp, 0.0, 1.0)
        return self.peak_logit * np.minimum(up, down)


@dataclass(frozen=True)
class SimConfig:
    days: int = 360
    txns_per_day: int = 1200
    base_logit: float = -7.6
    n_campaigns: int = 10
    campaign_start_max: float = 345.0
    # Report delay: log-normal (days), slower for small amounts (cardholders notice later).
    delay_median_days: float = 16.0
    delay_sigma: float = 0.85
    small_amount_threshold: float = 40.0
    small_amount_delay_factor: float = 2.2
    campaigns: tuple[Campaign, ...] | None = field(default=None)


def make_campaigns(cfg: SimConfig, rng: np.random.Generator) -> tuple[Campaign, ...]:
    out = []
    for _ in range(cfg.n_campaigns):
        lo = float(rng.choice([2.0, 10.0, 60.0, 150.0, 400.0]))
        out.append(
            Campaign(
                start=float(rng.uniform(0.0, cfg.campaign_start_max)),
                duration=float(rng.uniform(120.0, 260.0)),
                mcc=int(rng.integers(0, N_MCC)),
                channel=int(rng.integers(0, N_CHANNEL)),
                amount_lo=lo,
                amount_hi=lo * float(rng.uniform(4.0, 8.0)),
                peak_logit=float(rng.uniform(4.0, 5.5)),
            )
        )
    return tuple(sorted(out, key=lambda c: c.start))


def fraud_logit(df: pd.DataFrame, cfg: SimConfig, campaigns: tuple[Campaign, ...]) -> np.ndarray:
    """True log-odds of fraud. A stationary part plus time-varying campaigns."""
    logit = np.full(len(df), cfg.base_logit)
    logit += 1.8 * df["foreign"].to_numpy()
    logit += 2.2 * df["new_device"].to_numpy()
    logit += 1.2 * (df["hour"].to_numpy() <= 4)
    logit += 0.6 * np.minimum(df["velocity_24h"].to_numpy(), 8)
    logit += 1.6 * (df["card_age_days"].to_numpy() < 30)
    logit += 1.4 * (df["dist_km"].to_numpy() > 300)
    day = df["day"].to_numpy()
    amount = df["amount"].to_numpy()
    mcc = df["mcc"].to_numpy()
    channel = df["channel"].to_numpy()
    for c in campaigns:
        hit = (mcc == c.mcc) & (channel == c.channel) & (amount >= c.amount_lo) & (amount < c.amount_hi)
        logit += np.where(hit, c.intensity(day), 0.0)
    return logit


def simulate(cfg: SimConfig = SimConfig(), seed: int = 0) -> tuple[pd.DataFrame, tuple[Campaign, ...]]:
    rng = np.random.default_rng(seed)
    campaigns = cfg.campaigns if cfg.campaigns is not None else make_campaigns(cfg, rng)
    n = cfg.days * cfg.txns_per_day
    day = np.sort(rng.uniform(0.0, cfg.days, size=n))
    channel = rng.choice(N_CHANNEL, size=n, p=[0.55, 0.38, 0.07])
    df = pd.DataFrame(
        {
            "txn_id": np.arange(n, dtype=np.int64),
            "day": day,
            "amount": np.round(np.exp(rng.normal(3.4, 1.15, size=n)), 2),
            "mcc": rng.choice(N_MCC, size=n, p=_mcc_mix()),
            "channel": channel,
            "hour": rng.integers(0, 24, size=n),
            "card_age_days": rng.gamma(2.0, 400.0, size=n).astype(int),
            "dist_km": np.round(rng.exponential(25.0, size=n) * np.where(rng.random(n) < 0.04, 30, 1), 1),
            "velocity_24h": rng.poisson(1.3, size=n),
            "foreign": (rng.random(n) < 0.06).astype(int),
            "new_device": ((channel == 1) & (rng.random(n) < 0.12)).astype(int),
        }
    )
    p = 1.0 / (1.0 + np.exp(-fraud_logit(df, cfg, campaigns)))
    is_fraud = rng.random(n) < p
    median = np.where(
        df["amount"].to_numpy() < cfg.small_amount_threshold,
        cfg.delay_median_days * cfg.small_amount_delay_factor,
        cfg.delay_median_days,
    )
    delay = np.exp(np.log(median) + cfg.delay_sigma * rng.normal(size=n))
    delay = np.minimum(delay, REPORT_HORIZON_DAYS)
    df["true_p"] = p
    df["is_fraud"] = is_fraud.astype(int)
    df["report_day"] = np.where(is_fraud, day + delay, np.nan)
    return df, campaigns


def _mcc_mix() -> np.ndarray:
    w = np.array([18, 14, 12, 11, 10, 9, 8, 7, 6, 5], dtype=float)
    return w / w.sum()
