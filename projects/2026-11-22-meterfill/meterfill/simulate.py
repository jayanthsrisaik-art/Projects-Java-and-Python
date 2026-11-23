"""Synthetic smart-meter world with known ground truth.

Generates hourly weather, per-household true hourly consumption, and the
register readings a head-end system would actually receive: irregular
cadence, communication outages, duplicates, out-of-order arrival, misreads
(digit errors), register rollover and meter exchanges. Every fault is
labelled so the pipeline can be scored against it.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import HOUR, REGISTER_MAX
from .timeutil import day_of_week, fmt_ts, hour_of_day, parse_ts


@dataclass
class SimConfig:
    households: int = 150
    days: int = 56
    seed: int = 7
    start: str = "2026-01-05T00:00:00Z"  # a Monday in winter (electric heating matters)
    cadence_s: int = 1800
    jitter_s: int = 300
    outages_per_meter: float = 3.0  # mean number over the whole period
    outage_min_h: float = 2.5
    outage_max_h: float = 72.0
    misread_rate: float = 1 / 1500
    rollover_frac: float = 0.08
    exchange_frac: float = 0.04
    dup_rate: float = 0.02
    trailing_frac: float = 0.05
    max_arrival_delay_s: int = 6 * 3600


@dataclass
class World:
    config: SimConfig
    h0: int
    nh: int
    temps: np.ndarray  # (nh,)
    meter_ids: list[str]
    truth: np.ndarray  # (meters, nh) true kWh per hour
    readings: list[tuple[str, int, float]]  # in ARRIVAL order (meter_id, ts, register)
    labels: dict = field(default_factory=dict)  # meter_id -> {"misread": set(ts), "rollover": set, "reset": set}
    households: list[dict] = field(default_factory=list)

    def meter_index(self) -> dict[str, int]:
        return {m: i for i, m in enumerate(self.meter_ids)}


def _weather(rng: np.random.Generator, h0: int, nh: int) -> np.ndarray:
    hod = hour_of_day(h0, nh)
    t = np.arange(nh) / 24.0
    trend = 3.0 + 4.0 * np.sin(2 * math.pi * (t / 40.0) + rng.uniform(0, 2 * math.pi))
    daily = 4.0 * np.cos(2 * math.pi * (hod - 15) / 24.0)
    noise = np.zeros(nh)
    eps = rng.normal(0, 0.35, nh)
    for i in range(1, nh):
        noise[i] = 0.97 * noise[i - 1] + eps[i]
    return np.round(trend + daily + noise, 1)


def _bump(hod: np.ndarray, centre: float, width: float) -> np.ndarray:
    d = np.minimum(np.abs(hod - centre), 24 - np.abs(hod - centre))
    return np.exp(-0.5 * (d / width) ** 2)


ARCHETYPES = {
    # (weekday bumps, weekend bumps) as (centre, width, height)
    "commuter": ([(7, 1.0, 1.0), (19, 2.0, 1.6)], [(10, 2.5, 1.0), (19, 2.5, 1.4)]),
    "homebody": ([(9, 2.0, 0.8), (13, 2.5, 1.0), (19, 2.0, 1.2)], [(10, 2.5, 1.0), (18, 3.0, 1.2)]),
    "nightowl": ([(12, 2.0, 0.6), (23, 2.0, 1.5)], [(14, 3.0, 0.8), (0, 2.5, 1.5)]),
}


def _household(rng: np.random.Generator, h0: int, nh: int, temps: np.ndarray) -> tuple[np.ndarray, dict]:
    hod = hour_of_day(h0, nh).astype(float)
    dow = day_of_week(h0, nh)
    weekend = dow >= 5
    arche = rng.choice(list(ARCHETYPES), p=[0.5, 0.3, 0.2])
    wk, we = ARCHETYPES[arche]
    shift = rng.normal(0, 0.7)  # households differ in their exact routine
    act_wk = sum(h * _bump(hod, c + shift, w) for c, w, h in wk)
    act_we = sum(h * _bump(hod, c + shift, w) for c, w, h in we)
    activity = np.where(weekend, act_we, act_wk)

    base = rng.uniform(0.12, 0.45)
    amp = rng.uniform(0.3, 1.3)
    electric_heat = rng.random() < 0.35
    slope = rng.uniform(0.07, 0.22) if electric_heat else rng.uniform(0.0, 0.025)
    hdd = np.clip(16.0 - temps, 0, None)
    # heat pumps / heaters run harder when people are home
    heat = slope * hdd * (0.6 + 0.4 * np.minimum(activity, 1.5) / 1.5)

    kwh = base + amp * activity + heat
    kwh = kwh * rng.lognormal(0.0, 0.22, nh)
    # occasional appliances (oven, kettle, washer): more likely when active
    p_app = 0.04 * (0.3 + activity)
    kwh = kwh + (rng.random(nh) < p_app) * rng.uniform(0.8, 2.2, nh)

    has_ev = rng.random() < 0.2
    if has_ev:
        days = nh // 24
        for d in range(days):
            if rng.random() < 0.4:
                start = d * 24 + int(rng.integers(21, 26))  # 21:00 .. 01:00
                dur = rng.uniform(1.5, 5.0)
                full = int(dur)
                for k in range(full):
                    if start + k < nh:
                        kwh[start + k] += 7.2
                if start + full < nh:
                    kwh[start + full] += 7.2 * (dur - full)
    info = {"archetype": str(arche), "electric_heat": bool(electric_heat), "ev": bool(has_ev)}
    return np.round(kwh, 4), info


def _windows(rng: np.random.Generator, cfg: SimConfig, t_start: int, t_end: int) -> list[tuple[int, int]]:
    n = rng.poisson(cfg.outages_per_meter)
    out = []
    for _ in range(n):
        dur = math.exp(rng.uniform(math.log(cfg.outage_min_h), math.log(cfg.outage_max_h))) * HOUR
        lo, hi = t_start + 6 * HOUR, t_end - dur - 6 * HOUR
        if hi <= lo:
            continue
        s = int(rng.uniform(lo, hi))
        out.append((s, int(s + dur)))
    return out


def simulate(cfg: SimConfig | None = None) -> World:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    h0 = parse_ts(cfg.start)
    nh = cfg.days * 24
    t_end = h0 + nh * HOUR
    temps = _weather(rng, h0, nh)

    meter_ids = [f"MTR{1000 + i:05d}" for i in range(cfg.households)]
    truth = np.zeros((cfg.households, nh))
    readings: list[tuple[int, str, int, float]] = []  # (arrival, meter, ts, value)
    labels: dict[str, dict[str, set]] = {}
    households = []

    for mi, mid in enumerate(meter_ids):
        kwh, info = _household(rng, h0, nh, temps)
        truth[mi] = kwh
        cum_hour = np.concatenate([[0.0], np.cumsum(kwh)])  # cumulative at each hour start

        def cum_at(ts: float) -> float:
            x = (ts - h0) / HOUR
            i = min(int(x), nh - 1)
            return cum_hour[i] + kwh[i] * (x - i)

        lab = {"misread": set(), "rollover": set(), "reset": set()}
        labels[mid] = lab
        if rng.random() < cfg.rollover_frac:
            start_reg = REGISTER_MAX - rng.uniform(20, 0.6 * cum_hour[-1])
            info["rollover_planned"] = True
        else:
            start_reg = rng.uniform(500, 60_000)
        exchange_ts = None
        if rng.random() < cfg.exchange_frac:
            margin = min(7 * 24 * HOUR, nh * HOUR // 4)
            exchange_ts = int(rng.uniform(h0 + margin, t_end - margin))
            info["exchange_ts"] = fmt_ts(exchange_ts)

        # reading schedule
        k = np.arange(0, nh * HOUR // cfg.cadence_s)
        ts = h0 + k * cfg.cadence_s + rng.integers(-cfg.jitter_s, cfg.jitter_s + 1, k.size)
        ts = np.unique(ts[(ts > h0) & (ts < t_end)])
        keep = np.ones(ts.size, dtype=bool)
        for s, e in _windows(rng, cfg, h0, t_end):
            keep &= ~((ts >= s) & (ts < e))
        if exchange_ts is not None:  # installer visit: meter offline for 1-6 h
            keep &= ~((ts >= exchange_ts - int(rng.uniform(1, 6) * HOUR)) & (ts < exchange_ts))
        if rng.random() < cfg.trailing_frac:  # meter goes silent before the end
            keep &= ts < t_end - int(rng.uniform(6, 96) * HOUR)
        ts = ts[keep]

        values = []
        exchanged = False
        last_unwrapped = None
        for t in ts:
            if exchange_ts is not None and t >= exchange_ts:
                reg = cum_at(t) - cum_at(exchange_ts)  # new meter starts at 0.00
                if not exchanged:
                    lab["reset"].add(int(t))
                    exchanged = True
                    last_unwrapped = None
                unwrapped = reg
            else:
                unwrapped = start_reg + cum_at(t)
                reg = unwrapped % REGISTER_MAX
            if last_unwrapped is not None and int(unwrapped // REGISTER_MAX) > int(last_unwrapped // REGISTER_MAX):
                lab["rollover"].add(int(t))
            last_unwrapped = unwrapped
            values.append(math.floor(reg * 100) / 100)  # registers truncate, so they stay monotone

        # misreads: digit errors, never next to a reset or at the ends
        n = len(ts)
        reset_pos = {i for i, t in enumerate(ts) if int(t) in lab["reset"]}
        for i in range(3, n - 4):
            if rng.random() < cfg.misread_rate and not any(abs(i - r) <= 3 for r in reset_pos):
                if int(ts[i]) in lab["rollover"] or int(ts[i + 1]) in lab["rollover"]:
                    continue
                offset = int(rng.integers(1, 10)) * 10 ** int(rng.integers(3, 5)) * (1 if rng.random() < 0.5 else -1)
                values[i] = round((values[i] + offset) % REGISTER_MAX, 2)
                lab["misread"].add(int(ts[i]))

        for t, v in zip(ts, values):
            arrival = int(t) + int(rng.integers(0, cfg.max_arrival_delay_s))
            readings.append((arrival, mid, int(t), float(v)))
            if rng.random() < cfg.dup_rate:  # re-sent after a retry
                readings.append((arrival + int(rng.integers(60, 86_400)), mid, int(t), float(v)))
        households.append({"meter_id": mid, **info})

    readings.sort(key=lambda r: (r[0], r[1], r[2]))
    return World(
        config=cfg,
        h0=h0,
        nh=nh,
        temps=temps,
        meter_ids=meter_ids,
        truth=truth,
        readings=[(m, t, v) for _, m, t, v in readings],
        labels=labels,
        households=households,
    )


def write_world(world: World, out_dir: str | Path, include_truth: bool = True) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"readings": str(out / "readings.csv"), "weather": str(out / "weather.csv")}
    with open(paths["readings"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["meter_id", "ts", "register_kwh"])
        for m, t, v in world.readings:
            w.writerow([m, fmt_ts(t), f"{v:.2f}"])
    with open(paths["weather"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hour_ts", "temp_c"])
        for i, temp in enumerate(world.temps):
            w.writerow([fmt_ts(world.h0 + i * HOUR), f"{temp:.1f}"])
    if include_truth:
        paths["truth"] = str(out / "truth_hourly.csv")
        with open(paths["truth"], "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["meter_id", "hour_ts", "kwh"])
            for mi, m in enumerate(world.meter_ids):
                for i in range(world.nh):
                    w.writerow([m, fmt_ts(world.h0 + i * HOUR), f"{world.truth[mi, i]:.4f}"])
        paths["labels"] = str(out / "truth_faults.csv")
        with open(paths["labels"], "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["meter_id", "ts", "fault"])
            for m in world.meter_ids:
                for kind in ("misread", "rollover", "reset"):
                    for t in sorted(world.labels[m][kind]):
                        w.writerow([m, fmt_ts(t), kind])
    return paths
