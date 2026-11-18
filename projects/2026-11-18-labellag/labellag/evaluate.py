"""Rolling-origin backtest: train at each cutoff with what was knowable, score the next 14 days."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from labellag.labels import estimate_delay_curve, observe
from labellag.model import STRATEGIES, ModelConfig, fit, predict, rule_score, train
from labellag.simulate import REPORT_HORIZON_DAYS, SimConfig, simulate

METHODS = ("rules", *STRATEGIES, "oracle")


def metrics(y: np.ndarray, score: np.ndarray, budget: float = 0.005, probabilistic: bool = True) -> dict:
    """Average precision, recall/precision when reviewing the top ``budget`` share, calibration."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    k = max(1, math.ceil(budget * len(y)))
    top = np.argsort(-score, kind="stable")[:k]
    hits = int(y[top].sum())
    out = {
        "n": int(len(y)),
        "frauds": int(y.sum()),
        "avg_precision": float(average_precision_score(y, score)) if y.any() else float("nan"),
        "recall_at_budget": hits / max(1, int(y.sum())),
        "precision_at_budget": hits / k,
    }
    if probabilistic:
        out["brier"] = float(np.mean((score - y) ** 2))
        out["pred_rate_over_true_rate"] = float(score.mean() / y.mean()) if y.any() else float("nan")
    return out


@dataclass
class BacktestConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    cutoffs: tuple[float, ...] = (240.0, 270.0, 300.0, 330.0)
    test_days: float = 21.0
    window_days: float = 180.0
    budget: float = 0.005
    sim: SimConfig = field(default_factory=SimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def run_cutoff(df: pd.DataFrame, cutoff: float, bc: BacktestConfig, seed: int) -> list[dict]:
    obs = observe(df, cutoff)
    test = df.loc[(df["day"] > cutoff) & (df["day"] <= cutoff + bc.test_days)]
    if test["report_day"].max() > cutoff + bc.test_days + REPORT_HORIZON_DAYS + 1e-9:
        raise AssertionError("test labels exceed the report horizon")
    y = test["is_fraud"].to_numpy()
    cfg = ModelConfig(**{**bc.model.__dict__, "seed": seed})
    curve = estimate_delay_curve(obs, cutoff)

    # label completeness of the training window, measured against the truth
    win_truth = df.loc[(df["day"] <= cutoff) & (df["day"] > cutoff - bc.window_days)]
    young = win_truth["day"] > cutoff - 30
    seen = win_truth["report_day"] <= cutoff
    ctx = {
        "seed": seed,
        "cutoff": cutoff,
        "train_label_completeness": float(seen[win_truth["is_fraud"] == 1].mean()),
        "last30_label_completeness": float(seen[young & (win_truth["is_fraud"] == 1)].mean()),
    }
    rows = []
    rows.append({**ctx, "method": "rules", **metrics(y, rule_score(test), bc.budget, probabilistic=False)})
    for s in STRATEGIES:
        r = train(obs, cutoff, s, window_days=bc.window_days, cfg=cfg, curve=curve)
        m = metrics(y, predict(r.model, test), bc.budget)
        extra = {"train_rows": r.n_rows, "train_positives": round(r.n_positive, 1), **r.info}
        rows.append({**ctx, "method": s, **m, **extra})
    # oracle: same window, ground-truth labels. Not deployable; an upper reference.
    m_or = fit(win_truth, win_truth["is_fraud"], None, cfg)
    rows.append({**ctx, "method": "oracle", **metrics(y, predict(m_or, test), bc.budget)})
    # delay-curve check: estimated F vs. realised F for frauds in the 60 days before cutoff
    rec = win_truth.loc[(win_truth["is_fraud"] == 1) & (win_truth["day"] > cutoff - 60)]
    est = curve.reported_by(np.full(len(rec), 14.0), rec["amount"].to_numpy()).mean()
    real = float(((rec["report_day"] - rec["day"]) <= 14.0).mean())
    for r in rows:
        r["F14_estimated"] = float(est)
        r["F14_realised"] = real
    return rows


def backtest(bc: BacktestConfig, progress=None) -> pd.DataFrame:
    rows = []
    for seed in bc.seeds:
        df, _ = simulate(bc.sim, seed=seed)
        for c in bc.cutoffs:
            rows.extend(run_cutoff(df, c, bc, seed))
            if progress:
                progress(f"seed {seed} cutoff {c:g} done")
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    cols = ["avg_precision", "recall_at_budget", "precision_at_budget", "brier", "pred_rate_over_true_rate"]
    g = res.groupby("method", sort=False)[cols]
    mean = g.mean()
    # paired comparison against naive: in how many (seed, cutoff) runs did AP beat naive?
    base = res.loc[res["method"] == "naive"].set_index(["seed", "cutoff"])["avg_precision"]
    wins, diff, se = {}, {}, {}
    for m, sub in res.groupby("method", sort=False):
        a = sub.set_index(["seed", "cutoff"])["avg_precision"]
        d = (a - base.reindex(a.index)).to_numpy()
        diff[m] = float(d.mean())
        se[m] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        wins[m] = f"{int((d > 0).sum())}/{len(d)}" if m != "naive" else "-"
    mean["ap_minus_naive"] = pd.Series(diff)
    mean["ap_minus_naive_se"] = pd.Series(se)
    mean["ap_beats_naive"] = pd.Series(wins)
    return mean.loc[[m for m in METHODS if m in mean.index]]
