"""Model wrapper and the training strategies that differ only in how they treat young labels."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from labellag.labels import DelayCurve, estimate_delay_curve, posterior_unreported_fraud
from labellag.simulate import CATEGORICAL, FEATURES

STRATEGIES = ("naive", "mature_only", "drop_young_negatives", "ipw", "soft_label")


@dataclass
class ModelConfig:
    max_iter: int = 150
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 200
    l2_regularization: float = 5.0
    seed: int = 0


def make_model(cfg: ModelConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
        early_stopping=False,
        random_state=cfg.seed,
    )


def X(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES].to_numpy(dtype=float)


def fit(df: pd.DataFrame, y: np.ndarray, w: np.ndarray | None, cfg: ModelConfig) -> HistGradientBoostingClassifier:
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        raise ValueError("training data has a single class; widen the window")
    m = make_model(cfg)
    m.fit(X(df), y, sample_weight=w)
    return m


def predict(model: HistGradientBoostingClassifier, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X(df))[:, 1]


@dataclass
class TrainResult:
    strategy: str
    model: HistGradientBoostingClassifier
    cutoff: float
    n_rows: int
    n_positive: float
    curve: DelayCurve | None = None
    info: dict = field(default_factory=dict)


def train(
    obs: pd.DataFrame,
    cutoff: float,
    strategy: str,
    window_days: float = 180.0,
    mature_q: float = 0.90,
    em_rounds: int = 1,
    min_F: float = 0.1,
    cfg: ModelConfig = ModelConfig(),
    curve: DelayCurve | None = None,
) -> TrainResult:
    """Train one strategy on an ``observe``-d frame (no ground truth columns needed)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    win = obs.loc[obs["age"] < window_days].reset_index(drop=True)
    if strategy == "naive":
        m = fit(win, win["label"], None, cfg)
        return TrainResult(strategy, m, cutoff, len(win), float(win["label"].sum()))

    curve = curve or estimate_delay_curve(obs, cutoff)
    min_age = curve.quantile(mature_q)
    old = (win["age"] >= min_age).to_numpy()
    info = {"mature_age_days": min_age}

    if strategy == "mature_only":
        d = win.loc[old]
        m = fit(d, d["label"], None, cfg)
        return TrainResult(strategy, m, cutoff, len(d), float(d["label"].sum()), curve, info)

    if strategy == "drop_young_negatives":
        keep = old | (win["label"].to_numpy() == 1)
        d = win.loc[keep]
        m = fit(d, d["label"], None, cfg)
        return TrainResult(strategy, m, cutoff, len(d), float(d["label"].sum()), curve, info)

    F = curve.reported_by(win["age"].to_numpy(), win["amount"].to_numpy())
    lab = win["label"].to_numpy() == 1
    if strategy == "ipw":
        w = _ipw_weights(lab, F, min_F)
        m = fit(win, win["label"], w, cfg)
        info["ipw_max_weight"] = float(w.max())
        return TrainResult(strategy, m, cutoff, len(win), float(w[lab].sum()), curve, info)

    # soft_label: EM over "no chargeback yet" rows. Seed P(fraud | x) with cross-fitted IPW
    # models (so a row's own missing label never feeds its own seed score), then iterate.
    unl = ~lab
    fold = np.random.default_rng(cfg.seed).integers(0, 2, size=len(win))
    w_ipw = _ipw_weights(lab, F, min_F)
    p = _crossfit(win, fold, lambda d, idx: fit(d, d["label"], w_ipw[idx], cfg))
    for _ in range(em_rounds):
        q = np.where(unl, posterior_unreported_fraud(p, F), 1.0)
        p = _crossfit(win, fold, lambda d, idx, q=q: fit(*_soft_dataset(d, q[idx], unl[idx]), cfg))
    q = np.where(unl, posterior_unreported_fraud(p, F), 1.0)
    m = fit(*_soft_dataset(win, q, unl), cfg)
    info["expected_hidden_frauds"] = float(q[unl].sum())
    info["observed_frauds"] = float(lab.sum())
    return TrainResult(strategy, m, cutoff, len(win), float(q.sum()), curve, info)


def _ipw_weights(lab: np.ndarray, F: np.ndarray, min_F: float) -> np.ndarray:
    """Reported frauds count 1/F(age): a fraud seen at an age where only 25% of frauds
    have been reported stands in for four. Clipped at 1/min_F to bound variance."""
    return np.where(lab, 1.0 / np.maximum(F, min_F), 1.0)


def _crossfit(win: pd.DataFrame, fold: np.ndarray, fit_fn) -> np.ndarray:
    p = np.empty(len(win))
    for k in (0, 1):
        idx = np.flatnonzero(fold != k)
        p[fold == k] = predict(fit_fn(win.iloc[idx], idx), win.loc[fold == k])
    return p


def _soft_dataset(win: pd.DataFrame, q: np.ndarray, unl: np.ndarray):
    """Labelled positives at weight 1; each unlabelled row split into a positive copy
    with weight q and a negative copy with weight 1 - q (EM with soft targets)."""
    pos_copy = unl & (q > 1e-4)
    parts = [win, win.loc[pos_copy]]
    y = np.concatenate([np.where(unl, 0, 1), np.ones(pos_copy.sum(), dtype=int)])
    w = np.concatenate([np.where(unl, 1.0 - q, 1.0), q[pos_copy]])
    return pd.concat(parts, ignore_index=True), y, w


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """Hand-written risk rules of the sort a fraud team starts with (no learning)."""
    return (
        1.0 * df["foreign"].to_numpy()
        + 1.0 * df["new_device"].to_numpy()
        + 0.5 * (df["hour"].to_numpy() <= 4)
        + 0.3 * np.minimum(df["velocity_24h"].to_numpy(), 8)
        + 0.7 * (df["card_age_days"].to_numpy() < 30)
        + 0.5 * (df["dist_km"].to_numpy() > 300)
        + 0.002 * np.minimum(df["amount"].to_numpy(), 500)
    )
