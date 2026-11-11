"""Experiment: split, fit every model on train, evaluate on test, write a report."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from churnclock import features as F
from churnclock.metrics import (
    calibration_error,
    calibration_table,
    censoring_km,
    concordance_index,
    uno_c_index,
    brier_ipcw,
    integrated_brier,
)
from churnclock.survival import CoxPH, KaplanMeier
from churnclock.synth import EFFECTS, true_post_landmark_survival

HORIZONS = (90, 180, 365)
IBS_GRID = np.arange(15, 541, 15, dtype=float)
C_HORIZON = 180  # logistic models need one horizon to define a risk score


def split_ids(ids, test_frac: float = 0.25, seed: int = 2026) -> np.ndarray:
    """Deterministic customer-level split: True = test. Independent of row order."""
    h = [int(hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for i in ids]
    return np.array(h) < test_frac


@dataclass
class Predictions:
    surv_h: np.ndarray  # n x len(HORIZONS)
    surv_grid: np.ndarray  # n x len(IBS_GRID)
    risk: np.ndarray  # higher = churns sooner


class LogisticHorizon:
    """One logistic classifier per horizon: P(cancel within h days of the landmark).

    mode="censored_as_retained": customers censored before h are labelled 'did not churn'.
    mode="drop_censored": customers whose status at h is unknown are dropped from training.
    Both are common shortcuts; the evaluation measures what they cost."""

    def __init__(self, mode: str):
        if mode not in ("censored_as_retained", "drop_censored"):
            raise ValueError(mode)
        self.mode = mode
        self.models: dict[float, object] = {}

    def fit(self, X, duration, event, times):
        for h in times:
            y = ((event == 1) & (duration <= h)).astype(int)
            keep = np.ones_like(y, bool) if self.mode == "censored_as_retained" else (y == 1) | (duration > h)
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            m.fit(X[keep], y[keep])
            self.models[float(h)] = m
        return self

    def survival(self, X, times):
        return np.column_stack([1 - self.models[float(h)].predict_proba(X)[:, 1] for h in times])


def _fit_predict(name, train, test, cols, truth=None) -> tuple[Predictions, object]:
    Xtr, Xte = train[cols].to_numpy(float), test[cols].to_numpy(float)
    dtr, etr = train["duration"].to_numpy(float), train["event"].to_numpy(int)
    if name == "kaplan_meier":
        km = KaplanMeier().fit(dtr, etr)
        n = len(test)
        sh = np.tile(km.predict(np.array(HORIZONS, float)), (n, 1))
        sg = np.tile(km.predict(IBS_GRID), (n, 1))
        return Predictions(sh, sg, np.zeros(n)), km
    if name.startswith("logistic_"):
        mode = name.removeprefix("logistic_")
        times = sorted(set(HORIZONS) | set(IBS_GRID.tolist()))
        lg = LogisticHorizon(mode).fit(Xtr, dtr, etr, times)
        sh = lg.survival(Xte, HORIZONS)
        sg = lg.survival(Xte, IBS_GRID)
        # survival curves from independent per-horizon fits need not be monotone; enforce it
        sg = np.minimum.accumulate(sg, axis=1)
        return Predictions(sh, sg, 1 - lg.survival(Xte, [C_HORIZON])[:, 0]), lg
    if name.startswith("cox"):
        cox = CoxPH(l2=1.0).fit(Xtr, dtr, etr, feature_names=cols)
        return (
            Predictions(cox.survival(Xte, HORIZONS), cox.survival(Xte, IBS_GRID), cox.risk_score(Xte)),
            cox,
        )
    if name == "oracle":
        eta = test["eta"].to_numpy(float)
        return (
            Predictions(
                true_post_landmark_survival(eta, np.array(HORIZONS, float)),
                true_post_landmark_survival(eta, IBS_GRID),
                eta,
            ),
            None,
        )
    raise ValueError(name)


def evaluate(pred: Predictions, test: pd.DataFrame, G) -> dict:
    T, E = test["duration"].to_numpy(float), test["event"].to_numpy(int)
    out = {
        "c_index": round(concordance_index(T, E, pred.risk), 4),
        "uno_c_365": round(uno_c_index(T, E, pred.risk, G, tau=365), 4),
    }
    for k, h in enumerate(HORIZONS):
        out[f"brier_{h}"] = round(brier_ipcw(T, E, pred.surv_h[:, k], h, G), 4)
    out["ibs"] = round(integrated_brier(T, E, pred.surv_grid, IBS_GRID, G), 4)
    km = KaplanMeier().fit(T, E)
    for k, h in enumerate(HORIZONS):
        out[f"mean_pred_churn_{h}"] = round(float(1 - pred.surv_h[:, k].mean()), 4)
        out[f"km_test_churn_{h}"] = round(float(1 - km.predict(np.array([h]))[0]), 4)
    k180 = HORIZONS.index(180)
    cal = calibration_table(T, E, 1 - pred.surv_h[:, k180], 180, n_bins=5) if pred.risk.std() > 0 else []
    out["calibration_180"] = cal
    out["calibration_error_180"] = round(calibration_error(cal), 4) if cal else None
    return out


MODELS = ("kaplan_meier", "logistic_censored_as_retained", "logistic_drop_censored", "cox", "cox_leaky", "oracle")


def run(tables: dict[str, pd.DataFrame], seed: int = 2026, test_frac: float = 0.25) -> dict:
    customers, usage = tables["customers"], tables["usage"]
    clean = F.build_dataset(customers, usage, leaky=False)
    leaky = F.build_dataset(customers, usage, leaky=True)
    is_test = split_ids(clean["customer_id"], test_frac, seed)
    if "truth" in tables:
        clean = clean.merge(tables["truth"][["customer_id", "eta"]], on="customer_id", how="left")
    train, test = clean[~is_test].reset_index(drop=True), clean[is_test].reset_index(drop=True)
    ltrain, ltest = leaky[~is_test].reset_index(drop=True), leaky[is_test].reset_index(drop=True)
    assert not set(train.customer_id) & set(test.customer_id)

    G = censoring_km(train["duration"], train["event"])  # censoring distribution from TRAIN only
    cols = F.feature_columns()
    report = {
        "n_customers": int(len(customers)),
        "landmark_cohort": int(len(clean)),
        "excluded_before_landmark": int(len(customers) - len(clean)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_event_rate": round(float(train["event"].mean()), 4),
        "test_event_rate": round(float(test["event"].mean()), 4),
        "split": {"kind": "customer-level sha256 hash", "seed": seed, "test_frac": test_frac},
        "horizons_days_after_landmark": list(HORIZONS),
        "ibs_grid": [float(IBS_GRID[0]), float(IBS_GRID[-1]), float(IBS_GRID[1] - IBS_GRID[0])],
        "models": {},
    }
    fitted = {}
    for name in MODELS:
        if name == "oracle" and "eta" not in test:
            continue
        if name == "cox_leaky":
            pred, model = _fit_predict(name, ltrain, ltest, cols + F.LEAKY_EXTRA)
        else:
            pred, model = _fit_predict(name, train, test, cols)
        report["models"][name] = evaluate(pred, test, G)
        fitted[name] = model

    cox = fitted["cox"]
    coef = dict(zip(cols, cox.coef_))
    report["cox_coefficients"] = {
        c: {"estimated": round(float(coef[c]), 3), "true": EFFECTS.get(_TRUE_NAME.get(c, c))}
        for c in cols
    }
    report["cox_converged_iterations"] = cox.n_iter_
    return {"report": report, "cox": cox}


# model feature -> generator effect name (None when the generator has no direct effect)
_TRUE_NAME = {
    "tickets_30d": "tickets_0",
    "payment_failed_30d": "payment_failed_0",
    "discount": "discount",
    "plan_pro": "plan_pro",
    "plan_team": "plan_team",
    "channel_paid_social": "channel_paid_social",
    "channel_referral": "channel_referral",
}


def format_report(r: dict) -> str:
    lines = [
        f"customers={r['n_customers']}  landmark cohort={r['landmark_cohort']} "
        f"(excluded before day 30: {r['excluded_before_landmark']})  train={r['n_train']}  test={r['n_test']}",
        f"event rate train={r['train_event_rate']}  test={r['test_event_rate']}",
        "",
        f"{'model':32s} {'C-index':>8s} {'BS@90':>7s} {'BS@180':>7s} {'BS@365':>7s} {'IBS':>7s} "
        f"{'pred@180':>8s} {'KM@180':>7s} {'cal@180':>8s}",
    ]
    for name, m in r["models"].items():
        cal = "-" if m["calibration_error_180"] is None else f"{m['calibration_error_180']:.4f}"
        lines.append(
            f"{name:32s} {m['c_index']:8.4f} {m['brier_90']:7.4f} {m['brier_180']:7.4f} {m['brier_365']:7.4f} "
            f"{m['ibs']:7.4f} {m['mean_pred_churn_180']:8.4f} {m['km_test_churn_180']:7.4f} {cal:>8s}"
        )
    lines += ["", "Cox log-hazard ratios (original feature scale) vs generator truth:"]
    for c, v in r["cox_coefficients"].items():
        true = "n/a (proxy/none)" if v["true"] is None else f"{v['true']:+.3f}"
        lines.append(f"  {c:22s} est={v['estimated']:+.3f}  true={true}")
    return "\n".join(lines)


def write_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result["report"], indent=2) + "\n")
    (out_dir / "metrics.txt").write_text(format_report(result["report"]) + "\n")
    (out_dir / "cox.model.json").write_text(json.dumps(result["cox"].to_dict()) + "\n")
