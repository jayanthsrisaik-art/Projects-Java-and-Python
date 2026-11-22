"""Hourly shape predictors and training of the global gradient-boosting model.

Training never uses simulator ground truth. Targets are the meter's own
*actual* hourly values (from short intervals); gaps are created by deleting
readings inside random windows, the meter is rebuilt from what is left, and
the model learns to predict the deleted hours from the remaining ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .allocate import Coverage, coverage
from .clean import clean_meter
from .config import HOUR
from .features import FEATURES, feature_matrix, household_profile

METHODS = ("uniform", "profile", "model", "model_raw")


@dataclass
class ShapeModel:
    estimator: object
    features: list[str]
    meta: dict

    def predict_rel(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.estimator.predict(X), 0.02, None)

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump({"estimator": self.estimator, "features": self.features, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ShapeModel":
        import joblib

        d = joblib.load(path)
        if d["features"] != FEATURES:
            raise ValueError("model was trained with a different feature set; retrain it")
        return cls(d["estimator"], d["features"], d["meta"])


def predict_hourly(method: str, cov: Coverage, temps: np.ndarray, model: ShapeModel | None = None) -> np.ndarray:
    """Predicted kWh for every hour of the window, from the meter's own actual hours."""
    if method == "uniform":
        return np.full(cov.nh, cov.mean_rate())
    if method == "profile":
        return household_profile(cov, temps)["prof_how"]
    if method in ("model", "model_raw"):
        if model is None:
            raise ValueError(f"method {method!r} needs a trained model")
        X, prof = feature_matrix(cov, temps)
        return model.predict_rel(X) * prof["mean"]
    raise ValueError(f"unknown method {method!r}")


def choose_windows(
    cov: Coverage, rng: np.random.Generator, n: int, min_h: float = 3.0, max_h: float = 72.0
) -> list[tuple[int, int]]:
    """Random windows lying entirely in fully-actual hours, non-overlapping, 6 h apart."""
    full = cov.full_actual
    chosen: list[tuple[int, int]] = []
    for _ in range(n * 30):
        if len(chosen) >= n:
            break
        length = math.exp(rng.uniform(math.log(min_h), math.log(max_h))) * HOUR
        ws = cov.h0 + int(rng.uniform(HOUR, cov.nh * HOUR - length - HOUR))
        we = int(ws + length)
        i0, i1 = (ws - cov.h0) // HOUR, (we - 1 - cov.h0) // HOUR
        if not full[i0 : i1 + 1].all():
            continue
        if any(ws < b + 6 * HOUR and a < we + 6 * HOUR for a, b in chosen):
            continue
        chosen.append((ws, we))
    return sorted(chosen)


def drop_windows(rows: list[tuple[int, float]], windows: list[tuple[int, int]]) -> list[tuple[int, float]]:
    return [(t, v) for t, v in rows if not any(a <= t < b for a, b in windows)]


def hours_inside(windows: list[tuple[int, int]], h0: int, nh: int) -> np.ndarray:
    """Hour indices lying completely inside one of the windows."""
    out = []
    for a, b in windows:
        i0 = -(-(a - h0) // HOUR)
        i1 = (b - h0) // HOUR  # exclusive
        out.extend(range(max(i0, 0), min(i1, nh)))
    return np.array(sorted(out), dtype=np.int64)


def training_rows(
    rows: list[tuple[int, float]], temps: np.ndarray, h0: int, nh: int, rng: np.random.Generator, n_windows: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    full_cov = coverage(clean_meter(rows), h0, nh)
    windows = choose_windows(full_cov, rng, n_windows)
    if not windows:
        return np.empty((0, len(FEATURES))), np.empty(0)
    masked_cov = coverage(clean_meter(drop_windows(rows, windows)), h0, nh)
    X, prof = feature_matrix(masked_cov, temps)
    idx = hours_inside(windows, h0, nh)
    idx = idx[full_cov.full_actual[idx]]
    return X[idx], full_cov.actual_kwh[idx] / prof["mean"]


def train_model(
    meters_rows: dict[str, list[tuple[int, float]]],
    temps: np.ndarray,
    h0: int,
    nh: int,
    seed: int = 0,
    n_windows: int = 6,
    rounds: int = 2,
) -> ShapeModel:
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for _ in range(rounds):  # several independent maskings per meter
        for mid in sorted(meters_rows):
            X, y = training_rows(meters_rows[mid], temps, h0, nh, rng, n_windows)
            Xs.append(X)
            ys.append(y)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    est = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.06,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=[0, 1],
        random_state=seed,
    )
    est.fit(X, y)
    return ShapeModel(est, list(FEATURES), {"train_meters": len(meters_rows), "train_rows": int(len(y)), "seed": seed})
