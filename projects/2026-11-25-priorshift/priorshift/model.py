"""Fit the deployed classifier and everything the estimators need from the validation split."""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import estimators as est
from .calibration import BCTS, expected_calibration_error, nll
from .data import Splits, downsample


@dataclass
class ShiftModel:
    clf: object
    calibrator: BCTS
    n_classes: int
    train_priors: np.ndarray
    C_hard: np.ndarray
    C_soft: np.ndarray
    n_features: int
    factor: int = 1  # image downsampling applied by features(); see data.downsample
    diagnostics: dict = field(default_factory=dict)

    def features(self, X_pixels: np.ndarray) -> np.ndarray:
        """Raw 64-pixel rows (as in sklearn's digits) -> the model's input features."""
        X_pixels = np.asarray(X_pixels, dtype=float)
        if X_pixels.ndim != 2 or X_pixels.shape[1] != 64:
            raise ValueError(f"expected (n, 64) pixel rows, got {X_pixels.shape}")
        return downsample(X_pixels, self.factor)

    def raw_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"expected (n, {self.n_features}) features, got {X.shape}")
        return self.clf.predict_proba(X)

    def proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probabilities for the *training* class mix."""
        return self.calibrator.transform(self.raw_proba(X))

    def estimate_priors(self, probs: np.ndarray, method: str = "em") -> np.ndarray:
        return est.estimate(method, probs, self.train_priors, self.C_hard, self.C_soft)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "ShiftModel":
        obj = joblib.load(path)
        if not isinstance(obj, ShiftModel):
            raise TypeError("not a ShiftModel artifact")
        return obj


def fit(splits: Splits, n_classes: int, C: float = 1.0, factor: int = 1) -> ShiftModel:
    """`factor` is only recorded (splits are already downsampled) so features() can repeat it."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    clf.fit(splits.X_train, splits.y_train)

    # The class mix the probabilities are calibrated for. Calibration is fitted on the validation
    # split, so its class mix is the reference prior; train and val are stratified alike.
    train_priors = np.bincount(splits.y_val, minlength=n_classes) / len(splits.y_val)

    raw_val = clf.predict_proba(splits.X_val)
    cal = BCTS().fit(raw_val, splits.y_val)
    cal_val = cal.transform(raw_val)

    m = ShiftModel(
        clf=clf,
        calibrator=cal,
        n_classes=n_classes,
        train_priors=train_priors,
        C_hard=est.hard_confusion(cal_val, splits.y_val, n_classes),
        C_soft=est.soft_confusion(cal_val, splits.y_val, n_classes),
        n_features=splits.X_train.shape[1],
        factor=factor,
    )
    m.diagnostics = {
        "val_accuracy": float((cal_val.argmax(1) == splits.y_val).mean()),
        "val_nll_raw": nll(raw_val, splits.y_val),
        "val_nll_calibrated": nll(cal_val, splits.y_val),
        "val_ece_raw": expected_calibration_error(raw_val, splits.y_val),
        "val_ece_calibrated": expected_calibration_error(cal_val, splits.y_val),
        "temperature": cal.temperature,
    }
    return m
