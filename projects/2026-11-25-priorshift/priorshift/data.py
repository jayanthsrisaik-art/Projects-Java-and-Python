"""Data: scikit-learn's bundled handwritten digits (offline, 1,797 images, 10 classes).

Story: a form-processing service reads hand-written digit boxes. The classifier was trained on
one customer's forms, where digits are roughly balanced. New customers send forms whose digit
mix is very different (a postcode field full of 0-3, a quantity field that is mostly 1s), and
nobody labels production traffic.

To make the classifier imperfect enough that the class mix matters, images are optionally
downsampled (default 8x8 -> 4x4, i.e. a cheap low-resolution scanner). That is a property of
the input pipeline and is applied identically to every split.

Splits are stratified and seeded:
    train (50%)  fits the classifier
    val   (25%)  fits calibration and the confusion matrices
    pool  (25%)  is never touched during fitting; shifted test batches are drawn from it
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

N_CLASSES = 10


def downsample(X: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool flattened 8x8 images by `factor` (1 = unchanged, 2 -> 4x4)."""
    if factor == 1:
        return X.astype(float)
    if 8 % factor:
        raise ValueError("factor must divide 8")
    s = 8 // factor
    imgs = X.reshape(-1, s, factor, s, factor)
    return imgs.mean(axis=(2, 4)).reshape(len(X), s * s)


def load(factor: int = 2) -> tuple[np.ndarray, np.ndarray]:
    d = load_digits()
    return downsample(d.data, factor), d.target.astype(int)


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_pool: np.ndarray
    y_pool: np.ndarray


def make_splits(X: np.ndarray, y: np.ndarray, seed: int) -> Splits:
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
    X_val, X_pool, y_val, y_pool = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed + 10_000)
    return Splits(X_train, y_train, X_val, y_val, X_pool, y_pool)


def sample_batch(y_pool: np.ndarray, priors: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices into the pool for a batch whose label mix is drawn from `priors`.

    Class counts are multinomial(n, priors); rows are sampled with replacement within each class
    (the pool has only ~45 images per class, so large or skewed batches must repeat images).
    p(x|y) is unchanged by construction: this is pure label shift.
    """
    counts = rng.multinomial(n, priors)
    idx = []
    for k, c in enumerate(counts):
        if c:
            members = np.flatnonzero(y_pool == k)
            idx.append(rng.choice(members, size=c, replace=True))
    out = np.concatenate(idx)
    rng.shuffle(out)
    return out
