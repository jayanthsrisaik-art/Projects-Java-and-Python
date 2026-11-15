"""Pairwise linear rankers trained from click logs (naive or inverse-propensity weighted) or labels.

For every click on document d at rank k, each *non-clicked* document shown in the same impression
forms a preference pair (d > d'). The naive learner weights every pair 1; the IPS learner weights
it 1 / p_k, which makes the pairwise objective an unbiased estimate of the one we would get from
full-information relevance (Joachims, Swaminathan & Schnabel, 2017).
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .features import FEATURES, SearchIndex


@dataclass
class PairData:
    X: np.ndarray     # feature differences (preferred - other)
    w: np.ndarray
    n_clicks: int


def load_impressions(conn: sqlite3.Connection, queries: set[str] | None = None):
    """Yield (query, [product ids by position], {clicked positions}) per impression."""
    shown: dict[str, list[str]] = defaultdict(list)
    for rid, pid in conn.execute("SELECT request_id, product_id FROM shown ORDER BY request_id, position"):
        shown[rid].append(pid)
    clicked: dict[str, set[int]] = defaultdict(set)
    for rid, pos in conn.execute("SELECT request_id, position FROM clicks"):
        clicked[rid].add(pos)
    for rid, query in conn.execute("SELECT request_id, query FROM impressions ORDER BY request_id"):
        if queries is not None and query not in queries:
            continue
        yield query, shown[rid], clicked.get(rid, set())


def click_pairs(conn: sqlite3.Connection, index: SearchIndex, queries: set[str],
                propensity: dict[int, float] | None, clip: float = 0.02) -> PairData:
    agg: dict[tuple[str, str, str], float] = defaultdict(float)
    n_clicks = 0
    for query, shown, clicked in load_impressions(conn, queries):
        for k in clicked:
            w = 1.0 if propensity is None else 1.0 / max(propensity.get(k, clip), clip)
            n_clicks += 1
            d = shown[k - 1]
            for pos, other in enumerate(shown, start=1):
                if pos not in clicked:
                    agg[(query, d, other)] += w
    rows, weights = [], []
    for (query, d, other), w in agg.items():
        a, b = index.feature_row(query, d), index.feature_row(query, other)
        if a is None or b is None:          # shown item no longer retrievable: skip
            continue
        rows.append(a - b)
        weights.append(w)
    X = np.array(rows).reshape(-1, len(FEATURES))
    return PairData(X, np.array(weights), n_clicks)


def label_pairs(index: SearchIndex, queries: set[str], grades: dict[str, dict[str, int]]) -> PairData:
    rows, weights = [], []
    for q in sorted(queries):
        cs = index.candidates(q)
        g = [grades[q][pid] for pid in cs.product_ids]
        for i in range(len(g)):
            for j in range(len(g)):
                if g[i] > g[j]:
                    rows.append(cs.X[i] - cs.X[j])
                    weights.append(1.0)
    return PairData(np.array(rows).reshape(-1, len(FEATURES)), np.array(weights), 0)


def fit_pairwise(pairs: PairData, C: float = 1.0) -> np.ndarray:
    if len(pairs.X) == 0:
        raise ValueError("no training pairs")
    w = pairs.w / pairs.w.mean()            # same regularisation strength for every weighting
    X = np.vstack([pairs.X, -pairs.X])
    y = np.r_[np.ones(len(pairs.X)), np.zeros(len(pairs.X))]
    model = LogisticRegression(C=C, fit_intercept=False, max_iter=2000)
    model.fit(X, y, sample_weight=np.r_[w, w])
    return model.coef_.ravel()


def save_model(path: Path, weights: np.ndarray, method: str, extra: dict | None = None) -> None:
    path.write_text(json.dumps({"method": method, "features": FEATURES,
                                "weights": [round(float(x), 6) for x in weights], **(extra or {})}, indent=1))


def load_model(path: Path) -> tuple[np.ndarray, dict]:
    data = json.loads(path.read_text())
    if data["features"] != FEATURES:
        raise ValueError(f"model features {data['features']} do not match code features {FEATURES}")
    return np.array(data["weights"], dtype=float), data
