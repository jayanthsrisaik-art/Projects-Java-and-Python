"""Examination propensities p_k / p_1 from swap interventions (Joachims et al., WSDM 2017).

Under a random swap of ranks 1 and k, the document that would have been at rank 1 is shown at
rank k and vice versa. Because swaps are assigned at random, swapped and control impressions
have the same relevance distribution at every original rank, so

    clicks at k (orig rank-1 doc, swapped) / n_swap_k  ~  p_k * R1
    clicks at 1 (control)                   / n_ctrl    ~  p_1 * R1
    clicks at k (control)                   / n_ctrl    ~  p_k * Rk
    clicks at 1 (orig rank-k doc, swapped)  / n_swap_k  ~  p_1 * Rk

and the ratio of sums (p_k (R1 + Rk)) / (p_1 (R1 + Rk)) estimates p_k / p_1 using both documents.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class PropensityEstimate:
    raw: dict[int, float]
    smoothed: dict[int, float]   # non-increasing, clipped to [floor, 1]
    n_control: int
    n_swap: dict[int, int]


def _temp_query_filter(conn: sqlite3.Connection, queries: set[str] | None) -> str:
    if queries is None:
        return ""
    conn.execute("DROP TABLE IF EXISTS temp.qfilter")
    conn.execute("CREATE TEMP TABLE qfilter(query TEXT PRIMARY KEY)")
    conn.executemany("INSERT INTO qfilter VALUES (?)", [(q,) for q in sorted(queries)])
    return " AND i.query IN (SELECT query FROM temp.qfilter)"


def estimate_propensities(conn: sqlite3.Connection, queries: set[str] | None = None,
                          max_rank: int = 10, floor: float = 0.02) -> PropensityEstimate:
    where = _temp_query_filter(conn, queries)
    n_ctrl = conn.execute(
        f"SELECT COUNT(*) FROM impressions i WHERE i.swap_k IS NULL AND i.n_shown >= {max_rank}{where}"
    ).fetchone()[0]
    ctrl = dict(conn.execute(
        f"""SELECT c.position, COUNT(*) FROM clicks c JOIN impressions i USING (request_id)
            WHERE i.swap_k IS NULL AND i.n_shown >= {max_rank}{where} GROUP BY c.position""").fetchall())
    n_swap = dict(conn.execute(
        f"SELECT i.swap_k, COUNT(*) FROM impressions i WHERE i.swap_k IS NOT NULL AND i.n_shown >= {max_rank}{where} GROUP BY i.swap_k"
    ).fetchall())
    swap_clicks: dict[tuple[int, int], int] = {
        (k, pos): n for k, pos, n in conn.execute(
            f"""SELECT i.swap_k, c.position, COUNT(*) FROM clicks c JOIN impressions i USING (request_id)
                WHERE i.swap_k IS NOT NULL AND i.n_shown >= {max_rank}{where} GROUP BY i.swap_k, c.position""").fetchall()}

    raw = {1: 1.0}
    for k in range(2, max_rank + 1):
        ns = n_swap.get(k, 0)
        if ns == 0 or n_ctrl == 0:
            raw[k] = float("nan")
            continue
        num = swap_clicks.get((k, k), 0) / ns + ctrl.get(k, 0) / n_ctrl
        den = ctrl.get(1, 0) / n_ctrl + swap_clicks.get((k, 1), 0) / ns
        raw[k] = num / den if den > 0 else float("nan")

    ks = [k for k in range(1, max_rank + 1) if np.isfinite(raw[k])]
    weights = [n_ctrl if k == 1 else n_swap.get(k, 0) for k in ks]
    iso = IsotonicRegression(increasing=False, y_min=floor, y_max=1.0)
    fitted = iso.fit_transform(ks, [raw[k] for k in ks], sample_weight=weights)
    smoothed = dict(zip(ks, (float(v) for v in fitted)))
    for k in range(1, max_rank + 1):          # ranks with no data: carry the last known value
        if k not in smoothed:
            smoothed[k] = smoothed.get(k - 1, floor)
    smoothed[1] = 1.0
    return PropensityEstimate(raw, dict(sorted(smoothed.items())), n_ctrl,
                              {k: n_swap.get(k, 0) for k in range(2, max_rank + 1)})
