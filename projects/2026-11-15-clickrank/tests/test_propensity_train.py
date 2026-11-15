import sqlite3

import numpy as np
import pytest

from clickrank.experiment import evaluate, split_queries
from clickrank.features import FEATURES
from clickrank.ingest import SCHEMA
from clickrank.propensity import estimate_propensities
from clickrank.train import PairData, click_pairs, fit_pairwise, label_pairs


def _db(tmp_path, impressions, clicks):
    """impressions: (rid, query, swap_k, [pids]); clicks: (rid, position)."""
    conn = sqlite3.connect(tmp_path / "t.db")
    conn.executescript(SCHEMA)
    for rid, q, k, pids in impressions:
        conn.execute("INSERT INTO impressions VALUES (?,?,?,?,?,?,?,?)", (rid, "u", "s", 0.0, q, "p", k, len(pids)))
        conn.executemany("INSERT INTO shown VALUES (?,?,?)", [(rid, i, p) for i, p in enumerate(pids, 1)])
    for rid, pos in clicks:
        pid = conn.execute("SELECT product_id FROM shown WHERE request_id=? AND position=?", (rid, pos)).fetchone()[0]
        conn.execute("INSERT INTO clicks VALUES (?,?,?,?)", (rid, pos, pid, 1.0))
    conn.commit()
    return conn


def test_swap_estimator_formula_on_hand_built_log(tmp_path):
    ten = [f"P{i}" for i in range(10)]
    imps = [(f"c{i}", "q", None, ten) for i in range(4)] + [(f"s{i}", "q", 3, ten) for i in range(2)]
    # control: 2 clicks at rank 1, 1 at rank 3 over 4 impressions; swap(1,3): 1 click at rank 3, 0 at rank 1
    clicks = [("c0", 1), ("c1", 1), ("c2", 3), ("s0", 3)]
    conn = _db(tmp_path, imps, clicks)
    est = estimate_propensities(conn, max_rank=10)
    # (1/2 + 1/4) / (2/4 + 0/2) = 1.5 raw; isotonic smoothing caps at 1
    assert est.raw[3] == pytest.approx(1.5)
    assert est.n_control == 4 and est.n_swap[3] == 2
    assert np.isnan(est.raw[2])
    assert all(est.smoothed[k] >= est.smoothed[k + 1] for k in range(1, 10))
    assert est.smoothed[1] == 1.0


def test_propensities_recovered_from_simulated_log(world):
    est = estimate_propensities(world.conn())
    for k, true in world.truth["true_propensity"].items():
        assert est.smoothed[int(k)] == pytest.approx(true, abs=0.06), k


def test_ipw_pair_weights_are_inverse_propensity(world, tmp_path):
    q = world.catalog.queries[0].text
    pids = world.index.candidates(q).product_ids[:4]
    conn = _db(tmp_path, [("r1", q, None, pids)], [("r1", 2)])
    naive = click_pairs(conn, world.index, {q}, None)
    ipw = click_pairs(conn, world.index, {q}, {1: 1.0, 2: 0.25})
    assert naive.n_clicks == 1 and len(naive.X) == 3
    assert naive.w.tolist() == [1.0, 1.0, 1.0]
    assert ipw.w.tolist() == [4.0, 4.0, 4.0]
    np.testing.assert_allclose(ipw.X[0], world.index.feature_row(q, pids[1]) - world.index.feature_row(q, pids[0]))


def test_fit_pairwise_recovers_direction():
    rng = np.random.default_rng(0)
    true_w = np.zeros(len(FEATURES))
    true_w[0], true_w[2] = 2.0, -1.0
    X = rng.normal(size=(500, len(FEATURES)))
    X = X * np.sign(X @ true_w)[:, None]          # every row is a "preferred - other" difference
    w = fit_pairwise(PairData(X, np.ones(500), 0))
    assert w[0] > 0 and w[2] < 0
    assert np.argmax(np.abs(w)) == 0
    with pytest.raises(ValueError):
        fit_pairwise(PairData(np.zeros((0, len(FEATURES))), np.zeros(0), 0))


def test_split_is_reproducible_and_disjoint():
    qs = [f"q{i}" for i in range(100)]
    tr, te = split_queries(qs, 5)
    assert (tr, te) == split_queries(list(reversed(qs)), 5)
    assert not tr & te and len(te) == 30 and len(tr | te) == 100
    assert split_queries(qs, 6)[1] != te


def test_ipw_beats_naive_and_removes_popularity_bias(world):
    conn = world.conn()
    train_q, test_q = split_queries([q.text for q in world.catalog.queries], 7)
    est = estimate_propensities(conn, train_q)
    pop = FEATURES.index("log_popularity")
    w_naive = fit_pairwise(click_pairs(conn, world.index, train_q, None))
    w_ipw = fit_pairwise(click_pairs(conn, world.index, train_q, est.smoothed))
    w_oracle = fit_pairwise(label_pairs(world.index, train_q, world.catalog.grades))
    # popularity is irrelevant to true relevance; the naive learner absorbs the production ranker's boost
    assert w_naive[pop] > 2 * abs(w_ipw[pop])
    assert abs(w_oracle[pop]) < 0.2

    def mean_ndcg(w):
        vals = [v for v in evaluate(world.index, w, test_q, world.catalog.grades, 10).values() if v is not None]
        return sum(vals) / len(vals)

    n, i, o = mean_ndcg(w_naive), mean_ndcg(w_ipw), mean_ndcg(w_oracle)
    assert i > n + 0.03
    assert o >= i - 0.03
