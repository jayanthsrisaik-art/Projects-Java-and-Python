"""End-to-end experiment: simulate -> ingest -> estimate propensities -> train -> evaluate."""
from __future__ import annotations

import json
import random
import sqlite3
import statistics
from pathlib import Path

import numpy as np

from .catalog import build_catalog, vocabulary
from .features import FEATURES, SearchIndex, rank
from .ingest import ingest_file
from .metrics import ndcg_at_k
from .propensity import estimate_propensities
from .simulate import PRODUCTION_WEIGHTS, simulate
from .train import click_pairs, fit_pairwise, label_pairs, save_model

METHODS = ["random", "bm25", "production", "naive_clicks", "ipw_estimated", "ipw_true", "oracle_labels"]


def split_queries(queries: list[str], seed: int, test_fraction: float = 0.3) -> tuple[set[str], set[str]]:
    """Reproducible split by query text: no query contributes to both training and evaluation."""
    qs = sorted(queries)
    random.Random(f"split-{seed}").shuffle(qs)
    n_test = int(round(len(qs) * test_fraction))
    return set(qs[n_test:]), set(qs[:n_test])


def evaluate(index: SearchIndex, weights: np.ndarray | None, queries: set[str],
             grades: dict[str, dict[str, int]], k: int, seed: int = 0) -> dict[str, float | None]:
    """Per-query NDCG@k on the reranked BM25 candidates. weights=None means random order (20 draws)."""
    rng = random.Random(f"random-{seed}")
    out: dict[str, float | None] = {}
    for q in sorted(queries):
        cs = index.candidates(q)
        g = grades[q]
        if weights is None:
            vals = []
            for _ in range(20):
                ids = cs.product_ids[:]
                rng.shuffle(ids)
                vals.append(ndcg_at_k([g[p] for p in ids], k))
            out[q] = None if vals[0] is None else statistics.fmean(vals)
        else:
            out[q] = ndcg_at_k([g[p] for p in rank(cs, weights)], k)
    return out


def _paired(a: dict, b: dict, seed: int, n_boot: int = 2000) -> dict:
    qs = [q for q in a if a[q] is not None and b[q] is not None]
    d = np.array([a[q] - b[q] for q in qs])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {"mean_diff": float(d.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "wins": int((d > 1e-12).sum()), "losses": int((d < -1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum())}


def run(seed: int, workdir: Path, *, n_impressions: int = 60000, swap_rate: float = 0.1,
        eta: float = 1.0, artifacts: Path | None = None) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog(seed)
    index = SearchIndex(catalog.products, vocabulary())
    lines, simtruth = simulate(catalog, index, n_impressions, swap_rate=swap_rate, eta=eta, seed=seed)
    events = workdir / f"events_seed{seed}.jsonl"
    events.write_text("\n".join(lines) + "\n")
    db = workdir / f"logs_seed{seed}.db"
    report = ingest_file(events, db)

    conn = sqlite3.connect(db)
    try:
        train_q, test_q = split_queries([q.text for q in catalog.queries], seed)
        prop = estimate_propensities(conn, train_q)
        true_prop = {int(k): v for k, v in simtruth["true_propensity"].items()}
        naive = click_pairs(conn, index, train_q, None)
        ipw = click_pairs(conn, index, train_q, prop.smoothed)
        ipw_t = click_pairs(conn, index, train_q, true_prop)
        flagged = {u for (u,) in conn.execute("SELECT user_id FROM bot_users")}
        ctr = dict(conn.execute(
            """SELECT CASE WHEN i.swap_k IS NULL THEN 'control' ELSE 'swap' END,
                      1.0 * COUNT(c.request_id) / COUNT(DISTINCT i.request_id)
               FROM impressions i LEFT JOIN clicks c USING (request_id) GROUP BY 1""").fetchall())
    finally:
        conn.close()

    weights = {
        "bm25": np.array([1.0 if f == "bm25" else 0.0 for f in FEATURES]),
        "production": PRODUCTION_WEIGHTS,
        "naive_clicks": fit_pairwise(naive),
        "ipw_estimated": fit_pairwise(ipw),
        "ipw_true": fit_pairwise(ipw_t),
        "oracle_labels": fit_pairwise(label_pairs(index, train_q, catalog.grades)),
    }
    per_query = {m: {} for m in METHODS}
    results = {}
    for k in (5, 10):
        for m in METHODS:
            pq = evaluate(index, weights.get(m), test_q, catalog.grades, k, seed)
            if k == 10:
                per_query[m] = pq
            vals = [v for v in pq.values() if v is not None]
            results.setdefault(m, {})[f"ndcg@{k}"] = statistics.fmean(vals)
    n_eval = sum(v is not None for v in per_query["bm25"].values())

    if artifacts is not None:
        catalog.save(artifacts)
        save_model(artifacts / "model.json", weights["ipw_estimated"], "ipw_estimated",
                   {"propensity": prop.smoothed, "seed": seed})

    bots = set(simtruth["bots"])
    return {
        "seed": seed,
        "config": {"n_impressions": n_impressions, "swap_rate": swap_rate, "eta": eta,
                   "n_products": len(catalog.products), "n_queries": len(catalog.queries),
                   "n_train_queries": len(train_q), "n_test_queries": len(test_q),
                   "n_test_queries_evaluated": n_eval},
        "ingest": report,
        "bot_detection": {"flagged": len(flagged), "true_bots": len(bots),
                          "true_positives": len(flagged & bots)},
        "clicks_per_impression": ctr,
        "propensity": {"raw": prop.raw, "smoothed": prop.smoothed, "true": true_prop,
                       "n_control": prop.n_control, "n_swap": prop.n_swap},
        "training": {"naive_pairs": len(naive.X), "clicks_used": naive.n_clicks},
        "metrics": results,
        "ipw_vs_naive": _paired(per_query["ipw_estimated"], per_query["naive_clicks"], seed),
        "ipw_vs_production": _paired(per_query["ipw_estimated"], per_query["production"], seed),
        "weights": {m: dict(zip(FEATURES, (round(float(x), 4) for x in w))) for m, w in weights.items()},
    }


def aggregate(runs: list[dict]) -> dict:
    summary = {}
    for m in METHODS:
        for metric in ("ndcg@5", "ndcg@10"):
            vals = [r["metrics"][m][metric] for r in runs]
            summary.setdefault(m, {})[metric] = {"mean": statistics.fmean(vals), "min": min(vals), "max": max(vals)}
    return summary


def write_report(runs: list[dict], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary = aggregate(runs)
    doc = {"summary": summary, "runs": runs}
    (out / "report.json").write_text(json.dumps(doc, indent=1, default=float))
    lines = ["# clickrank results", "",
             f"Seeds: {[r['seed'] for r in runs]}; {runs[0]['config']['n_impressions']} impressions per seed, "
             f"swap rate {runs[0]['config']['swap_rate']}, true eta {runs[0]['config']['eta']}. "
             f"Held-out test queries per seed: {[r['config']['n_test_queries_evaluated'] for r in runs]}.", "",
             "## Test-query NDCG (reranking the 20 BM25 candidates, graded ground truth)", "",
             "| method | NDCG@5 mean (min-max) | NDCG@10 mean (min-max) |", "|---|---|---|"]
    for m in METHODS:
        s5, s10 = summary[m]["ndcg@5"], summary[m]["ndcg@10"]
        lines.append(f"| {m} | {s5['mean']:.4f} ({s5['min']:.4f}-{s5['max']:.4f}) | "
                     f"{s10['mean']:.4f} ({s10['min']:.4f}-{s10['max']:.4f}) |")
    lines += ["", "## IPS vs naive, per seed (NDCG@10 on test queries, bootstrap 95% CI over queries)", "",
              "| seed | mean diff | 95% CI | wins / losses / ties |", "|---|---|---|---|"]
    for r in runs:
        p = r["ipw_vs_naive"]
        lines.append(f"| {r['seed']} | {p['mean_diff']:+.4f} | [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | "
                     f"{p['wins']} / {p['losses']} / {p['ties']} |")
    lines += ["", "## Estimated vs true examination propensity (seed " + str(runs[0]["seed"]) + ")", "",
              "| rank | raw estimate | smoothed | true |", "|---|---|---|---|"]
    p0 = runs[0]["propensity"]
    for k in range(1, 11):
        lines.append(f"| {k} | {p0['raw'][k]:.3f} | {p0['smoothed'][k]:.3f} | {p0['true'][k]:.3f} |")
    lines += ["", "## Learned weights (seed " + str(runs[0]["seed"]) + ")", "",
              "| feature | " + " | ".join(m for m in runs[0]["weights"]) + " |",
              "|---|" + "---|" * len(runs[0]["weights"])]
    for f in FEATURES:
        lines.append(f"| {f} | " + " | ".join(f"{runs[0]['weights'][m][f]:+.3f}" for m in runs[0]["weights"]) + " |")
    lines += ["", "## Ingestion and bot filtering", "", "| seed | " + " | ".join(
        k for k in sorted(runs[0]["ingest"])) + " | bots flagged / true / TP |",
        "|---|" + "---|" * (len(runs[0]["ingest"]) + 1)]
    for r in runs:
        b = r["bot_detection"]
        lines.append(f"| {r['seed']} | " + " | ".join(str(r["ingest"].get(k, 0)) for k in sorted(runs[0]["ingest"]))
                     + f" | {b['flagged']} / {b['true_bots']} / {b['true_positives']} |")
    lines += ["", "## Cost of the swap interventions (clicks per impression)", "",
              "| seed | control | swapped |", "|---|---|---|"]
    for r in runs:
        c = r["clicks_per_impression"]
        lines.append(f"| {r['seed']} | {c.get('control', 0):.4f} | {c.get('swap', 0):.4f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return doc
