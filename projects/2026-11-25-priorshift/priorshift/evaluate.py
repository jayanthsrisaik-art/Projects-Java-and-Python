"""Benchmark: simulate label-shifted production batches and score every estimator.

For each split seed the classifier, calibrator and confusion matrices are fitted on train/val
only. Batches are then drawn from the untouched pool with a known label mix. Estimators see the
batch's predicted probabilities only; target labels are used solely to score the result.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import data
from . import estimators as est
from .calibration import nll
from .detect import detect_shift, total_variation
from .model import fit

K = data.N_CLASSES

# name -> (prior estimator, smooth?)
VARIANTS = {
    "none": ("none", False),
    "cc": ("cc", False),
    "pcc": ("pcc", False),
    "acc": ("acc", False),
    "pacc": ("pacc", False),
    "em": ("em", False),
    "acc+smooth": ("acc", True),
    "pacc+smooth": ("pacc", True),
    "em+smooth": ("em", True),
    "em+smooth gated": ("em", True),  # adapt only when the shift test fires
    "oracle": (None, False),          # true batch priors: an upper bound, not a method
}


def scenario_priors(name: str, ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if name == "no_shift":
        return ref.copy()
    if name == "surge":  # one class jumps to 40%, the rest share the remainder equally
        q = np.full(K, 0.6 / (K - 1))
        q[rng.integers(K)] = 0.4
        return q
    if name == "missing":  # three classes never occur, the rest are uniform
        q = np.ones(K)
        q[rng.choice(K, size=3, replace=False)] = 0.0
        return q / q.sum()
    if name == "dirichlet_1":
        return rng.dirichlet(np.ones(K))
    if name == "dirichlet_0.3":  # severe, sparse mixes
        return rng.dirichlet(np.full(K, 0.3))
    raise ValueError(name)


SCENARIOS = ("no_shift", "surge", "missing", "dirichlet_1", "dirichlet_0.3")


@dataclass
class Config:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    batch_sizes: tuple[int, ...] = (100, 500, 2000)
    repeats: int = 20
    factor: int = 2
    alpha: float = 0.01
    min_tv: float = 0.05
    scenarios: tuple[str, ...] = SCENARIOS
    extra: dict = field(default_factory=dict)


def run(cfg: Config) -> dict:
    t0 = time.time()
    X, y = data.load(cfg.factor)
    rows = []  # one dict per (batch, variant)
    detect_rows = []
    diagnostics = []
    for seed in cfg.seeds:
        splits = data.make_splits(X, y, seed)
        m = fit(splits, K, factor=cfg.factor)
        diag = dict(m.diagnostics, seed=seed)
        pool_probs = m.proba(splits.X_pool)  # computed once; batches index into it
        diag["pool_accuracy"] = float((pool_probs.argmax(1) == splits.y_pool).mean())
        diagnostics.append(diag)
        rng = np.random.default_rng(1_000 + seed)
        for scen in cfg.scenarios:
            for n in cfg.batch_sizes:
                for r in range(cfg.repeats):
                    q_true = scenario_priors(scen, m.train_priors, rng)
                    idx = data.sample_batch(splits.y_pool, q_true, n, rng)
                    P, yb = pool_probs[idx], splits.y_pool[idx]
                    q_emp = np.bincount(yb, minlength=K) / n  # realised mix of this batch
                    test = detect_shift(P, m.train_priors, alpha=cfg.alpha, min_tv=cfg.min_tv)
                    detect_rows.append({
                        "seed": seed, "scenario": scen, "n": n,
                        "p_value": test.p_value, "tv_hat": test.tv_distance,
                        "true_tv": total_variation(q_emp, m.train_priors),
                        "sig_only": test.p_value < cfg.alpha, "alarm": test.shifted,
                    })
                    cache = {}
                    for name, (method, smooth) in VARIANTS.items():
                        if method is None:
                            q_hat = q_emp
                        else:
                            if method not in cache:
                                cache[method] = m.estimate_priors(P, method)
                            q_hat = cache[method]
                            if smooth:
                                q_hat = est.smooth_priors(q_hat, n)
                            if name.endswith("gated") and not test.shifted:
                                q_hat = m.train_priors
                        adj = est.adjust_posteriors(P, m.train_priors, q_hat)
                        rows.append({
                            "seed": seed, "scenario": scen, "n": n, "variant": name,
                            "tv": total_variation(q_hat, q_emp),
                            "acc": float((adj.argmax(1) == yb).mean()),
                            "nll": nll(adj, yb),
                        })
    return summarise(cfg, rows, detect_rows, diagnostics, time.time() - t0)


def _ms(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "count": int(a.size)}


def summarise(cfg: Config, rows, detect_rows, diagnostics, seconds) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["n"], r["variant"])].append(r)
    table = []
    for (scen, n, var), rs in sorted(by.items(), key=lambda kv: (cfg.scenarios.index(kv[0][0]), kv[0][1], list(VARIANTS).index(kv[0][2]))):
        table.append({
            "scenario": scen, "n": n, "variant": var,
            "tv": _ms([r["tv"] for r in rs]),
            "acc": _ms([r["acc"] for r in rs]),
            "nll": _ms([r["nll"] for r in rs]),
        })

    # Over all shifted scenarios: averages and paired win-rate against no adaptation.
    shifted = [r for r in rows if r["scenario"] != "no_shift"]
    base = _index_batches(shifted, "none")  # rows are appended batch by batch, so lists align
    overall = {}
    for var in VARIANTS:
        rs = [r for r in shifted if r["variant"] == var]
        mine = _index_batches(shifted, var)
        wins = [m["nll"] < b["nll"] for m, b in zip(mine, base, strict=True)]
        overall[var] = {
            "tv": _ms([r["tv"] for r in rs]),
            "acc": _ms([r["acc"] for r in rs]),
            "nll": _ms([r["nll"] for r in rs]),
            "nll_win_rate_vs_none": float(np.mean(wins)),
        }

    det = defaultdict(list)
    for d in detect_rows:
        det[(d["scenario"], d["n"])].append(d)
    detection = [{
        "scenario": s, "n": n,
        "alarm_rate": float(np.mean([d["alarm"] for d in ds])),
        "significant_only_rate": float(np.mean([d["sig_only"] for d in ds])),
        "mean_true_tv": float(np.mean([d["true_tv"] for d in ds])),
    } for (s, n), ds in sorted(det.items(), key=lambda kv: (cfg.scenarios.index(kv[0][0]), kv[0][1]))]

    return {
        "config": {
            "seeds": list(cfg.seeds), "batch_sizes": list(cfg.batch_sizes), "repeats": cfg.repeats,
            "downsample_factor": cfg.factor, "alpha": cfg.alpha, "min_tv": cfg.min_tv,
            "scenarios": list(cfg.scenarios),
            "batches": len(detect_rows),
        },
        "model_diagnostics": diagnostics,
        "overall_shifted": overall,
        "by_scenario": table,
        "detection": detection,
        "runtime_seconds": round(seconds, 1),
    }


def _index_batches(rows, variant):
    return [r for r in rows if r["variant"] == variant]


# --------------------------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------------------------


def to_markdown(rep: dict) -> str:
    c = rep["config"]
    out = ["# priorshift benchmark report", ""]
    out.append(f"Generated by `python -m priorshift benchmark`. {c['batches']} simulated batches: "
               f"{len(c['seeds'])} split seeds x {len(c['scenarios'])} scenarios x batch sizes {c['batch_sizes']} "
               f"x {c['repeats']} repeats. Images downsampled by {c['downsample_factor']}. Runtime {rep['runtime_seconds']} s.")
    out += ["", "## Classifier (validation split, per seed)", "",
            "| seed | val acc | pool acc | val NLL raw | val NLL calibrated | ECE raw | ECE calibrated | T |",
            "|---|---|---|---|---|---|---|---|"]
    for d in rep["model_diagnostics"]:
        out.append(f"| {d['seed']} | {d['val_accuracy']:.3f} | {d['pool_accuracy']:.3f} | {d['val_nll_raw']:.3f} | "
                   f"{d['val_nll_calibrated']:.3f} | {d['val_ece_raw']:.3f} | {d['val_ece_calibrated']:.3f} | {d['temperature']:.2f} |")
    out += ["", "## All shifted scenarios combined", "",
            "TV = total-variation distance between estimated and realised batch mix (lower is better). "
            "Accuracy and NLL are of the Bayes-adjusted posteriors. Win rate = share of batches where the "
            "variant's NLL is lower than no adaptation.", "",
            "| variant | prior TV error | accuracy | NLL | NLL win rate vs none |", "|---|---|---|---|---|"]
    for var, v in rep["overall_shifted"].items():
        out.append(f"| {var} | {v['tv']['mean']:.3f} | {v['acc']['mean']:.4f} | {v['nll']['mean']:.4f} | {v['nll_win_rate_vs_none']:.1%} |")
    out += ["", "## By scenario and batch size", "",
            "Mean over seeds x repeats (std across batches in brackets).", ""]
    current = None
    for row in rep["by_scenario"]:
        key = (row["scenario"], row["n"])
        if key != current:
            current = key
            out += ["", f"### {row['scenario']}, n = {row['n']}", "",
                    "| variant | prior TV error | accuracy | NLL |", "|---|---|---|---|"]
        out.append(f"| {row['variant']} | {row['tv']['mean']:.3f} ({row['tv']['std']:.3f}) | "
                   f"{row['acc']['mean']:.4f} ({row['acc']['std']:.4f}) | {row['nll']['mean']:.4f} ({row['nll']['std']:.4f}) |")
    out += ["", "## Shift detection", "",
            f"Likelihood-ratio test on the EM fit, alpha = {c['alpha']}. `alarm` additionally requires an "
            f"estimated TV shift >= {c['min_tv']}. On `no_shift` the rate is the false-alarm rate (batches are still "
            "multinomial draws, so their realised mix differs slightly from training).", "",
            "| scenario | n | mean true TV | significant only | alarm (significant and TV >= min) |", "|---|---|---|---|---|"]
    for d in rep["detection"]:
        out.append(f"| {d['scenario']} | {d['n']} | {d['mean_true_tv']:.3f} | {d['significant_only_rate']:.1%} | {d['alarm_rate']:.1%} |")
    return "\n".join(out) + "\n"


def write(rep: dict, out_dir) -> None:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    (out / "REPORT.md").write_text(to_markdown(rep))
