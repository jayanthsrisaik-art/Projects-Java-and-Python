"""Baselines and evaluation against hand-labelled truth and the perturbation oracle."""

from __future__ import annotations

import json
from pathlib import Path

from .analyzer import referenced_tables
from .graph import LineageGraph
from .oracle import OracleResult
from .project import ModelResult, Project

Pairs = dict[str, set[str]]  # model column -> source columns


def colineage_pairs(graph: LineageGraph, value_only: bool = False) -> Pairs:
    out = {}
    for node in graph.model_columns():
        out[node] = {s for s, d in graph.trace(node).items() if not value_only or d.value is not None}
    return out


def table_level_pairs(project: Project, results: dict[str, ModelResult]) -> Pairs:
    """Table lineage only (what a model DAG gives you): every output column depends on every
    column of every source table reachable upstream."""
    reach: dict[str, set[str]] = {}
    for name, r in results.items():  # topological order
        tables, _ = referenced_tables(r.sql)
        srcs: set[str] = set()
        for t in tables:
            if t in project.sources:
                srcs |= {f"{t}.{c}" for c in project.sources[t].columns}
            elif t in reach:
                srcs |= reach[t]
        reach[name] = srcs
    return {f"{m}.{c}": set(reach[m]) for m, r in results.items() if r.status == "ok" for c in r.columns}


def name_match_pairs(project: Project, results: dict[str, ModelResult]) -> Pairs:
    """Name matching (common in catalogs without a SQL parser): output column X depends on
    columns literally named X in the relations the model reads, recursively."""
    memo: dict[str, set[str]] = {}

    def resolve(model: str, col: str) -> set[str]:
        key = f"{model}.{col}"
        if key in memo:
            return memo[key]
        memo[key] = set()
        tables, _ = referenced_tables(results[model].sql)
        out: set[str] = set()
        for t in sorted(tables):
            if t in project.sources and col in project.sources[t].columns:
                out.add(f"{t}.{col}")
            elif t in results and results[t].status == "ok" and col in results[t].columns:
                out |= resolve(t, col)
        memo[key] = out
        return out

    return {f"{m}.{c}": resolve(m, c) for m, r in results.items() if r.status == "ok" for c in r.columns}


def load_truth(path: str | Path) -> tuple[Pairs, Pairs]:
    raw = json.loads(Path(path).read_text())["columns"]
    union = {k: set(v["value"]) | set(v["control"]) for k, v in raw.items()}
    value = {k: set(v["value"]) for k, v in raw.items()}
    return union, value


def prf(pred: Pairs, truth: Pairs) -> dict:
    keys = set(truth) | set(pred)
    tp = sum(len(pred.get(k, set()) & truth.get(k, set())) for k in keys)
    fp = sum(len(pred.get(k, set()) - truth.get(k, set())) for k in keys)
    fn = sum(len(truth.get(k, set()) - pred.get(k, set())) for k in keys)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    exact = sum(pred.get(k, set()) == truth.get(k, set()) for k in truth) / len(truth) if truth else 1.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 4), "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "exact_columns": round(exact, 4),
    }


def soundness(pred: Pairs, oracle: OracleResult) -> dict:
    missed = {k: sorted(v - pred.get(k, set())) for k, v in oracle.observed.items() if v - pred.get(k, set())}
    observed = sum(len(v) for v in oracle.observed.values())
    caught = observed - sum(len(v) for v in missed.values())
    return {
        "observed_pairs": observed,
        "covered": caught,
        "recall_vs_oracle": round(caught / observed, 4) if observed else 1.0,
        "missed": missed,
    }


def evaluate(project: Project, results, graph: LineageGraph, truth_path, oracle: OracleResult | None) -> dict:
    truth_union, truth_value = load_truth(truth_path)
    methods = {
        "colineage": colineage_pairs(graph),
        "colineage_value_only": colineage_pairs(graph, value_only=True),
        "name_match": name_match_pairs(project, results),
        "table_level": table_level_pairs(project, results),
    }
    report: dict = {"impact_vs_truth": {}, "value_role_vs_truth": {}, "vs_oracle": {}}
    for name, pairs in methods.items():
        report["impact_vs_truth"][name] = prf(pairs, truth_union)
        if oracle is not None:
            s = soundness(pairs, oracle)
            report["vs_oracle"][name] = {k: v for k, v in s.items() if k != "missed"}
            report["vs_oracle"][name]["missed_examples"] = dict(list(s["missed"].items())[:5])
    report["value_role_vs_truth"]["colineage"] = prf(methods["colineage_value_only"], truth_value)
    cl, tu = methods["colineage"], truth_union
    report["colineage_false_positives"] = {k: sorted(cl[k] - tu.get(k, set())) for k in cl if cl[k] - tu.get(k, set())}
    report["colineage_false_negatives"] = {k: sorted(tu[k] - cl.get(k, set())) for k in tu if tu[k] - cl.get(k, set())}
    if oracle is not None:
        report["truth_vs_oracle"] = soundness(truth_union, oracle)
        report["oracle_unconfirmed_colineage_pairs"] = sum(
            len(cl[k] - oracle.observed.get(k, set())) for k in cl
        )
    return report
