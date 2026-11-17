"""Critic stress test: inject realistic text-to-SQL mistakes into gold SQL, measure detection.

Each mutation models a failure mode seen in LLM-written SQL. A mutant only counts if it is
*effective* (its result differs from gold, or it fails). Detection is measured twice: with the
gold contract (critic quality in isolation) and with the contract inferred from the question
(what happens in the pipeline). False rejects are measured on the unmodified gold SQL.

The two "semantic" mutations change meaning without changing shape; they are included on
purpose to show what a schema/contract critic cannot see.
"""
from __future__ import annotations

import re
import sqlite3

import sqlglot
from collections import Counter, defaultdict
from typing import Callable

from .contract import infer_contract
from .critic import Critic
from .evaluate import results_match
from .execute import run_sql
from .golden import Example


def _fanout_join(sql: str) -> str | None:
    if "order_items" in sql or "FROM orders o" not in sql:
        return None
    return sql.replace("FROM orders o", "FROM orders o JOIN order_items oi ON oi.order_id = o.order_id", 1)


def _drop_distinct(sql: str) -> str | None:
    return sql.replace("COUNT(DISTINCT ", "COUNT(", 1) if "COUNT(DISTINCT " in sql else None


def _drop_label(sql: str) -> str | None:
    if "GROUP BY" not in sql:
        return None
    tree = sqlglot.parse_one(sql, read="sqlite")
    tree.set("expressions", tree.expressions[1:])
    return tree.sql(dialect="sqlite")


def _drop_limit(sql: str) -> str | None:
    out = re.sub(r"\s+LIMIT\s+\d+\s*$", "", sql)
    return out if out != sql else None


def _hallucinate_column(sql: str) -> str | None:
    for real, fake in (("o.order_date", "o.created_at"), ("p.category", "p.category_name"),
                       ("r.name", "r.region_name"), ("c.segment", "c.customer_segment"),
                       ("p.unit_price", "p.price")):
        if real in sql:
            return sql.replace(real, fake)
    return None


def _destructive(sql: str) -> str | None:
    return sql + "; DROP TABLE orders"


def _extra_group(sql: str) -> str | None:
    # only when orders is in the outer query (not just inside a NOT EXISTS subquery)
    if "GROUP BY" in sql or "EXISTS" in sql or ("FROM orders o" not in sql and "JOIN orders o" not in sql):
        return None
    return sql + " GROUP BY o.status"


def _wrong_year(sql: str) -> str | None:
    m = re.search(r"= '(20\d\d)'", sql)
    if not m:
        return None
    other = "2023" if m.group(1) != "2023" else "2024"
    return sql.replace(f"= '{m.group(1)}'", f"= '{other}'")


def _drop_status_filter(sql: str) -> str | None:
    out = sql.replace("o.status = 'completed' AND ", "")
    return out if out != sql else None


MUTATIONS: dict[str, tuple[str, Callable[[str], str | None]]] = {
    "fanout_join":        ("structural", _fanout_join),
    "drop_distinct":      ("structural", _drop_distinct),
    "drop_label_column":  ("structural", _drop_label),
    "drop_limit":         ("structural", _drop_limit),
    "hallucinated_column": ("structural", _hallucinate_column),
    "destructive_stmt":   ("safety", _destructive),
    "spurious_group_by":  ("structural", _extra_group),
    "wrong_year":         ("semantic", _wrong_year),
    "drop_status_filter": ("semantic", _drop_status_filter),
}


def run_stress(examples: list[Example], con: sqlite3.Connection, critic: Critic | None = None) -> dict:
    critic = critic or Critic()
    per = defaultdict(Counter)
    gold_rejects = Counter()
    split_n = Counter()
    for ex in examples:
        inferred = infer_contract(ex.question)
        gold_res = run_sql(con, ex.gold_sql)
        split_n[ex.split] += 1
        gold_rejects[("gold_contract", ex.split)] += not critic.review(ex.gold_sql, gold_res, ex.contract).ok
        gold_rejects[("inferred_contract", ex.split)] += not critic.review(ex.gold_sql, gold_res, inferred).ok
        for name, (_, fn) in MUTATIONS.items():
            mutant = fn(ex.gold_sql)
            if mutant is None:
                continue
            res = run_sql(con, mutant) if ";" not in mutant else None
            effective = res is None or not res.ok or not results_match(ex, res.rows, len(res.columns), con)
            if not effective:
                per[name]["ineffective"] += 1
                continue
            per[name]["n"] += 1
            v_gold = critic.review(mutant, res, ex.contract)
            v_inf = critic.review(mutant, res, inferred)
            per[name]["detected_gold_contract"] += not v_gold.ok
            per[name]["detected_inferred_contract"] += not v_inf.ok
            for code in v_gold.codes()[:1]:
                per[name][f"code:{code}"] += 1
    rows = {}
    for name, c in per.items():
        n = c["n"]
        rows[name] = {
            "kind": MUTATIONS[name][0], "n": n, "ineffective_skipped": c["ineffective"],
            "recall_gold_contract": round(c["detected_gold_contract"] / n, 3) if n else None,
            "recall_inferred_contract": round(c["detected_inferred_contract"] / n, 3) if n else None,
            "first_issue_codes": {k[5:]: v for k, v in c.items() if k.startswith("code:")},
        }
    return {"examples": dict(split_n),
            "false_reject_rate": {
                f"{kind}/{sp}": round(gold_rejects[(kind, sp)] / n, 4)
                for kind in ("gold_contract", "inferred_contract") for sp, n in sorted(split_n.items())},
            "mutations": rows}


def to_markdown(rep: dict) -> str:
    lines = ["# Critic stress test (mutated gold SQL)", "",
             f"Golden examples: {rep['examples']} (each wording is checked, because the inferred "
             "contract depends on wording; counts below are mutant instances over all examples).", "",
             "False-reject rate on unmodified gold SQL: " + ", ".join(
                 f"{k} {v:.3f}" for k, v in rep["false_reject_rate"].items()) + ".",
             "", "| mutation | kind | effective mutants | recall (gold contract) | "
             "recall (inferred contract) | first issue raised |", "|---|---|---:|---:|---:|---|"]
    for name, r in rep["mutations"].items():
        codes = ", ".join(f"{k}×{v}" for k, v in sorted(r["first_issue_codes"].items())) or "-"
        lines.append(f"| {name} | {r['kind']} | {r['n']} | {r['recall_gold_contract']:.3f} | "
                     f"{r['recall_inferred_contract']:.3f} | {codes} |")
    return "\n".join(lines) + "\n"
