"""Evaluation harness: execution accuracy against the golden set, per split and per arm.

Correctness is judged on executed results, not SQL text:
  * column count must match gold; column order may differ
  * floats compared after rounding to 4 decimals
  * scalar / grouped / list: multiset equality of rows
  * series: ordered equality
  * top-k: the predicted values must equal gold's top-k values in order, and every predicted
    (label, value) row must appear in gold's full ranking - so ties at the cut-off are fine

Headline metrics per (split, mode):
  accuracy        correct answers / questions
  wrong_shown     answered but wrong / questions   <- what the critic exists to drive down
  abstain_rate    abstained / questions
  precision       correct / answered
"""
from __future__ import annotations

import itertools
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .contract import infer_contract
from .execute import run_sql
from .golden import Example
from .graph import MODES, Answer, Pipeline
from .writers import Writer


def _norm(v):
    if isinstance(v, float):
        return round(v, 4) + 0.0
    return v


def _rows(rows) -> list[tuple]:
    return [tuple(_norm(v) for v in r) for r in rows]


def _permutations(pred: list[tuple], n: int):
    for perm in itertools.permutations(range(n)):
        yield [tuple(r[i] for i in perm) for r in pred]


def results_match(ex: Example, pred_rows: list[tuple], pred_cols: int, con: sqlite3.Connection) -> bool:
    gold = run_sql(con, ex.gold_sql)
    if not gold.ok:
        raise RuntimeError(f"gold SQL failed for {ex.id}: {gold.error}")
    n = len(gold.columns)
    if pred_cols != n:
        return False
    g = _rows(gold.rows)
    p_all = _rows(pred_rows)
    shape = ex.contract.shape
    for p in _permutations(p_all, n):
        if shape == "series":
            if p == g:
                return True
        elif shape == "topk":
            full = _rows(run_sql(con, re.sub(r"\s+LIMIT\s+\d+\s*$", "", ex.gold_sql)).rows)
            vidx = n - 1
            if [r[vidx] for r in p] == [r[vidx] for r in g] and not (Counter(p) - Counter(full)):
                return True
        elif Counter(p) == Counter(g):
            return True
    return False


@dataclass
class Scored:
    example: Example
    answer: Answer
    correct: bool
    first_attempt_correct: bool


def score(ex: Example, ans: Answer, con: sqlite3.Connection) -> Scored:
    correct = ans.status == "answered" and results_match(ex, ans.rows, len(ans.columns), con)
    first = ans.attempts[0].sql if ans.attempts else ""
    r = run_sql(con, first) if first else None
    first_ok = bool(r and r.ok and results_match(ex, r.rows, len(r.columns), con))
    return Scored(ex, ans, correct, first_ok)


def summarize(scored: list[Scored]) -> dict:
    n = len(scored)
    answered = [s for s in scored if s.answer.status == "answered"]
    correct = sum(s.correct for s in scored)
    wrong = sum(1 for s in answered if not s.correct)
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "wrong_shown": round(wrong / n, 4),
        "abstain_rate": round((n - len(answered)) / n, 4),
        "precision": round(correct / len(answered), 4) if answered else None,
        "first_attempt_accuracy": round(sum(s.first_attempt_correct for s in scored) / n, 4),
        "recovered_by_retry": sum(1 for s in scored if s.correct and not s.first_attempt_correct),
        "correct_first_then_rejected": sum(1 for s in scored if s.first_attempt_correct
                                           and s.answer.attempts and s.answer.attempts[0].issues),
        "mean_attempts": round(sum(len(s.answer.attempts) for s in scored) / n, 3),
    }


def contract_accuracy(examples: list[Example]) -> dict:
    hits = sum(1 for e in examples
               if (c := infer_contract(e.question)).shape == e.contract.shape and c.k == e.contract.k)
    return {"n": len(examples), "accuracy": round(hits / len(examples), 4) if examples else None}


def run_eval(examples: list[Example], writer_factory, con: sqlite3.Connection,
             modes=MODES, max_attempts: int = 3) -> dict:
    report: dict = {"modes": {}, "contract_inference": {}, "per_family": {}, "failures": {}}
    splits = sorted({e.split for e in examples})
    for split in splits:
        report["contract_inference"][split] = contract_accuracy([e for e in examples if e.split == split])
    for mode in modes:
        pipe = Pipeline(writer_factory(), con, mode, max_attempts)
        scored = [score(e, pipe.ask(e.question), con) for e in examples]
        report["modes"][mode] = {sp: summarize([s for s in scored if s.example.split == sp])
                                 for sp in splits}
        fam: dict = defaultdict(lambda: defaultdict(list))
        for s in scored:
            fam[s.example.family][s.example.split].append(s)
        report["per_family"][mode] = {
            f: {sp: {"n": len(v), "accuracy": round(sum(x.correct for x in v) / len(v), 3),
                     "wrong_shown": sum(1 for x in v if x.answer.status == "answered" and not x.correct)}
                for sp, v in d.items()} for f, d in sorted(fam.items())}
        report["failures"][mode] = [
            {"id": s.example.id, "question": s.example.question, "status": s.answer.status,
             "sql": s.answer.sql, "attempts": [{"sql": a.sql, "issues": [i["code"] for i in a.issues]}
                                               for a in s.answer.attempts]}
            for s in scored if not s.correct][:40]
    return report


def check_baseline(report: dict, baseline: dict, max_drop: float = 0.0) -> list[str]:
    """Regression gate: accuracy may not drop and wrong_shown may not rise by more than max_drop."""
    problems = []
    for mode, splits in baseline.get("modes", {}).items():
        for split, base in splits.items():
            cur = report["modes"].get(mode, {}).get(split)
            if cur is None:
                continue
            if cur["accuracy"] < base["accuracy"] - max_drop:
                problems.append(f"{mode}/{split}: accuracy {cur['accuracy']} < baseline {base['accuracy']}")
            if cur["wrong_shown"] > base["wrong_shown"] + max_drop:
                problems.append(f"{mode}/{split}: wrong_shown {cur['wrong_shown']} > baseline "
                                f"{base['wrong_shown']}")
    return problems


def to_markdown(report: dict, title: str) -> str:
    lines = [f"# {title}", "",
             "| split | mode | accuracy | wrong answers shown | abstain | precision | "
             "1st-attempt acc | recovered by retry | correct 1st rejected | mean attempts |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, splits in report["modes"].items():
        for split, m in splits.items():
            prec = f"{m['precision']:.3f}" if m["precision"] is not None else "-"
            lines.append(f"| {split} | {mode} | {m['accuracy']:.3f} | {m['wrong_shown']:.3f} | "
                         f"{m['abstain_rate']:.3f} | {prec} | {m['first_attempt_accuracy']:.3f} | "
                         f"{m['recovered_by_retry']} | {m['correct_first_then_rejected']} | "
                         f"{m['mean_attempts']:.2f} |")
    lines += ["", "Contract inference accuracy (shape and k, from the question alone): " +
              ", ".join(f"{sp} {v['accuracy']:.3f} (n={v['n']})"
                        for sp, v in report["contract_inference"].items())]
    if "critic" in report["per_family"]:
        fams = report["per_family"]["critic"]
        lines += ["", "## Per family (critic mode): accuracy / wrong answers shown", "",
                  "| family | dev | test |", "|---|---:|---:|"]
        for f, d in fams.items():
            cells = [f"{d[sp]['accuracy']:.2f} / {d[sp]['wrong_shown']}" if sp in d else "-"
                     for sp in ("dev", "test")]
            lines.append(f"| {f} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save(report: dict, out_dir: str | Path, name: str, title: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (out / f"{name}.md").write_text(to_markdown(report, title))
