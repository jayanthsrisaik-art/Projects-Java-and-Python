"""Evaluation harness: generate held-out synthetic corpora, answer every question, score, and explain errors."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date

from .corpus import DEFAULT_REF_DATE, Family, generate
from .index import Corpus, parse_doc
from .pipeline import Answer, Answerer
from .reader import ExtractiveReader
from .textutil import norm_number

DEV_SEEDS = (0, 1, 2)
TEST_SEEDS = (100, 101, 102, 103, 104)
STRATEGIES = ("naive", "latest", "asof_nostatus", "asof")
QTYPES = ("current", "point", "range", "relative", "future", "previous", "when")


def is_correct(q: dict, a: Answer) -> bool:
    if q["abstain"]:
        return a.abstain
    if a.abstain:
        return False
    if q["kind"] == "date":
        return a.date == q["gold_date"]
    return a.values == q["gold_values"]


def classify_error(q: dict, a: Answer, fam: Family) -> str:
    if q["abstain"]:
        return "missed_abstention"
    if a.abstain:
        return "false_abstention"
    if a.policy != q["policy"]:
        return "wrong_policy"
    by_val = {norm_number(str(v.value)): v for v in fam.versions}
    if q["kind"] == "date":
        return "wrong_date"
    gold_docs = [v for v in fam.versions if v.doc_id in q["gold_docs"]]
    lo = min(v.effective for v in gold_docs)
    hi = max(v.effective for v in gold_docs)
    kinds = []
    for val in a.values:
        v = by_val.get(val)
        if v is None:
            kinds.append("misread_value")
        elif v.status != "active":
            kinds.append("unapproved_version")
        elif v.effective < lo:
            kinds.append("older_version")
        elif v.effective > hi:
            kinds.append("newer_version")
    if kinds:
        return kinds[0]
    return "incomplete_range"  # every value is a right one, but some are missing (or extra order)


def run_split(seeds, split: str, strategies=STRATEGIES, ref: date = DEFAULT_REF_DATE, reader_factory=None):
    rows = []
    for seed in seeds:
        fams, docs, qs = generate(seed, split, ref)
        corpus = Corpus([parse_doc(t) for t in docs.values()])
        fam_by = {f.topic.key: f for f in fams}
        for strat in strategies:
            reader = reader_factory(strat) if reader_factory else ExtractiveReader()
            ans = Answerer(corpus, ref, strat, reader)
            for q in qs:
                a = ans.answer(q["question"])
                ok = is_correct(q, a)
                rows.append({
                    "seed": seed, "strategy": strat, "qid": q["qid"], "qtype": q["qtype"],
                    "question": q["question"], "gold_abstain": q["abstain"],
                    "gold": q["gold_date"] if q["kind"] == "date" else q["gold_values"],
                    "pred": a.date if a.kind == "date" else a.values, "pred_abstain": a.abstain,
                    "intent": a.intent, "policy_ok": a.policy == q["policy"], "correct": ok,
                    "error": None if ok else classify_error(q, a, fam_by[q["policy"]]),
                })
    return rows


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    by = defaultdict(list)
    for r in rows:
        by[r["strategy"]].append(r)
    for strat, rs in by.items():
        n = len(rs)
        per_type = {}
        for t in QTYPES:
            sub = [r for r in rs if r["qtype"] == t]
            if sub:
                per_type[t] = {"n": len(sub), "accuracy": round(sum(r["correct"] for r in sub) / len(sub), 4)}
        abst = [r for r in rs if r["gold_abstain"]]
        pred_abst = [r for r in rs if r["pred_abstain"]]
        out[strat] = {
            "n": n,
            "accuracy": round(sum(r["correct"] for r in rs) / n, 4),
            "policy_accuracy": round(sum(r["policy_ok"] for r in rs) / n, 4),
            "per_qtype": per_type,
            "abstention": {
                "gold_abstain": len(abst),
                "recall": round(sum(r["pred_abstain"] for r in abst) / len(abst), 4) if abst else None,
                "precision": round(sum(r["gold_abstain"] for r in pred_abst) / len(pred_abst), 4) if pred_abst else None,
            },
            "errors": dict(Counter(r["error"] for r in rs if r["error"]).most_common()),
            "per_seed_accuracy": {
                str(s): round(sum(r["correct"] for r in rs if r["seed"] == s) / max(1, sum(1 for r in rs if r["seed"] == s)), 4)
                for s in sorted({r["seed"] for r in rs})
            },
        }
    return out


def render_report(summary: dict, title: str) -> str:
    strats = [s for s in STRATEGIES if s in summary] + [s for s in summary if s not in STRATEGIES]
    lines = [f"## {title}", "", f"questions per strategy: {summary[strats[0]]['n']}", ""]
    lines.append("| strategy | accuracy | policy found | " + " | ".join(QTYPES) + " | abstain recall | abstain precision |")
    lines.append("|---" * (5 + len(QTYPES)) + "|")
    for s in strats:
        m = summary[s]
        cells = [f"{m['per_qtype'][t]['accuracy']:.3f}" if t in m["per_qtype"] else "–" for t in QTYPES]
        ab = m["abstention"]
        fmt = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
        lines.append(f"| {s} | **{m['accuracy']:.3f}** | {m['policy_accuracy']:.3f} | " + " | ".join(cells)
                     + f" | {fmt(ab['recall'])} | {fmt(ab['precision'])} |")
    lines += ["", "Error breakdown (count of wrong answers by cause):", ""]
    for s in strats:
        errs = ", ".join(f"{k} {v}" for k, v in summary[s]["errors"].items()) or "none"
        lines.append(f"- **{s}**: {errs}")
    lines += ["", "Per-seed accuracy:", ""]
    for s in strats:
        lines.append(f"- **{s}**: " + ", ".join(f"seed {k}: {v:.3f}" for k, v in summary[s]["per_seed_accuracy"].items()))
    return "\n".join(lines) + "\n"


def ablation_history_filter(seeds, split: str, ref: date = DEFAULT_REF_DATE) -> dict:
    """asof with and without the reader's history-sentence filter."""
    res = {}
    for name, hf in (("asof", True), ("asof (no history filter)", False)):
        rows = run_split(seeds, split, ("asof",), ref, reader_factory=lambda _s, hf=hf: ExtractiveReader(hf))
        res[name] = round(sum(r["correct"] for r in rows) / len(rows), 4)
    return res


def dump_rows(rows: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
