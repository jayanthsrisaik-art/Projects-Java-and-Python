"""Command line interface.

    python -m colineage analyze  PROJECT [--out lineage.json]
    python -m colineage trace    PROJECT model.column
    python -m colineage impact   PROJECT source.column [--value-only]
    python -m colineage explain  PROJECT model.column source.column
    python -m colineage check    PROJECT [--drop t.c ...] [--rename t.c=new ...]
    python -m colineage verify   PROJECT [--seeds 0 1 2]
    python -m colineage bench    PROJECT [PROJECT ...] [--out-dir results]

Exit codes: 0 ok, 1 a check found breakage / a verification failed, 2 usage or model errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer import KIND_NAMES
from .evaluate import colineage_pairs, evaluate, soundness
from .graph import LineageGraph
from .oracle import run_oracle
from .project import Project, analyze_project, results_to_json


def _load(path: str):
    project = Project.load(path)
    results = analyze_project(project)
    return project, results, LineageGraph(results, set(project.sources))


def _fmt_dep(d) -> str:
    parts = []
    if d.value is not None:
        parts.append(KIND_NAMES[d.value])
    if d.control:
        parts.append("control")
    return "+".join(parts)


def _report_errors(results) -> int:
    bad = [r for r in results.values() if r.status != "ok"]
    for r in bad:
        print(f"  {r.status.upper():7} {r.name}: {r.error}", file=sys.stderr)
    return len(bad)


def cmd_analyze(a) -> int:
    project, results, graph = _load(a.project)
    doc = {"models": results_to_json(results), "end_to_end": {}}
    for node in graph.model_columns():
        doc["end_to_end"][node] = {s: d.to_json() for s, d in sorted(graph.trace(node).items())}
    text = json.dumps(doc, indent=2) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)
    ok = sum(r.status == "ok" for r in results.values())
    print(f"analyzed {ok}/{len(results)} model(s), {len(graph.model_columns())} column(s)", file=sys.stderr)
    for r in results.values():
        for w in r.warnings:
            print(f"  WARNING {r.name}: {w}", file=sys.stderr)
    return 2 if _report_errors(results) else 0


def cmd_trace(a) -> int:
    _, results, graph = _load(a.project)
    try:
        deps = graph.trace(a.column)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2
    print(f"{a.column} <- {len(deps)} source column(s)")
    for s, d in sorted(deps.items(), key=lambda kv: (kv[1].value is None, kv[0])):
        print(f"  {s:32} {_fmt_dep(d)}")
    return 0


def cmd_impact(a) -> int:
    project, results, graph = _load(a.project)
    t, _, c = a.column.partition(".")
    if t not in project.sources or c not in project.sources[t].columns:
        print(f"no such source column: {a.column}", file=sys.stderr)
        return 2
    hits = graph.impact(a.column, value_only=a.value_only)
    models = sorted({n.split(".")[0] for n in hits})
    print(f"{a.column} -> {len(hits)} column(s) in {len(models)} model(s)")
    for n, d in sorted(hits.items()):
        print(f"  {n:32} {_fmt_dep(d)}")
    return 0


def cmd_explain(a) -> int:
    _, _, graph = _load(a.project)
    try:
        paths = graph.explain(a.column, a.source)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2
    if not paths:
        print(f"{a.column} does not depend on {a.source}")
        return 1
    for p in paths:
        print(a.column + "".join(f"\n  <-[{role}]- {node}" for node, role in p))
    return 0


def cmd_check(a) -> int:
    project = Project.load(a.project)
    before = analyze_project(project)
    rename = {}
    for r in a.rename or []:
        old, _, new = r.partition("=")
        rename[old.lower()] = new.lower()
    try:
        changed = project.with_source_changes([d.lower() for d in a.drop or []], rename)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    after = analyze_project(changed)
    graph = LineageGraph(before, set(project.sources))
    broken, reshaped = [], []
    for name, r in after.items():
        b = before[name]
        if b.status == "ok" and r.status != "ok":
            broken.append((name, r.status, r.error))
        elif b.status == "ok" and r.status == "ok" and list(b.columns) != list(r.columns):
            reshaped.append((name, list(b.columns), list(r.columns)))
    affected: dict[str, set[str]] = {}
    for col in list(a.drop or []) + list(rename):
        for node in graph.impact(col.lower()):
            affected.setdefault(node, set()).add(col.lower())
    print(f"schema change: drop={a.drop or []} rename={a.rename or []}")
    print(f"  breaks {len(broken)} model(s), reshapes {len(reshaped)}, affects {len(affected)} column(s)")
    for name, status, err in broken:
        print(f"  {status.upper():7} {name}: {err}")
    for name, old, new in reshaped:
        print(f"  RESHAPE {name}: {old} -> {new}")
    if a.verbose:
        for node in sorted(affected):
            print(f"  affected {node} (via {', '.join(sorted(affected[node]))})")
    return 1 if broken or reshaped else 0


def cmd_verify(a) -> int:
    project, results, graph = _load(a.project)
    if _report_errors(results):
        return 2
    oracle = run_oracle(project, results, seeds=tuple(a.seeds))
    s = soundness(colineage_pairs(graph), oracle)
    print(
        f"perturbation oracle: {s['observed_pairs']} observed (model column, source column) "
        f"dependencies over seeds {a.seeds}; static lineage covers {s['covered']}"
    )
    for node, srcs in s["missed"].items():
        print(f"  MISSED {node} <- {', '.join(srcs)}")
    print("sound on this sample" if not s["missed"] else "UNSOUND: static lineage missed observed dependencies")
    return 0 if not s["missed"] else 1


def _md(name: str, rep: dict) -> str:
    lines = [f"## {name}", "", "Impact pairs (model column, source column) vs hand-labelled truth:", ""]
    lines.append("| method | TP | FP | FN | precision | recall | F1 | columns exactly right |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m, v in rep["impact_vs_truth"].items():
        lines.append(
            f"| {m} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']:.3f} | {v['recall']:.3f} "
            f"| {v['f1']:.3f} | {v['exact_columns']:.3f} |"
        )
    if rep.get("vs_oracle"):
        lines += ["", "Recall against dependencies observed by the perturbation oracle:", ""]
        lines.append("| method | observed | covered | recall |")
        lines.append("|---|---:|---:|---:|")
        for m, v in rep["vs_oracle"].items():
            lines.append(f"| {m} | {v['observed_pairs']} | {v['covered']} | {v['recall_vs_oracle']:.3f} |")
        tvo = rep["truth_vs_oracle"]
        lines += ["", f"Truth labels vs oracle: {tvo['covered']}/{tvo['observed_pairs']} observed pairs are labelled."]
    v = rep["value_role_vs_truth"]["colineage"]
    lines += ["", f"Value-role edges (colineage value-only vs truth `value`): TP {v['tp']}, FP {v['fp']}, FN {v['fn']}."]
    if rep["colineage_false_positives"]:
        lines += ["", "colineage false positives (conservative over-approximation):", ""]
        for k, s in rep["colineage_false_positives"].items():
            lines.append(f"- `{k}`: {', '.join(s)}")
    if rep["colineage_false_negatives"]:
        lines += ["", "colineage false negatives:", ""]
        for k, s in rep["colineage_false_negatives"].items():
            lines.append(f"- `{k}`: {', '.join(s)}")
    return "\n".join(lines) + "\n"


def cmd_bench(a) -> int:
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# colineage benchmark", "", "Generated by `python -m colineage bench`. Seeds: " + str(a.seeds), ""]
    status = 0
    for path in a.projects:
        project, results, graph = _load(path)
        if _report_errors(results):
            return 2
        oracle = run_oracle(project, results, seeds=tuple(a.seeds))
        rep = evaluate(project, results, graph, Path(path) / "truth.json", oracle)
        name = Path(path).name
        (out_dir / f"{name}.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
        md.append(_md(name, rep))
        if rep["vs_oracle"]["colineage"]["recall_vs_oracle"] < 1.0:
            status = 1
        c = rep["impact_vs_truth"]["colineage"]
        print(f"{name}: colineage P={c['precision']:.3f} R={c['recall']:.3f} "
              f"oracle recall={rep['vs_oracle']['colineage']['recall_vs_oracle']:.3f}")
    (out_dir / "bench.md").write_text("\n".join(md))
    print(f"wrote {out_dir}/bench.md")
    return status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="colineage", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze", help="per-model and end-to-end lineage as JSON")
    p.add_argument("project")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("trace", help="source columns behind a model column")
    p.add_argument("project")
    p.add_argument("column")
    p.set_defaults(fn=cmd_trace)
    p = sub.add_parser("impact", help="model columns affected by a source column")
    p.add_argument("project")
    p.add_argument("column")
    p.add_argument("--value-only", action="store_true")
    p.set_defaults(fn=cmd_impact)
    p = sub.add_parser("explain", help="edge paths from a model column to a source column")
    p.add_argument("project")
    p.add_argument("column")
    p.add_argument("source")
    p.set_defaults(fn=cmd_explain)
    p = sub.add_parser("check", help="what a source schema change breaks")
    p.add_argument("project")
    p.add_argument("--drop", nargs="*")
    p.add_argument("--rename", nargs="*", help="table.col=new_name")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_check)
    for name, fn, h in (("verify", cmd_verify, "check soundness with the perturbation oracle"),):
        p = sub.add_parser(name, help=h)
        p.add_argument("project")
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        p.set_defaults(fn=fn)
    p = sub.add_parser("bench", help="compare against baselines, truth labels and the oracle")
    p.add_argument("projects", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out-dir", default="results")
    p.set_defaults(fn=cmd_bench)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
