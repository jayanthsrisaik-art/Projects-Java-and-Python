"""Load a project (sources.json + models/*.sql), order models, analyze each one."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .analyzer import KIND_NAMES, LineageError, analyze_query, referenced_tables

VALID_TYPES = {"INTEGER", "REAL", "TEXT"}


@dataclass
class SourceTable:
    name: str
    rows: int
    columns: dict[str, dict]  # column -> spec ({"type": ..., generator hints})


@dataclass
class ModelResult:
    name: str
    sql: str
    depends_on: list[str] = field(default_factory=list)
    columns: dict[str, dict[str, int]] = field(default_factory=dict)
    control: set[str] = field(default_factory=set)
    column_control: dict[str, set[str]] = field(default_factory=dict)
    error: str | None = None
    status: str = "ok"  # ok | error | blocked
    warnings: list[str] = field(default_factory=list)
    qualified_sql: str = ""


@dataclass
class Project:
    sources: dict[str, SourceTable]
    models_sql: dict[str, str]

    @staticmethod
    def load(path: str | Path) -> "Project":
        path = Path(path)
        raw = json.loads((path / "sources.json").read_text())
        sources = {}
        for tname, spec in raw.items():
            tname = tname.lower()
            cols = {}
            for cname, cspec in spec["columns"].items():
                if isinstance(cspec, str):
                    cspec = {"type": cspec}
                t = cspec.get("type", "TEXT").upper()
                if t not in VALID_TYPES:
                    raise ValueError(f"{tname}.{cname}: type must be one of {sorted(VALID_TYPES)}")
                cols[cname.lower()] = {**cspec, "type": t}
            sources[tname] = SourceTable(tname, int(spec.get("rows", 100)), cols)
        models = {}
        for f in sorted((path / "models").glob("*.sql")):
            name = f.stem.lower()
            if name in sources:
                raise ValueError(f"model {name} shadows a source table")
            models[name] = f.read_text()
        return Project(sources, models)

    def with_source_changes(self, drop: list[str] = (), rename: dict[str, str] | None = None) -> "Project":
        """Copy of the project with source columns dropped/renamed (for `check`)."""
        rename = rename or {}
        new = {}
        for t in self.sources.values():
            cols = {}
            for c, spec in t.columns.items():
                key = f"{t.name}.{c}"
                if key in drop:
                    continue
                cols[rename.get(key, c)] = spec
            new[t.name] = SourceTable(t.name, t.rows, cols)
        for key in list(drop) + list(rename):
            t, _, c = key.partition(".")
            if t not in self.sources or c not in self.sources[t].columns:
                raise ValueError(f"no such source column: {key}")
        return Project(new, dict(self.models_sql))


def model_order(project: Project) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    """Topological order of models (Kahn, alphabetical tie-break), deps, and per-model errors
    for models that cannot even be parsed or that sit on a cycle."""
    deps: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    for name, sql in project.models_sql.items():
        try:
            tables, _ = referenced_tables(sql)
        except LineageError as e:
            errors[name] = str(e)
            tables = set()
        deps[name] = sorted(t for t in tables if t in project.models_sql)
    indeg = {m: len(d) for m, d in deps.items()}
    children: dict[str, list[str]] = {m: [] for m in deps}
    for m, ds in deps.items():
        for d in ds:
            children[d].append(m)
    ready = sorted(m for m, n in indeg.items() if n == 0)
    order: list[str] = []
    while ready:
        m = ready.pop(0)
        order.append(m)
        for c in sorted(children[m]):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
        ready.sort()
    for m in sorted(set(deps) - set(order)):
        errors.setdefault(m, "dependency cycle: " + " -> ".join(_cycle_from(m, deps)))
        order.append(m)
    return order, deps, errors


def _cycle_from(start: str, deps: dict[str, list[str]]) -> list[str]:
    path, seen = [start], {start}
    cur = start
    while True:
        nxt = next((d for d in deps[cur] if d in deps), None)
        if nxt is None:
            return path
        path.append(nxt)
        if nxt in seen:
            return path[path.index(nxt):]
        seen.add(nxt)
        cur = nxt


def analyze_project(project: Project) -> dict[str, ModelResult]:
    order, deps, pre_errors = model_order(project)
    catalog: dict[str, dict[str, str]] = {
        t.name: {c: s["type"] for c, s in t.columns.items()} for t in project.sources.values()
    }
    keys = {t.name: {c for c, s in t.columns.items() if s.get("pk")} for t in project.sources.values()}
    results: dict[str, ModelResult] = {}
    for name in order:
        sql = project.models_sql[name]
        r = ModelResult(name=name, sql=sql, depends_on=deps[name])
        results[name] = r
        if name in pre_errors:
            r.status, r.error = "error", pre_errors[name]
            continue
        bad = [d for d in deps[name] if results[d].status != "ok"]
        if bad:
            r.status, r.error = "blocked", f"upstream model(s) failed: {', '.join(bad)}"
            continue
        try:
            ql = analyze_query(sql, catalog, set(project.models_sql), keys)
        except LineageError as e:
            r.status, r.error = "error", str(e)
            continue
        r.columns, r.control, r.column_control = ql.columns, ql.control, ql.column_control
        r.warnings, r.qualified_sql = ql.warnings, ql.qualified_sql
        catalog[name] = {c: "UNKNOWN" for c in ql.columns}
    return results


def results_to_json(results: dict[str, ModelResult]) -> dict:
    out = {}
    for name, r in results.items():
        out[name] = {
            "status": r.status,
            "depends_on": r.depends_on,
            "sql_sha1": hashlib.sha1(r.sql.encode()).hexdigest()[:12],
        }
        if r.error:
            out[name]["error"] = r.error
        if r.status == "ok":
            out[name]["columns"] = {
                c: {n: KIND_NAMES[k] for n, k in sorted(d.items())} for c, d in r.columns.items()
            }
            out[name]["control"] = sorted(r.control)
            if r.column_control:
                out[name]["column_control"] = {c: sorted(v) for c, v in r.column_control.items()}
        if r.warnings:
            out[name]["warnings"] = r.warnings
    return out
