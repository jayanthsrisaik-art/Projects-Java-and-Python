"""The critic agent: decides whether a writer's SQL and result may reach the user.

It does not see gold SQL. It checks three layers and returns structured issues that are fed
back to the writer:

1. static   - parses, single read-only SELECT, every table/column exists in the catalog,
              no aggregate over rows duplicated by a one-to-many join (fan-out)
2. runtime  - executes without error inside the sandbox budget
3. contract - the executed result matches the shape the question implies
              (column count, numeric measure, row count, unique group labels, ordering, LIMIT)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify

from .contract import Contract
from .db import CATALOG, descendants
from .execute import ExecResult

ENTITY_TABLE = {"customers": "customers", "orders": "orders"}


@dataclass(frozen=True)
class Issue:
    code: str
    message: str

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


@dataclass
class Verdict:
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def codes(self) -> list[str]:
        return [i.code for i in self.issues]


# ----------------------------------------------------------------- static checks

def static_issues(sql: str, contract: Contract | None = None) -> list[Issue]:
    try:
        trees = [t for t in sqlglot.parse(sql, read="sqlite") if t is not None]
    except ParseError as exc:
        return [Issue("parse_error", f"SQL does not parse: {str(exc).splitlines()[0]}")]
    if len(trees) != 1:
        return [Issue("not_read_only", "exactly one statement is allowed")]
    tree = trees[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)) or tree.find(
            exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter):
        return [Issue("not_read_only", "only read-only SELECT queries are allowed")]

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if name not in CATALOG and name not in cte_names:
            return [Issue("unknown_table", f"table '{t.name}' does not exist; tables: "
                          + ", ".join(sorted(CATALOG)))]
    try:
        qualified = qualify(tree.copy(), schema=CATALOG, dialect="sqlite",
                            validate_qualify_columns=True, quote_identifiers=False)
    except OptimizeError as exc:
        return [Issue("unknown_column", f"column reference problem: {exc}")]
    return _fanout_issues(qualified, contract)


def _scope_tables(select: exp.Select) -> dict[str, str]:
    """alias -> table for the FROM/JOIN of this SELECT (not subqueries); FROM table first."""
    out = {}
    sources = []
    frm = select.args.get("from") or select.args.get("from_")   # key differs across sqlglot versions
    if frm is not None:
        sources.append(frm.this)
    sources += [j.this for j in select.args.get("joins") or []]
    for src in sources:
        if isinstance(src, exp.Table):
            out[src.alias_or_name.lower()] = src.name.lower()
    return out


def _fanout_issues(tree: exp.Expression, contract: Contract | None) -> list[Issue]:
    issues = []
    for select in tree.find_all(exp.Select):
        tables = _scope_tables(select)
        present = set(tables.values())
        base = next(iter(tables.values()), None)
        for agg in select.find_all(exp.Count, exp.Sum, exp.Avg):
            if agg.parent_select is not select:
                continue
            arg = agg.this
            if isinstance(arg, exp.Distinct):
                continue
            if isinstance(agg, exp.Count) and isinstance(arg, exp.Star):
                ent = contract and contract.count_entity
                # counted table: the entity the question names, else the query's FROM table
                table = ENTITY_TABLE.get(ent) if ent else base
                if table in present and descendants(table) & present:
                    dup = sorted(descendants(table) & present)
                    what = ent or f"{table} rows"
                    issues.append(Issue("fanout_count",
                        f"COUNT(*) counts joined rows, but the join to {', '.join(dup)} repeats "
                        f"each {table} row (counting {what}) - use COUNT(DISTINCT ...) or drop the join"))
                continue
            for col in arg.find_all(exp.Column):
                table = tables.get(col.table.lower())
                if table and descendants(table) & present:
                    dup = sorted(descendants(table) & present)
                    code = "fanout_count" if isinstance(agg, exp.Count) else "fanout_aggregate"
                    issues.append(Issue(code,
                        f"{agg.key.upper()} over {table}.{col.name} is inflated: the join to "
                        f"{', '.join(dup)} repeats each {table} row"))
                    break
    return issues


# ----------------------------------------------------------------- result checks

def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def contract_issues(sql: str, result: ExecResult, contract: Contract) -> list[Issue]:
    cols, rows = result.columns, result.rows
    issues: list[Issue] = []
    want = contract.n_cols
    if want is not None and len(cols) != want:
        issues.append(Issue("column_count",
            f"expected {contract.describe()}; got {len(cols)} columns ({', '.join(cols)})"))
        return issues
    shape = contract.shape
    if shape == "scalar":
        if len(rows) != 1:
            issues.append(Issue("row_count", f"expected a single row, got {len(rows)}"))
        elif rows[0][0] is None:
            issues.append(Issue("null_result", "the answer is NULL - filters probably match nothing"))
        elif not _is_number(rows[0][0]):
            issues.append(Issue("measure_not_numeric", f"expected a number, got {rows[0][0]!r}"))
        return issues
    if shape == "list":
        return issues
    # grouped / topk / series: (label, value)
    if not rows:
        issues.append(Issue("empty_result", "no rows returned - filters probably match nothing"))
        return issues
    value_idx = [i for i in range(2) if all(_is_number(r[i]) or r[i] is None for r in rows)]
    if not value_idx:
        issues.append(Issue("measure_not_numeric", "no numeric value column in the result"))
        return issues
    label_idx = 1 - value_idx[-1]
    labels = [r[label_idx] for r in rows]
    if len(set(labels)) != len(labels):
        issues.append(Issue("duplicate_group_keys",
            "group labels repeat - GROUP BY does not match the selected label column"))
    upper = sql.upper()
    if shape == "topk":
        if len(rows) > contract.k:
            issues.append(Issue("row_count", f"expected at most {contract.k} rows, got {len(rows)}"))
        if "ORDER BY" not in upper or "LIMIT" not in upper:
            issues.append(Issue("missing_order_limit", "a top-k answer needs ORDER BY ... LIMIT"))
        vals = [r[value_idx[-1]] for r in rows]
        if vals != sorted(vals, reverse=True):
            issues.append(Issue("wrong_order", "top-k rows must be sorted by value, descending"))
    if shape == "series":
        if labels != sorted(labels, key=lambda x: (x is None, x)):
            issues.append(Issue("wrong_order", "a time series must be ordered by period"))
        if len(rows) > 12:
            issues.append(Issue("row_count", f"expected at most 12 periods, got {len(rows)}"))
    return issues


class Critic:
    def review(self, sql: str, result: ExecResult | None, contract: Contract) -> Verdict:
        issues = static_issues(sql, contract)
        if issues:
            return Verdict(issues)
        if result is None or not result.ok:
            err = result.error if result else "not executed"
            return Verdict([Issue("exec_error", f"execution failed: {err}")])
        return Verdict(contract_issues(sql, result, contract))
