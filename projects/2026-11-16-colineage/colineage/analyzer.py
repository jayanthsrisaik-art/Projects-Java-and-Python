"""Single-query column lineage on top of sqlglot's qualified AST.

``analyze_query`` takes one SELECT (or set operation), qualifies it against a catalog
(``{relation: {column: type}}``) and returns, for every output column:

* **value** dependencies: ``{"relation.column": kind}``, the upstream columns whose values flow
  into this column's values. ``kind`` is the strongest operation on the path:
  ``identity`` < ``transform`` < ``window`` < ``aggregate``.
* plus one relation-wide **control** set: upstream columns that decide which rows exist or how
  they are grouped / deduplicated / matched (WHERE, JOIN ON, GROUP BY, HAVING, QUALIFY,
  DISTINCT, ORDER BY under LIMIT, EXISTS/IN subqueries, set-operation dedupe), including the
  control sets of every relation read in FROM/JOIN.

The analysis is deliberately conservative (sound): if a change to an upstream column could
change an output column's values, the upstream column appears in value or control. The
perturbation oracle in ``oracle.py`` checks this property against SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify

DIALECT = "sqlite"

IDENTITY, TRANSFORM, WINDOW, AGGREGATE = 0, 1, 2, 3
KIND_NAMES = {IDENTITY: "identity", TRANSFORM: "transform", WINDOW: "window", AGGREGATE: "aggregate"}
KIND_BY_NAME = {v: k for k, v in KIND_NAMES.items()}

Deps = dict[str, int]  # node "relation.column" -> kind


class LineageError(Exception):
    """A model cannot be analyzed (parse error, unknown table/column, unsupported SQL)."""


@dataclass
class Rel:
    """Lineage of a relation as seen by the query reading it."""

    columns: dict[str, Deps]  # ordered output column -> value deps
    control: set[str] = field(default_factory=set)  # decides which rows exist
    # extra control that applies to one column only (on top of ``control``)
    col_control: dict[str, set[str]] = field(default_factory=dict)

    def column(self, name: str) -> tuple[Deps, set[str]]:
        return self.columns[name], self.col_control.get(name, set())


def merge_deps(into: Deps, other: Deps, floor: int = IDENTITY) -> Deps:
    """Add ``other`` into ``into``; each kind is raised to at least ``floor``; keep the max."""
    for node, kind in other.items():
        k = max(kind, floor)
        if into.get(node, -1) < k:
            into[node] = k
    return into


def referenced_tables(sql: str) -> tuple[set[str], set[str]]:
    """(tables referenced, CTE names defined) from an unqualified parse."""
    tree = _parse(sql)
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = {t.name.lower() for t in tree.find_all(exp.Table) if t.name}
    return tables - ctes, ctes


def _parse(sql: str) -> exp.Expression:
    try:
        statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
    except ParseError as e:
        raise LineageError(f"parse error: {_first_line(str(e))}") from None
    if len(statements) != 1:
        raise LineageError(f"expected exactly one statement, found {len(statements)}")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise LineageError(f"model must be a SELECT query, got {tree.key.upper()}")
    return tree


def _first_line(s: str) -> str:
    return s.strip().splitlines()[0] if s.strip() else s


def qualify_query(sql: str, catalog: dict[str, dict[str, str]]) -> exp.Query:
    tree = _parse(sql)
    tables, ctes = referenced_tables(sql)
    unknown = sorted(t for t in tables if t not in catalog)
    if unknown:
        raise LineageError(f"unknown relation(s): {', '.join(unknown)}")
    if any(isinstance(w, exp.With) and w.args.get("recursive") for w in tree.find_all(exp.With)):
        raise LineageError("recursive CTEs are not supported")
    try:
        q = qualify(
            tree,
            schema=catalog,
            dialect=DIALECT,
            validate_qualify_columns=True,
            identify=False,
        )
    except OptimizeError as e:
        raise LineageError(_explain_optimize_error(str(e), tree, catalog)) from None
    for star in q.find_all(exp.Star):
        if not isinstance(star.parent, exp.Count) and not isinstance(star.parent, exp.AggFunc):
            raise LineageError("could not expand SELECT * (unknown relation columns)")
    return q


def _explain_optimize_error(msg: str, tree: exp.Expression, catalog) -> str:
    msg = _first_line(msg)
    # "Column 'x' could not be resolved" -> say which in-scope relations have it (ambiguity)
    if "could not be resolved" in msg and "'" in msg:
        col = msg.split("'")[1].lower()
        rels = sorted(
            {t.name.lower() for t in tree.find_all(exp.Table) if col in catalog.get(t.name.lower(), {})}
        )
        if len(rels) > 1:
            return f"ambiguous column '{col}': present in {', '.join(rels)}; qualify it"
        if not rels:
            return f"unknown column '{col}'"
    return msg


class _Scope:
    def __init__(self, sources: dict[str, Rel], outer: "_Scope | None"):
        self.sources = sources
        self.outer = outer
        self.projections: dict[str, Deps] = {}
        self.proj_control: dict[str, set[str]] = {}

    def resolve(self, col: exp.Column) -> tuple[Deps, set[str]]:
        table = col.table.lower() if col.table else ""
        name = col.name.lower()
        scope: _Scope | None = self
        while scope is not None:
            if table:
                rel = scope.sources.get(table)
                if rel is not None:
                    if name not in rel.columns:
                        raise LineageError(f"column {table}.{name} does not exist")
                    return rel.column(name)
            else:
                # unqualified after qualify() == reference to an output alias (ORDER BY / HAVING)
                if name in scope.projections:
                    return scope.projections[name], scope.proj_control.get(name, set())
            scope = scope.outer
        raise LineageError(f"cannot resolve column {col.sql(dialect=DIALECT)}")


class QueryAnalyzer:
    def __init__(
        self,
        catalog: dict[str, dict[str, str]],
        row_nodes: set[str] = frozenset(),
        keys: dict[str, set[str]] | None = None,
    ):
        self.catalog = catalog
        self.row_nodes = row_nodes  # relations whose row set is itself filtered (models)
        self.keys = keys or {}  # relation -> single-column unique keys (declared PKs)
        self.warnings: list[str] = []

    # -- entry -----------------------------------------------------------------------------
    def analyze(self, q: exp.Query, ctes: dict[str, Rel] | None = None, outer: _Scope | None = None) -> Rel:
        ctes = dict(ctes or {})
        with_ = q.args.get("with_") or q.args.get("with")
        if with_ is not None:
            for cte in with_.expressions:
                ctes[cte.alias_or_name.lower()] = self.analyze(cte.this, ctes, outer)
        if isinstance(q, exp.Subquery):
            return self.analyze(q.this, ctes, outer)
        if isinstance(q, exp.SetOperation):
            return self._set_op(q, ctes, outer)
        if isinstance(q, exp.Select):
            return self._select(q, ctes, outer)
        raise LineageError(f"unsupported query node {q.key}")

    # -- set operations ------------------------------------------------------------------------
    def _set_op(self, q: exp.SetOperation, ctes, outer) -> Rel:
        left = self.analyze(q.left, ctes, outer)
        right = self.analyze(q.right, ctes, outer)
        if len(left.columns) != len(right.columns):
            raise LineageError(
                f"{q.key.upper()} branches have {len(left.columns)} and {len(right.columns)} columns"
            )
        control = set(left.control) | set(right.control)
        names = list(left.columns)
        cols: dict[str, Deps] = {}
        col_control: dict[str, set[str]] = {}
        right_names = list(right.columns)
        right_deps = list(right.columns.values())
        for i, name in enumerate(names):
            d: Deps = dict(left.columns[name])
            extra = set(left.col_control.get(name, set()))
            r_extra = right.col_control.get(right_names[i], set())
            if isinstance(q, exp.Union):
                merge_deps(d, right_deps[i])
                extra |= r_extra
            else:  # INTERSECT / EXCEPT: right side only filters
                control.update(right_deps[i])
                control.update(r_extra)
            cols[name] = d
            col_control[name] = extra
        distinct = q.args.get("distinct", True) or not isinstance(q, exp.Union)
        if distinct:  # dedupe: every column decides which rows survive
            for d in list(cols.values()) + right_deps:
                control.update(d)
            for e in col_control.values():
                control.update(e)
        return Rel(cols, control, col_control)

    # -- SELECT ----------------------------------------------------------------------------------
    def _relation(self, node: exp.Expression, ctes, outer) -> tuple[str, Rel]:
        if isinstance(node, exp.Table):
            name = node.name.lower()
            alias = (node.alias or node.name).lower()
            if name in ctes and not node.args.get("db"):
                return alias, ctes[name]
            if name in self.catalog:
                # reading a model: which rows it has is decided by that model's own control set,
                # represented by the row-existence node "model.*"
                control = {f"{name}.*"} if name in self.row_nodes else set()
                return alias, Rel({c: {f"{name}.{c}": IDENTITY} for c in self.catalog[name]}, control)
            raise LineageError(f"unknown relation {name}")
        if isinstance(node, exp.Subquery):
            alias = node.alias_or_name.lower()
            return alias, self.analyze(node.this, ctes, outer)
        raise LineageError(f"unsupported FROM item: {node.sql(dialect=DIALECT)[:60]}")

    def _select(self, q: exp.Select, ctes, outer) -> Rel:
        sources: dict[str, Rel] = {}
        from_ = q.args.get("from_") or q.args.get("from")
        items = []
        if from_ is not None:
            items.append(from_.this)
        for j in q.args.get("joins") or []:
            items.append(j.this)
        for item in items:
            alias, rel = self._relation(item, ctes, outer)
            sources[alias] = rel
        scope = _Scope(sources, outer)
        control: set[str] = set()
        for rel in sources.values():
            control.update(rel.control)

        cols: dict[str, Deps] = {}
        extras: dict[str, set[str]] = {}
        inners: dict[str, exp.Expression] = {}
        for proj in q.expressions:
            name = proj.alias_or_name.lower()
            if name in cols:
                raise LineageError(f"duplicate output column '{name}'")
            inner = proj.this if isinstance(proj, exp.Alias) else proj
            bare = inner
            while isinstance(bare, exp.Paren):
                bare = bare.this
            base = IDENTITY if isinstance(bare, exp.Column) else TRANSFORM
            deps: Deps = {}
            extra: set[str] = set()
            self._value(inner, scope, ctes, base, False, deps, extra)
            cols[name], extras[name], inners[name] = deps, extra, inner
        scope.projections, scope.proj_control = cols, extras

        def ctl(node: exp.Expression | None) -> None:
            if node is not None:
                d: Deps = {}
                self._value(node, scope, ctes, TRANSFORM, False, d, control)
                control.update(d)

        for j in q.args.get("joins") or []:
            ctl(j.args.get("on"))
            for u in j.args.get("using") or []:  # qualify() normally rewrites USING to ON
                ctl(u)
        ctl(q.args.get("where"))
        preserved = self._key_preserved(q, from_, sources)
        if preserved is None:
            ctl(q.args.get("group"))
        # else: grouping by a unique key of the preserved relation (plus columns of that same
        # relation) yields one group per row, whatever the grouped values are
        ctl(q.args.get("having"))
        ctl(q.args.get("qualify"))
        if q.args.get("distinct"):
            for d in cols.values():
                control.update(d)
        if q.args.get("limit") or q.args.get("offset"):
            if q.args.get("order") is None:
                self.warnings.append("LIMIT without ORDER BY: result rows are nondeterministic")
            ctl(q.args.get("order"))
        if q.args.get("distinct"):
            for e in extras.values():
                control.update(e)

        if preserved is None:
            return Rel(cols, control, extras)
        # LEFT JOINs + GROUP BY a unique key of the FROM relation, nothing else filtering:
        # exactly one output row per row of that relation. Non-aggregate columns computed only
        # from that relation are independent of the joined tables and of the grouping.
        row_control = set(sources[preserved].control)
        col_control = {}
        for name, inner in inners.items():
            if self._only_from(inner, preserved):
                col_control[name] = extras[name]
            else:
                col_control[name] = extras[name] | control
        return Rel(cols, row_control, col_control)

    def _key_preserved(self, q: exp.Select, from_, sources: dict[str, Rel]) -> str | None:
        if from_ is None or not isinstance(from_.this, exp.Table) or q.args.get("group") is None:
            return None
        if any(q.args.get(k) for k in ("where", "having", "qualify", "distinct", "limit", "offset")):
            return None
        base = from_.this.name.lower()
        alias = (from_.this.alias or from_.this.name).lower()
        keys = self.keys.get(base, set())
        if not keys or sources.get(alias) is None:
            return None
        for j in q.args.get("joins") or []:
            if (j.args.get("side") or "").upper() != "LEFT" or j.args.get("kind"):
                return None
        group = q.args["group"].expressions
        if not all(isinstance(g, exp.Column) and g.table.lower() == alias for g in group):
            return None
        if not any(g.name.lower() in keys for g in group):
            return None
        return alias

    @staticmethod
    def _only_from(node: exp.Expression, alias: str) -> bool:
        if any(isinstance(n, (exp.AggFunc, exp.Window, exp.Query, exp.Subquery)) for n in node.walk()):
            return False
        return all(c.table.lower() == alias for c in node.find_all(exp.Column))

    # -- expressions ------------------------------------------------------------------------------
    def _value(
        self, node, scope: _Scope, ctes, kind: int, in_window: bool, out: Deps, ctl: set[str]
    ) -> None:
        """Collect value deps of ``node`` into ``out`` and per-column control into ``ctl``."""
        if isinstance(node, exp.Column):
            if isinstance(node.this, exp.Star):
                return
            deps, extra = scope.resolve(node)
            merge_deps(out, deps, kind)
            ctl.update(extra)
            return
        if isinstance(node, (exp.Subquery, exp.Query)) and not isinstance(node, exp.Values):
            sub = self.analyze(node if not isinstance(node, exp.Subquery) else node.this, ctes, scope)
            for d in sub.columns.values():
                merge_deps(out, d, max(kind, TRANSFORM))
            # a scalar/EXISTS/IN subquery's filters shape the value itself
            for n in set(sub.control).union(*sub.col_control.values()):
                merge_deps(out, {n: TRANSFORM}, max(kind, TRANSFORM))
            return
        if isinstance(node, exp.Window):
            k = max(kind, WINDOW)
            fn = node.this
            if isinstance(fn, exp.AggFunc):
                for child in fn.iter_expressions():
                    self._value(child, scope, ctes, k, True, out, ctl)
            else:
                self._value(fn, scope, ctes, k, True, out, ctl)
            for key in ("partition_by", "order", "spec"):
                v = node.args.get(key)
                if isinstance(v, list):
                    for child in v:
                        self._value(child, scope, ctes, k, True, out, ctl)
                elif v is not None:
                    self._value(v, scope, ctes, k, True, out, ctl)
            return
        if isinstance(node, exp.AggFunc) and not in_window:
            kind = max(kind, AGGREGATE)
        elif not isinstance(node, (exp.Column, exp.Paren)) and kind == IDENTITY:
            kind = TRANSFORM
        for child in node.iter_expressions():
            self._value(child, scope, ctes, kind, in_window, out, ctl)


@dataclass
class QueryLineage:
    columns: dict[str, Deps]
    control: set[str]
    column_control: dict[str, set[str]]
    upstream: set[str]
    qualified_sql: str
    warnings: list[str]


def analyze_query(
    sql: str,
    catalog: dict[str, dict[str, str]],
    row_nodes: set[str] = frozenset(),
    keys: dict[str, set[str]] | None = None,
) -> QueryLineage:
    q = qualify_query(sql, catalog)
    qa = QueryAnalyzer(catalog, row_nodes, keys)
    rel = qa.analyze(q)
    tables, _ = referenced_tables(sql)
    return QueryLineage(
        columns=rel.columns,
        control=rel.control,
        column_control={c: set(v) for c, v in rel.col_control.items() if v},
        upstream=tables,
        qualified_sql=q.sql(dialect=DIALECT, pretty=True),
        warnings=qa.warnings,
    )
