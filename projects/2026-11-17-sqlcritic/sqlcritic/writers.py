"""Writer agents. A writer sees the question (never the contract or gold SQL) plus the
critic's feedback on its earlier attempts, and returns one SQL string.

* HeuristicWriter - offline, deterministic keyword parser. Its vocabulary was built from the
  dev-split wording only, so its test-split failures are genuine. On critic feedback it moves
  to its next-ranked interpretation (a different shape after a shape complaint, the same shape
  otherwise). It is a baseline for exercising the loop, not a claim about LLM accuracy.
* ClaudeWriter    - live mode through the Anthropic SDK (needs credentials).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from .db import CATEGORIES, REGIONS, SCHEMA_DOC, SEGMENTS, STATUSES


@dataclass
class Attempt:
    sql: str
    issues: list[dict] = field(default_factory=list)   # [{"code", "message"}]


class WriterError(RuntimeError):
    pass


class Writer(Protocol):
    name: str

    def write(self, question: str, history: list[Attempt]) -> str: ...


# ======================================================================= heuristic writer

SHAPE_CODES = {"column_count", "row_count", "duplicate_group_keys", "missing_order_limit",
               "wrong_order", "empty_result"}

TABLE_ALIAS = {"order_items": "oi", "orders": "o", "customers": "c", "regions": "r", "products": "p"}
# the schema's FK graph is a tree; edges with their join condition
EDGES = {
    ("order_items", "orders"): "o.order_id = oi.order_id",
    ("order_items", "products"): "p.product_id = oi.product_id",
    ("orders", "customers"): "c.customer_id = o.customer_id",
    ("customers", "regions"): "r.region_id = c.region_id",
}
REV = "SUM(oi.quantity * oi.unit_price)"


def _neighbors(t: str) -> list[str]:
    return [b if a == t else a for a, b in EDGES if t in (a, b)]


def _path(a: str, b: str) -> list[str]:
    prev, frontier = {a: None}, [a]
    while frontier:
        nxt = []
        for t in frontier:
            for n in _neighbors(t):
                if n not in prev:
                    prev[n] = t
                    nxt.append(n)
        frontier = nxt
    out, cur = [], b
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    return out[::-1]


def _from_clause(base: str, needed: set[str]) -> tuple[str, set[str]]:
    tables = [base]
    for t in sorted(needed):
        for step in _path(base, t):
            if step not in tables:
                tables.append(step)
    sql = f"{base} {TABLE_ALIAS[base]}"
    for t in tables[1:]:
        cond = next(c for (a, b), c in EDGES.items()
                    if t in (a, b) and ({a, b} - {t}) <= set(tables[:tables.index(t)]))
        sql += f" JOIN {t} {TABLE_ALIAS[t]} ON {cond}"
    return sql, set(tables)


@dataclass
class Slots:
    year: str | None = None
    region: str | None = None
    category: str | None = None
    segment: str | None = None
    status: str | None = None
    price: str | None = None
    k: int | None = None
    purchase: bool = False


def parse_slots(q: str) -> Slots:
    ql = q.lower()
    s = Slots()
    if m := re.search(r"\b(20\d\d)\b", ql):
        s.year = m.group(1)
    s.region = next((r for r in REGIONS if re.search(rf"\b{r.lower()}\b", ql)), None)
    s.category = next((c for c in CATEGORIES if re.search(rf"\b{c}\b", ql)), None)
    s.segment = next((x for x in SEGMENTS if re.search(rf"\b{x}\b", ql)), None)
    s.status = next((x for x in STATUSES if re.search(rf"\b{x}\b", ql)), None)
    if m := re.search(r"\b(?:over|above)\s*\$?(\d+(?:\.\d+)?)", ql):
        s.price = m.group(1)
    if m := re.search(r"\b(?:top|which)\s+(\d{1,2})\b", ql):
        s.k = int(m.group(1))
    s.purchase = bool(re.search(r"\b(bought|purchased)\b", ql))
    return s


def detect_measures(q: str) -> list[str]:
    ql = q.lower()
    rules = [
        ("aov", r"average order value"),
        ("cancel_share", r"\b(share|fraction)\b"),
        ("no_orders", r"\bnever\b|\bno orders\b"),
        ("units", r"\b(units|quantity)\b"),
        ("revenue", r"\brevenue\b"),
        ("customers", r"\b(distinct|unique) customers\b"),
        ("products_list", r"\bprice[d]?\b"),
        ("orders", r"\borders?\b"),
        ("customers", r"\bcustomers\b"),
    ]
    found = []
    for name, pat in rules:
        if re.search(pat, ql) and name not in found:
            found.append(name)
    for fallback in ("orders", "revenue", "customers"):
        if fallback not in found:
            found.append(fallback)
    return found


def detect_groups(q: str, slots: Slots) -> list[str | None]:
    ql = q.lower()
    primary: list[str | None] = []
    if m := re.search(r"\b(?:by|per|each)\s+(region|status|category|segment)\b", ql):
        primary.append(m.group(1))
    if re.search(r"\b(monthly|per month)\b", ql):
        primary.append("month")
    if slots.k:
        if "products" in ql:
            primary.append("product")
        if "customers" in ql:
            primary.append("customer")
    out: list[str | None] = primary or [None]
    for word, g in [("region", "region"), ("status", "status"), ("categor", "category"),
                    ("segment", "segment"), ("month", "month"), ("product", "product"),
                    ("customer", "customer")]:
        if word in ql and g not in out:
            out.append(g)
    if None not in out:
        out.append(None)
    return out


GROUP_EXPR = {
    "region": ("r.name", "regions", "r.name"),
    "status": ("o.status", "orders", "o.status"),
    "category": ("p.category", "products", "p.category"),
    "segment": ("c.segment", "customers", "c.segment"),
    "month": ("strftime('%m', o.order_date)", "orders", "strftime('%m', o.order_date)"),
    "product": ("p.name", "products", "p.product_id, p.name"),
    "customer": ("c.name", "customers", "c.customer_id, c.name"),
}


@dataclass(frozen=True)
class Candidate:
    sql: str
    shape: str


def build_sql(measure: str, group: str | None, s: Slots) -> Candidate | None:
    where: list[str] = []
    needed: set[str] = set()
    completed = measure in ("revenue", "units", "aov") or (
        s.purchase and measure in ("customers", "orders"))

    if measure == "products_list":
        conds = []
        if s.category:
            conds.append(f"p.category = '{s.category}'")
        if s.price:
            conds.append(f"p.unit_price > {s.price}")
        sql = "SELECT p.name, p.unit_price FROM products p"
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        return Candidate(sql, "list")

    base = {"revenue": "order_items", "units": "order_items", "aov": "order_items",
            "orders": "orders", "cancel_share": "orders", "customers": "customers",
            "no_orders": "customers"}[measure]
    if s.year:
        needed.add("orders")
        where.append(f"strftime('%Y', o.order_date) = '{s.year}'")
    if s.region:
        needed.add("regions")
        where.append(f"r.name = '{s.region}'")
    if s.category:
        needed.add("products")
        where.append(f"p.category = '{s.category}'")
    if s.segment:
        needed.add("customers")
        where.append(f"c.segment = '{s.segment}'")
    if completed:
        needed.add("orders")
        where.append("o.status = 'completed'")
    elif s.status and measure in ("orders", "customers") and s.status != "completed":
        needed.add("orders")
        where.append(f"o.status = '{s.status}'")
    if measure == "no_orders":
        needed.discard("orders")
        where = [w for w in where if not w.startswith(("o.", "strftime"))]
        where.append("NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id)")
    if group:
        needed.add(GROUP_EXPR[group][1])
    if measure == "customers" and needed - {"customers", "regions"}:
        base = "orders"   # reach products/items through orders, count customers distinctly
    needed.discard(base)
    frm, tables = _from_clause(base, needed)

    if measure == "revenue":
        m = REV
    elif measure == "units":
        m = "SUM(oi.quantity)"
    elif measure == "aov":
        m = f"{REV} * 1.0 / COUNT(DISTINCT o.order_id)"
    elif measure == "cancel_share":
        m = "AVG(CASE WHEN o.status = 'cancelled' THEN 1.0 ELSE 0.0 END)"
    elif measure == "orders":
        m = "COUNT(DISTINCT o.order_id)" if "order_items" in tables else "COUNT(*)"
    elif measure == "customers":
        key = "o.customer_id" if base == "orders" else "c.customer_id"
        m = f"COUNT(DISTINCT {key})" if base == "orders" or tables - {"customers", "regions"} \
            else "COUNT(*)"
    else:  # no_orders
        m = "COUNT(*)"

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    if group is None:
        return Candidate(f"SELECT {m} FROM {frm}{where_sql}", "scalar")
    label, _, gby = GROUP_EXPR[group]
    sql = f"SELECT {label} AS {group}, {m} AS value FROM {frm}{where_sql} GROUP BY {gby}"
    if s.k and group in ("product", "customer"):
        return Candidate(sql + f" ORDER BY value DESC LIMIT {s.k}", "topk")
    if group == "month":
        return Candidate(sql + " ORDER BY month", "series")
    return Candidate(sql, "grouped")


class HeuristicWriter:
    name = "heuristic"

    def candidates(self, question: str) -> list[Candidate]:
        slots = parse_slots(question)
        measures = detect_measures(question)
        groups = detect_groups(question, slots)
        ranked = sorted(((mi + gi, mi, gi, m, g) for mi, m in enumerate(measures)
                         for gi, g in enumerate(groups)), key=lambda t: t[:3])
        out, seen = [], set()
        for *_, m, g in ranked:
            cand = build_sql(m, g, slots)
            if cand and cand.sql not in seen:
                seen.add(cand.sql)
                out.append(cand)
        return out[:12]

    def write(self, question: str, history: list[Attempt]) -> str:
        cands = self.candidates(question)
        tried = {a.sql for a in history}
        remaining = [c for c in cands if c.sql not in tried]
        if not remaining:
            raise WriterError("no further interpretations")
        if history:
            last = history[-1]
            last_shape = next((c.shape for c in cands if c.sql == last.sql), None)
            shape_complaint = any(i["code"] in SHAPE_CODES for i in last.issues)
            pref = [c for c in remaining if (c.shape != last_shape) == shape_complaint]
            if pref:
                return pref[0].sql
        return remaining[0].sql


# ======================================================================= Claude writer

SYSTEM_PROMPT = f"""You translate business questions into a single SQLite SELECT query.

{SCHEMA_DOC}
Rules:
- Return exactly one read-only SELECT statement inside a ```sql code block, nothing else needed.
- Follow the business definitions above exactly (e.g. revenue counts completed orders only).
- When a join to order_items repeats order or customer rows, count with COUNT(DISTINCT ...).
- For "top N" questions return (label, value) ordered by value descending with LIMIT N.
- For breakdowns return (label, value) with one row per label.
- If a reviewer reports problems with your previous query, fix exactly those problems."""

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    m = _SQL_BLOCK.search(text)
    sql = (m.group(1) if m else text).strip().rstrip(";").strip()
    if not sql:
        raise WriterError("model returned no SQL")
    return sql


def build_user_message(question: str, history: list[Attempt]) -> str:
    parts = [f"Question: {question}"]
    for i, a in enumerate(history, 1):
        feedback = "\n".join(f"  - [{x['code']}] {x['message']}" for x in a.issues) or "  - (none)"
        parts.append(f"Attempt {i}:\n```sql\n{a.sql}\n```\nReviewer feedback:\n{feedback}")
    if history:
        parts.append("Write a corrected query.")
    return "\n\n".join(parts)


class ClaudeWriter:
    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", client=None):
        self.model = model
        self.effort = effort
        if client is None:
            import anthropic   # imported lazily so offline mode needs no SDK credentials
            client = anthropic.Anthropic()
        self.client = client

    def write(self, question: str, history: list[Attempt]) -> str:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": build_user_message(question, history)}],
        )
        if response.stop_reason == "refusal":
            raise WriterError("model declined the request")
        if response.stop_reason == "max_tokens":
            raise WriterError("model output was truncated")
        text = "".join(b.text for b in response.content if b.type == "text")
        return extract_sql(text)


def make_writer(name: str, **kw) -> Writer:
    if name == "heuristic":
        return HeuristicWriter()
    if name == "claude":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or kw.get("client")):
            # the SDK can also use an `ant auth login` profile; let it try and report clearly
            pass
        return ClaudeWriter(**kw)
    raise ValueError(f"unknown writer {name!r}")
