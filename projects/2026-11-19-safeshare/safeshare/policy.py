"""Release policy: a fail-closed declaration of what happens to every column.

Any column present in the data but absent from the policy is an error, so a
new upstream column (say ``ssn``) can never slip into a shared extract by
default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .qi import KINDS

ACTIONS = ("drop", "token", "date_shift", "qi", "sensitive", "keep")


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ColumnRule:
    table: str
    column: str
    action: str
    domain: str | None = None      # token / date_shift
    by: str | None = None          # date_shift: column identifying the subject
    kind: str | None = None        # qi
    allow_pii: bool = False        # keep: acknowledge a PII-scanner hit


@dataclass
class Policy:
    k: int
    l: int
    subject_table: str
    subject_key: str
    tables: dict            # table -> {column -> ColumnRule}
    max_date_shift_days: int = 180
    suppression_limit: float = 0.02
    digest: str = ""

    # ------------------------------------------------------------ accessors
    def rules(self, table: str) -> dict:
        return self.tables[table]

    def qi_rules(self) -> list[ColumnRule]:
        return [r for r in self.tables[self.subject_table].values() if r.action == "qi"]

    def sensitive_rule(self) -> ColumnRule | None:
        found = [r for r in self.tables[self.subject_table].values()
                 if r.action == "sensitive"]
        return found[0] if found else None

    @property
    def subject_domain(self) -> str:
        return self.tables[self.subject_table][self.subject_key].domain

    def check_columns(self, table: str, header: list[str]) -> None:
        """Fail closed: the data header must match the declared columns exactly."""
        if table not in self.tables:
            raise PolicyError(f"table {table!r} is not declared in the policy")
        declared = set(self.tables[table])
        present = set(header)
        undeclared = sorted(present - declared)
        missing = sorted(declared - present)
        if len(header) != len(present):
            raise PolicyError(f"{table}: duplicate column names in header")
        if undeclared:
            raise PolicyError(f"{table}: undeclared column(s) {undeclared} - every "
                              "column needs an explicit action (fail-closed)")
        if missing:
            raise PolicyError(f"{table}: declared column(s) {missing} not in data")


def _rule(table: str, column: str, spec: dict) -> ColumnRule:
    if not isinstance(spec, dict) or "action" not in spec:
        raise PolicyError(f"{table}.{column}: rule must be an object with 'action'")
    action = spec["action"]
    if action not in ACTIONS:
        raise PolicyError(f"{table}.{column}: unknown action {action!r}")
    allowed = {"action", "domain", "by", "type", "allow_pii"}
    extra = set(spec) - allowed
    if extra:
        raise PolicyError(f"{table}.{column}: unknown key(s) {sorted(extra)}")
    domain = spec.get("domain")
    if action in ("token", "date_shift") and not domain:
        raise PolicyError(f"{table}.{column}: {action} needs a 'domain'")
    if action == "date_shift" and not spec.get("by"):
        raise PolicyError(f"{table}.{column}: date_shift needs 'by' (subject column)")
    kind = spec.get("type")
    if action == "qi" and kind not in KINDS:
        raise PolicyError(f"{table}.{column}: qi 'type' must be one of {KINDS}")
    return ColumnRule(table, column, action, domain, spec.get("by"), kind,
                      bool(spec.get("allow_pii", False)))


def parse_policy(doc: dict) -> Policy:
    try:
        k = int(doc.get("k", 5))
        l_div = int(doc.get("l", 1))
        subject_table = doc["subject_table"]
        subject_key = doc["subject_key"]
        tables_doc = doc["tables"]
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyError(f"policy missing/invalid field: {exc}") from exc
    if k < 2:
        raise PolicyError("k must be >= 2")
    if l_div < 1:
        raise PolicyError("l must be >= 1")
    tables: dict = {}
    for table, tdoc in tables_doc.items():
        cols = tdoc.get("columns") if isinstance(tdoc, dict) else None
        if not cols:
            raise PolicyError(f"table {table!r} needs a non-empty 'columns' object")
        tables[table] = {c: _rule(table, c, s) for c, s in cols.items()}
    if subject_table not in tables:
        raise PolicyError(f"subject_table {subject_table!r} not declared")
    srules = tables[subject_table]
    if subject_key not in srules or srules[subject_key].action != "token":
        raise PolicyError("subject_key must be a 'token' column of the subject table")
    sdomain = srules[subject_key].domain
    for table, rules in tables.items():
        for r in rules.values():
            if r.action in ("qi", "sensitive") and table != subject_table:
                raise PolicyError(f"{table}.{r.column}: '{r.action}' columns are only "
                                  "allowed in the subject table")
            if r.action == "date_shift":
                by = rules.get(r.by)
                if by is None or by.action != "token" or by.domain != sdomain:
                    raise PolicyError(f"{table}.{r.column}: 'by' must name a token "
                                      f"column of domain {sdomain!r} in the same table")
                if r.domain != sdomain:
                    raise PolicyError(f"{table}.{r.column}: date_shift domain must be "
                                      f"the subject domain {sdomain!r}")
    if not any(r.action == "qi" for r in srules.values()):
        raise PolicyError("the subject table needs at least one 'qi' column")
    if sum(r.action == "sensitive" for r in srules.values()) > 1:
        raise PolicyError("at most one 'sensitive' column is supported")
    if l_div > 1 and not any(r.action == "sensitive" for r in srules.values()):
        raise PolicyError("l > 1 requires a 'sensitive' column")
    shift = int(doc.get("max_date_shift_days", 180))
    supp = float(doc.get("suppression_limit", 0.02))
    if shift < 1:
        raise PolicyError("max_date_shift_days must be >= 1")
    if not 0.0 <= supp < 1.0:
        raise PolicyError("suppression_limit must be in [0, 1)")
    digest = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]
    return Policy(k, l_div, subject_table, subject_key, tables, shift, supp, digest)


def load_policy(path: str | Path, k: int | None = None, l: int | None = None) -> Policy:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"cannot read policy {path}: {exc}") from exc
    if k is not None:
        doc["k"] = k
    if l is not None:
        doc["l"] = l
    return parse_policy(doc)
