"""Scan columns that would be released verbatim for identifiers that leaked in.

Rules: e-mail addresses, phone numbers, SSN-shaped strings, and *known
identifier values* - any value (or word) of a column the policy drops or
tokenises, e.g. a patient's surname inside a free-text note.  Findings never
contain the offending value itself; only a masked preview.
"""

from __future__ import annotations

import re

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?!\d)")
SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
WORD = re.compile(r"[A-Za-z0-9@._+-]+")
MIN_KNOWN_LEN = 4
MAX_ID_WORDS = 3


def mask(value: str) -> str:
    value = str(value)
    if len(value) <= 2:
        return "*" * len(value)
    return value[0] + "*" * (len(value) - 2) + value[-1]


def _identifier_like(rows, i) -> bool:
    """Short values (names, emails, ids) are identifier sources; long free text
    (a dropped ``notes`` column) is not - its ordinary words would flag every
    other column that shares vocabulary with it."""
    words = [len(WORD.findall(r[i])) for r in rows if r[i]]
    return bool(words) and sum(words) / len(words) <= MAX_ID_WORDS


def known_identifiers(tables: dict, policy) -> set[str]:
    """Lower-cased values and words of every dropped / tokenised column."""
    known: set[str] = set()
    for table, (header, rows) in tables.items():
        rules = policy.rules(table)
        cols = [i for i, c in enumerate(header) if rules[c].action in ("drop", "token")
                and not rules[c].allow_pii and _identifier_like(rows, i)]
        for row in rows:
            for i in cols:
                v = row[i].strip().lower()
                if len(v) >= MIN_KNOWN_LEN:
                    known.add(v)
                for w in WORD.findall(v):
                    if len(w) >= MIN_KNOWN_LEN and not w.isdigit():
                        known.add(w)
    return known


def scan(tables: dict, policy, max_examples: int = 3) -> list[dict]:
    """Return findings for columns released verbatim (keep / sensitive)."""
    known = known_identifiers(tables, policy)
    findings = []
    for table, (header, rows) in tables.items():
        rules = policy.rules(table)
        for ci, col in enumerate(header):
            rule = rules[col]
            if rule.action not in ("keep", "sensitive") or rule.allow_pii:
                continue
            hits: dict = {}
            for row in rows:
                v = row[ci]
                if not v:
                    continue
                found = []           # (rule, matched fragment)
                for kind, rx in (("email", EMAIL), ("phone", PHONE), ("ssn", SSN)):
                    m = rx.search(v)
                    if m:
                        found.append((kind, m.group()))
                low = v.strip().lower()
                if low in known:
                    found.append(("known_identifier", low))
                else:
                    w = next((w for w in WORD.findall(low)
                              if len(w) >= MIN_KNOWN_LEN and w in known), None)
                    if w:
                        found.append(("known_identifier", w))
                for kind, frag in found:
                    h = hits.setdefault(kind, {"rows": 0, "examples": []})
                    h["rows"] += 1
                    if len(h["examples"]) < max_examples:
                        h["examples"].append(mask(frag))
            for kind, h in sorted(hits.items()):
                findings.append({"table": table, "column": col, "rule": kind,
                                 "rows": h["rows"], "examples": h["examples"]})
    return findings
