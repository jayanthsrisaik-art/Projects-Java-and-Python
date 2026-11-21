"""Extraction and canonicalisation of *atoms*: the security-relevant values that
appear in tool-call arguments (email addresses, bank accounts, money amounts,
URL hosts and file paths).

Provenance is decided per atom: an argument value is safe only if every atom in
it can be found, in canonical form, in a sufficiently trusted part of the
context. Canonicalisation therefore has to be the same on both sides (the
argument and the context text), which is why every kind exposes one function,
``extract(kind, text)``, used for both.
"""

from __future__ import annotations

import posixpath
import re
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit

KINDS = ("email", "account", "amount", "host", "path")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
# Internal account ids (ACCT-1234-5678) and IBAN-style ids, optionally grouped by spaces.
ACCT_RE = re.compile(r"\bACCT-\d{4}-\d{4}\b", re.IGNORECASE)
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")
# Money must be *marked* (currency symbol before or currency word after); bare
# numbers in text (dates, ids, phone numbers) would otherwise attest amounts by
# coincidence.
_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?"
MONEY_PREFIX_RE = re.compile(r"[$€£]\s?" + _NUM)
MONEY_SUFFIX_RE = re.compile(_NUM + r"\s?(?:USD|EUR|GBP|dollars?|euros?)\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s<>\"'()\[\]{}]+", re.IGNORECASE)
DOMAIN_RE = re.compile(r"(?<![\w@.-])(?:[a-z0-9-]+\.)+(?:com|org|net|io|dev|co|example|test|info|biz)\b",
                       re.IGNORECASE)
FILE_EXT = r"(?:txt|md|csv|pdf|docx|xlsx|json|log|key|pem|env|db|zip|bak|ini|yaml|yml)"
PATH_RE = re.compile(r"(?<![\w@/.:-])(?:~/|/)?(?:[\w-]+/)*[\w-]+\." + FILE_EXT + r"\b", re.IGNORECASE)


def _money(whole: str, frac: str | None) -> str | None:
    try:
        value = Decimal(whole.replace(",", "") + "." + (frac or "0"))
    except InvalidOperation:
        return None
    return f"{value:.2f}"


def canon_host(host: str) -> str:
    host = host.lower().strip().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def canon_path(path: str) -> str:
    path = path.strip()
    if path.startswith("~/"):
        path = path[2:]
    path = posixpath.normpath(path)
    return path.lstrip("/").lower()


def _blank(pattern: re.Pattern[str], text: str) -> str:
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def extract(kind: str, text: str) -> set[str]:
    """Return the canonical atoms of ``kind`` found in free text."""
    if kind == "email":
        return {m.group(0).lower().rstrip(".") for m in EMAIL_RE.finditer(text)}
    if kind == "account":
        found = {m.group(0).upper() for m in ACCT_RE.finditer(text)}
        found |= {m.group(0).replace(" ", "").upper() for m in IBAN_RE.finditer(text)}
        return found
    if kind == "amount":
        found = set()
        for rx in (MONEY_PREFIX_RE, MONEY_SUFFIX_RE):
            for m in rx.finditer(text):
                value = _money(m.group(1), m.group(2))
                if value is not None:
                    found.add(value)
        return found
    if kind == "host":
        found = set()
        for m in URL_RE.finditer(text):
            host = urlsplit(m.group(0)).hostname
            if host:
                found.add(canon_host(host))
        rest = _blank(EMAIL_RE, _blank(URL_RE, text))
        found |= {canon_host(m.group(0)) for m in DOMAIN_RE.finditer(rest)}
        return found
    if kind == "path":
        rest = _blank(EMAIL_RE, _blank(URL_RE, text))
        return {canon_path(m.group(0)) for m in PATH_RE.finditer(rest)}
    raise ValueError(f"unknown atom kind: {kind}")


def atoms_of_value(kind: str, value: object) -> set[str]:
    """Canonical atoms of a tool-call argument value.

    Arguments are more structured than prose, so a few forms that ``extract``
    deliberately ignores in prose are accepted here: a bare number for an
    amount, and a bare host or path.
    """
    if value is None:
        return set()
    if isinstance(value, (list, tuple)):
        out: set[str] = set()
        for item in value:
            out |= atoms_of_value(kind, item)
        return out
    if kind == "amount" and isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return {f"{Decimal(str(value)):.2f}"}
    text = str(value)
    found = extract(kind, text)
    if found:
        return found
    stripped = text.strip()
    if kind == "amount":
        m = re.fullmatch(_NUM, stripped.replace("$", "").replace("€", "").replace("£", "").strip())
        if m:
            value_s = _money(m.group(1), m.group(2))
            return {value_s} if value_s else set()
    if kind == "host" and re.fullmatch(r"(?:[a-z0-9-]+\.)+[a-z]{2,}", stripped, re.IGNORECASE):
        return {canon_host(stripped)}
    if kind == "path" and stripped and " " not in stripped:
        return {canon_path(stripped)}
    return set()


def all_atoms(text: str) -> set[tuple[str, str]]:
    """Every (kind, atom) pair in ``text``; used by the mock model to decide
    whether it has *seen* the values a plan step needs."""
    return {(kind, atom) for kind in KINDS for atom in extract(kind, text)}
