"""Tokenisation, stemming and quantity normalisation shared by every component."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

STOPWORDS = frozenset(
    """a an the of to in on at for by with and or is are was were be been being do does did
    what which who whom how much many when where why will would can could should may might
    this that these those it its as from into than then there their they them our we you your
    has have had not no any all each per s""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def stem(tok: str) -> str:
    """A deliberately tiny suffix stripper (good enough for an English policy corpus)."""
    for suf in ("ing", "ly", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def tokens(text: str, drop_stop: bool = True) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if drop_stop and t in STOPWORDS:
            continue
        out.append(stem(t))
    return out


def norm_number(raw: str) -> str:
    """'$2,000' -> '2000', '0.670' -> '0.67', '25' -> '25'."""
    s = raw.replace(",", "").replace("$", "").strip()
    try:
        d = Decimal(s)
    except InvalidOperation:
        return s
    d = d.normalize()
    txt = format(d, "f")
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return txt


# A quantity "with a unit": money, percentages, or counts of days/weeks/months.
QUANTITY = re.compile(
    r"\$\s?(?P<money>\d[\d,]*(?:\.\d+)?)"
    r"|(?P<num>\d[\d,]*(?:\.\d+)?)\s?(?:[a-z]+\s){0,2}?(?P<unit>%|percent\b|days?\b|weeks?\b|months?\b)"
)


def quantities(text: str) -> list[str]:
    """Normalised unit-bearing quantities, in order of appearance."""
    out = []
    for m in QUANTITY.finditer(text):
        out.append(norm_number(m.group("money") or m.group("num")))
    return out


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()]
