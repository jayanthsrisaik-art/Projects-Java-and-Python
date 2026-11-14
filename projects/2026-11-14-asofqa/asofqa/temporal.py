"""Rule-based temporal intent parser.

Turns a question into one of:

* ``point``    – a single day (explicit date, "N years ago", "currently", or no time words at all)
* ``range``    – a period (a year, month, quarter, half-year, "last year", "this year", "last month")
* ``previous`` – the value before the version currently in effect
* ``when``     – "when did X become <value>?" (``value`` holds the normalised number)

Everything is resolved against an explicit reference date so results are reproducible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from .textutil import QUANTITY, norm_number

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
WORDNUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_N = r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_YEAR = r"((?:19|20)\d\d)"


@dataclass(frozen=True)
class Intent:
    kind: str  # point | range | previous | when
    start: date | None = None
    end: date | None = None
    value: str | None = None
    source: str = ""  # the phrase that decided it (for explanations)
    span: tuple[int, int] | None = None  # character span of that phrase in the question


def _month_end(y: int, m: int) -> date:
    return (date(y + (m == 12), m % 12 + 1, 1)) - timedelta(days=1)


def _shift_years(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year - n)
    except ValueError:  # 29 Feb
        return d.replace(year=d.year - n, day=28)


def _shift_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 - n, 12)
    y += d.year
    m += 1
    return date(y, m, min(d.day, _month_end(y, m).day))


def _num(tok: str) -> int:
    return int(tok) if tok.isdigit() else WORDNUM[tok]


def parse(question: str, ref: date) -> Intent:
    q = question.lower()

    # --- "when did X become V" -----------------------------------------------------------
    if re.search(r"\b(when (did|does|will|was|has)|since when|(on )?what date|which date)\b", q):
        qty = [m for m in QUANTITY.finditer(question)]
        if qty:
            m = qty[-1]
            return Intent("when", value=norm_number(m.group("money") or m.group("num")),
                          source=m.group(0), span=m.span())

    # --- explicit dates (most specific first) -----------------------------------------------
    rules = [
        (r"\b(\d{4})-(\d{2})-(\d{2})\b", lambda g: ("point", date(int(g[0]), int(g[1]), int(g[2])))),
        (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", lambda g: ("point", date(int(g[2]), int(g[0]), int(g[1])))),
        (rf"\b{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+{_YEAR}\b",
         lambda g: ("point", date(int(g[2]), MONTHS[g[0][:3] if g[0] not in MONTHS else g[0]], int(g[1])))),
        (rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MON}\.?,?\s+{_YEAR}\b",
         lambda g: ("point", date(int(g[2]), MONTHS[g[1] if g[1] in MONTHS else g[1][:3]], int(g[0])))),
        (rf"\bq([1-4])\s+{_YEAR}\b",
         lambda g: ("range", date(int(g[1]), 3 * int(g[0]) - 2, 1), _month_end(int(g[1]), 3 * int(g[0])))),
        (rf"\b(first|second|1st|2nd)\s+half\s+of\s+{_YEAR}\b|\bh([12])\s+{_YEAR}\b", "half"),
        (rf"\b{_MON}\s+(?:of\s+)?{_YEAR}\b",
         lambda g: ("range", date(int(g[1]), MONTHS[g[0] if g[0] in MONTHS else g[0][:3]], 1),
                    _month_end(int(g[1]), MONTHS[g[0] if g[0] in MONTHS else g[0][:3]]))),
        (rf"(?<![$\d,.])\b{_YEAR}\b(?!\s*(?:%|days?|weeks?|months?|dollars|usd))",
         lambda g: ("range", date(int(g[0]), 1, 1), date(int(g[0]), 12, 31))),
    ]
    for pat, fn in rules:
        m = re.search(pat, q)
        if not m:
            continue
        if fn == "half":
            g = m.groups()
            which, year = (g[0], g[1]) if g[0] else (("first" if g[2] == "1" else "second"), g[3])
            y = int(year)
            first = which in ("first", "1st")
            return Intent("range", date(y, 1 if first else 7, 1), date(y, 6, 30) if first else date(y, 12, 31),
                          source=m.group(0), span=m.span())
        try:
            res = fn(m.groups())
        except (ValueError, KeyError):
            continue  # e.g. 2/30/2024 – not a real date; keep looking
        kind, s, *rest = res
        e = rest[0] if rest else s
        return Intent(kind, s, e, source=m.group(0), span=m.span())

    # --- "previous" ----------------------------------------------------------------------
    m = re.search(r"\b(before the (most recent|latest|last) (change|update|revision)|previous(ly)?|"
                  r"prior (value|figure|allowance|rate|policy)|used to be|before it was (last )?"
                  r"(changed|revised|updated))\b", q)
    if m:
        return Intent("previous", source=m.group(0), span=m.span())

    # --- relative expressions --------------------------------------------------------------
    m = re.search(rf"\b{_N}\s+(year|month|week)s?\s+ago\b", q)
    if m:
        n, unit = _num(m.group(1)), m.group(2)
        d = (_shift_years(ref, n) if unit == "year" else _shift_months(ref, n) if unit == "month"
             else ref - timedelta(weeks=n))
        return Intent("point", d, d, source=m.group(0), span=m.span())
    m = re.search(r"\b(last|previous|this|past) (year|month)\b", q)
    if m:
        which, unit = m.groups()
        if unit == "year":
            if which == "this":
                s, e = date(ref.year, 1, 1), ref
            elif which == "past":
                s, e = _shift_years(ref, 1), ref
            else:
                s, e = date(ref.year - 1, 1, 1), date(ref.year - 1, 12, 31)
        else:
            if which == "this":
                s, e = date(ref.year, ref.month, 1), ref
            elif which == "past":
                s, e = _shift_months(ref, 1), ref
            else:
                p = _shift_months(date(ref.year, ref.month, 1), 1)
                s, e = p, _month_end(p.year, p.month)
        return Intent("range", s, e, source=m.group(0), span=m.span())

    m = re.search(r"\b(now|currently|current|today|at present|right now|these days|nowadays)\b", q)
    return Intent("point", ref, ref, source=m.group(0) if m else "(default: reference date)",
                  span=m.span() if m else None)


def strip_temporal(question: str, intent: Intent) -> str:
    """Question text with the temporal phrase removed (so dates do not bias retrieval)."""
    if intent.span and intent.kind != "when":
        a, b = intent.span
        return (question[:a] + " " + question[b:]).strip()
    return question
