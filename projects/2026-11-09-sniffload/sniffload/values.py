"""Scalar parsers: nulls, booleans, locale-aware numbers, dates and datetimes.

Every parser returns ``None`` when the text does not match. None of them guess on
its own: the column-level code in ``columns.py`` decides which decimal separator
or date order applies, and these functions only apply that decision.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

NULL_TOKENS = frozenset({"", "na", "n/a", "null", "none", "nan", "nil", "-", "--", "#n/a"})
TRUE_TOKENS = frozenset({"true", "yes", "y", "t", "ja", "oui"})
FALSE_TOKENS = frozenset({"false", "no", "n", "f", "nein", "non"})

CURRENCY = "$€£¥"
# Thousands separators allowed with each decimal separator. NBSP and narrow NBSP
# are what French/Swiss locales and many spreadsheets emit.
THOUSANDS = {".": ",'   ", ",": ".   "}


def is_null(text: str) -> bool:
    return text.strip().lower() in NULL_TOKENS


def parse_bool(text: str) -> bool | None:
    t = text.strip().lower()
    if t in TRUE_TOKENS:
        return True
    if t in FALSE_TOKENS:
        return False
    return None


def _number_regex(dec: str) -> re.Pattern[str]:
    d = re.escape(dec)
    t = "[" + re.escape(THOUSANDS[dec]) + "]"
    grouped = rf"[1-9][0-9]{{0,2}}(?P<t>{t})[0-9]{{3}}(?:(?P=t)[0-9]{{3}})*"
    plain = r"[0-9]+"
    exp = r"(?:[eE][+-]?[0-9]+)?" if dec == "." else ""
    body = rf"(?:(?P<int>{grouped}|{plain})(?:{d}(?P<frac>[0-9]*))?{exp}|{d}(?P<frac2>[0-9]+){exp})"
    return re.compile(rf"^{body}$")


_NUMBER_RE = {".": _number_regex("."), ",": _number_regex(",")}


class Number:
    """Result of a successful number parse."""

    __slots__ = ("value", "has_fraction", "leading_zero", "grouped")

    def __init__(self, value: Decimal, has_fraction: bool, leading_zero: bool, grouped: bool):
        self.value = value
        self.has_fraction = has_fraction
        self.leading_zero = leading_zero
        self.grouped = grouped


def parse_number(text: str, dec: str) -> Number | None:
    """Parse ``text`` with decimal separator ``dec`` ('.' or ',').

    Accepts a leading sign, accounting parentheses ``(1,234.50)``, one currency
    symbol before or after the number, and thousands separators in groups of
    three (one separator character per value).
    """
    s = text.strip()
    if not s:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1].strip()
    if s and s[0] in "+-":
        negative, s = (s[0] == "-") != negative, s[1:].lstrip()
    if s and s[0] in CURRENCY:
        s = s[1:].lstrip()
    elif s and s[-1] in CURRENCY:
        s = s[:-1].rstrip()
    if s and s[0] in "+-" and not negative:  # "$-12.00"
        negative, s = s[0] == "-", s[1:]
    m = _NUMBER_RE[dec].match(s)
    if not m:
        return None
    int_part = m.group("int") or ""
    grouped = m.group("t") is not None
    digits_int = re.sub(r"[^0-9]", "", int_part)
    frac = m.group("frac") if m.group("frac") is not None else m.group("frac2")
    has_dec_sep = dec in s
    exp_part = ""
    if dec == "." and ("e" in s or "E" in s):
        exp_part = s[s.lower().index("e"):]
    canonical = (digits_int or "0") + ("." + frac if frac else "") + exp_part
    try:
        value = Decimal(canonical)
    except InvalidOperation:  # pragma: no cover - regex guarantees validity
        return None
    if negative:
        value = -value
    leading_zero = len(digits_int) > 1 and digits_int[0] == "0" and not grouped
    return Number(value, has_dec_sep or bool(exp_part), leading_zero, grouped)


# ---------------------------------------------------------------- dates
# (format, order). Order is the day/month ambiguity class: YMD formats can never
# be confused with each other, DMY and MDY can.
DATE_FORMATS: list[tuple[str, str]] = [
    ("%Y-%m-%d", "YMD"),
    ("%Y/%m/%d", "YMD"),
    ("%Y.%m.%d", "YMD"),
    ("%d/%m/%Y", "DMY"),
    ("%m/%d/%Y", "MDY"),
    ("%d.%m.%Y", "DMY"),
    ("%m.%d.%Y", "MDY"),
    ("%d-%m-%Y", "DMY"),
    ("%m-%d-%Y", "MDY"),
    ("%d/%m/%y", "DMY"),
    ("%m/%d/%y", "MDY"),
    ("%d.%m.%y", "DMY"),
    ("%d %b %Y", "DMY"),
    ("%d-%b-%Y", "DMY"),
    ("%b %d %Y", "MDY"),
    ("%b %d, %Y", "MDY"),
]
DATETIME_FORMATS: list[tuple[str, str]] = [
    ("iso", "YMD"),  # datetime.fromisoformat, with time part required
    ("%d/%m/%Y %H:%M", "DMY"),
    ("%m/%d/%Y %H:%M", "MDY"),
    ("%d/%m/%Y %H:%M:%S", "DMY"),
    ("%m/%d/%Y %H:%M:%S", "MDY"),
    ("%d.%m.%Y %H:%M", "DMY"),
    ("%d.%m.%Y %H:%M:%S", "DMY"),
]
_ISO_DT = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
# Cheap pre-filter so strptime is not attempted on obviously non-date text.
_DATEISH = re.compile(r"^[0-9A-Za-z]{1,4}[-/. ][0-9A-Za-z]{1,3}[-/., ]{1,2}[0-9]{2,4}$")


def parse_date(text: str, fmt: str) -> date | None:
    s = text.strip()
    if not _DATEISH.match(s):
        return None
    try:
        return datetime.strptime(s, fmt).date()
    except ValueError:
        return None


def parse_datetime(text: str, fmt: str) -> datetime | None:
    s = text.strip()
    if fmt == "iso":
        if not _ISO_DT.match(s):
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00") if s.endswith("Z") else s)
        except ValueError:
            return None
    if len(s) < 12 or not s[0].isdigit():
        return None
    try:
        return datetime.strptime(s, fmt)
    except ValueError:
        return None
