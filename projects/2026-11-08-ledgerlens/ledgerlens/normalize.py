"""Raw bank descriptor -> (processor, clean merchant name, stable merchant key).

The merchant key is what everything downstream groups on: user rules, recurring
detection, and the train/test split. It must be identical for every transaction of
one merchant (store numbers, dates, card digits and processors vary) and different
across merchants.

Key = first 16 letters of the cleaned name with spaces and punctuation removed.
That makes ``SQ *BLUE BOTTLE COF`` and ``PAYPAL *BLUEBOTTLECOFFEE`` agree, and is
robust to the bank truncating the name field at ~20 characters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

KEY_LEN = 16

US_STATES = set("""AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE
NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC""".split())

# Small city gazetteer. A production system would use a full list (e.g. GeoNames);
# here it covers the synthetic data plus a few common names.
CITY_GAZETTEER = {
    "SPRINGFIELD", "RIVERTON", "FAIRVIEW", "MADISON", "GEORGETOWN", "SALEM", "FRANKLIN",
    "CLINTON", "ASHLAND", "BRISTOL", "DOVER", "AUBURN", "SAN JOSE", "NEW HAVEN", "LAKE CITY",
    "ANN ARBOR", "SAN FRANCISCO", "NEW YORK", "LOS ANGELES", "CHICAGO", "SEATTLE", "AUSTIN",
    "BOSTON", "DENVER", "PORTLAND", "OAKLAND", "SAN DIEGO", "SALT LAKE CITY",
}

_PREFIXES: list[tuple[str, re.Pattern]] = [
    ("POS", re.compile(r"^POS DEBIT \d{4} ")),
    ("DEBITCARD", re.compile(r"^DEBIT CARD PURCHASE\s*-\s*")),
    ("SQ", re.compile(r"^SQ \*\s*")),
    ("TST", re.compile(r"^TST\*\s*")),
    ("PAYPAL", re.compile(r"^PAYPAL \*\s*")),
    ("RECURRING", re.compile(r"^RECURRING PMT ")),
    ("ACH", re.compile(r"^ACH CREDIT ")),
]
_NOISE = [
    re.compile(r"\bCARD \d{4}\b"),
    re.compile(r"#\s?\d+"),
    re.compile(r"\bREF\d+\b"),
    re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),
    re.compile(r"\b(PAYROLL|DIR DEP)\b"),
    re.compile(r"\S*\d\S*"),                # any remaining token containing a digit
]
_WEB = re.compile(r"\.COM\b")


@dataclass(frozen=True)
class Normalized:
    processor: str
    name: str
    key: str


def _strip_location(tokens: list[str], processor: str) -> list[str]:
    if len(tokens) < 2 or tokens[-1] not in US_STATES:
        return tokens
    for width in (3, 2, 1):  # "SALT LAKE CITY UT", "SAN JOSE CA", "DOVER DE"
        if len(tokens) > width + 1 and " ".join(tokens[-width - 1:-1]) in CITY_GAZETTEER:
            return tokens[:-width - 1]
    # A bare trailing state is only trusted for web billers ("NETFLIX.COM 866-... CA");
    # elsewhere "GOLDEN & CO" must keep its "CO".
    return tokens[:-1] if processor == "WEB" else tokens


def normalize(descriptor: str) -> Normalized:
    text = re.sub(r"\s+", " ", descriptor.upper()).strip()
    processor = "OTHER"
    for name, pat in _PREFIXES:
        if pat.search(text):
            processor, text = name, pat.sub("", text, count=1)
            break
    if processor == "OTHER" and _WEB.search(text):
        processor = "WEB"
    text = _WEB.sub("", text)
    for pat in _NOISE:
        text = pat.sub(" ", text)
    tokens = _strip_location(text.split(), processor)
    name = re.sub(r"[^A-Z& ]", "", " ".join(tokens))
    name = re.sub(r"\s+", " ", name).strip(" &")
    key = re.sub(r"[^A-Z]", "", name)[:KEY_LEN]
    return Normalized(processor, name, key or "UNKNOWN")
