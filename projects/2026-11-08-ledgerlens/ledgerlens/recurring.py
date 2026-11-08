"""Recurring-charge (subscription / bill / payroll) detection.

A series is one (account, merchant_key, sign). It is recurring when
1. it has at least ``min_occurrences`` transactions,
2. the median gap between consecutive charges matches a cadence and at least 75% of
   gaps fall inside that cadence's tolerance band, and
3. at least 80% of amounts lie within ``amount_tol`` (relative) of the median amount.

Rule 3 is what rejects "groceries every Saturday": the timing is perfectly regular
but the basket total is not.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from statistics import median

CADENCES = {
    "weekly": (7, 6, 8),
    "biweekly": (14, 12, 16),
    "monthly": (30, 25, 35),
    "yearly": (365, 350, 380),
}


@dataclass
class Recurring:
    account_id: str
    merchant_key: str
    cadence: str
    occurrences: int
    typical_amount: float
    last_date: str
    next_expected: str


def _cadence(gaps: list[int]) -> str | None:
    med = median(gaps)
    for name, (_, lo, hi) in CADENCES.items():
        if lo <= med <= hi:
            inside = sum(lo <= g <= hi for g in gaps)
            return name if inside >= 0.75 * len(gaps) else None
    return None


def detect(rows, min_occurrences: int = 3, amount_tol: float = 0.2,
           key_fn=lambda r: r["merchant_key"]) -> list[Recurring]:
    """``rows``: iterable of dicts with account_id, posted_date, amount and a merchant key."""
    series: dict[tuple, list[tuple[date, float]]] = defaultdict(list)
    for r in rows:
        sign = 1 if r["amount"] > 0 else -1
        series[(r["account_id"], key_fn(r), sign)].append((date.fromisoformat(r["posted_date"]), r["amount"]))
    found: list[Recurring] = []
    for (acct, key, _), items in series.items():
        # one charge per day per series (duplicate deliveries must not create 0-day gaps)
        by_day = sorted({d: a for d, a in items}.items())
        if len(by_day) < min_occurrences:
            continue
        dates = [d for d, _ in by_day]
        amounts = [a for _, a in by_day]
        cad = _cadence([(b - a).days for a, b in zip(dates, dates[1:])])
        if cad is None:
            continue
        med_amt = median(amounts)
        close = sum(abs(a - med_amt) <= max(amount_tol * abs(med_amt), 0.01) for a in amounts)
        if close < 0.8 * len(amounts):
            continue
        found.append(Recurring(acct, key, cad, len(dates), round(med_amt, 2), dates[-1].isoformat(),
                               (dates[-1] + timedelta(days=CADENCES[cad][0])).isoformat()))
    found.sort(key=lambda r: (r.account_id, r.merchant_key))
    return found


def as_dicts(items: list[Recurring]) -> list[dict]:
    return [asdict(i) for i in items]
