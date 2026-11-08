import json
import random
from collections import defaultdict
from datetime import date, timedelta

import pytest

from conftest import ROOT
from ledgerlens.normalize import normalize
from ledgerlens.recurring import detect
from ledgerlens.synth import make_merchants, render_descriptor

CASES = json.loads((ROOT / "samples" / "normalization_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["descriptor"][:30] for c in CASES])
def test_normalization_cases(case):
    n = normalize(case["descriptor"])
    assert (n.processor, n.name, n.key) == (case["processor"], case["name"], case["key"])


def test_state_code_inside_name_is_kept_without_city():
    # "& CO" must not be stripped as Colorado unless a known city precedes it
    assert normalize("TST* LITTLE & CO 1911").key == "LITTLECO"
    assert normalize("POS DEBIT 0101 LITTLE & CO #12 DOVER DE").key == "LITTLECO"


def test_every_rendering_of_a_merchant_maps_to_one_key_and_keys_do_not_collide():
    rng = random.Random(3)
    merchants = make_merchants(rng, per_category=20)
    keys_by_m = defaultdict(set)
    for m in merchants:
        for _ in range(30):  # all processors, random store numbers / cards / dates
            keys_by_m[m.merchant_id].add(normalize(render_descriptor(m, date(2026, 3, 4), rng)).key)
    assert all(len(v) == 1 for v in keys_by_m.values())
    all_keys = [next(iter(v)) for v in keys_by_m.values()]
    assert len(set(all_keys)) == len(all_keys)


def _series(start, gaps, amounts, key="K", acct="a1"):
    d, out = start, []
    for i, amt in enumerate(amounts):
        out.append({"account_id": acct, "posted_date": d.isoformat(), "amount": amt, "merchant_key": key})
        if i < len(gaps):
            d += timedelta(days=gaps[i])
    return out


def test_monthly_subscription_detected_with_next_date():
    rows = _series(date(2026, 1, 15), [31, 28, 31, 30], [-9.99] * 5)
    [r] = detect(rows)
    assert (r.cadence, r.occurrences, r.typical_amount) == ("monthly", 5, -9.99)
    assert r.next_expected == (date(2026, 5, 15) + timedelta(days=30)).isoformat()


def test_price_increase_and_one_missed_month_still_recurring():
    rows = _series(date(2026, 1, 3), [31, 59, 30, 31], [-9.99, -9.99, -10.99, -10.99, -10.99])
    assert [r.cadence for r in detect(rows)] == ["monthly"]


def test_weekly_same_day_with_variable_amount_is_not_recurring():
    rows = _series(date(2026, 1, 3), [7] * 9, [-42.1, -88.0, -61.5, -120.3, -35.0, -70.2, -99.9, -51.0, -64.4, -80.0])
    assert detect(rows) == []


def test_biweekly_income_detected_and_signs_kept_apart():
    pay = _series(date(2026, 1, 2), [14] * 5, [2500.0] * 6, key="ACMECORP")
    fee = _series(date(2026, 1, 9), [14] * 5, [-2500.0] * 6, key="ACMECORP")  # same key, debits
    found = detect(pay + fee)
    assert [(r.cadence, r.typical_amount) for r in found] in (
        [("biweekly", 2500.0), ("biweekly", -2500.0)], [("biweekly", -2500.0), ("biweekly", 2500.0)])
    # the opposite-sign series are separate: mixing them would make the amounts inconsistent
    assert len(found) == 2


def test_duplicate_delivery_does_not_create_zero_day_gaps():
    rows = _series(date(2026, 1, 1), [30, 31, 28], [-15.0] * 4)
    assert len(detect(rows + rows)) == 1


def test_too_few_or_irregular_charges_are_not_recurring():
    assert detect(_series(date(2026, 1, 1), [30], [-5.0, -5.0])) == []
    assert detect(_series(date(2026, 1, 1), [3, 40, 11, 70], [-5.0] * 5)) == []
