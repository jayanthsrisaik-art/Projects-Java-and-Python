"""Synthetic bank-transaction generator.

Produces raw card/ACH rows that look like real bank exports: processor prefixes
(``SQ *``, ``TST*``, ``PAYPAL *``), store numbers, card last-4 digits, phone numbers,
city/state suffixes and truncated merchant names. Every row carries the hidden
ground truth (true merchant id, category, and whether the account/merchant series is
a genuine recurring charge) so normalisation, classification and recurring detection
can all be measured.

Deliberately awkward properties:
* category vocabulary overlaps ("GOLDEN", "URBAN", "GAS & ELECTRIC" vs "GAS"),
* ~30% of merchants are invented brand words with no category word at all,
* weekly grocery runs at a fixed weekday with *variable* amounts (not recurring),
* exact duplicate deliveries of the same txn_id,
* a small rate of label noise (users mis-tag things).
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

CATEGORIES = [
    "groceries", "dining", "fuel", "transport", "utilities",
    "subscriptions", "shopping", "health", "travel", "income",
]

_WORDS = {
    "groceries": (["Green", "Fresh", "Valley", "Sunny", "Corner", "Harvest", "Golden", "Family",
                   "Daily", "Oak", "Urban", "Hilltop"],
                  ["Market", "Grocers", "Foods", "Supermarket", "Food Mart", "Produce", "Pantry",
                   "Express", "& Co"]),
    "dining": (["Blue", "Red", "Lucky", "Little", "Happy", "Urban", "Old Town", "Golden", "Smoky",
                "Rustic", "Corner", "Sunny"],
               ["Cafe", "Grill", "Pizzeria", "Taqueria", "Bistro", "Diner", "Noodle House",
                "Burger Bar", "Sushi", "Bakery", "Coffee", "Kitchen", "& Co"]),
    "fuel": (["Speedy", "Star", "Eagle", "Prairie", "Coastal", "Metro", "Pioneer", "Summit", "Valley"],
             ["Fuel", "Gas", "Petro", "Gas & Go", "Fuel Stop", "Energy", "Service Station", "Express"]),
    "transport": (["City", "Metro", "Swift", "Zip", "Urban", "Rapid", "Star"],
                  ["Rides", "Transit", "Cab", "Taxi", "Parking", "Rail", "Scooters", "Shuttle"]),
    "utilities": (["Northern", "Pacific", "Tri-County", "Riverside", "Lakeshore", "Mountain", "Valley"],
                  ["Power", "Electric", "Water District", "Gas & Electric", "Broadband", "Telecom",
                   "Wireless", "Energy"]),
    "subscriptions": (["Stream", "Cloud", "Tune", "Pixel", "Byte", "Nova", "Echo", "Star"],
                      ["Flix", "Box", "Music", "Plus", "Stream", "TV", "Drive", "Pro"]),
    "shopping": (["Home", "Urban", "Style", "Mega", "Bright", "Main Street", "Outlet", "Golden"],
                 ["Outfitters", "Depot", "Emporium", "Goods", "Store", "Mart", "Supply", "Boutique",
                  "& Co", "Plus"]),
    "health": (["Care", "Well", "Family", "Lakeside", "Cedar", "Vital", "Sunny"],
               ["Pharmacy", "Drugs", "Dental", "Clinic", "Medical", "Optometry", "Health", "Plus"]),
    "travel": (["Sky", "Jet", "Grand", "Coastal", "Harbor", "Royal", "Summit"],
               ["Airlines", "Air", "Hotel", "Inn", "Resorts", "Suites", "Lodge", "Car Rental"]),
    "income": (["Acme", "Globex", "Initech", "Northwind", "Contoso", "Umbrella", "Vandelay",
                "Hooli", "Soylent", "Wonka"],
               ["Corp", "LLC", "Inc", "Industries", "Labs", "Group"]),
}

_SYLLABLES = ["ka", "ze", "lo", "vi", "ran", "tor", "mi", "qua", "bel", "zen", "dro", "ni",
              "sa", "vo", "lex", "ta", "rio", "fi", "mon", "gu"]

CITIES = [("SPRINGFIELD", "IL"), ("RIVERTON", "WY"), ("FAIRVIEW", "TX"), ("MADISON", "WI"),
          ("GEORGETOWN", "KY"), ("SALEM", "OR"), ("FRANKLIN", "TN"), ("CLINTON", "IA"),
          ("ASHLAND", "OH"), ("BRISTOL", "CT"), ("DOVER", "DE"), ("AUBURN", "AL"),
          ("SAN JOSE", "CA"), ("NEW HAVEN", "CT"), ("LAKE CITY", "FL"), ("ANN ARBOR", "MI")]

# processors allowed per category (the format is a real signal, but overlaps)
_PROCESSORS = {
    "groceries": ["POS", "DEBITCARD"], "dining": ["SQ", "TST", "POS"],
    "fuel": ["POS", "DEBITCARD"], "transport": ["SQ", "WEB", "POS"],
    "utilities": ["RECURRING", "WEB"], "subscriptions": ["PAYPAL", "WEB"],
    "shopping": ["POS", "DEBITCARD", "PAYPAL", "WEB"], "health": ["POS", "DEBITCARD"],
    "travel": ["WEB", "DEBITCARD"], "income": ["ACH"],
}

NAME_LIMIT = 20  # banks truncate the merchant field


@dataclass
class Merchant:
    merchant_id: str
    name: str
    category: str
    processors: list[str]
    city: tuple[str, str]
    phone: str
    brand_only: bool


@dataclass
class Txn:
    txn_id: str
    account_id: str
    posted_date: str
    descriptor: str
    amount: float
    category: str          # label as the user tagged it (may be noisy)
    true_category: str
    merchant_id: str
    recurring: bool        # ground truth: is this account/merchant series a real recurring charge
    extra: dict = field(default_factory=dict)


def _brand(rng: random.Random) -> str:
    return "".join(rng.choice(_SYLLABLES) for _ in range(rng.choice([2, 3]))).capitalize()


def make_merchants(rng: random.Random, per_category: int = 24, brand_rate: float = 0.3) -> list[Merchant]:
    merchants: list[Merchant] = []
    used: set[str] = set()
    for cat in CATEGORIES:
        prefixes, suffixes = _WORDS[cat]
        n = 0
        while n < per_category:
            brand_only = cat != "income" and rng.random() < brand_rate
            if brand_only:
                name = _brand(rng) if rng.random() < 0.6 else f"{_brand(rng)} {_brand(rng)}"
            else:
                name = f"{rng.choice(prefixes)} {rng.choice(suffixes)}"
            key = "".join(ch for ch in name.upper() if ch.isalpha())[:16]
            if key in used:
                continue
            used.add(key)
            procs = rng.sample(_PROCESSORS[cat], k=min(len(_PROCESSORS[cat]), rng.choice([1, 2])))
            phone = f"{rng.randint(800, 888)}-555-{rng.randint(0, 9999):04d}"
            merchants.append(Merchant(f"m{len(merchants):04d}", name, cat, procs,
                                      rng.choice(CITIES), phone, brand_only))
            n += 1
    return merchants


def render_descriptor(m: Merchant, d: date, rng: random.Random) -> str:
    """Render one raw bank descriptor for merchant ``m`` on day ``d``."""
    proc = rng.choice(m.processors)
    upper = m.name.upper()
    trunc = upper[:NAME_LIMIT].rstrip()
    squashed = "".join(ch for ch in upper if ch.isalnum())[:NAME_LIMIT]
    city, st = m.city
    store = rng.randint(1, 9999)
    if proc == "POS":
        return f"POS DEBIT {d:%m%d} {trunc} #{store:04d} {city} {st}"
    if proc == "DEBITCARD":
        return f"DEBIT CARD PURCHASE - {trunc} {city} {st} CARD {rng.randint(1000, 9999)}"
    if proc == "SQ":
        return f"SQ *{trunc} {city} {st}"
    if proc == "TST":
        return f"TST* {trunc} {store}"
    if proc == "PAYPAL":
        return f"PAYPAL *{squashed} {rng.randint(10**9, 10**10 - 1)}"
    if proc == "WEB":
        return f"{squashed}.COM {m.phone} {st}"
    if proc == "RECURRING":
        return f"RECURRING PMT {trunc} REF{rng.randint(100000, 999999)}"
    if proc == "ACH":
        kind = rng.choice(["PAYROLL", "DIR DEP"])
        return f"ACH CREDIT {trunc} {kind} {rng.randint(10**7, 10**8 - 1)}"
    raise ValueError(proc)


def _money(x: float) -> float:
    return round(x + 0.0, 2)


def generate(seed: int = 7, n_accounts: int = 60, days: int = 182, per_category: int = 24,
             start: date = date(2026, 1, 1), label_noise: float = 0.015,
             dup_rate: float = 0.01) -> tuple[list[Txn], list[Merchant]]:
    rng = random.Random(seed)
    merchants = make_merchants(rng, per_category)
    by_cat: dict[str, list[Merchant]] = {c: [m for m in merchants if m.category == c] for c in CATEGORIES}
    end = start + timedelta(days=days - 1)
    rows: list[Txn] = []

    def add(acct: str, m: Merchant, d: date, amount: float, recurring: bool) -> None:
        if d < start or d > end:
            return
        rows.append(Txn("", acct, d.isoformat(), render_descriptor(m, d, rng), _money(amount),
                        m.category, m.category, m.merchant_id, recurring))

    def random_visits(acct: str, m: Merchant, mean_gap: float, amount_fn) -> None:
        d = start + timedelta(days=rng.randint(0, int(mean_gap)))
        while d <= end:
            add(acct, m, d, amount_fn(), False)
            d += timedelta(days=max(1, round(rng.expovariate(1 / mean_gap))))

    for a in range(n_accounts):
        acct = f"acct{a:03d}"
        # groceries: 1-2 random stores, sometimes one strict weekly run with variable totals
        for m in rng.sample(by_cat["groceries"], rng.randint(1, 2)):
            random_visits(acct, m, 6, lambda: -rng.lognormvariate(4.0, 0.5))
        if rng.random() < 0.4:
            m = rng.choice(by_cat["groceries"])
            d = start + timedelta(days=rng.randint(0, 6))
            while d <= end:  # same weekday every week, but amounts vary a lot -> NOT recurring
                add(acct, m, d, -rng.lognormvariate(4.2, 0.45), False)
                d += timedelta(days=7)
        for m in rng.sample(by_cat["dining"], rng.randint(3, 6)):
            random_visits(acct, m, 14, lambda: -rng.uniform(6, 80))
        for m in rng.sample(by_cat["fuel"], rng.randint(1, 2)):
            random_visits(acct, m, 10, lambda: -rng.uniform(25, 75))
        for m in rng.sample(by_cat["transport"], rng.randint(1, 2)):
            random_visits(acct, m, 9, lambda: -rng.uniform(3, 40))
        for m in rng.sample(by_cat["shopping"], rng.randint(2, 4)):
            random_visits(acct, m, 25, lambda: -rng.lognormvariate(3.8, 0.8))
        for m in rng.sample(by_cat["health"], rng.randint(1, 2)):
            random_visits(acct, m, 40, lambda: -rng.uniform(5, 150))
        for m in rng.sample(by_cat["travel"], rng.randint(0, 2)):
            for _ in range(rng.randint(1, 2)):
                add(acct, m, start + timedelta(days=rng.randint(0, days - 1)), -rng.uniform(80, 900), False)
        # utilities: monthly, variable amount (+-12%), fixed day of month +-2 -> recurring
        for m in rng.sample(by_cat["utilities"], 2):
            base, dom = rng.uniform(40, 200), rng.randint(1, 26)
            skipped = rng.randint(2, 6) if rng.random() < 0.15 else None  # one missed bill
            for mo in range(1, 13):
                if mo == skipped:
                    continue
                d = date(start.year, mo, dom) + timedelta(days=rng.randint(-2, 2))
                add(acct, m, d, -base * rng.uniform(0.88, 1.12), True)
        # subscriptions: monthly fixed price -> recurring
        for m in rng.sample(by_cat["subscriptions"], rng.randint(1, 4)):
            price = rng.choice([4.99, 7.99, 9.99, 11.99, 14.99, 15.49, 19.99])
            dom = rng.randint(1, 28)
            hike = rng.randint(3, 5) if rng.random() < 0.25 else 13  # price increase mid-series
            for mo in range(1, 13):
                add(acct, m, date(start.year, mo, dom), -(price + (1.0 if mo >= hike else 0.0)), True)
        # income: biweekly fixed payroll -> recurring
        emp = rng.choice(by_cat["income"])
        pay, d = round(rng.uniform(1400, 4200), 2), start + timedelta(days=rng.randint(0, 13))
        while d <= end:
            add(acct, emp, d, pay, True)
            d += timedelta(days=14)

    rows.sort(key=lambda t: (t.posted_date, t.account_id, t.descriptor))
    for i, t in enumerate(rows):
        t.txn_id = f"t{i:06d}"
        if rng.random() < label_noise:
            t.category = rng.choice([c for c in CATEGORIES if c != t.true_category])
    dups = [Txn(**{**t.__dict__, "extra": {"duplicate": True}}) for t in rows if rng.random() < dup_rate]
    rows.extend(dups)
    rows.sort(key=lambda t: (t.posted_date, t.txn_id))
    return rows, merchants


RAW_FIELDS = ["txn_id", "account_id", "posted_date", "descriptor", "amount"]
LABEL_FIELDS = ["category", "true_category", "merchant_id", "recurring"]


def write_csv(rows: list[Txn], path: str | Path, with_labels: bool = True) -> None:
    fields = RAW_FIELDS + (LABEL_FIELDS if with_labels else [])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in rows:
            w.writerow({k: getattr(t, k) for k in fields})


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as fh:
        out = []
        for r in csv.DictReader(fh):
            r["amount"] = float(r["amount"])
            if "recurring" in r:
                r["recurring"] = r["recurring"] == "True"
            out.append(r)
        return out
