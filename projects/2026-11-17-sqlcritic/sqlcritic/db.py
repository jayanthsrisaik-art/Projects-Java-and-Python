"""Deterministic synthetic retail warehouse in SQLite, plus the schema doc the writer sees."""
from __future__ import annotations

import datetime as dt
import random
import sqlite3
from pathlib import Path

REGIONS = ["North", "South", "East", "West"]
CATEGORIES = ["electronics", "furniture", "grocery", "apparel", "toys"]
SEGMENTS = ["consumer", "smb", "enterprise"]
STATUSES = ["completed", "cancelled", "returned"]
YEARS = [2023, 2024, 2025]

DDL = """
CREATE TABLE regions (region_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    region_id INTEGER NOT NULL REFERENCES regions(region_id),
    segment TEXT NOT NULL, signup_date TEXT NOT NULL);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL, unit_price REAL NOT NULL);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL, unit_price REAL NOT NULL,
    PRIMARY KEY (order_id, product_id));
"""

# column catalog used by the critic (sqlglot schema format)
CATALOG = {
    "regions": {"region_id": "INT", "name": "TEXT"},
    "customers": {"customer_id": "INT", "name": "TEXT", "region_id": "INT", "segment": "TEXT",
                  "signup_date": "TEXT"},
    "products": {"product_id": "INT", "name": "TEXT", "category": "TEXT", "unit_price": "REAL"},
    "orders": {"order_id": "INT", "customer_id": "INT", "order_date": "TEXT", "status": "TEXT"},
    "order_items": {"order_id": "INT", "product_id": "INT", "quantity": "INT", "unit_price": "REAL"},
}

# child -> parents (many-to-one foreign keys). A join from a parent to a child repeats parent rows.
FOREIGN_KEYS = {
    "customers": {"regions": ("region_id", "region_id")},
    "orders": {"customers": ("customer_id", "customer_id")},
    "order_items": {"orders": ("order_id", "order_id"), "products": ("product_id", "product_id")},
}

SCHEMA_DOC = """Tables (SQLite):
  regions(region_id PK, name)                         -- North, South, East, West
  customers(customer_id PK, name, region_id FK->regions, segment, signup_date)
                                                      -- segment in: consumer, smb, enterprise
  products(product_id PK, name, category, unit_price) -- category in: electronics, furniture,
                                                      --   grocery, apparel, toys
  orders(order_id PK, customer_id FK->customers, order_date 'YYYY-MM-DD', status)
                                                      -- status in: completed, cancelled, returned
  order_items(order_id FK->orders, product_id FK->products, quantity, unit_price)
                                                      -- one row per product per order

Business definitions:
  revenue            = SUM(order_items.quantity * order_items.unit_price) over COMPLETED orders only
  units sold         = SUM(order_items.quantity) over COMPLETED orders only
  average order value = revenue / number of distinct completed orders
  a purchase / "bought" means a COMPLETED order
  order counts, status breakdowns and cancellation shares use ALL orders regardless of status
  year filters use strftime('%Y', orders.order_date)
"""


def _children() -> dict[str, set[str]]:
    kids: dict[str, set[str]] = {t: set() for t in CATALOG}
    for child, parents in FOREIGN_KEYS.items():
        for parent in parents:
            kids[parent].add(child)
    return kids


def descendants(table: str) -> set[str]:
    """Tables whose join to ``table`` repeats ``table``'s rows (transitive one-to-many)."""
    kids = _children()
    out: set[str] = set()
    stack = [table]
    while stack:
        for c in kids.get(stack.pop(), ()):   # CTEs / unknown names: no FK children
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


_FIRST = ["Ava", "Ben", "Chloe", "Dev", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jamal", "Kira",
          "Liam", "Maya", "Noah", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara", "Uma", "Victor",
          "Wen", "Ximena", "Yusuf", "Zoe"]
_LAST = ["Abara", "Berg", "Costa", "Dubois", "Eze", "Fischer", "Gupta", "Haddad", "Ito", "Jensen",
         "Kowalski", "Lopez", "Moreau", "Nakamura", "Okafor", "Petrov", "Quispe", "Rossi", "Silva",
         "Tanaka", "Usman", "Varga", "Wright", "Xu", "Yilmaz", "Zhou"]
_NOUNS = {
    "electronics": ["Headphones", "Monitor", "Router", "Tablet", "Speaker", "Webcam", "Keyboard",
                    "Charger", "Drone", "Projector", "Smartwatch", "SSD"],
    "furniture": ["Desk", "Chair", "Bookshelf", "Sofa", "Lamp", "Cabinet", "Stool", "Bench",
                  "Wardrobe", "Nightstand", "Ottoman", "Shelf"],
    "grocery": ["Coffee", "Olive Oil", "Rice", "Tea", "Pasta", "Honey", "Granola", "Salsa",
                "Almonds", "Chocolate", "Vinegar", "Oats"],
    "apparel": ["Jacket", "Sneakers", "Scarf", "Hoodie", "Jeans", "Boots", "Cap", "Gloves",
                "Raincoat", "Sweater", "Belt", "Socks"],
    "toys": ["Puzzle", "Robot Kit", "Kite", "Blocks", "Board Game", "Yo-yo", "Plush Bear",
             "Train Set", "Marbles", "Slime Kit", "Rocket", "Dollhouse"],
}
_ADJ = ["Classic", "Pro", "Eco", "Urban", "Nordic", "Compact", "Deluxe", "Basic"]
_PRICE = {"electronics": (30, 900), "furniture": (40, 700), "grocery": (2, 40),
          "apparel": (10, 150), "toys": (5, 120)}


def build_db(path: str | Path, seed: int = 7, n_customers: int = 400, n_orders: int = 4000) -> Path:
    path = Path(path)
    if path.exists():
        path.unlink()
    rng = random.Random(seed)
    con = sqlite3.connect(path)
    con.executescript(DDL)
    con.executemany("INSERT INTO regions VALUES (?, ?)", list(enumerate(REGIONS, start=1)))

    names: set[str] = set()
    customers = []
    for cid in range(1, n_customers + 1):
        while True:
            name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
            if name not in names:
                break
            name = f"{name} {cid}"
            if name not in names:
                break
        names.add(name)
        region = rng.choices(range(1, 5), weights=[4, 3, 2, 3])[0]
        segment = rng.choices(SEGMENTS, weights=[6, 3, 1])[0]
        signup = dt.date(2022, 1, 1) + dt.timedelta(days=rng.randrange(0, 1300))
        customers.append((cid, name, region, segment, signup.isoformat()))
    con.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

    products = []
    pid = 1
    for cat in CATEGORIES:
        lo, hi = _PRICE[cat]
        for noun in _NOUNS[cat]:
            products.append((pid, f"{rng.choice(_ADJ)} {noun}", cat, round(rng.uniform(lo, hi), 2)))
            pid += 1
    con.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

    active = [c[0] for c in customers if rng.random() < 0.85]   # ~15% never order
    start = dt.date(2023, 1, 1)
    span = (dt.date(2025, 12, 31) - start).days + 1
    orders, items = [], []
    for oid in range(1, n_orders + 1):
        day = start + dt.timedelta(days=rng.randrange(span))
        status = rng.choices(STATUSES, weights=[85, 10, 5])[0]
        orders.append((oid, rng.choice(active), day.isoformat(), status))
        for prod in rng.sample(products, rng.randint(1, 4)):
            discount = rng.choice([0, 0, 0, 0.1])
            items.append((oid, prod[0], rng.randint(1, 5), round(prod[3] * (1 - discount), 2)))
    con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
    con.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?)", items)
    con.commit()
    con.close()
    return path


def connect_readonly(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    return con
