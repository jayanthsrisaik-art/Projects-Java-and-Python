"""Synthetic product catalogue, query set and hidden ground-truth relevance.

The observable catalogue (titles, brand, category, attributes, price, rating, view counts) is
what a real search team would have. ``quality`` and the relevance grades are hidden: they drive
the simulated users and are used *only* for offline evaluation, never as model features.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .text import normalize_query, tokenize

BRANDS = ["aurora", "kestrel", "nimbus", "volta", "cobalt", "maple",
          "orbit", "summit", "tandem", "zephyr", "quill", "ember"]

# category -> (noun used in titles and queries, attribute vocabulary, price range)
CATEGORIES: dict[str, tuple[str, list[str], tuple[float, float]]] = {
    "headphones": ("headphones", ["wireless", "noise", "earbuds", "bluetooth", "sport", "foldable"], (20, 400)),
    "laptop": ("laptop", ["gaming", "ultralight", "touchscreen", "business", "compact"], (300, 2500)),
    "shoes": ("shoes", ["running", "waterproof", "trail", "leather", "lightweight"], (40, 250)),
    "coffee": ("coffeemaker", ["espresso", "programmable", "compact", "stainless", "grinder"], (25, 900)),
    "backpack": ("backpack", ["waterproof", "laptop", "hiking", "leather", "rolling"], (25, 300)),
    "lamp": ("lamp", ["led", "dimmable", "rechargeable", "clamp", "smart"], (15, 200)),
    "keyboard": ("keyboard", ["mechanical", "wireless", "ergonomic", "gaming", "compact"], (20, 300)),
    "tent": ("tent", ["ultralight", "waterproof", "family", "instant", "insulated"], (60, 900)),
    "bottle": ("bottle", ["insulated", "stainless", "collapsible", "filter", "sport"], (8, 60)),
    "case": ("case", ["leather", "waterproof", "rugged", "slim", "wallet"], (8, 80)),
}
FILLER = ["pro", "max", "classic", "edition", "series", "plus", "lite", "x", "one", "air"]


@dataclass
class Product:
    id: str
    title: str
    category: str          # "accessory" for accessories that merely mention another category
    brand: str
    attrs: list[str]
    price: float
    rating: float
    views: int


@dataclass
class Query:
    text: str              # normalised
    category: str
    brand: str | None
    attrs: list[str] = field(default_factory=list)
    weight: float = 1.0    # relative traffic share


@dataclass
class Catalog:
    products: list[Product]
    queries: list[Query]
    quality: dict[str, float]           # hidden
    grades: dict[str, dict[str, int]]   # hidden: query text -> product id -> grade (0..3)

    def product_map(self) -> dict[str, Product]:
        return {p.id: p for p in self.products}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "catalog.json").write_text(json.dumps(
            {"products": [asdict(p) for p in self.products], "vocabulary": vocabulary()}, indent=1))
        (directory / "truth.json").write_text(json.dumps({
            "queries": [asdict(q) for q in self.queries],
            "quality": self.quality,
            "grades": self.grades,
        }))


def vocabulary() -> dict:
    """Observable taxonomy: category nouns, brand names, attribute words."""
    return {
        "nouns": {cat: noun for cat, (noun, _, _) in CATEGORIES.items()},
        "brands": BRANDS,
        "attrs": sorted({a for _, attrs, _ in CATEGORIES.values() for a in attrs}),
    }


def load_products(directory: Path) -> tuple[list[Product], dict]:
    data = json.loads((directory / "catalog.json").read_text())
    return [Product(**p) for p in data["products"]], data["vocabulary"]


def load_truth(directory: Path) -> dict:
    return json.loads((directory / "truth.json").read_text())


def relevance(query: Query, product: Product, quality: float) -> int:
    """Hidden graded relevance. Category is a hard gate; attributes, brand and quality refine."""
    if product.category != query.category:
        return 0
    s = 1.0 + 0.7 * quality
    if query.attrs:
        s += 1.2 * sum(a in product.attrs for a in query.attrs) / len(query.attrs) - 0.3
    if query.brand:
        s += 0.6 if product.brand == query.brand else -1.0
    return int(min(3, max(0, round(s))))


def build_catalog(seed: int = 0, per_category: int = 55, n_accessories: int = 60,
                  n_queries: int = 400) -> Catalog:
    rng = random.Random(seed)
    hype = {b: rng.gauss(0, 1) for b in BRANDS}   # brand marketing spend: drives views, not quality
    products: list[Product] = []
    quality: dict[str, float] = {}

    def add(category: str, brand: str, attrs: list[str], title: str, lo: float, hi: float) -> None:
        pid = f"P{len(products) + 1:04d}"
        q = rng.gauss(0, 1)
        price = round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 2)
        rating = round(min(5.0, max(1.0, 3.6 + 0.5 * q + rng.gauss(0, 0.35))), 1)
        log_views = 6.0 + 1.2 * hype[brand] + 0.9 * rng.gauss(0, 1) + 0.1 * q
        products.append(Product(pid, title, category, brand, sorted(attrs), price, rating,
                                int(math.exp(log_views))))
        quality[pid] = q

    for cat, (noun, attr_vocab, (lo, hi)) in CATEGORIES.items():
        brands = rng.sample(BRANDS, 5)
        for _ in range(per_category):
            brand = rng.choice(brands)
            attrs = rng.sample(attr_vocab, rng.choice([1, 2, 2, 3]))
            words = [brand, *attrs, noun] + ([rng.choice(FILLER)] if rng.random() < 0.5 else [])
            add(cat, brand, attrs, " ".join(words), lo, hi)

    # Accessories mention another category's noun (and maybe its attributes): lexical traps.
    for _ in range(n_accessories):
        target = rng.choice(list(CATEGORIES))
        noun, attr_vocab, _ = CATEGORIES[target]
        attrs = rng.sample(attr_vocab, 1)
        kind = rng.choice(["replacement cushions for", "carrying pouch for", "cleaning kit for",
                           "stand for", "spare parts for", "sticker set for"])
        brand = rng.choice(BRANDS)
        add("accessory", brand, attrs, f"{brand} {kind} {attrs[0]} {noun}", 5, 40)

    queries: dict[str, Query] = {}
    while len(queries) < n_queries:
        cat = rng.choice(list(CATEGORIES))
        noun, attr_vocab, _ = CATEGORIES[cat]
        cat_brands = sorted({p.brand for p in products if p.category == cat})
        brand = rng.choice(cat_brands) if rng.random() < 0.35 else None
        r = rng.random()
        attrs = rng.sample(attr_vocab, 2) if r < 0.15 else rng.sample(attr_vocab, 1) if r < 0.75 else []
        text = normalize_query(" ".join(([brand] if brand else []) + attrs + [noun]))
        if text not in queries:
            queries[text] = Query(text, cat, brand, attrs)
    qlist = list(queries.values())
    order = list(range(len(qlist)))
    rng.shuffle(order)
    for rank, i in enumerate(order, start=1):         # Zipf-like traffic
        qlist[i].weight = 1.0 / rank ** 0.8

    grades = {q.text: {p.id: relevance(q, p, quality[p.id]) for p in products} for q in qlist}
    return Catalog(products, qlist, quality, grades)


def product_tokens(p: Product) -> list[str]:
    return tokenize(p.title)
