"""Golden set: questions with gold SQL and the true result contract.

Each family has four wordings. Wordings 0-1 form the **dev** split (used while building the
heuristic writer and the critic's contract rules); wordings 2-3 form the **test** split and
are only scored. Test therefore measures generalisation to unseen phrasing of known intents,
not to new intents: all 14 families appear in both splits.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .contract import Contract
from .db import CATEGORIES, REGIONS, SEGMENTS, YEARS

REV = "SUM(oi.quantity * oi.unit_price)"
YEAR = "strftime('%Y', o.order_date) = '{y}'"


@dataclass(frozen=True)
class Example:
    id: str
    family: str
    split: str
    question: str
    gold_sql: str
    contract: Contract

    def as_dict(self) -> dict:
        d = asdict(self)
        d["contract"] = self.contract.as_dict()
        return d


# family -> (slot names, wordings[4], gold SQL template, contract factory)
FAMILIES: dict[str, tuple] = {
    "orders_in_year": (
        ["y"],
        ["How many orders were placed in {y}?",
         "Count the orders from {y}.",
         "What was the total number of orders in {y}?",
         "Number of orders we received during {y}?"],
        f"SELECT COUNT(*) FROM orders o WHERE {YEAR}",
        lambda s: Contract("scalar", count_entity="orders")),
    "customers_in_region": (
        ["r"],
        ["How many customers are in the {r} region?",
         "Count customers located in {r}.",
         "What is the number of customers based in the {r} region?",
         "How many {r} customers do we have?"],
        "SELECT COUNT(*) FROM customers c JOIN regions r ON r.region_id = c.region_id "
        "WHERE r.name = '{r}'",
        lambda s: Contract("scalar", count_entity="customers")),
    "revenue_category_year": (
        ["c", "y"],
        ["What was total revenue from {c} in {y}?",
         "How much revenue did {c} products generate in {y}?",
         "Total {c} sales in {y}?",
         "What were {c} sales for {y}?"],
        f"SELECT {REV} FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
        f"JOIN products p ON p.product_id = oi.product_id "
        f"WHERE o.status = 'completed' AND p.category = '{{c}}' AND {YEAR}",
        lambda s: Contract("scalar")),
    "revenue_by_region_year": (
        ["y"],
        ["Show revenue by region for {y}.",
         "What was revenue per region in {y}?",
         "Break down {y} sales across regions.",
         "How did each region do on revenue in {y}?"],
        f"SELECT r.name, {REV} FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
        f"JOIN customers c ON c.customer_id = o.customer_id "
        f"JOIN regions r ON r.region_id = c.region_id "
        f"WHERE o.status = 'completed' AND {YEAR} GROUP BY r.name",
        lambda s: Contract("grouped")),
    "top_products_revenue_year": (
        ["k", "y"],
        ["What are the top {k} products by revenue in {y}?",
         "Top {k} products by revenue for {y}.",
         "Which {k} products brought in the most sales in {y}?",
         "Name the {k} best-selling products by revenue in {y}."],
        f"SELECT p.name, {REV} AS revenue FROM order_items oi "
        f"JOIN orders o ON o.order_id = oi.order_id JOIN products p ON p.product_id = oi.product_id "
        f"WHERE o.status = 'completed' AND {YEAR} GROUP BY p.product_id, p.name "
        f"ORDER BY revenue DESC LIMIT {{k}}",
        lambda s: Contract("topk", k=s["k"])),
    "top_customers_orders": (
        ["k"],
        ["Which {k} customers placed the most orders?",
         "Top {k} customers by number of orders.",
         "Who are our {k} most frequent buyers by order count?",
         "List the {k} customers with the highest order counts."],
        "SELECT c.name, COUNT(*) AS n_orders FROM orders o "
        "JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.name "
        "ORDER BY n_orders DESC LIMIT {k}",
        lambda s: Contract("topk", k=s["k"])),
    "aov_segment": (
        ["s"],
        ["What is the average order value for {s} customers?",
         "Average order value among {s} customers?",
         "How much does a typical {s} customer order cost on average?",
         "What's the mean order size in dollars for the {s} segment?"],
        f"SELECT {REV} * 1.0 / COUNT(DISTINCT o.order_id) FROM order_items oi "
        f"JOIN orders o ON o.order_id = oi.order_id JOIN customers c ON c.customer_id = o.customer_id "
        f"WHERE o.status = 'completed' AND c.segment = '{{s}}'",
        lambda s: Contract("scalar")),
    "buyers_of_category": (
        ["c"],
        ["How many distinct customers bought {c} products?",
         "How many unique customers purchased {c}?",
         "How many shoppers have bought something from {c}?",
         "Number of buyers of {c} items?"],
        "SELECT COUNT(DISTINCT o.customer_id) FROM orders o "
        "JOIN order_items oi ON oi.order_id = o.order_id "
        "JOIN products p ON p.product_id = oi.product_id "
        "WHERE o.status = 'completed' AND p.category = '{c}'",
        lambda s: Contract("scalar", count_entity="customers")),
    "monthly_orders_year": (
        ["y"],
        ["Show monthly order counts for {y}.",
         "How many orders per month in {y}?",
         "Give me the month-by-month number of orders in {y}.",
         "Orders each month during {y}?"],
        f"SELECT strftime('%m', o.order_date) AS month, COUNT(*) FROM orders o WHERE {YEAR} "
        f"GROUP BY month ORDER BY month",
        lambda s: Contract("series")),
    "cancel_share_region": (
        ["r"],
        ["What share of orders in {r} were cancelled?",
         "What fraction of {r} orders got cancelled?",
         "What is the cancellation rate for the {r} region?",
         "What proportion of orders from {r} customers were canceled?"],
        "SELECT AVG(CASE WHEN o.status = 'cancelled' THEN 1.0 ELSE 0.0 END) FROM orders o "
        "JOIN customers c ON c.customer_id = o.customer_id "
        "JOIN regions r ON r.region_id = c.region_id WHERE r.name = '{r}'",
        lambda s: Contract("scalar")),
    "customers_without_orders": (
        ["s"],
        ["How many {s} customers have never placed an order?",
         "Count {s} customers with no orders.",
         "How many {s} accounts haven't ordered anything yet?",
         "Number of {s} customers without a single order?"],
        "SELECT COUNT(*) FROM customers c WHERE c.segment = '{s}' AND NOT EXISTS "
        "(SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id)",
        lambda s: Contract("scalar", count_entity="customers")),
    "units_category_year": (
        ["c", "y"],
        ["How many units of {c} were sold in {y}?",
         "Total quantity of {c} items sold in {y}?",
         "How many {c} pieces did we sell in {y}?",
         "What was the {c} unit volume in {y}?"],
        "SELECT SUM(oi.quantity) FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
        "JOIN products p ON p.product_id = oi.product_id "
        f"WHERE o.status = 'completed' AND p.category = '{{c}}' AND {YEAR}",
        lambda s: Contract("scalar")),
    "products_over_price": (
        ["c", "x"],
        ["List {c} products priced over ${x}.",
         "Show {c} products with a price above ${x}.",
         "Which {c} products cost more than ${x}?",
         "{c} items more expensive than ${x}?"],
        "SELECT p.name, p.unit_price FROM products p WHERE p.category = '{c}' AND p.unit_price > {x}",
        lambda s: Contract("list")),
    "orders_by_status_year": (
        ["y"],
        ["Break down {y} orders by status.",
         "How many orders were in each status in {y}?",
         "What's the status mix of orders placed in {y}?",
         "Count {y} orders per status."],
        f"SELECT o.status, COUNT(*) FROM orders o WHERE {YEAR} GROUP BY o.status",
        lambda s: Contract("grouped")),
}

SLOT_VALUES = {"y": YEARS, "r": REGIONS, "c": CATEGORIES, "s": SEGMENTS, "k": [3, 5, 10],
               "x": [20, 50, 100]}


def generate() -> list[Example]:
    out = []
    for family, (slots, wordings, sql_t, contract_f) in FAMILIES.items():
        for values in itertools.product(*(SLOT_VALUES[s] for s in slots)):
            slot = dict(zip(slots, values))
            for w, wording in enumerate(wordings):
                q = wording.format(**slot)
                q = q[0].upper() + q[1:]
                ex_id = f"{family}-{'-'.join(map(str, values))}-w{w}"
                out.append(Example(ex_id, family, "dev" if w < 2 else "test", q,
                                   sql_t.format(**slot), contract_f(slot)))
    return out


def load(split: str | None = None) -> list[Example]:
    exs = generate()
    return [e for e in exs if split in (None, "all", e.split)]


def export_jsonl(path: str | Path) -> int:
    exs = generate()
    with open(path, "w") as fh:
        for e in exs:
            fh.write(json.dumps(e.as_dict()) + "\n")
    return len(exs)
