"""Result contracts: the expected *shape* of an answer, inferred from the question alone.

The critic never sees gold SQL. It derives what a correct answer must look like (one number,
one row per group, at most k ranked rows, a 12-month series, or a list) and checks the
writer's executed result against that. The golden set stores the true contract for each
question, so contract inference accuracy is measured separately in the eval.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

SHAPES = ("scalar", "grouped", "topk", "series", "list")


@dataclass(frozen=True)
class Contract:
    shape: str                  # one of SHAPES
    k: int | None = None        # topk only
    count_entity: str | None = None   # "customers" / "orders" when the question counts entities

    @property
    def n_cols(self) -> int | None:
        return {"scalar": 1, "grouped": 2, "topk": 2, "series": 2}.get(self.shape)

    def describe(self) -> str:
        return {
            "scalar": "exactly one row with one numeric column",
            "grouped": "one row per group: a label column plus one numeric column, labels unique",
            "topk": f"at most {self.k} rows (label, numeric value) ordered by the value, using LIMIT",
            "series": "one row per period (period label, numeric value) ordered by period",
            "list": "a list of matching rows (may be empty)",
        }[self.shape]

    def as_dict(self) -> dict:
        return asdict(self)


# Cues come from the dev-split wording plus generic phrasing; test wording is held out.
_TOPK = re.compile(r"\b(?:top|which)\s+(\d{1,2})\b")
_SERIES = re.compile(r"\b(monthly|per month|each month|by month)\b")
_GROUPED = re.compile(r"\b(?:by|per|each)\s+(?:region|status|category|segment)e?s?\b"
                      r"|\bbreak(?:s|ing)?\s*down\b")
_LIST = re.compile(r"^\s*(?:list|show|which)\b.*\bproducts?\b")
_COUNT = re.compile(r"\b(how many|number of|count)\b")


def infer_contract(question: str) -> Contract:
    q = question.lower()
    entity = None
    if _COUNT.search(q):
        if re.search(r"\bcustomers\b", q):
            entity = "customers"
        elif re.search(r"\borders\b", q):
            entity = "orders"
    m = _TOPK.search(q)
    if m and 0 < int(m.group(1)) <= 50:
        return Contract("topk", k=int(m.group(1)))
    if _SERIES.search(q):
        return Contract("series")
    if _GROUPED.search(q):
        return Contract("grouped")
    if _LIST.search(q) and not _COUNT.search(q):
        return Contract("list")
    return Contract("scalar", count_entity=entity)
