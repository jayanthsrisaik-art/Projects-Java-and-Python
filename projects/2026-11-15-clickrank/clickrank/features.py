"""Candidate retrieval (BM25) and query-document features from observable data only.

No click-derived features (per-product CTR etc.) are used: those would leak the logging
policy's position bias straight into the model and, for evaluation, leak test-query clicks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .catalog import Product, product_tokens
from .text import BM25, tokenize

FEATURES = ["bm25", "coverage", "category_match", "brand_match", "brand_mismatch",
            "attr_coverage", "rating", "log_popularity", "log_price"]
N_CANDIDATES = 20


@dataclass
class ParsedQuery:
    tokens: list[str]
    category: str | None
    brand: str | None
    attrs: list[str]


@dataclass
class CandidateSet:
    query: str
    product_ids: list[str]
    X: np.ndarray            # shape (n, len(FEATURES))


class SearchIndex:
    def __init__(self, products: list[Product], vocab: dict, n_candidates: int = N_CANDIDATES):
        self.products = {p.id: p for p in products}
        self.n_candidates = n_candidates
        self.noun_to_cat = {noun: cat for cat, noun in vocab["nouns"].items()}
        self.brands = set(vocab["brands"])
        self.attrs = set(vocab["attrs"])
        self.tokens = {p.id: product_tokens(p) for p in products}
        self.bm25 = BM25(self.tokens)
        pops = np.array([math.log1p(p.views) for p in products])
        prices = np.array([math.log(p.price) for p in products])
        self._pop = (pops.mean(), pops.std() or 1.0)
        self._price = (prices.mean(), prices.std() or 1.0)
        self._cache: dict[str, CandidateSet] = {}

    def parse(self, query: str) -> ParsedQuery:
        toks = tokenize(query)
        cat = next((self.noun_to_cat[t] for t in toks if t in self.noun_to_cat), None)
        brand = next((t for t in toks if t in self.brands), None)
        attrs = [t for t in toks if t in self.attrs and t not in self.noun_to_cat]
        return ParsedQuery(toks, cat, brand, attrs)

    def features(self, pq: ParsedQuery, pid: str, bm25: float, bm25_max: float) -> list[float]:
        p = self.products[pid]
        ptoks = set(self.tokens[pid])
        qset = set(pq.tokens)
        return [
            bm25 / bm25_max if bm25_max > 0 else 0.0,
            len(qset & ptoks) / len(qset) if qset else 0.0,
            1.0 if pq.category is not None and p.category == pq.category else 0.0,
            1.0 if pq.brand is not None and p.brand == pq.brand else 0.0,
            1.0 if pq.brand is not None and p.brand != pq.brand else 0.0,
            (sum(a in p.attrs for a in pq.attrs) / len(pq.attrs)) if pq.attrs else 0.0,
            p.rating - 3.5,
            (math.log1p(p.views) - self._pop[0]) / self._pop[1],
            (math.log(p.price) - self._price[0]) / self._price[1],
        ]

    def candidates(self, query: str) -> CandidateSet:
        if query in self._cache:
            return self._cache[query]
        pq = self.parse(query)
        top = self.bm25.top(pq.tokens, self.n_candidates)
        bmax = top[0][1] if top else 0.0
        X = np.array([self.features(pq, pid, s, bmax) for pid, s in top], dtype=float).reshape(-1, len(FEATURES))
        cs = CandidateSet(query, [pid for pid, _ in top], X)
        self._cache[query] = cs
        return cs

    def feature_row(self, query: str, pid: str) -> np.ndarray | None:
        cs = self.candidates(query)
        try:
            return cs.X[cs.product_ids.index(pid)]
        except ValueError:
            return None


def rank(cs: CandidateSet, weights: np.ndarray) -> list[str]:
    """Order candidates by linear score; ties broken by BM25 order (stable sort)."""
    if not cs.product_ids:
        return []
    scores = cs.X @ weights
    order = sorted(range(len(cs.product_ids)), key=lambda i: -scores[i])
    return [cs.product_ids[i] for i in order]
