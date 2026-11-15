"""Tokenisation and a small BM25 index used for candidate retrieval."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def normalize_query(text: str) -> str:
    """Canonical form used as the query key everywhere (logs, truth, splits)."""
    return " ".join(tokenize(text))


class BM25:
    """Okapi BM25 over pre-tokenised documents. Deterministic tie-breaking by doc id."""

    def __init__(self, docs: dict[str, list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {d: Counter(toks) for d, toks in docs.items()}
        self.length = {d: len(toks) for d, toks in docs.items()}
        self.n_docs = len(docs)
        self.avgdl = (sum(self.length.values()) / self.n_docs) if self.n_docs else 0.0
        self.postings: dict[str, list[str]] = defaultdict(list)
        for d, counts in self.tf.items():
            for term in counts:
                self.postings[term].append(d)

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: list[str]) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for term in set(query_tokens):
            idf = self.idf(term)
            for d in self.postings.get(term, ()):
                tf = self.tf[d][term]
                norm = tf + self.k1 * (1 - self.b + self.b * self.length[d] / self.avgdl)
                out[d] += idf * tf * (self.k1 + 1) / norm
        return dict(out)

    def top(self, query_tokens: list[str], n: int) -> list[tuple[str, float]]:
        ranked = sorted(self.scores(query_tokens).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:n]
