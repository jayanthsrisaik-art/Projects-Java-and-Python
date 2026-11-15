import json
import math

import pytest

from clickrank.catalog import Product, Query, build_catalog, relevance
from clickrank.metrics import dcg, ndcg_at_k
from clickrank.text import BM25, normalize_query, tokenize


def test_tokenize_and_normalize():
    assert tokenize("Noise-Cancelling  HEADPHONES!") == ["noise", "cancelling", "headphones"]
    assert normalize_query("  Wireless   HEADphones ") == "wireless headphones"
    assert normalize_query("   ") == ""


def test_bm25_prefers_rare_terms_and_breaks_ties_by_id():
    docs = {"a": ["red", "lamp"], "b": ["blue", "lamp"], "c": ["red", "tent"], "d": ["green", "tent"]}
    bm = BM25(docs)
    assert bm.idf("lamp") == pytest.approx(bm.idf("red"))
    top = bm.top(["red", "lamp"], 4)
    assert top[0][0] == "a"                    # matches both terms
    assert [d for d, _ in top[1:]] == ["b", "c"]  # equal scores, ordered by id
    assert "d" not in dict(top)


def test_dcg_and_ndcg_known_values():
    assert dcg([3, 2, 0], 3) == pytest.approx(7 + 3 / math.log2(3))
    assert ndcg_at_k([3, 2, 1, 0]) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 0]) is None
    worst = ndcg_at_k([0, 1, 3], k=3)
    assert 0 < worst < 1
    assert ndcg_at_k([0, 3], k=1) == 0.0


def _p(category="lamp", brand="aurora", attrs=("led",)):
    return Product("P1", "t", category, brand, list(attrs), 10.0, 4.0, 100)


def test_relevance_rules():
    q = Query("aurora led lamp", "lamp", "aurora", ["led"])
    assert relevance(q, _p(category="accessory"), quality=3.0) == 0          # category gate
    good = relevance(q, _p(), quality=0.0)
    assert relevance(q, _p(brand="volta"), quality=0.0) < good               # brand mismatch hurts
    assert relevance(q, _p(attrs=("smart",)), quality=0.0) < good            # missing attribute hurts
    assert relevance(q, _p(), quality=2.0) >= good                           # quality helps
    assert all(0 <= relevance(q, _p(), quality=x) <= 3 for x in (-5, 0, 5))


def test_catalog_is_deterministic_and_hides_truth(tmp_path):
    a, b = build_catalog(3), build_catalog(3)
    assert [p.title for p in a.products] == [p.title for p in b.products]
    assert [q.text for q in a.queries] == [q.text for q in b.queries]
    assert len({q.text for q in a.queries}) == len(a.queries) == 400
    a.save(tmp_path)
    observable = json.loads((tmp_path / "catalog.json").read_text())
    assert "quality" not in json.dumps(observable)
    assert set(observable["products"][0]) == {"id", "title", "category", "brand", "attrs", "price", "rating", "views"}
    assert "grades" in json.loads((tmp_path / "truth.json").read_text())
