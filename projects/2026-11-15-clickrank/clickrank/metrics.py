"""Ranking metrics."""
from __future__ import annotations

import math
from typing import Sequence


def dcg(grades: Sequence[int], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(list(grades)[:k]))


def ndcg_at_k(ranked_grades: Sequence[int], k: int = 10) -> float | None:
    """NDCG@k of a ranking against the ideal ordering of the same items. None if nothing is relevant."""
    ideal = dcg(sorted(ranked_grades, reverse=True), k)
    if ideal == 0:
        return None
    return dcg(ranked_grades, k) / ideal
