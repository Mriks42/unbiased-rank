"""Ranking metrics.

All metrics are computed *per query* and returned as arrays rather than
pre-averaged. Two reasons:

1. The experiment compares arms on the same queries, so the statistics layer
   needs per-query values to form paired differences. Averaging first would
   discard exactly the pairing that makes the comparison sensitive.
2. Per-query values are needed for the bootstrap and for segment breakdowns
   (head vs tail queries).

Grades follow ESCI: E=3, S=2, C=1, I=0. The mapping is an ordering choice, not
a measurement, and is stated in FINDINGS.md alongside results.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

# ESCI label -> graded relevance. Exact is most relevant; Irrelevant is zero.
GRADE_MAP: Final[dict[str, int]] = {"E": 3, "S": 2, "C": 1, "I": 0}

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def grades_from_labels(labels: npt.NDArray[np.str_] | list[str]) -> IntArray:
    """Map ESCI letter labels to integer grades."""
    return np.array([GRADE_MAP[str(label)] for label in labels], dtype=np.int64)


def dcg_at_k(grades: FloatArray, k: int) -> float:
    """Discounted cumulative gain using the 2^g - 1 gain formulation.

    The exponential gain is the standard choice for graded relevance; it makes
    a single Exact match worth more than several Complements, which matches how
    product search is actually judged.
    """
    top = grades[:k]
    if top.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, top.size + 2))
    return float(np.sum((np.power(2.0, top) - 1.0) * discounts))


def ndcg_at_k(grades_in_rank_order: FloatArray, k: int) -> float:
    """NDCG@k for one query.

    Args:
        grades_in_rank_order: relevance grades ordered by the system's ranking.
        k: cutoff.

    Returns:
        NDCG in [0, 1]. A query with no relevant documents scores 0.0 by
        convention; it carries no information about ranking quality, and
        including it as 0 rather than dropping it keeps the query set identical
        across arms so the paired comparison stays valid.
    """
    ideal = np.sort(grades_in_rank_order)[::-1]
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(grades_in_rank_order, k) / idcg


def reciprocal_rank(grades_in_rank_order: FloatArray, relevant_threshold: float = 1.0) -> float:
    """Reciprocal rank of the first document at or above `relevant_threshold`."""
    hits = np.flatnonzero(grades_in_rank_order >= relevant_threshold)
    if hits.size == 0:
        return 0.0
    return 1.0 / float(hits[0] + 1)


def recall_at_k(
    grades_in_rank_order: FloatArray, k: int, relevant_threshold: float = 1.0
) -> float:
    """Fraction of this query's relevant documents that appear in the top k."""
    total_relevant = int(np.count_nonzero(grades_in_rank_order >= relevant_threshold))
    if total_relevant == 0:
        return 0.0
    retrieved = int(np.count_nonzero(grades_in_rank_order[:k] >= relevant_threshold))
    return retrieved / total_relevant


def rank_by_score(scores: FloatArray, grades: FloatArray) -> FloatArray:
    """Reorder `grades` by descending `scores`.

    Ties are broken by ascending original position, which is deterministic.
    Without a defined tie-break, two arms producing identical scores could
    report different metrics purely from sort instability, and that difference
    would look like a real effect.
    """
    order = np.lexsort((np.arange(scores.size), -scores))
    return grades[order]


__all__ = [
    "GRADE_MAP",
    "dcg_at_k",
    "grades_from_labels",
    "ndcg_at_k",
    "rank_by_score",
    "recall_at_k",
    "reciprocal_rank",
]
