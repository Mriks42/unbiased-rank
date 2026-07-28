"""Tests for ranking metrics.

Several cases here are hand-computed rather than compared against another
implementation, so a shared misunderstanding of the formula cannot pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from unbiased_rank.evaluation.metrics import (
    GRADE_MAP,
    dcg_at_k,
    grades_from_labels,
    ndcg_at_k,
    rank_by_score,
    recall_at_k,
    reciprocal_rank,
)


def test_grade_map_orders_esci_labels_correctly() -> None:
    assert GRADE_MAP["E"] > GRADE_MAP["S"] > GRADE_MAP["C"] > GRADE_MAP["I"]
    assert GRADE_MAP["I"] == 0


def test_grades_from_labels() -> None:
    got = grades_from_labels(["E", "S", "C", "I"])
    assert list(got) == [3, 2, 1, 0]


def test_dcg_hand_computed() -> None:
    """DCG for grades [3, 0] = (2^3-1)/log2(2) + (2^0-1)/log2(3) = 7.0."""
    assert dcg_at_k(np.array([3.0, 0.0]), k=2) == pytest.approx(7.0)


def test_dcg_second_position_is_discounted() -> None:
    """Same gain lower down must contribute less."""
    first = dcg_at_k(np.array([0.0, 3.0]), k=2)
    assert first == pytest.approx(7.0 / np.log2(3))
    assert first < dcg_at_k(np.array([3.0, 0.0]), k=2)


def test_ndcg_is_one_for_perfect_ordering() -> None:
    assert ndcg_at_k(np.array([3.0, 2.0, 1.0, 0.0]), k=4) == pytest.approx(1.0)


def test_ndcg_is_less_than_one_for_reversed_ordering() -> None:
    assert ndcg_at_k(np.array([0.0, 1.0, 2.0, 3.0]), k=4) < 1.0


def test_ndcg_is_zero_when_nothing_is_relevant() -> None:
    """No relevant documents carries no ranking signal; scored 0 by convention.

    Such queries are kept rather than dropped so the query set stays identical
    across arms and the paired comparison remains valid.
    """
    assert ndcg_at_k(np.array([0.0, 0.0, 0.0]), k=3) == 0.0


def test_ndcg_respects_the_cutoff() -> None:
    """A relevant document beyond k must not contribute."""
    truncated = ndcg_at_k(np.array([0.0, 0.0, 3.0]), k=2)
    assert truncated == 0.0


def test_ndcg_is_bounded() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        grades = rng.integers(0, 4, size=10).astype(np.float64)
        assert 0.0 <= ndcg_at_k(grades, k=10) <= 1.0


def test_reciprocal_rank_positions() -> None:
    assert reciprocal_rank(np.array([3.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert reciprocal_rank(np.array([0.0, 2.0, 0.0])) == pytest.approx(0.5)
    assert reciprocal_rank(np.array([0.0, 0.0, 1.0])) == pytest.approx(1 / 3)
    assert reciprocal_rank(np.array([0.0, 0.0, 0.0])) == 0.0


def test_recall_at_k() -> None:
    grades = np.array([3.0, 0.0, 2.0, 0.0, 1.0])  # three relevant
    assert recall_at_k(grades, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(grades, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(grades, k=5) == pytest.approx(1.0)


def test_recall_is_zero_when_no_relevant_documents_exist() -> None:
    assert recall_at_k(np.array([0.0, 0.0]), k=2) == 0.0


def test_rank_by_score_orders_descending() -> None:
    scores = np.array([0.1, 0.9, 0.5])
    grades = np.array([1.0, 3.0, 2.0])
    assert list(rank_by_score(scores, grades)) == [3.0, 2.0, 1.0]


def test_rank_by_score_tie_break_is_deterministic() -> None:
    """Equal scores must resolve by original position, not sort instability.

    Without a defined tie-break, two arms producing identical scores could
    report different metrics, and that artifact would read as a real effect.
    """
    scores = np.array([0.5, 0.5, 0.5])
    grades = np.array([1.0, 2.0, 3.0])
    first = rank_by_score(scores, grades)
    assert list(first) == [1.0, 2.0, 3.0]
    assert list(rank_by_score(scores, grades)) == list(first)


def test_ranking_a_perfect_scorer_gives_ndcg_one() -> None:
    """End-to-end: scores that mirror grades must produce a perfect ranking."""
    grades = np.array([0.0, 3.0, 1.0, 2.0])
    ordered = rank_by_score(scores=grades.copy(), grades=grades)
    assert ndcg_at_k(ordered, k=4) == pytest.approx(1.0)
