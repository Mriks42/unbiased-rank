"""Tests for reciprocal rank fusion."""

from __future__ import annotations

import numpy as np
import pytest

from unbiased_rank.indexing.fusion import (
    DEFAULT_RRF_K,
    ranks_from_scores,
    reciprocal_rank_fusion,
)


class TestRanksFromScores:
    def test_highest_score_gets_rank_one(self) -> None:
        assert list(ranks_from_scores(np.array([0.1, 0.9, 0.5]))) == [3, 1, 2]

    def test_ties_resolve_by_original_position(self) -> None:
        assert list(ranks_from_scores(np.array([0.5, 0.5, 0.5]))) == [1, 2, 3]

    def test_ranks_are_a_permutation(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.normal(size=50)
        assert sorted(ranks_from_scores(scores)) == list(range(1, 51))


class TestReciprocalRankFusion:
    def test_agreeing_runs_preserve_the_ordering(self) -> None:
        a = np.array([0.9, 0.5, 0.1])
        b = np.array([3.0, 2.0, 1.0])  # different scale, same order
        fused = reciprocal_rank_fusion([a, b])
        assert list(ranks_from_scores(fused)) == [1, 2, 3]

    def test_fusion_ignores_score_magnitude(self) -> None:
        """The point of RRF: only ranks matter, so scale is irrelevant.

        This is why RRF avoids the per-corpus calibration that score-based
        fusion of BM25 and cosine similarity would require.
        """
        a = np.array([0.9, 0.5, 0.1])
        scaled = a * 1000.0
        assert np.allclose(reciprocal_rank_fusion([a]), reciprocal_rank_fusion([scaled]))

    def test_extreme_ranks_beat_consensus_middle_ranks(self) -> None:
        """Counterintuitive but correct: (1st, 3rd) outscores (2nd, 2nd).

        Because 1/(k+r) is convex in r, a candidate one run loves and another
        dislikes beats a candidate both rank middling:

            1/61 + 1/63 = 0.032266  >  2/62 = 0.032258

        This is worth asserting explicitly. The folk description of RRF as
        "rewarding consensus" is wrong in exactly this case, and a test written
        from that assumption fails against a correct implementation.
        """
        a = np.array([1.0, 0.9, 0.0])  # ranks: 1, 2, 3
        b = np.array([0.0, 0.9, 1.0])  # ranks: 3, 2, 1
        fused = reciprocal_rank_fusion([a, b])

        assert fused[0] == pytest.approx(fused[2])  # symmetric pair
        assert fused[0] > fused[1]

    def test_consensus_top_rank_wins_outright(self) -> None:
        """RRF does reward agreement when the agreement is at the top."""
        a = np.array([1.0, 0.5, 0.0])  # ranks: 1, 2, 3
        b = np.array([1.0, 0.5, 0.0])  # ranks: 1, 2, 3
        fused = reciprocal_rank_fusion([a, b])
        assert fused[0] > fused[1] > fused[2]

    def test_single_list_is_rank_equivalent_to_the_input(self) -> None:
        scores = np.array([0.2, 0.7, 0.4, 0.9])
        fused = reciprocal_rank_fusion([scores])
        assert list(ranks_from_scores(fused)) == list(ranks_from_scores(scores))

    def test_larger_k_compresses_differences(self) -> None:
        scores = np.array([1.0, 0.5, 0.1])
        small_k = reciprocal_rank_fusion([scores], k=1.0)
        large_k = reciprocal_rank_fusion([scores], k=1000.0)
        assert np.ptp(small_k) > np.ptp(large_k)

    def test_default_k_is_the_published_value(self) -> None:
        assert DEFAULT_RRF_K == 60.0

    def test_mismatched_sizes_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same candidates"):
            reciprocal_rank_fusion([np.zeros(3), np.zeros(4)])

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one score list"):
            reciprocal_rank_fusion([])

    def test_empty_candidate_set_returns_empty(self) -> None:
        assert reciprocal_rank_fusion([np.zeros(0)]).size == 0
