"""Tests for candidate-set assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unbiased_rank.data.splits import SPLIT_COLUMN
from unbiased_rank.ranking.candidates import (
    CandidateSet,
    add_sampled_negatives,
    build_candidate_sets,
    build_product_row_lookup,
    candidate_size_summary,
)


@pytest.fixture
def judgments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2],
            "query": ["red shoes"] * 3 + ["blue hat"] * 2,
            "product_id": ["P1", "P2", "P3", "P2", "P4"],
            "esci_label": ["E", "I", "S", "C", "E"],
            SPLIT_COLUMN: ["test"] * 3 + ["train"] * 2,
        }
    )


@pytest.fixture
def lookup() -> dict[str, int]:
    return build_product_row_lookup(np.array(["P1", "P2", "P3", "P4"], dtype=object))


def test_lookup_maps_ids_to_positional_rows() -> None:
    got = build_product_row_lookup(np.array(["A", "B", "C"], dtype=object))
    assert got == {"A": 0, "B": 1, "C": 2}


def test_groups_by_query(judgments: pd.DataFrame, lookup: dict[str, int]) -> None:
    sets = build_candidate_sets(judgments, lookup)

    assert len(sets) == 2
    assert [c.query_id for c in sets] == [1, 2]
    assert len(sets[0]) == 3
    assert len(sets[1]) == 2


def test_grades_map_from_esci_labels(judgments: pd.DataFrame, lookup: dict[str, int]) -> None:
    sets = build_candidate_sets(judgments, lookup)
    assert list(sets[0].grades) == [3, 0, 2]  # E, I, S


def test_product_rows_are_positional_indices(
    judgments: pd.DataFrame, lookup: dict[str, int]
) -> None:
    sets = build_candidate_sets(judgments, lookup)
    assert list(sets[0].product_rows) == [0, 1, 2]
    assert list(sets[1].product_rows) == [1, 3]


def test_rows_and_grades_stay_aligned(judgments: pd.DataFrame, lookup: dict[str, int]) -> None:
    """Positional alignment is the contract every scorer depends on."""
    sets = build_candidate_sets(judgments, lookup)
    for candidate in sets:
        assert candidate.product_rows.size == candidate.grades.size


def test_split_filter(judgments: pd.DataFrame, lookup: dict[str, int]) -> None:
    test_sets = build_candidate_sets(judgments, lookup, split="test")
    train_sets = build_candidate_sets(judgments, lookup, split="train")

    assert [c.query_id for c in test_sets] == [1]
    assert [c.query_id for c in train_sets] == [2]


def test_split_filter_requires_the_column(lookup: dict[str, int]) -> None:
    frame = pd.DataFrame(
        {"query_id": [1], "query": ["q"], "product_id": ["P1"], "esci_label": ["E"]}
    )
    with pytest.raises(KeyError, match="split_assignment"):
        build_candidate_sets(frame, lookup, split="test")


def test_unknown_product_raises_rather_than_dropping(lookup: dict[str, int]) -> None:
    """Silently dropping unmatched judgments would shrink candidate sets and
    change every metric without any visible signal."""
    frame = pd.DataFrame(
        {
            "query_id": [1],
            "query": ["q"],
            "product_id": ["P_MISSING"],
            "esci_label": ["E"],
        }
    )
    with pytest.raises(KeyError, match="out of sync"):
        build_candidate_sets(frame, lookup)


def test_has_relevant_flag() -> None:
    relevant = CandidateSet(1, "q", np.array([0]), np.array([2]))
    irrelevant = CandidateSet(2, "q", np.array([0, 1]), np.array([0, 0]))

    assert relevant.has_relevant
    assert not irrelevant.has_relevant


def test_summary_reports_distribution(judgments: pd.DataFrame, lookup: dict[str, int]) -> None:
    summary = candidate_size_summary(build_candidate_sets(judgments, lookup))

    assert summary["n_queries"] == 2
    assert summary["mean_candidates"] == pytest.approx(2.5)
    assert summary["min_candidates"] == 2
    assert summary["max_candidates"] == 3
    assert summary["fraction_with_relevant"] == pytest.approx(1.0)


def test_summary_handles_empty_input() -> None:
    assert candidate_size_summary([]) == {"n_queries": 0.0}


class TestSampledNegatives:
    @pytest.fixture
    def base(self) -> list[CandidateSet]:
        return [
            CandidateSet(1, "q1", np.array([0, 1, 2]), np.array([3, 2, 0])),
            CandidateSet(2, "q2", np.array([5, 6]), np.array([3, 1])),
        ]

    @pytest.fixture
    def pool(self) -> np.ndarray:
        return np.arange(500, dtype=np.int64)

    def test_pads_to_target_size(self, base, pool) -> None:  # type: ignore[no-untyped-def]
        out = add_sampled_negatives(base, pool, target_size=50, seed=0)
        assert all(len(c) == 50 for c in out)

    def test_negatives_are_graded_irrelevant(self, base, pool) -> None:  # type: ignore[no-untyped-def]
        out = add_sampled_negatives(base, pool, target_size=50, seed=0)
        original = base[0]
        # Original judgments are preserved in their original positions.
        assert list(out[0].grades[: len(original)]) == list(original.grades)
        # Everything appended is grade 0.
        assert np.all(out[0].grades[len(original) :] == 0)

    def test_negatives_never_collide_with_judged_products(self, pool) -> None:  # type: ignore[no-untyped-def]
        """A sampled negative that is actually judged relevant for this query
        would be mislabelled as irrelevant, corrupting the ground truth."""
        judged = np.arange(20, dtype=np.int64)
        sets = [CandidateSet(1, "q", judged, np.full(20, 3, dtype=np.int64))]

        out = add_sampled_negatives(sets, pool, target_size=200, seed=0)

        appended = out[0].product_rows[20:]
        assert not set(appended.tolist()) & set(judged.tolist())

    def test_no_duplicate_candidates(self, base, pool) -> None:  # type: ignore[no-untyped-def]
        out = add_sampled_negatives(base, pool, target_size=100, seed=0)
        for candidate in out:
            rows = candidate.product_rows.tolist()
            assert len(rows) == len(set(rows))

    def test_rows_and_grades_stay_aligned(self, base, pool) -> None:  # type: ignore[no-untyped-def]
        out = add_sampled_negatives(base, pool, target_size=60, seed=0)
        for candidate in out:
            assert candidate.product_rows.size == candidate.grades.size

    def test_sampling_is_reproducible(self, base, pool) -> None:  # type: ignore[no-untyped-def]
        first = add_sampled_negatives(base, pool, target_size=40, seed=7)
        second = add_sampled_negatives(base, pool, target_size=40, seed=7)
        for a, b in zip(first, second, strict=True):
            assert np.array_equal(a.product_rows, b.product_rows)

    def test_different_seeds_sample_differently(self, base, pool) -> None:  # type: ignore[no-untyped-def]
        a = add_sampled_negatives(base, pool, target_size=40, seed=1)
        b = add_sampled_negatives(base, pool, target_size=40, seed=2)
        assert not np.array_equal(a[0].product_rows, b[0].product_rows)

    def test_sampling_is_independent_of_iteration_order(self, base, pool) -> None:  # type: ignore[no-untyped-def]
        """Each query seeds from its own id, so reordering the input must not
        change what any individual query receives."""
        forward = add_sampled_negatives(base, pool, target_size=40, seed=3)
        backward = add_sampled_negatives(list(reversed(base)), pool, target_size=40, seed=3)

        by_id = {c.query_id: c for c in backward}
        for candidate in forward:
            assert np.array_equal(candidate.product_rows, by_id[candidate.query_id].product_rows)

    def test_sets_already_at_target_are_untouched(self, pool) -> None:  # type: ignore[no-untyped-def]
        big = [CandidateSet(1, "q", np.arange(30, dtype=np.int64), np.ones(30, dtype=np.int64))]
        out = add_sampled_negatives(big, pool, target_size=10, seed=0)
        assert np.array_equal(out[0].product_rows, big[0].product_rows)

    def test_empty_pool_is_rejected(self, base) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="pool is empty"):
            add_sampled_negatives(base, np.array([], dtype=np.int64), target_size=50)

    def test_padding_lowers_random_ranking_score(self, pool) -> None:  # type: ignore[no-untyped-def]
        """The point of the exercise: restore headroom to the metric.

        With mostly-relevant judged sets, random ranking scores ~0.8 and leaves
        almost nothing for any effect to show up in.
        """
        from unbiased_rank.evaluation.metrics import ndcg_at_k, rank_by_score

        rng = np.random.default_rng(0)
        judged = [
            CandidateSet(i, "q", np.arange(16, dtype=np.int64), np.full(16, 3, dtype=np.int64))
            for i in range(50)
        ]
        padded = add_sampled_negatives(judged, np.arange(5_000, dtype=np.int64), 100, seed=0)

        def random_ndcg(sets: list[CandidateSet]) -> float:
            return float(
                np.mean(
                    [
                        ndcg_at_k(
                            rank_by_score(rng.random(len(c)), c.grades.astype(np.float64)), 10
                        )
                        for c in sets
                    ]
                )
            )

        assert random_ndcg(judged) == pytest.approx(1.0)  # cannot go wrong
        assert random_ndcg(padded) < 0.6  # real headroom restored
