"""Tests for candidate-set assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unbiased_rank.data.splits import SPLIT_COLUMN
from unbiased_rank.ranking.candidates import (
    CandidateSet,
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
