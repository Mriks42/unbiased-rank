"""Property tests for the query-level split.

These encode the invariants the experiment depends on. The leakage invariant in
particular is the one silent-failure mode that would invalidate every number the
project reports, so it is asserted rather than assumed.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.conftest import make_examples
from unbiased_rank.config import SplitConfig
from unbiased_rank.data.splits import (
    SPLIT_COLUMN,
    SPLIT_NAMES,
    assign_split,
    attach_split,
    build_split,
    find_leaked_queries,
    manifest_hash,
    query_hash_unit,
)

QUERY_IDS = st.lists(
    st.integers(min_value=0, max_value=10_000_000), min_size=1, max_size=300, unique=True
)

PERMISSIVE = settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)


def _config(seed: int = 20260727) -> SplitConfig:
    return SplitConfig(
        seed=seed,
        train_fraction=0.70,
        val_fraction=0.10,
        test_fraction=0.20,
        min_test_queries=0,
    )


@given(query_id=st.integers(min_value=0, max_value=10_000_000), seed=st.integers(0, 1_000_000))
def test_hash_unit_is_in_unit_interval(query_id: int, seed: int) -> None:
    assert 0.0 <= query_hash_unit(query_id, seed) < 1.0


@given(query_id=st.integers(min_value=0, max_value=10_000_000))
def test_assignment_is_a_valid_split_name(query_id: int) -> None:
    assert assign_split(query_id, _config()) in SPLIT_NAMES


@given(query_id=st.integers(min_value=0, max_value=10_000_000))
def test_assignment_is_deterministic(query_id: int) -> None:
    """Same input must give the same answer across calls and processes.

    Guards against the builtin `hash()` regression: string hashing is randomised
    per process, so a switch to it would break this only across runs.
    """
    first = assign_split(query_id, _config())
    assert all(assign_split(query_id, _config()) == first for _ in range(5))


@given(query_ids=QUERY_IDS)
@PERMISSIVE
def test_no_query_appears_in_two_splits(query_ids: list[int]) -> None:
    """The leakage invariant: no query_id may straddle a split boundary."""
    frame = pd.DataFrame(
        {
            "query_id": [q for q in query_ids for _ in range(3)],
            "example_id": range(len(query_ids) * 3),
        }
    )
    report = build_split(frame, _config())
    labelled = attach_split(frame, report.assignments)

    assert find_leaked_queries(labelled).empty
    assert not labelled[SPLIT_COLUMN].isna().any()


@given(query_ids=QUERY_IDS)
@PERMISSIVE
def test_split_is_independent_of_row_order(query_ids: list[int]) -> None:
    """Shuffling the input frame must not change the assignment."""
    frame = pd.DataFrame({"query_id": query_ids})
    forward = build_split(frame, _config())
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    backward = build_split(reversed_frame, _config())

    assert forward.manifest_hash == backward.manifest_hash


@given(query_ids=QUERY_IDS, extra=QUERY_IDS)
@PERMISSIVE
def test_existing_queries_are_never_reassigned_when_data_grows(
    query_ids: list[int], extra: list[int]
) -> None:
    """Adding queries must not move queries that were already assigned.

    This is why assignment hashes the id instead of shuffling: a shuffle would
    reshuffle the whole corpus whenever the dataset changed size, invalidating
    any previously trained model's split.
    """
    original = build_split(pd.DataFrame({"query_id": query_ids}), _config())
    grown_ids = sorted(set(query_ids) | set(extra))
    grown = build_split(pd.DataFrame({"query_id": grown_ids}), _config())

    merged = original.assignments.merge(
        grown.assignments, on="query_id", suffixes=("_before", "_after")
    )
    assert (merged[f"{SPLIT_COLUMN}_before"] == merged[f"{SPLIT_COLUMN}_after"]).all()


@given(query_ids=QUERY_IDS)
@PERMISSIVE
def test_manifest_hash_is_order_invariant_but_content_sensitive(query_ids: list[int]) -> None:
    report = build_split(pd.DataFrame({"query_id": query_ids}), _config())
    shuffled = report.assignments.sample(frac=1.0, random_state=7).reset_index(drop=True)

    assert manifest_hash(shuffled) == report.manifest_hash

    mutated = report.assignments.copy()
    current = mutated.loc[0, SPLIT_COLUMN]
    mutated.loc[0, SPLIT_COLUMN] = "test" if current != "test" else "train"
    assert manifest_hash(mutated) != report.manifest_hash


def test_split_proportions_track_configured_fractions() -> None:
    """With enough queries, realised proportions should approach the config."""
    frame = make_examples(n_queries=20_000, products_per_query=1)
    report = build_split(frame, _config())

    total = report.n_queries
    assert abs(report.query_counts["train"] / total - 0.70) < 0.02
    assert abs(report.query_counts["val"] / total - 0.10) < 0.02
    assert abs(report.query_counts["test"] / total - 0.20) < 0.02


def test_different_seeds_produce_different_splits() -> None:
    frame = make_examples(n_queries=2_000, products_per_query=1)
    a = build_split(frame, _config(seed=1))
    b = build_split(frame, _config(seed=2))
    assert a.manifest_hash != b.manifest_hash
