"""Unit tests for split construction, floors and reporting."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import make_examples
from unbiased_rank.config import SplitConfig
from unbiased_rank.data.splits import (
    SPLIT_COLUMN,
    SplitFloorError,
    attach_split,
    build_split,
    find_leaked_queries,
)


def test_build_split_requires_query_id_column() -> None:
    with pytest.raises(KeyError, match="query_id"):
        build_split(pd.DataFrame({"example_id": [1, 2]}), _config())


def test_row_counts_sum_to_input_rows(examples: pd.DataFrame, split_config: SplitConfig) -> None:
    report = build_split(examples, split_config)
    assert sum(report.row_counts.values()) == len(examples)


def test_query_counts_sum_to_distinct_queries(
    examples: pd.DataFrame, split_config: SplitConfig
) -> None:
    report = build_split(examples, split_config)
    assert sum(report.query_counts.values()) == examples["query_id"].nunique()
    assert report.n_queries == examples["query_id"].nunique()


def test_all_rows_of_a_query_share_one_split(
    examples: pd.DataFrame, split_config: SplitConfig
) -> None:
    report = build_split(examples, split_config)
    labelled = attach_split(examples, report.assignments)
    assert find_leaked_queries(labelled).empty


def test_split_floor_raises_when_test_set_too_small() -> None:
    """The floor exists because the power analysis sizes the test set.

    Shipping a smaller test set silently would leave the primary comparison
    underpowered while still producing plausible-looking numbers.
    """
    frame = make_examples(n_queries=100, products_per_query=1)
    config = _config(min_test_queries=5_000)

    with pytest.raises(SplitFloorError, match="below the required minimum"):
        build_split(frame, config)


def test_split_floor_passes_when_satisfied() -> None:
    frame = make_examples(n_queries=30_000, products_per_query=1)
    report = build_split(frame, _config(min_test_queries=5_000))
    assert report.query_counts["test"] >= 5_000


def test_attach_split_labels_every_row(
    examples: pd.DataFrame, split_config: SplitConfig
) -> None:
    report = build_split(examples, split_config)
    labelled = attach_split(examples, report.assignments)

    assert len(labelled) == len(examples)
    assert not labelled[SPLIT_COLUMN].isna().any()


def test_find_leaked_queries_detects_an_injected_leak() -> None:
    """Negative control: the leak detector must actually fire when there is one.

    Without this, a detector that always returned empty would pass every other
    leakage test in the suite.
    """
    labelled = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2],
            SPLIT_COLUMN: ["train", "test", "val", "val"],
        }
    )
    leaked = find_leaked_queries(labelled)

    assert list(leaked) == [1]


def _config(min_test_queries: int = 0) -> SplitConfig:
    return SplitConfig(
        seed=20260727,
        train_fraction=0.70,
        val_fraction=0.10,
        test_fraction=0.20,
        min_test_queries=min_test_queries,
    )
