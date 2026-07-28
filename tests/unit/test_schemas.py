"""Tests for schema validation and quarantine partitioning."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from tests.conftest import make_examples, make_products
from unbiased_rank.data.schemas import (
    examples_schema,
    products_schema,
    validate_partition,
)


def test_clean_frame_passes_with_nothing_quarantined(examples: pd.DataFrame) -> None:
    result = validate_partition(examples, examples_schema())

    assert result.n_quarantined == 0
    assert result.n_valid == len(examples)
    assert result.quarantine_rate == 0.0
    assert result.failure_summary().empty


def test_bad_label_is_quarantined_not_dropped_silently() -> None:
    frame = make_examples(n_queries=10, products_per_query=2)
    frame.loc[3, "esci_label"] = "Z"  # not one of E/S/C/I

    result = validate_partition(frame, examples_schema())

    assert result.n_quarantined == 1
    assert result.n_valid == len(frame) - 1
    assert result.quarantined.iloc[0]["esci_label"] == "Z"
    summary = result.failure_summary()
    assert summary.loc[0, "column"] == "esci_label"
    assert summary.loc[0, "n_rows"] == 1


def test_empty_strings_are_quarantined() -> None:
    frame = make_examples(n_queries=10, products_per_query=2)
    frame.loc[1, "query"] = ""
    frame.loc[5, "product_id"] = ""

    result = validate_partition(frame, examples_schema())

    assert result.n_quarantined == 2
    assert set(result.failure_summary()["column"]) == {"query", "product_id"}


def test_multiple_failures_on_one_row_quarantine_it_once() -> None:
    frame = make_examples(n_queries=10, products_per_query=2)
    frame.loc[4, "query"] = ""
    frame.loc[4, "esci_label"] = "Z"

    result = validate_partition(frame, examples_schema())

    assert result.n_quarantined == 1
    assert result.n_valid == len(frame) - 1


def test_missing_column_raises_rather_than_quarantining() -> None:
    """A structural problem means the input is not the dataset we expect.

    Quarantining every row would report a 100% quarantine rate and imply the
    data was merely dirty, hiding the real cause.
    """
    frame = make_examples(n_queries=5, products_per_query=2).drop(columns=["esci_label"])

    with pytest.raises(SchemaError, match="structural validation failure"):
        validate_partition(frame, examples_schema())


def test_quarantine_rate_is_a_fraction_of_input() -> None:
    frame = make_examples(n_queries=10, products_per_query=10)  # 100 rows
    frame.loc[0:9, "esci_label"] = "Z"  # 10 rows, inclusive slice

    result = validate_partition(frame, examples_schema())

    assert result.n_quarantined == 10
    assert result.quarantine_rate == pytest.approx(0.10)


def test_products_tolerate_sparse_optional_fields() -> None:
    """Sparse descriptions/brands are normal in ESCI, not corruption."""
    examples = make_examples(n_queries=5, products_per_query=2)
    products = make_products(examples)
    products.loc[0, "product_description"] = None
    products.loc[1, "product_brand"] = None

    result = validate_partition(products, products_schema())

    assert result.n_quarantined == 0


def test_products_reject_missing_title() -> None:
    examples = make_examples(n_queries=5, products_per_query=2)
    products = make_products(examples)
    products.loc[2, "product_title"] = ""

    result = validate_partition(products, products_schema())

    assert result.n_quarantined == 1
