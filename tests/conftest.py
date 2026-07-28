"""Shared fixtures.

Tests run against synthetic frames shaped like ESCI so the suite works without
the real dataset on disk. Tests that genuinely need the download are marked
`requires_data` and skip when it is absent.
"""

from __future__ import annotations

import pandas as pd
import pytest

from unbiased_rank.config import SplitConfig


@pytest.fixture
def split_config() -> SplitConfig:
    return SplitConfig(
        seed=20260727,
        train_fraction=0.70,
        val_fraction=0.10,
        test_fraction=0.20,
        min_test_queries=0,  # Floor is exercised explicitly in its own test.
    )


def make_examples(n_queries: int = 50, products_per_query: int = 4) -> pd.DataFrame:
    """Build an ESCI-shaped judgment frame."""
    labels = ("E", "S", "C", "I")
    rows = []
    example_id = 0
    for q in range(n_queries):
        for p in range(products_per_query):
            rows.append(
                {
                    "example_id": example_id,
                    "query_id": q,
                    "query": f"query text {q}",
                    "product_id": f"P{q:05d}{p:02d}",
                    "product_locale": "us",
                    "esci_label": labels[(q + p) % len(labels)],
                }
            )
            example_id += 1
    return pd.DataFrame(rows)


def make_products(examples: pd.DataFrame) -> pd.DataFrame:
    """Build a catalog covering every product referenced by `examples`."""
    product_ids = examples["product_id"].unique()
    return pd.DataFrame(
        {
            "product_id": product_ids,
            "product_title": [f"title for {pid}" for pid in product_ids],
            "product_locale": "us",
            "product_description": [f"description for {pid}" for pid in product_ids],
            "product_brand": ["acme"] * len(product_ids),
        }
    )


@pytest.fixture
def examples() -> pd.DataFrame:
    return make_examples()


@pytest.fixture
def products(examples: pd.DataFrame) -> pd.DataFrame:
    return make_products(examples)
