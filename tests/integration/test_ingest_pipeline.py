"""End-to-end ingestion tests.

These write real parquet files to a temporary directory and run the real
pipeline. The filesystem is deliberately not mocked: parquet round-tripping,
dtype preservation and the orphan-join are exactly the places this stage can
break, and a mocked filesystem would hide all three.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import make_examples, make_products
from unbiased_rank.config import DataConfig, SplitConfig
from unbiased_rank.data.ingest import RawDataNotFoundError, ingest
from unbiased_rank.data.splits import SPLIT_COLUMN, find_leaked_queries


@pytest.fixture
def staged(tmp_path: Path) -> DataConfig:
    """Write ESCI-shaped parquet files and return a config pointing at them."""
    raw = tmp_path / "raw"
    raw.mkdir()
    examples = make_examples(n_queries=400, products_per_query=5)
    products = make_products(examples)
    examples.to_parquet(raw / "shopping_queries_dataset_examples.parquet", index=False)
    products.to_parquet(raw / "shopping_queries_dataset_products.parquet", index=False)

    return DataConfig(
        raw_dir=raw,
        interim_dir=tmp_path / "interim",
        quarantine_dir=tmp_path / "quarantine",
        locale="us",
        use_small_version=False,
        split=SplitConfig(
            seed=20260727,
            train_fraction=0.70,
            val_fraction=0.10,
            test_fraction=0.20,
            min_test_queries=0,
        ),
        valid_esci_labels=["E", "S", "C", "I"],
    )


def test_ingest_writes_all_expected_artifacts(staged: DataConfig) -> None:
    report = ingest(staged)

    assert (staged.interim_dir / "examples.parquet").exists()
    assert (staged.interim_dir / "products.parquet").exists()
    assert (staged.interim_dir / "split_assignments.parquet").exists()
    assert (staged.interim_dir / "ingest_report.json").exists()

    on_disk = json.loads((staged.interim_dir / "ingest_report.json").read_text(encoding="utf-8"))
    assert on_disk["split_manifest_hash"] == report.split_manifest_hash
    assert on_disk["n_queries"] == 400


def test_ingested_examples_carry_a_leak_free_split(staged: DataConfig) -> None:
    ingest(staged)
    labelled = pd.read_parquet(staged.interim_dir / "examples.parquet")

    assert SPLIT_COLUMN in labelled.columns
    assert not labelled[SPLIT_COLUMN].isna().any()
    assert find_leaked_queries(labelled).empty


def test_ingest_is_reproducible_across_runs(staged: DataConfig) -> None:
    """Same inputs and seed must give an identical split hash.

    This is the executable form of the M1 reproducibility criterion.
    """
    first = ingest(staged)
    second = ingest(staged)

    assert first.split_manifest_hash == second.split_manifest_hash
    assert first.query_counts == second.query_counts
    assert first.row_counts == second.row_counts


def test_bad_rows_are_quarantined_and_counted(staged: DataConfig) -> None:
    examples = pd.read_parquet(staged.examples_path)
    examples.loc[0:4, "esci_label"] = "Z"
    examples.to_parquet(staged.examples_path, index=False)

    report = ingest(staged)

    assert report.examples_rows_quarantined == 5
    assert report.examples_rows_valid == report.examples_rows_in - 5
    quarantined = pd.read_parquet(staged.quarantine_dir / "examples_quarantined.parquet")
    assert len(quarantined) == 5
    assert (quarantined["esci_label"] == "Z").all()


def test_judgments_without_a_catalog_product_are_diverted(staged: DataConfig) -> None:
    """Orphaned judgments cannot be scored, so they must not reach training."""
    products = pd.read_parquet(staged.products_path)
    kept = products.iloc[:-10]
    dropped_ids = set(products.iloc[-10:]["product_id"])
    kept.to_parquet(staged.products_path, index=False)

    report = ingest(staged)
    labelled = pd.read_parquet(staged.interim_dir / "examples.parquet")

    assert not set(labelled["product_id"]) & dropped_ids
    assert (staged.quarantine_dir / "examples_orphaned.parquet").exists()
    assert report.examples_rows_valid == len(labelled)


def test_other_locales_are_excluded(staged: DataConfig) -> None:
    examples = pd.read_parquet(staged.examples_path)
    examples.loc[0:9, "product_locale"] = "jp"
    examples.to_parquet(staged.examples_path, index=False)

    ingest(staged)
    labelled = pd.read_parquet(staged.interim_dir / "examples.parquet")

    assert (labelled["product_locale"] == "us").all()


def test_missing_raw_data_gives_an_actionable_error(tmp_path: Path, staged: DataConfig) -> None:
    empty = DataConfig(
        raw_dir=tmp_path / "nonexistent",
        interim_dir=staged.interim_dir,
        quarantine_dir=staged.quarantine_dir,
        locale="us",
        use_small_version=False,
        split=staged.split,
        valid_esci_labels=["E", "S", "C", "I"],
    )

    with pytest.raises(RawDataNotFoundError, match="esci-data"):
        ingest(empty)


def test_split_floor_blocks_an_undersized_test_set(staged: DataConfig) -> None:
    demanding = staged.model_copy(
        update={"split": staged.split.model_copy(update={"min_test_queries": 5_000})}
    )

    with pytest.raises(ValueError, match="below the required minimum"):
        ingest(demanding)
