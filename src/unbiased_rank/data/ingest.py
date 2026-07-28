"""ESCI ingestion: load, validate, quarantine, split, persist.

The ESCI parquet files are not vendored. Point `raw_dir` in `configs/data.yaml`
at a local copy (see that file for the source and the licensing caveat).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from unbiased_rank.config import DataConfig, load_data_config
from unbiased_rank.data.schemas import (
    ValidationResult,
    examples_schema,
    products_schema,
    validate_partition,
)
from unbiased_rank.data.splits import SPLIT_COLUMN, SplitReport, attach_split, build_split

logger = logging.getLogger(__name__)


class RawDataNotFoundError(FileNotFoundError):
    """Raised when the ESCI parquet files are absent from `raw_dir`."""


@dataclass(frozen=True)
class IngestReport:
    """Counts and hashes describing one ingestion run.

    Persisted next to the outputs so any trained model can be traced back to the
    exact data state that produced it.
    """

    examples_rows_in: int
    examples_rows_valid: int
    examples_rows_quarantined: int
    products_rows_in: int
    products_rows_valid: int
    products_rows_quarantined: int
    locale: str
    n_queries: int
    split_manifest_hash: str
    query_counts: dict[str, int]
    row_counts: dict[str, int]

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


def load_raw_examples(config: DataConfig) -> pd.DataFrame:
    """Read the judgment table, filtered to the configured locale."""
    frame = _read_parquet(config.examples_path)
    frame = frame[frame["product_locale"] == config.locale]
    if config.use_small_version and "small_version" in frame.columns:
        frame = frame[frame["small_version"] == 1]
    return frame.reset_index(drop=True)


def load_raw_products(config: DataConfig) -> pd.DataFrame:
    """Read the product catalog, filtered to the configured locale."""
    frame = _read_parquet(config.products_path)
    frame = frame[frame["product_locale"] == config.locale]
    return frame.reset_index(drop=True)


def ingest(config: DataConfig | None = None) -> IngestReport:
    """Run the full M1 pipeline and write interim + quarantine artifacts."""
    cfg = config if config is not None else load_data_config()

    raw_examples = load_raw_examples(cfg)
    raw_products = load_raw_products(cfg)
    logger.info(
        "loaded raw data: %d examples, %d products (locale=%s)",
        len(raw_examples),
        len(raw_products),
        cfg.locale,
    )

    examples_result = validate_partition(
        raw_examples, examples_schema(tuple(cfg.valid_esci_labels))
    )
    products_result = validate_partition(raw_products, products_schema())
    _persist_quarantine(cfg.quarantine_dir, "examples", examples_result)
    _persist_quarantine(cfg.quarantine_dir, "products", products_result)

    # Judgments referencing a quarantined or absent product cannot be scored,
    # so they are dropped here rather than surfacing as nulls during training.
    known_products = set(products_result.valid["product_id"])
    examples = examples_result.valid
    orphaned = ~examples["product_id"].isin(known_products)
    if int(orphaned.sum()) > 0:
        logger.warning("dropping %d judgments with no catalog product", int(orphaned.sum()))
        _write_parquet(
            cfg.quarantine_dir / "examples_orphaned.parquet", examples.loc[orphaned]
        )
        examples = examples.loc[~orphaned].reset_index(drop=True)

    split_report: SplitReport = build_split(examples, cfg.split)
    labelled = attach_split(examples, split_report.assignments)

    cfg.interim_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(cfg.interim_dir / "examples.parquet", labelled)
    _write_parquet(cfg.interim_dir / "products.parquet", products_result.valid)
    _write_parquet(cfg.interim_dir / "split_assignments.parquet", split_report.assignments)

    report = IngestReport(
        examples_rows_in=len(raw_examples),
        examples_rows_valid=len(labelled),
        examples_rows_quarantined=examples_result.n_quarantined,
        products_rows_in=len(raw_products),
        products_rows_valid=products_result.n_valid,
        products_rows_quarantined=products_result.n_quarantined,
        locale=cfg.locale,
        n_queries=split_report.n_queries,
        split_manifest_hash=split_report.manifest_hash,
        query_counts=split_report.query_counts,
        row_counts=split_report.row_counts,
    )
    report.to_json(cfg.interim_dir / "ingest_report.json")
    logger.info(
        "ingest complete: %d queries, split hash %s, test queries %d",
        report.n_queries,
        report.split_manifest_hash,
        report.query_counts["test"],
    )
    return report


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RawDataNotFoundError(
            f"expected ESCI parquet at {path}. Download the dataset from "
            "https://github.com/amazon-science/esci-data and set `raw_dir` in "
            "configs/data.yaml. Review its license before public deployment."
        )
    return pd.read_parquet(path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _persist_quarantine(directory: Path, name: str, result: ValidationResult) -> None:
    """Write quarantined rows and their failure summary, if any."""
    if result.n_quarantined == 0:
        logger.info("%s: no rows quarantined", name)
        return
    logger.warning(
        "%s: quarantined %d rows (%.4f%% of input)",
        name,
        result.n_quarantined,
        100 * result.quarantine_rate,
    )
    _write_parquet(directory / f"{name}_quarantined.parquet", result.quarantined)
    _write_parquet(directory / f"{name}_failure_summary.parquet", result.failure_summary())


__all__ = [
    "IngestReport",
    "RawDataNotFoundError",
    "SPLIT_COLUMN",
    "ingest",
    "load_raw_examples",
    "load_raw_products",
]
