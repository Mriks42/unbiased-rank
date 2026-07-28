"""Configuration loading.

Config lives in YAML under `configs/` and is validated into typed models here,
so a malformed config fails at load time rather than midway through a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


def repo_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parents[2]


class SplitConfig(BaseModel):
    """Query-level split parameters."""

    seed: int
    train_fraction: float = Field(gt=0.0, lt=1.0)
    val_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    min_test_queries: int = Field(ge=0)

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> SplitConfig:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        # Float tolerance: config authors write 0.7/0.1/0.2, which does not sum
        # to exactly 1.0 in binary floating point.
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1.0, got {total!r}")
        return self


class DataConfig(BaseModel):
    """ESCI ingestion configuration."""

    raw_dir: Path
    interim_dir: Path
    quarantine_dir: Path
    locale: str
    use_small_version: bool
    split: SplitConfig
    valid_esci_labels: list[str]

    @model_validator(mode="after")
    def _resolve_paths(self) -> DataConfig:
        """Make relative paths absolute against the repo root.

        Without this, behaviour would depend on the caller's working directory.
        """
        root = repo_root()
        for field in ("raw_dir", "interim_dir", "quarantine_dir"):
            value: Path = getattr(self, field)
            if not value.is_absolute():
                object.__setattr__(self, field, root / value)
        return self

    @property
    def examples_path(self) -> Path:
        return self.raw_dir / "shopping_queries_dataset_examples.parquet"

    @property
    def products_path(self) -> Path:
        return self.raw_dir / "shopping_queries_dataset_products.parquet"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(
            f"expected a mapping at the top level of {path}, got {type(loaded).__name__}"
        )
    return loaded


def load_data_config(path: Path | None = None) -> DataConfig:
    """Load and validate `configs/data.yaml` (or an explicit override path)."""
    config_path = path if path is not None else repo_root() / "configs" / "data.yaml"
    return DataConfig.model_validate(load_yaml(config_path))
