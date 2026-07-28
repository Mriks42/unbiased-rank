"""Pandera schemas and quarantine partitioning for ESCI data.

Validation here is deliberately *non-fatal for row-level problems*: rows that
fail a value check are diverted to a quarantine frame and counted, rather than
aborting the run or being silently coerced. Structural problems (a missing or
mistyped column) still raise, because those mean the input is not the dataset
we think it is and nothing downstream can be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

# ESCI relevance grades: Exact, Substitute, Complement, Irrelevant.
ESCI_LABELS: tuple[str, ...] = ("E", "S", "C", "I")


def examples_schema(valid_labels: tuple[str, ...] = ESCI_LABELS) -> pa.DataFrameSchema:
    """Schema for the query-product judgment table."""
    return pa.DataFrameSchema(
        {
            "example_id": pa.Column(int, nullable=False),
            "query_id": pa.Column(int, nullable=False),
            "query": pa.Column(str, checks=pa.Check.str_length(min_value=1), nullable=False),
            "product_id": pa.Column(str, checks=pa.Check.str_length(min_value=1), nullable=False),
            "product_locale": pa.Column(str, nullable=False),
            "esci_label": pa.Column(str, checks=pa.Check.isin(valid_labels), nullable=False),
        },
        strict=False,  # ESCI carries extra columns (small_version, split, ...) we pass through.
        coerce=False,  # Coercion would mask exactly the malformations we want to quarantine.
        name="esci_examples",
    )


def products_schema() -> pa.DataFrameSchema:
    """Schema for the product catalog table.

    Only `product_id` and `product_title` are required to be present; ESCI
    legitimately has sparse descriptions, bullets, brands and colors, and
    treating that sparsity as corruption would quarantine most of the catalog.
    """
    return pa.DataFrameSchema(
        {
            "product_id": pa.Column(str, checks=pa.Check.str_length(min_value=1), nullable=False),
            "product_title": pa.Column(
                str, checks=pa.Check.str_length(min_value=1), nullable=False
            ),
            "product_locale": pa.Column(str, nullable=False),
            "product_description": pa.Column(str, nullable=True, required=False),
            "product_bullet_point": pa.Column(str, nullable=True, required=False),
            "product_brand": pa.Column(str, nullable=True, required=False),
            "product_color": pa.Column(str, nullable=True, required=False),
        },
        strict=False,
        coerce=False,
        name="esci_products",
    )


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of partitioning a frame into valid and quarantined rows."""

    valid: pd.DataFrame
    quarantined: pd.DataFrame
    failure_cases: pd.DataFrame

    @property
    def n_valid(self) -> int:
        return len(self.valid)

    @property
    def n_quarantined(self) -> int:
        return len(self.quarantined)

    @property
    def quarantine_rate(self) -> float:
        total = self.n_valid + self.n_quarantined
        return 0.0 if total == 0 else self.n_quarantined / total

    def failure_summary(self) -> pd.DataFrame:
        """Counts per (column, check) so quarantine causes are inspectable."""
        if self.failure_cases.empty:
            return pd.DataFrame(columns=["column", "check", "n_rows"])
        return (
            self.failure_cases.groupby(["column", "check"], dropna=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values("n_rows", ascending=False, ignore_index=True)
        )


def validate_partition(frame: pd.DataFrame, schema: pa.DataFrameSchema) -> ValidationResult:
    """Validate `frame`, diverting row-level failures to quarantine.

    Raises:
        SchemaError: the frame is structurally wrong (missing/mistyped column),
            so no meaningful partition into valid/invalid rows exists.
    """
    try:
        validated = schema.validate(frame, lazy=True)
    except SchemaErrors as errors:
        failure_cases = errors.failure_cases
        bad_index = _row_level_failure_index(failure_cases)

        # A failure with no row index is structural, not a bad value.
        if bad_index is None:
            # pandera ships SchemaError without type annotations.
            raise SchemaError(  # type: ignore[no-untyped-call]
                schema,
                frame,
                f"structural validation failure for schema {schema.name!r}; "
                f"cannot partition rows:\n{failure_cases}",
            ) from errors

        quarantined = frame.loc[frame.index.isin(bad_index)]
        valid = frame.loc[~frame.index.isin(bad_index)]
        return ValidationResult(valid=valid, quarantined=quarantined, failure_cases=failure_cases)

    empty_failures = pd.DataFrame(columns=["column", "check", "index"])
    return ValidationResult(
        valid=validated,
        quarantined=frame.iloc[0:0],
        failure_cases=empty_failures,
    )


def _row_level_failure_index(failure_cases: pd.DataFrame) -> pd.Index | None:
    """Extract the failing row labels, or None if any failure is structural."""
    if "index" not in failure_cases.columns:
        return None
    index_values = failure_cases["index"]
    if index_values.isna().any():
        # Mixed structural + row-level failures: treat as structural. Partial
        # quarantining against an unreliable structure would hide the real problem.
        return None
    return pd.Index(index_values.unique())
