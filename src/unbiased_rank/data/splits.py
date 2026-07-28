"""Query-level train/val/test splitting.

Why query-level and not row-level
---------------------------------
ESCI holds many judged products per query. Splitting on rows would place some
judgments for query Q in train and others in test. A ranker could then memorise
Q's relevant products during training and be rewarded for it at evaluation,
inflating every downstream metric with no way to notice from the numbers alone.
Splitting on `query_id` makes train and test share no queries at all.

Why hashing and not shuffling
-----------------------------
Assignment is a pure function of (seed, query_id):

  * reproducible without persisting a split file,
  * independent of row order and of how the frame was loaded,
  * stable under dataset growth — adding queries never reassigns existing ones.

`hashlib` is used rather than the builtin `hash()`, which is randomised per
process for `str` inputs and would silently produce a different split on every
run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from unbiased_rank.config import SplitConfig

SPLIT_COLUMN: Final[str] = "split_assignment"
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "val", "test")

_DIGEST_BYTES: Final[int] = 8
_UNIT_SCALE: Final[float] = float(1 << (8 * _DIGEST_BYTES))


def query_hash_unit(query_id: int, seed: int) -> float:
    """Map (seed, query_id) to a deterministic value in [0, 1)."""
    payload = f"{seed}:{query_id}".encode()
    digest = hashlib.blake2b(payload, digest_size=_DIGEST_BYTES).digest()
    return int.from_bytes(digest, "big") / _UNIT_SCALE


def assign_split(query_id: int, config: SplitConfig) -> str:
    """Assign a single query to train, val or test."""
    unit = query_hash_unit(query_id, config.seed)
    if unit < config.train_fraction:
        return "train"
    if unit < config.train_fraction + config.val_fraction:
        return "val"
    return "test"


@dataclass(frozen=True)
class SplitReport:
    """Split outcome plus the evidence needed to assert reproducibility."""

    assignments: pd.DataFrame  # columns: query_id, split_assignment
    manifest_hash: str
    query_counts: dict[str, int]
    row_counts: dict[str, int]

    @property
    def n_queries(self) -> int:
        return len(self.assignments)


class SplitFloorError(ValueError):
    """Raised when a split yields fewer test queries than the design requires."""


def build_split(frame: pd.DataFrame, config: SplitConfig) -> SplitReport:
    """Assign every distinct `query_id` in `frame` to a split.

    Raises:
        SplitFloorError: fewer test queries than `config.min_test_queries`. The
            Stage 3.1 power analysis sizes the test set so a 0.005 NDCG@10
            difference stays detectable; silently shipping a smaller test set
            would leave the headline comparison underpowered.
    """
    if "query_id" not in frame.columns:
        raise KeyError("frame must contain a 'query_id' column")

    unique_ids = np.sort(frame["query_id"].unique())
    assignments = pd.DataFrame(
        {
            "query_id": unique_ids,
            SPLIT_COLUMN: [assign_split(int(qid), config) for qid in unique_ids],
        }
    )

    query_counts = _counts_by_split(assignments[SPLIT_COLUMN])
    labelled = attach_split(frame, assignments)
    row_counts = _counts_by_split(labelled[SPLIT_COLUMN])

    if query_counts["test"] < config.min_test_queries:
        raise SplitFloorError(
            f"test split has {query_counts['test']} queries, "
            f"below the required minimum of {config.min_test_queries}. "
            "Either ingest more data or lower min_test_queries and re-derive the "
            "minimum detectable effect in the power analysis."
        )

    return SplitReport(
        assignments=assignments,
        manifest_hash=manifest_hash(assignments),
        query_counts=query_counts,
        row_counts=row_counts,
    )


def attach_split(frame: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Join split labels onto a judgment frame."""
    return frame.merge(assignments, on="query_id", how="left", validate="many_to_one")


def manifest_hash(assignments: pd.DataFrame) -> str:
    """Stable digest of the full assignment, for reproducibility assertions.

    Sorted by query_id so the digest depends on the assignment itself and not
    on frame ordering.
    """
    ordered = assignments.sort_values("query_id", ignore_index=True)
    digest = hashlib.blake2b(digest_size=16)
    for query_id, split in zip(ordered["query_id"], ordered[SPLIT_COLUMN], strict=True):
        digest.update(f"{int(query_id)}:{split}\n".encode())
    return digest.hexdigest()


def find_leaked_queries(labelled: pd.DataFrame) -> pd.Index[Any]:
    """Return query_ids appearing under more than one split label.

    Should always be empty by construction; used as an executable assertion
    rather than a comment claiming the property holds.
    """
    per_query = labelled.groupby("query_id")[SPLIT_COLUMN].nunique()
    leaked: pd.Index[Any] = pd.Index(per_query[per_query > 1].index)
    return leaked


def _counts_by_split(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in SPLIT_NAMES}
