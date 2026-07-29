"""Per-query candidate sets for the re-ranking evaluation setting.

Why re-ranking rather than full-corpus retrieval
------------------------------------------------
ESCI judges roughly 19 products per query out of a 1.2M catalogue. If we
retrieved from the whole corpus, most returned products would be *unjudged*,
and scoring them as irrelevant would systematically punish any system that
surfaces good-but-unjudged products. That biases the comparison in a direction
that has nothing to do with position bias, which is what this project measures.

Ranking the judged set instead keeps every document's relevance known. This is
the standard arrangement for learning-to-rank benchmarks -- LETOR, MSLR and
Istella all ship query-document feature vectors rather than a corpus.

Full-corpus retrieval is still measured separately, as a retrieval sanity
check, and reported as Recall@k on a query sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from unbiased_rank.data.splits import SPLIT_COLUMN
from unbiased_rank.evaluation.metrics import GRADE_MAP

logger = logging.getLogger(__name__)

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class CandidateSet:
    """One query and the judged products to be ranked for it."""

    query_id: int
    query_text: str
    product_rows: IntArray  # positional indices into the catalogue / embeddings
    grades: IntArray

    def __len__(self) -> int:
        return int(self.product_rows.size)

    @property
    def has_relevant(self) -> bool:
        """Whether any candidate is above Irrelevant.

        Queries without a relevant product score 0 on every metric and carry no
        ranking signal, but they are *kept* so that all arms are evaluated on an
        identical query set and the paired comparison stays valid.
        """
        return bool(np.any(self.grades > 0))


def build_product_row_lookup(product_ids: npt.NDArray[np.object_]) -> dict[str, int]:
    """Map product_id to its positional row in the catalogue/embedding matrix."""
    return {str(pid): row for row, pid in enumerate(product_ids)}


def build_candidate_sets(
    examples: pd.DataFrame,
    product_row_lookup: dict[str, int],
    split: str | None = None,
) -> list[CandidateSet]:
    """Group judgments into per-query candidate sets.

    Args:
        examples: judgment frame carrying query_id, query, product_id, esci_label.
        product_row_lookup: product_id to catalogue row.
        split: optional split filter (train/val/test).

    Raises:
        KeyError: a judgment references a product absent from the catalogue.
            Ingestion already diverts orphans, so reaching this means the
            catalogue and judgments came from different runs -- silently
            dropping those rows would quietly shrink candidate sets.
    """
    frame = examples
    if split is not None:
        if SPLIT_COLUMN not in frame.columns:
            raise KeyError(f"examples frame has no {SPLIT_COLUMN!r} column")
        frame = frame[frame[SPLIT_COLUMN] == split]

    missing = set(frame["product_id"].astype(str)) - product_row_lookup.keys()
    if missing:
        raise KeyError(
            f"{len(missing)} judged products are absent from the catalogue "
            f"(e.g. {sorted(missing)[:3]}). The catalogue and judgments are out of sync; "
            "re-run ingestion so both come from the same snapshot."
        )

    rows: list[CandidateSet] = []
    for query_id, group in frame.groupby("query_id", sort=True):
        product_rows = np.fromiter(
            (product_row_lookup[str(pid)] for pid in group["product_id"]),
            dtype=np.int64,
            count=len(group),
        )
        grades = np.fromiter(
            (GRADE_MAP[str(label)] for label in group["esci_label"]),
            dtype=np.int64,
            count=len(group),
        )
        rows.append(
            CandidateSet(
                query_id=int(query_id),  # type: ignore[arg-type]  # groupby key is typed loosely
                query_text=str(group["query"].iloc[0]),
                product_rows=product_rows,
                grades=grades,
            )
        )

    logger.info(
        "built %d candidate sets (split=%s), %.1f candidates per query on average",
        len(rows),
        split,
        float(np.mean([len(r) for r in rows])) if rows else 0.0,
    )
    return rows


def candidate_size_summary(candidate_sets: list[CandidateSet]) -> dict[str, float]:
    """Descriptive stats on candidate-set sizes, for the evaluation writeup."""
    if not candidate_sets:
        return {"n_queries": 0.0}
    sizes = np.array([len(c) for c in candidate_sets], dtype=np.float64)
    with_relevant = sum(1 for c in candidate_sets if c.has_relevant)
    return {
        "n_queries": float(sizes.size),
        "mean_candidates": float(sizes.mean()),
        "median_candidates": float(np.median(sizes)),
        "min_candidates": float(sizes.min()),
        "max_candidates": float(sizes.max()),
        "queries_with_relevant": float(with_relevant),
        "fraction_with_relevant": float(with_relevant / sizes.size),
    }


__all__ = [
    "CandidateSet",
    "build_candidate_sets",
    "build_product_row_lookup",
    "candidate_size_summary",
]
