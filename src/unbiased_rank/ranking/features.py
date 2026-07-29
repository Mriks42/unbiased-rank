"""Feature extraction for learning-to-rank.

One shared module used by *both* training and serving. That is deliberate: the
classic production failure in ranking is training-serving skew, where features
are computed slightly differently in the two paths and the deployed model
silently underperforms its offline evaluation. Sharing the code makes that
impossible by construction rather than by discipline.

Feature set is intentionally small. The experiment measures what position bias
does to *label quality*, so feature richness is a confounder to keep fixed, not
a knob to tune. A stronger feature set would lift every arm equally.

ESCI carries no price or category taxonomy, so features are limited to text,
brand and the two retrieval signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from unbiased_rank.indexing.text import tokenize

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "bm25",
    "dense_cosine",
    "token_coverage",
    "token_jaccard",
    "exact_title_prefix",
    "brand_match",
    "title_token_count",
    "query_token_count",
)

N_FEATURES: Final[int] = len(FEATURE_NAMES)


@dataclass(frozen=True)
class ProductText:
    """Pre-tokenised product text, indexed by catalogue row.

    Tokenising 313k titles takes seconds and is done once; recomputing it per
    query would dominate feature extraction.
    """

    titles: list[str]
    brands: list[str]
    title_tokens: list[frozenset[str]]
    title_lengths: npt.NDArray[np.int32]

    @classmethod
    def from_catalogue(cls, products: pd.DataFrame) -> ProductText:
        titles = products["product_title"].fillna("").astype(str).tolist()
        brands = (
            products["product_brand"].fillna("").astype(str).tolist()
            if "product_brand" in products.columns
            else [""] * len(products)
        )
        tokens = [frozenset(tokenize(t)) for t in titles]
        lengths = np.array([len(t) for t in tokens], dtype=np.int32)
        return cls(titles=titles, brands=brands, title_tokens=tokens, title_lengths=lengths)


def extract_features(
    query_text: str,
    product_rows: IntArray,
    bm25_scores: FloatArray,
    dense_scores: FloatArray,
    catalogue: ProductText,
) -> FloatArray:
    """Build the (n_candidates, N_FEATURES) matrix for one query.

    Args:
        query_text: raw query string.
        product_rows: catalogue rows of the candidates.
        bm25_scores: BM25 score per candidate, aligned to product_rows.
        dense_scores: cosine similarity per candidate, aligned to product_rows.
        catalogue: pre-tokenised product text.
    """
    n = product_rows.size
    if bm25_scores.size != n or dense_scores.size != n:
        raise ValueError(
            f"score arrays must align with candidates: got {n} candidates, "
            f"{bm25_scores.size} bm25, {dense_scores.size} dense"
        )

    query_tokens = frozenset(tokenize(query_text))
    query_length = len(query_tokens)
    query_lower = query_text.strip().lower()

    features = np.zeros((n, N_FEATURES), dtype=np.float64)
    features[:, 0] = bm25_scores
    features[:, 1] = dense_scores

    for i, row in enumerate(product_rows.tolist()):
        title_tokens = catalogue.title_tokens[row]
        shared = len(query_tokens & title_tokens)

        # Coverage: what fraction of the query the title accounts for. Asymmetric
        # on purpose -- a long title that contains the whole query is a good
        # match, and Jaccard alone would penalise it for being long.
        features[i, 2] = shared / query_length if query_length else 0.0
        union = len(query_tokens | title_tokens)
        features[i, 3] = shared / union if union else 0.0
        features[i, 4] = float(catalogue.titles[row].strip().lower().startswith(query_lower))
        features[i, 5] = float(
            bool(catalogue.brands[row]) and catalogue.brands[row].lower() in query_lower
        )
        features[i, 6] = float(catalogue.title_lengths[row])

    features[:, 7] = float(query_length)
    return features


def feature_importance_frame(importances: FloatArray) -> pd.DataFrame:
    """Label raw importances with feature names, sorted descending."""
    if importances.size != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} importances, got {importances.size}")
    return (
        pd.DataFrame({"feature": FEATURE_NAMES, "importance": importances})
        .sort_values("importance", ascending=False, ignore_index=True)
    )


__all__ = [
    "FEATURE_NAMES",
    "N_FEATURES",
    "ProductText",
    "extract_features",
    "feature_importance_frame",
]
