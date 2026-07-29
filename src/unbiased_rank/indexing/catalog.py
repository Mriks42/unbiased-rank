"""Catalogue and query embedding, with caching and incremental subsets.

Encoding is the most expensive step in the project: ~1.2M products at roughly
165 texts/s on 6 CPU cores is about two hours. Most milestones do not need the
whole catalogue -- the test split touches 313k products, test+val 446k -- so
encoding is scoped to the products a split actually references.

Row space
---------
The *catalogue* row order (as loaded from `products.parquet`) is the canonical
index used by BM25 and by candidate sets. A dense embedding matrix may cover
only a subset of it, so every embedding file is written alongside the catalogue
rows it covers. Downstream code maps catalogue row -> embedding row through
that array rather than assuming the two are aligned; assuming alignment on a
partial encode would silently score the wrong products.

Run:

    python -m unbiased_rank.indexing.catalog --splits test
    python -m unbiased_rank.indexing.catalog --splits test val train
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from unbiased_rank.config import DataConfig, load_data_config
from unbiased_rank.data.splits import SPLIT_COLUMN
from unbiased_rank.indexing.dense import DEFAULT_MODEL, DenseEncoder, FloatArray
from unbiased_rank.indexing.text import product_text

logger = logging.getLogger(__name__)

IntArray = npt.NDArray[np.int64]

# Product titles average ~20 words (~27 word-piece tokens); p95 is 33 words.
# 64 covers essentially all of them, and shortening the model's maximum
# sequence length from the 256 default trims padding work at no measured
# quality cost for text this short.
MAX_SEQ_LENGTH = 64


def embeddings_dir(config: DataConfig) -> Path:
    return config.interim_dir / "embeddings"


def product_texts(products: pd.DataFrame) -> list[str]:
    """Indexed text per product, in catalogue row order."""
    titles = products["product_title"].fillna("").astype(str)
    brands = (
        products["product_brand"].fillna("").astype(str)
        if "product_brand" in products.columns
        else pd.Series([""] * len(products), index=products.index)
    )
    return [product_text(t, b or None) for t, b in zip(titles, brands, strict=True)]


def unique_queries(examples: pd.DataFrame) -> pd.DataFrame:
    """Distinct (query_id, query) pairs, sorted by id for a stable ordering."""
    queries = examples[["query_id", "query"]].drop_duplicates(subset=["query_id"])
    return queries.sort_values("query_id", ignore_index=True)


def catalogue_rows_for_splits(
    examples: pd.DataFrame, products: pd.DataFrame, splits: list[str] | None
) -> IntArray:
    """Catalogue row indices referenced by the given splits (all if None)."""
    if splits is None:
        return np.arange(len(products), dtype=np.int64)

    if SPLIT_COLUMN not in examples.columns:
        raise KeyError(f"examples frame has no {SPLIT_COLUMN!r} column")
    wanted = examples[examples[SPLIT_COLUMN].isin(splits)]["product_id"].unique()

    row_of = {str(pid): row for row, pid in enumerate(products["product_id"])}
    rows = np.fromiter(
        (row_of[str(pid)] for pid in wanted if str(pid) in row_of),
        dtype=np.int64,
    )
    return np.sort(rows)


def load_product_embeddings(config: DataConfig) -> tuple[FloatArray, IntArray]:
    """Load the embedding matrix and the catalogue rows it covers."""
    directory = embeddings_dir(config)
    vectors: FloatArray = np.load(directory / "products.npy")
    rows: IntArray = np.load(directory / "product_rows.npy")
    if vectors.shape[0] != rows.size:
        raise ValueError(
            f"product embeddings ({vectors.shape[0]}) and covered rows ({rows.size}) disagree; "
            "re-run `python -m unbiased_rank.indexing.catalog`."
        )
    return vectors, rows


def catalogue_row_to_embedding_row(covered_rows: IntArray, n_catalogue: int) -> IntArray:
    """Reverse index: catalogue row -> embedding row, or -1 when not encoded."""
    mapping = np.full(n_catalogue, -1, dtype=np.int64)
    mapping[covered_rows] = np.arange(covered_rows.size, dtype=np.int64)
    return mapping


def build_embeddings(
    config: DataConfig | None = None,
    splits: list[str] | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
) -> dict[str, FloatArray]:
    """Encode queries and the products referenced by `splits`."""
    cfg = config if config is not None else load_data_config()
    out_dir = embeddings_dir(cfg)
    encoder = DenseEncoder(model_name=model_name, batch_size=batch_size)
    encoder.max_seq_length = MAX_SEQ_LENGTH

    products = pd.read_parquet(cfg.interim_dir / "products.parquet")
    examples = pd.read_parquet(cfg.interim_dir / "examples.parquet")
    queries = unique_queries(examples)

    logger.info("encoding %d queries", len(queries))
    started = time.perf_counter()
    query_vectors = encoder.encode_cached(
        queries["query"].astype(str).tolist(), out_dir / "queries.npy"
    )
    logger.info("queries done in %.1fs", time.perf_counter() - started)

    rows = catalogue_rows_for_splits(examples, products, splits)
    texts = product_texts(products.iloc[rows])
    logger.info(
        "encoding %d of %d products (splits=%s) -- the slow step",
        len(texts),
        len(products),
        splits or "all",
    )
    started = time.perf_counter()
    product_vectors = encoder.encode_cached(texts, out_dir / "products.npy")
    elapsed = time.perf_counter() - started
    logger.info("products done in %.1fs (%.0f texts/s)", elapsed, len(texts) / max(elapsed, 1e-9))

    np.save(out_dir / "product_rows.npy", rows)
    np.save(out_dir / "query_ids.npy", queries["query_id"].to_numpy(dtype=np.int64))

    return {"queries": query_vectors, "products": product_vectors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build catalogue and query embeddings.")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["test"],
        help="splits whose products to encode; pass 'all' for the whole catalogue",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    splits = None if args.splits == ["all"] else list(args.splits)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = build_embeddings(splits=splits, model_name=args.model, batch_size=args.batch_size)
    for name, matrix in result.items():
        logger.info("%s embeddings: shape=%s dtype=%s", name, matrix.shape, matrix.dtype)


if __name__ == "__main__":
    main()


__all__ = [
    "MAX_SEQ_LENGTH",
    "build_embeddings",
    "catalogue_row_to_embedding_row",
    "catalogue_rows_for_splits",
    "embeddings_dir",
    "load_product_embeddings",
    "product_texts",
    "unique_queries",
]
