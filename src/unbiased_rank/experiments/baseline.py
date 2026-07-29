"""M2 retrieval baseline: BM25 vs dense vs RRF, in the re-ranking setting.

Establishes the reference points the position-bias experiment is measured
against. Every number carries a bootstrap confidence interval, and arm-to-arm
differences are paired, because the whole project turns on distinguishing real
effects from noise.

Run:

    python -m unbiased_rank.experiments.baseline --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from unbiased_rank.config import DataConfig, load_data_config
from unbiased_rank.evaluation.metrics import ndcg_at_k, rank_by_score, reciprocal_rank
from unbiased_rank.evaluation.statistics import (
    Estimate,
    bootstrap_mean,
    paired_bootstrap_difference,
    paired_difference_sd,
    required_sample_size,
)
from unbiased_rank.indexing.catalog import (
    catalogue_row_to_embedding_row,
    embeddings_dir,
    load_product_embeddings,
    product_texts,
)
from unbiased_rank.indexing.dense import cosine_scores
from unbiased_rank.indexing.fusion import reciprocal_rank_fusion
from unbiased_rank.indexing.lexical import BM25Index
from unbiased_rank.ranking.candidates import (
    CandidateSet,
    add_sampled_negatives,
    build_candidate_sets,
    build_product_row_lookup,
    candidate_size_summary,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
NDCG_CUTOFF = 10


@dataclass
class ArmResult:
    """Per-arm metrics with per-query values retained for paired tests."""

    name: str
    ndcg: Estimate
    mrr: Estimate
    per_query_ndcg: FloatArray = field(repr=False)

    def summary(self) -> dict[str, float]:
        return {
            "ndcg@10": self.ndcg.value,
            "ndcg@10_ci_low": self.ndcg.ci_low,
            "ndcg@10_ci_high": self.ndcg.ci_high,
            "mrr": self.mrr.value,
            "n_queries": float(self.ndcg.n),
        }


def evaluate_arm(
    name: str,
    scores_per_query: list[FloatArray],
    candidate_sets: list[CandidateSet],
    seed: int = 0,
) -> ArmResult:
    """Compute NDCG@10 and MRR for one scoring arm."""
    ndcg = np.empty(len(candidate_sets), dtype=np.float64)
    mrr = np.empty(len(candidate_sets), dtype=np.float64)

    for i, (scores, candidate) in enumerate(zip(scores_per_query, candidate_sets, strict=True)):
        ordered = rank_by_score(scores, candidate.grades.astype(np.float64))
        ndcg[i] = ndcg_at_k(ordered, NDCG_CUTOFF)
        mrr[i] = reciprocal_rank(ordered)

    return ArmResult(
        name=name,
        ndcg=bootstrap_mean(ndcg, seed=seed),
        mrr=bootstrap_mean(mrr, seed=seed),
        per_query_ndcg=ndcg,
    )


def score_bm25(index: BM25Index, candidate_sets: list[CandidateSet]) -> list[FloatArray]:
    return [index.score(c.query_text, c.product_rows) for c in candidate_sets]


def score_dense(
    query_vectors: npt.NDArray[np.float32],
    product_vectors: npt.NDArray[np.float32],
    query_row_lookup: dict[int, int],
    catalogue_to_embedding: npt.NDArray[np.int64],
    candidate_sets: list[CandidateSet],
) -> list[FloatArray]:
    """Cosine similarity per candidate, via the catalogue -> embedding mapping.

    Embeddings may cover only a subset of the catalogue, so candidate rows are
    translated rather than used directly. An uncovered candidate would index
    the wrong embedding, so it is an error rather than a silent zero.
    """
    scores: list[FloatArray] = []
    for candidate in candidate_sets:
        embedding_rows = catalogue_to_embedding[candidate.product_rows]
        if np.any(embedding_rows < 0):
            missing = int(np.count_nonzero(embedding_rows < 0))
            raise KeyError(
                f"query {candidate.query_id} has {missing} candidates with no embedding. "
                "Re-run `python -m unbiased_rank.indexing.catalog --splits ...` covering "
                "this split."
            )
        query_vector = query_vectors[query_row_lookup[candidate.query_id]]
        similarity = cosine_scores(query_vector, product_vectors[embedding_rows])
        scores.append(similarity.astype(np.float64))
    return scores


def score_fusion(
    lexical: list[FloatArray], dense: list[FloatArray]
) -> list[FloatArray]:
    return [reciprocal_rank_fusion([a, b]) for a, b in zip(lexical, dense, strict=True)]


def score_random(candidate_sets: list[CandidateSet], seed: int = 0) -> list[FloatArray]:
    """Random ordering: the floor any real system must clear.

    Without this, a weak-looking NDCG is hard to interpret -- short candidate
    sets make even random ranking score surprisingly well, and only the random
    floor reveals how much of an arm's score is genuine signal.
    """
    rng = np.random.default_rng(seed)
    return [rng.random(len(c)) for c in candidate_sets]


def run_baseline(
    config: DataConfig | None = None,
    split: str = "test",
    sample_queries: int | None = None,
    candidates_per_query: int = 100,
    seed: int = 0,
) -> dict[str, object]:
    """Build indexes, score every arm, and report metrics with CIs."""
    cfg = config if config is not None else load_data_config()
    emb_dir = embeddings_dir(cfg)

    products = pd.read_parquet(cfg.interim_dir / "products.parquet")
    examples = pd.read_parquet(cfg.interim_dir / "examples.parquet")
    product_vectors, covered_rows = load_product_embeddings(cfg)
    query_vectors = np.load(emb_dir / "queries.npy")
    query_ids = np.load(emb_dir / "query_ids.npy")
    catalogue_to_embedding = catalogue_row_to_embedding_row(covered_rows, len(products))

    _assert_alignment(examples, query_vectors, query_ids)

    lookup = build_product_row_lookup(products["product_id"].to_numpy(dtype=object))
    candidate_sets = build_candidate_sets(examples, lookup, split=split)
    if sample_queries is not None and sample_queries < len(candidate_sets):
        rng = np.random.default_rng(seed)
        picked = rng.choice(len(candidate_sets), size=sample_queries, replace=False)
        candidate_sets = [candidate_sets[i] for i in sorted(picked)]
        logger.info("sampled %d queries for a faster run", len(candidate_sets))

    judged_only = candidate_sets
    if candidates_per_query > 0:
        # Negatives are drawn from the encoded pool, so every candidate is
        # guaranteed to have an embedding for the dense arm to score.
        candidate_sets = add_sampled_negatives(
            candidate_sets, covered_rows, target_size=candidates_per_query, seed=seed
        )

    logger.info("building BM25 index over %d products", len(products))
    started = time.perf_counter()
    bm25 = BM25Index(product_texts(products))
    logger.info(
        "BM25 index built in %.1fs (vocabulary %d)",
        time.perf_counter() - started,
        bm25.vocabulary_size,
    )

    query_row_lookup = {int(qid): row for row, qid in enumerate(query_ids)}

    logger.info("scoring %d queries", len(candidate_sets))
    timings: dict[str, float] = {}
    started = time.perf_counter()
    lexical_scores = score_bm25(bm25, candidate_sets)
    timings["bm25_total_s"] = time.perf_counter() - started

    started = time.perf_counter()
    dense_scores = score_dense(
        query_vectors, product_vectors, query_row_lookup, catalogue_to_embedding, candidate_sets
    )
    timings["dense_total_s"] = time.perf_counter() - started

    fusion_scores = score_fusion(lexical_scores, dense_scores)
    random_scores = score_random(candidate_sets, seed=seed)

    arms = {
        "random": evaluate_arm("random", random_scores, candidate_sets, seed=seed),
        "bm25": evaluate_arm("bm25", lexical_scores, candidate_sets, seed=seed),
        "dense": evaluate_arm("dense", dense_scores, candidate_sets, seed=seed),
        "rrf": evaluate_arm("rrf", fusion_scores, candidate_sets, seed=seed),
    }

    comparisons = _paired_comparisons(arms, seed=seed)
    per_query_latency_ms = 1000.0 * timings["bm25_total_s"] / max(len(candidate_sets), 1)

    return {
        "split": split,
        "candidates_per_query": candidates_per_query,
        "candidates": candidate_size_summary(candidate_sets),
        "candidates_judged_only": candidate_size_summary(judged_only),
        "arms": {name: arm.summary() for name, arm in arms.items()},
        "comparisons": comparisons,
        "timings": {
            **timings,
            "bm25_ms_per_query": per_query_latency_ms,
        },
        "power": _power_note(arms, seed=seed),
    }


def _paired_comparisons(arms: dict[str, ArmResult], seed: int) -> dict[str, dict[str, float]]:
    """Paired NDCG differences against BM25, the conventional reference."""
    reference = arms["bm25"]
    out: dict[str, dict[str, float]] = {}
    for name, arm in arms.items():
        if name == "bm25":
            continue
        diff = paired_bootstrap_difference(
            arm.per_query_ndcg, reference.per_query_ndcg, seed=seed
        )
        out[f"{name}_minus_bm25"] = {
            "delta_ndcg@10": diff.value,
            "ci_low": diff.ci_low,
            "ci_high": diff.ci_high,
            "significant": float(diff.excludes_zero()),
        }
    return out


def _power_note(arms: dict[str, ArmResult], seed: int) -> dict[str, float]:
    """Measured sigma_d and the implied sample size for a 0.005 effect.

    sigma_d is measured here rather than assumed; the Stage 3.1 protocol
    requires the pilot to set it before M4 sizing.
    """
    sigma_d = paired_difference_sd(arms["rrf"].per_query_ndcg, arms["bm25"].per_query_ndcg)
    return {
        "sigma_d_rrf_vs_bm25": sigma_d,
        "required_n_for_mde_0.005": float(required_sample_size(sigma_d, 0.005)),
        "required_n_for_mde_0.010": float(required_sample_size(sigma_d, 0.010)),
    }


def _assert_alignment(
    examples: pd.DataFrame,
    query_vectors: npt.NDArray[np.float32],
    query_ids: npt.NDArray[np.int64],
) -> None:
    """Fail loudly if query embeddings and judgments disagree.

    Product alignment is enforced structurally by the covered-rows mapping;
    queries have no such mapping, so their count is checked directly. A silent
    mismatch would score the wrong query and still produce plausible metrics.
    """
    if query_vectors.shape[0] != query_ids.size:
        raise ValueError(
            f"query embeddings ({query_vectors.shape[0]}) and query ids ({query_ids.size}) differ"
        )
    expected_queries = examples["query_id"].nunique()
    if query_ids.size != expected_queries:
        raise ValueError(
            f"query embeddings cover {query_ids.size} queries but judgments contain "
            f"{expected_queries}. Re-run `python -m unbiased_rank.indexing.catalog`."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M2 retrieval baseline.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample-queries", type=int, default=None)
    parser.add_argument(
        "--candidates-per-query",
        type=int,
        default=100,
        help="pad candidate sets with sampled negatives to this size; 0 disables padding",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/baseline.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run_baseline(
        split=args.split,
        sample_queries=args.sample_queries,
        candidates_per_query=args.candidates_per_query,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["ArmResult", "evaluate_arm", "run_baseline"]
