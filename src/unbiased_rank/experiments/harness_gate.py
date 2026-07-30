"""M3 validation gate: does the simulation harness behave under no bias?

The check
---------
With position bias switched off (eta=0), no click noise, and enough impressions
per query, a ranker trained on *simulated clicks* should be statistically
indistinguishable from one trained on *true grades*. Under those settings clicks
are an unbiased (if noisy) sample of relevance, so any large gap means the
harness is broken -- clicks are being generated, aggregated, or joined wrongly.

Why it matters
--------------
Every M4 number is a difference between arms that consume this machinery. A
subtly wrong simulator would produce a full set of plausible, internally
consistent, and completely invalid results. This gate is the only thing standing
between that outcome and a published finding.

It is deliberately a *falsifiable* check with a stated pass criterion, not a
plot to eyeball.

Run:

    python -m unbiased_rank.experiments.harness_gate
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from unbiased_rank.config import DataConfig, load_data_config
from unbiased_rank.evaluation.metrics import ndcg_at_k, rank_by_score
from unbiased_rank.evaluation.statistics import (
    Estimate,
    bootstrap_mean,
    paired_bootstrap_difference,
)
from unbiased_rank.indexing.catalog import (
    catalogue_row_to_embedding_row,
    embeddings_dir,
    load_product_embeddings,
    product_texts,
)
from unbiased_rank.indexing.dense import cosine_scores
from unbiased_rank.indexing.lexical import BM25Index
from unbiased_rank.ranking.candidates import (
    CandidateSet,
    add_sampled_negatives,
    build_candidate_sets,
    build_product_row_lookup,
)
from unbiased_rank.ranking.features import ProductText, extract_features
from unbiased_rank.ranking.labels import binarise, click_labels, grade_labels
from unbiased_rank.ranking.lambdamart import LambdaMartRanker, RankerParams, stack_training_data
from unbiased_rank.simulation.click_model import ClickModel
from unbiased_rank.simulation.logger import LogConfig, simulate_click_log
from unbiased_rank.simulation.position_bias import PositionBiasModel

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
NDCG_CUTOFF = 10

# Pass criterion. Chosen before running: the two arms may differ by at most this
# much in NDCG@10 for the harness to be considered sound.
GATE_TOLERANCE = 0.02


@dataclass
class GateResult:
    """Outcome of the validation gate."""

    grades_ndcg: Estimate
    clicks_ndcg: Estimate
    difference: Estimate
    tolerance: float

    @property
    def passed(self) -> bool:
        """Pass when the gap is small in magnitude.

        Deliberately *not* "the CI contains zero". With thousands of queries even
        a trivial gap becomes statistically detectable, so a significance test
        would fail the gate for differences too small to matter. The question
        here is whether the harness is broken, not whether two arms differ at
        all.
        """
        return abs(self.difference.value) <= self.tolerance

    def to_dict(self) -> dict[str, object]:
        return {
            "grades_ndcg@10": self.grades_ndcg.value,
            "grades_ci": [self.grades_ndcg.ci_low, self.grades_ndcg.ci_high],
            "clicks_ndcg@10": self.clicks_ndcg.value,
            "clicks_ci": [self.clicks_ndcg.ci_low, self.clicks_ndcg.ci_high],
            "difference": self.difference.value,
            "difference_ci": [self.difference.ci_low, self.difference.ci_high],
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


def _build_feature_blocks(
    candidate_sets: list[CandidateSet],
    bm25: BM25Index,
    query_vectors: npt.NDArray[np.float32],
    product_vectors: npt.NDArray[np.float32],
    query_row_lookup: dict[int, int],
    catalogue_to_embedding: npt.NDArray[np.int64],
    catalogue_text: ProductText,
) -> list[FloatArray]:
    blocks: list[FloatArray] = []
    for candidate in candidate_sets:
        lexical = bm25.score(candidate.query_text, candidate.product_rows)
        embedding_rows = catalogue_to_embedding[candidate.product_rows]
        query_vector = query_vectors[query_row_lookup[candidate.query_id]]
        dense = cosine_scores(query_vector, product_vectors[embedding_rows]).astype(np.float64)
        blocks.append(
            extract_features(
                candidate.query_text, candidate.product_rows, lexical, dense, catalogue_text
            )
        )
    return blocks


def _per_query_ndcg(
    scores: list[FloatArray], candidate_sets: list[CandidateSet]
) -> FloatArray:
    values = np.empty(len(candidate_sets), dtype=np.float64)
    for i, (score, candidate) in enumerate(zip(scores, candidate_sets, strict=True)):
        ordered = rank_by_score(score, candidate.grades.astype(np.float64))
        values[i] = ndcg_at_k(ordered, NDCG_CUTOFF)
    return values


def run_gate(
    config: DataConfig | None = None,
    n_train_queries: int = 6000,
    n_eval_queries: int = 4000,
    candidates_per_query: int = 100,
    impressions_per_query: int = 20,
    seed: int = 0,
) -> GateResult:
    """Train on grades and on unbiased clicks; compare on held-out queries."""
    cfg = config if config is not None else load_data_config()

    products = pd.read_parquet(cfg.interim_dir / "products.parquet")
    examples = pd.read_parquet(cfg.interim_dir / "examples.parquet")
    product_vectors, covered_rows = load_product_embeddings(cfg)
    query_vectors = np.load(embeddings_dir(cfg) / "queries.npy")
    query_ids = np.load(embeddings_dir(cfg) / "query_ids.npy")
    catalogue_to_embedding = catalogue_row_to_embedding_row(covered_rows, len(products))
    query_row_lookup = {int(q): i for i, q in enumerate(query_ids)}

    lookup = build_product_row_lookup(products["product_id"].to_numpy(dtype=object))
    all_sets = build_candidate_sets(examples, lookup, split="test")
    needed = n_train_queries + n_eval_queries
    if len(all_sets) < needed:
        raise ValueError(f"need {needed} queries, only {len(all_sets)} available")

    # Disjoint train/eval slices. Both come from the test split because the
    # gate is about the harness, not about generalisation.
    train_sets = add_sampled_negatives(
        all_sets[:n_train_queries], covered_rows, candidates_per_query, seed=seed
    )
    eval_sets = add_sampled_negatives(
        all_sets[n_train_queries:needed], covered_rows, candidates_per_query, seed=seed
    )

    logger.info("building BM25 index over %d products", len(products))
    bm25 = BM25Index(product_texts(products))
    catalogue_text = ProductText.from_catalogue(products)

    args = (
        bm25,
        query_vectors,
        product_vectors,
        query_row_lookup,
        catalogue_to_embedding,
        catalogue_text,
    )
    train_features = _build_feature_blocks(train_sets, *args)
    eval_features = _build_feature_blocks(eval_sets, *args)

    # Unbiased logging: eta=0, no noise, many impressions. Policy is BM25.
    policy = [block[:, 0] for block in train_features]
    log = simulate_click_log(
        train_sets,
        policy,
        PositionBiasModel(eta=0.0),
        ClickModel(noise=0.0),
        LogConfig(
            top_k=candidates_per_query,
            impressions_per_query=impressions_per_query,
            seed=seed,
        ),
    )

    grade_blocks = grade_labels(train_sets)
    click_blocks = binarise(click_labels(train_sets, log))

    params = RankerParams(seed=seed)
    grades_model = LambdaMartRanker(params).fit(
        stack_training_data(train_features, grade_blocks.labels)
    )
    clicks_model = LambdaMartRanker(params).fit(
        stack_training_data(train_features, click_blocks.labels)
    )

    grades_ndcg = _per_query_ndcg(grades_model.score_blocks(eval_features), eval_sets)
    clicks_ndcg = _per_query_ndcg(clicks_model.score_blocks(eval_features), eval_sets)

    return GateResult(
        grades_ndcg=bootstrap_mean(grades_ndcg, seed=seed),
        clicks_ndcg=bootstrap_mean(clicks_ndcg, seed=seed),
        difference=paired_bootstrap_difference(clicks_ndcg, grades_ndcg, seed=seed),
        tolerance=GATE_TOLERANCE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M3 harness validation gate.")
    parser.add_argument("--train-queries", type=int, default=6000)
    parser.add_argument("--eval-queries", type=int, default=4000)
    parser.add_argument("--impressions", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("outputs/harness_gate.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_gate(
        n_train_queries=args.train_queries,
        n_eval_queries=args.eval_queries,
        impressions_per_query=args.impressions,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    print("\n=== M3 HARNESS VALIDATION GATE ===")
    print(f"trained on true grades : NDCG@10 {result.grades_ndcg}")
    print(f"trained on clicks(eta=0): NDCG@10 {result.clicks_ndcg}")
    print(f"difference             : {result.difference}")
    print(f"tolerance              : +/-{result.tolerance}")
    print(f"\nRESULT: {'PASS' if result.passed else 'FAIL'}")
    if not result.passed:
        print(
            "The harness is not sound. Do not proceed to M4 -- a broken simulator "
            "produces a full set of plausible, internally consistent, invalid results."
        )


if __name__ == "__main__":
    main()


__all__ = ["GATE_TOLERANCE", "GateResult", "run_gate"]
