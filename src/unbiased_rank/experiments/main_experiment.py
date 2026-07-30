"""M4: the four-arm unbiased learning-to-rank experiment.

Four rankers that differ *only* in where their training labels come from:

| Arm | Labels | Represents |
|---|---|---|
| A ceiling   | true ESCI grades          | best achievable |
| B floor     | BM25 score, no learning   | doing nothing |
| C naive     | raw biased clicks         | what teams ship without thinking |
| D corrected | biased clicks + IPS       | the textbook fix |

Headline quantity is the **recovery fraction**:

    (D - C) / (A - C)

the share of the naive-to-ceiling gap that correction closes. 1.0 means
correction fully repairs the damage; 0.0 means it does nothing; *negative* means
it made things worse, which is a real possible outcome when propensities are
badly estimated.

Everything except labels is held fixed -- same features, same model capacity,
same seeds, same evaluation queries -- so a measured difference is attributable
to the label source and nothing else.

Run:

    python -m unbiased_rank.experiments.main_experiment --etas 1.0 --seeds 3
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
from unbiased_rank.propensity.estimators import (
    MisspecifiedPropensity,
    OraclePropensity,
    PropensityEstimator,
    RandomizationPropensity,
    RegressionEMPropensity,
    estimation_error,
)
from unbiased_rank.ranking.candidates import (
    CandidateSet,
    add_sampled_negatives,
    build_candidate_sets,
    build_product_row_lookup,
)
from unbiased_rank.ranking.features import ProductText, extract_features
from unbiased_rank.ranking.labels import binarise, click_labels, grade_labels
from unbiased_rank.ranking.lambdamart import (
    LambdaMartRanker,
    RankerParams,
    stack_training_data,
)
from unbiased_rank.simulation.click_model import ClickModel
from unbiased_rank.simulation.logger import LogConfig, simulate_click_log
from unbiased_rank.simulation.position_bias import PositionBiasModel

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
NDCG_CUTOFF = 10
RANDOMIZE_FRACTION = 0.10  # Intervention slice for the randomization estimator.


@dataclass
class Fixture:
    """Everything reusable across conditions.

    Features and the BM25 index are expensive and label-independent, so they are
    built once. Rebuilding them per condition would also risk them differing
    between arms, which would break the "only labels differ" guarantee.
    """

    train_sets: list[CandidateSet]
    eval_sets: list[CandidateSet]
    train_features: list[FloatArray]
    eval_features: list[FloatArray]
    train_policy: list[FloatArray]
    top_k: int


@dataclass
class ArmMetrics:
    name: str
    ndcg: Estimate
    per_query: FloatArray = field(repr=False)


@dataclass
class ConditionResult:
    """One (eta, estimator, clip) cell of the sweep."""

    eta: float
    estimator: str
    clip: float | None
    seed: int
    naive: ArmMetrics
    corrected: ArmMetrics
    ceiling_ndcg: float
    floor_ndcg: float
    click_rate: float
    propensity_error: dict[str, float]
    corrected_minus_naive: Estimate

    @property
    def recovery_fraction(self) -> float:
        """Share of the naive-to-ceiling gap that correction closes.

        Undefined when naive already matches or exceeds the ceiling, in which
        case there is no damage to repair; reported as NaN rather than a
        misleading number.
        """
        gap = self.ceiling_ndcg - self.naive.ndcg.value
        if abs(gap) < 1e-9:
            return float("nan")
        return (self.corrected.ndcg.value - self.naive.ndcg.value) / gap

    def to_dict(self) -> dict[str, object]:
        return {
            "eta": self.eta,
            "estimator": self.estimator,
            "clip": self.clip,
            "seed": self.seed,
            "ceiling_ndcg@10": self.ceiling_ndcg,
            "floor_ndcg@10": self.floor_ndcg,
            "naive_ndcg@10": self.naive.ndcg.value,
            "naive_ci": [self.naive.ndcg.ci_low, self.naive.ndcg.ci_high],
            "corrected_ndcg@10": self.corrected.ndcg.value,
            "corrected_ci": [self.corrected.ndcg.ci_low, self.corrected.ndcg.ci_high],
            "corrected_minus_naive": self.corrected_minus_naive.value,
            "corrected_minus_naive_ci": [
                self.corrected_minus_naive.ci_low,
                self.corrected_minus_naive.ci_high,
            ],
            "significant": self.corrected_minus_naive.excludes_zero(),
            "recovery_fraction": self.recovery_fraction,
            "click_rate": self.click_rate,
            "propensity_error": self.propensity_error,
        }


def build_fixture(
    config: DataConfig | None = None,
    n_train_queries: int = 6000,
    n_eval_queries: int = 4000,
    candidates_per_query: int = 100,
    top_k: int = 20,
    seed: int = 0,
) -> Fixture:
    """Load data and precompute features for both query slices."""
    cfg = config if config is not None else load_data_config()

    products = pd.read_parquet(cfg.interim_dir / "products.parquet")
    examples = pd.read_parquet(cfg.interim_dir / "examples.parquet")
    product_vectors, covered_rows = load_product_embeddings(cfg)
    query_vectors = np.load(embeddings_dir(cfg) / "queries.npy")
    query_ids = np.load(embeddings_dir(cfg) / "query_ids.npy")
    cat2emb = catalogue_row_to_embedding_row(covered_rows, len(products))
    query_row = {int(q): i for i, q in enumerate(query_ids)}

    lookup = build_product_row_lookup(products["product_id"].to_numpy(dtype=object))
    all_sets = build_candidate_sets(examples, lookup, split="test")
    needed = n_train_queries + n_eval_queries
    if len(all_sets) < needed:
        raise ValueError(f"need {needed} queries, only {len(all_sets)} available")

    train_sets = add_sampled_negatives(
        all_sets[:n_train_queries], covered_rows, candidates_per_query, seed=seed
    )
    eval_sets = add_sampled_negatives(
        all_sets[n_train_queries:needed], covered_rows, candidates_per_query, seed=seed
    )

    logger.info("building BM25 index over %d products", len(products))
    bm25 = BM25Index(product_texts(products))
    catalogue_text = ProductText.from_catalogue(products)

    def features_for(sets: list[CandidateSet]) -> list[FloatArray]:
        blocks = []
        for candidate in sets:
            lexical = bm25.score(candidate.query_text, candidate.product_rows)
            dense = cosine_scores(
                query_vectors[query_row[candidate.query_id]],
                product_vectors[cat2emb[candidate.product_rows]],
            ).astype(np.float64)
            blocks.append(
                extract_features(
                    candidate.query_text, candidate.product_rows, lexical, dense, catalogue_text
                )
            )
        return blocks

    logger.info(
        "extracting features for %d train / %d eval queries", len(train_sets), len(eval_sets)
    )
    train_features = features_for(train_sets)
    eval_features = features_for(eval_sets)

    return Fixture(
        train_sets=train_sets,
        eval_sets=eval_sets,
        train_features=train_features,
        eval_features=eval_features,
        train_policy=[block[:, 0] for block in train_features],  # feature 0 is BM25
        top_k=top_k,
    )


def _per_query_ndcg(scores: list[FloatArray], sets: list[CandidateSet]) -> FloatArray:
    values = np.empty(len(sets), dtype=np.float64)
    for i, (score, candidate) in enumerate(zip(scores, sets, strict=True)):
        ordered = rank_by_score(score, candidate.grades.astype(np.float64))
        values[i] = ndcg_at_k(ordered, NDCG_CUTOFF)
    return values


def _train_and_score(
    fixture: Fixture,
    labels: list[FloatArray],
    weights: list[FloatArray] | None,
    seed: int,
) -> FloatArray:
    ranker = LambdaMartRanker(RankerParams(seed=seed)).fit(
        stack_training_data(fixture.train_features, labels, weights)
    )
    return _per_query_ndcg(ranker.score_blocks(fixture.eval_features), fixture.eval_sets)


def make_estimator(name: str, true_eta: float, misspecified_eta: float) -> PropensityEstimator:
    """Build an estimator by name."""
    if name == "oracle":
        return OraclePropensity(PositionBiasModel(eta=true_eta))
    if name == "misspecified":
        return MisspecifiedPropensity(assumed_eta=misspecified_eta)
    if name == "randomization":
        return RandomizationPropensity()
    if name == "regression_em":
        return RegressionEMPropensity()
    raise ValueError(f"unknown estimator {name!r}")


def run_condition(
    fixture: Fixture,
    eta: float,
    estimator_name: str,
    ceiling_ndcg: FloatArray,
    floor_ndcg: FloatArray,
    clip: float | None = 0.05,
    misspecified_eta: float = 1.0,
    impressions_per_query: int = 10,
    seed: int = 0,
) -> ConditionResult:
    """Simulate a log at this eta, then train and evaluate arms C and D."""
    bias = PositionBiasModel(eta=eta)
    log = simulate_click_log(
        fixture.train_sets,
        fixture.train_policy,
        bias,
        ClickModel(noise=0.0),
        LogConfig(
            top_k=fixture.top_k,
            impressions_per_query=impressions_per_query,
            seed=seed,
            randomize_fraction=RANDOMIZE_FRACTION,
        ),
    )

    estimator = make_estimator(estimator_name, eta, misspecified_eta)
    estimated = estimator.estimate(log, fixture.top_k)
    error = estimation_error(estimated, bias.propensities(fixture.top_k))

    # Overwrite the logged (true) propensity with the estimate, so arm D uses
    # only what a production system could actually know.
    estimated_log = log.copy()
    estimated_log["propensity"] = estimated[log["rank"].to_numpy(dtype=np.int64) - 1]

    naive_labels = binarise(click_labels(fixture.train_sets, log))
    corrected = binarise(
        click_labels(fixture.train_sets, estimated_log, propensity_weights=True, clip=clip)
    )

    naive_ndcg = _train_and_score(fixture, naive_labels.labels, None, seed)
    corrected_ndcg = _train_and_score(fixture, corrected.labels, corrected.weights, seed)

    return ConditionResult(
        eta=eta,
        estimator=estimator_name,
        clip=clip,
        seed=seed,
        naive=ArmMetrics("naive", bootstrap_mean(naive_ndcg, seed=seed), naive_ndcg),
        corrected=ArmMetrics(
            "corrected", bootstrap_mean(corrected_ndcg, seed=seed), corrected_ndcg
        ),
        ceiling_ndcg=float(ceiling_ndcg.mean()),
        floor_ndcg=float(floor_ndcg.mean()),
        click_rate=float(log["clicked"].mean()),
        propensity_error=error,
        corrected_minus_naive=paired_bootstrap_difference(corrected_ndcg, naive_ndcg, seed=seed),
    )


def run_experiment(
    etas: list[float],
    estimators: list[str],
    seeds: list[int],
    config: DataConfig | None = None,
    n_train_queries: int = 6000,
    n_eval_queries: int = 4000,
    clip: float | None = 0.05,
    misspecified_eta: float = 1.0,
    impressions_per_query: int = 10,
) -> dict[str, object]:
    """Run the full sweep."""
    started = time.perf_counter()
    fixture = build_fixture(
        config, n_train_queries=n_train_queries, n_eval_queries=n_eval_queries
    )

    # Arms A and B are label-independent of eta, so they are computed once.
    ceiling = _train_and_score(fixture, grade_labels(fixture.train_sets).labels, None, seeds[0])
    floor = _per_query_ndcg(
        [block[:, 0] for block in fixture.eval_features], fixture.eval_sets
    )
    logger.info(
        "ceiling NDCG@10 %.4f, floor (BM25) %.4f", float(ceiling.mean()), float(floor.mean())
    )

    results: list[ConditionResult] = []
    total = len(etas) * len(estimators) * len(seeds)
    done = 0
    for eta in etas:
        for estimator_name in estimators:
            for seed in seeds:
                done += 1
                logger.info(
                    "[%d/%d] eta=%.2f estimator=%s seed=%d", done, total, eta, estimator_name, seed
                )
                results.append(
                    run_condition(
                        fixture,
                        eta,
                        estimator_name,
                        ceiling,
                        floor,
                        clip=clip,
                        misspecified_eta=misspecified_eta,
                        impressions_per_query=impressions_per_query,
                    )
                )

    return {
        "settings": {
            "n_train_queries": len(fixture.train_sets),
            "n_eval_queries": len(fixture.eval_sets),
            "top_k": fixture.top_k,
            "clip": clip,
            "misspecified_eta": misspecified_eta,
            "impressions_per_query": impressions_per_query,
            "randomize_fraction": RANDOMIZE_FRACTION,
            "etas": etas,
            "estimators": estimators,
            "seeds": seeds,
        },
        "arms": {
            "ceiling_ndcg@10": float(ceiling.mean()),
            "floor_ndcg@10": float(floor.mean()),
        },
        "conditions": [r.to_dict() for r in results],
        "runtime_seconds": time.perf_counter() - started,
    }


def summarise(report: dict[str, object]) -> pd.DataFrame:
    """Aggregate across seeds: mean recovery plus its spread."""
    conditions: list[dict[str, object]] = report["conditions"]  # type: ignore[assignment]
    frame = pd.DataFrame.from_records(conditions)
    aggregated: pd.DataFrame = (
        frame.groupby(["eta", "estimator"], sort=True)
        .agg(
            naive=("naive_ndcg@10", "mean"),
            corrected=("corrected_ndcg@10", "mean"),
            delta=("corrected_minus_naive", "mean"),
            recovery=("recovery_fraction", "mean"),
            recovery_sd=("recovery_fraction", "std"),
            click_rate=("click_rate", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M4 four-arm experiment.")
    parser.add_argument("--etas", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=["oracle", "misspecified", "randomization", "regression_em"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-queries", type=int, default=6000)
    parser.add_argument("--eval-queries", type=int, default=4000)
    parser.add_argument("--impressions", type=int, default=10)
    parser.add_argument("--clip", type=float, default=0.05)
    parser.add_argument("--misspecified-eta", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/experiment.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run_experiment(
        etas=args.etas,
        estimators=args.estimators,
        seeds=args.seeds,
        n_train_queries=args.train_queries,
        n_eval_queries=args.eval_queries,
        clip=args.clip,
        misspecified_eta=args.misspecified_eta,
        impressions_per_query=args.impressions,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    arms: dict[str, float] = report["arms"]  # type: ignore[assignment]
    runtime: float = report["runtime_seconds"]  # type: ignore[assignment]
    print(f"\nceiling (true grades) NDCG@10 = {arms['ceiling_ndcg@10']:.4f}")
    print(f"floor   (BM25 only)   NDCG@10 = {arms['floor_ndcg@10']:.4f}\n")
    print(summarise(report).to_string(index=False))
    print(f"\nwrote {args.out}  ({runtime:.0f}s)")


if __name__ == "__main__":
    main()


__all__ = [
    "ConditionResult",
    "Fixture",
    "build_fixture",
    "run_condition",
    "run_experiment",
    "summarise",
]
