"""Calibrate the click simulator against real ESCI candidate sets.

The unit tests assert calibration under a *random* logging policy, which
isolates the propensity curve from relevance. This script runs the
configuration M4 will actually use -- BM25 as the logging policy -- and records
what that looks like, because the result demonstrates the core difficulty the
project exists to address.

Run:

    python -m unbiased_rank.experiments.calibrate_simulator
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from unbiased_rank.config import DataConfig, load_data_config
from unbiased_rank.indexing.catalog import load_product_embeddings, product_texts
from unbiased_rank.indexing.lexical import BM25Index
from unbiased_rank.ranking.candidates import (
    add_sampled_negatives,
    build_candidate_sets,
    build_product_row_lookup,
)
from unbiased_rank.simulation.click_model import ClickModel
from unbiased_rank.simulation.logger import (
    LogConfig,
    observed_click_rate_by_rank,
    simulate_click_log,
)
from unbiased_rank.simulation.position_bias import ETA_SWEEP, PositionBiasModel

logger = logging.getLogger(__name__)


def calibrate(
    config: DataConfig | None = None,
    n_queries: int = 4000,
    candidates_per_query: int = 100,
    top_k: int = 20,
    impressions_per_query: int = 5,
    seed: int = 7,
) -> dict[str, object]:
    """Measure CTR-by-rank across the eta sweep under a BM25 logging policy."""
    cfg = config if config is not None else load_data_config()
    products = pd.read_parquet(cfg.interim_dir / "products.parquet")
    examples = pd.read_parquet(cfg.interim_dir / "examples.parquet")
    _, covered = load_product_embeddings(cfg)

    lookup = build_product_row_lookup(products["product_id"].to_numpy(dtype=object))
    sets = build_candidate_sets(examples, lookup, split="test")[:n_queries]
    sets = add_sampled_negatives(sets, covered, target_size=candidates_per_query, seed=0)

    logger.info("building BM25 logging policy over %d products", len(products))
    bm25 = BM25Index(product_texts(products))
    policy = [bm25.score(c.query_text, c.product_rows) for c in sets]

    sweep: list[dict[str, float]] = []
    curves: dict[str, list[dict[str, float]]] = {}
    for eta in ETA_SWEEP:
        log = simulate_click_log(
            sets,
            policy,
            PositionBiasModel(eta=eta),
            ClickModel(noise=0.0),
            LogConfig(top_k=top_k, impressions_per_query=impressions_per_query, seed=seed),
        )
        ctr = observed_click_rate_by_rank(log)
        implied = ctr["ctr"] / ctr["propensity"]

        sweep.append(
            {
                "eta": float(eta),
                "overall_click_rate": float(log["clicked"].mean()),
                "ctr_rank_1": float(ctr.iloc[0]["ctr"]),
                "ctr_rank_last": float(ctr.iloc[-1]["ctr"]),
                "top_to_bottom_ratio": float(ctr.iloc[0]["ctr"] / max(ctr.iloc[-1]["ctr"], 1e-12)),
                "implied_relevance_cv": float(implied.std() / implied.mean()),
            }
        )
        curves[f"eta_{eta}"] = [
            {
                "rank": int(row["rank"]),
                "propensity": float(row["propensity"]),
                "ctr": float(row["ctr"]),
                "implied_relevance": float(row["ctr"] / row["propensity"]),
            }
            for _, row in ctr.iterrows()
        ]

    return {
        "settings": {
            "n_queries": len(sets),
            "candidates_per_query": candidates_per_query,
            "top_k": top_k,
            "impressions_per_query": impressions_per_query,
            "logging_policy": "bm25",
            "click_noise": 0.0,
            "seed": seed,
        },
        "sweep": sweep,
        "ctr_curves": curves,
        "interpretation": (
            "implied_relevance = ctr / propensity declines with rank because BM25 is a "
            "competent policy: it places genuinely relevant products first, and the "
            "sampled negatives sink to the bottom. That decline is real relevance "
            "signal, not a simulator defect -- the unit tests verify a flat ratio under "
            "a random policy. It is also precisely why naive click training fails: "
            "observed CTR conflates position bias with relevance, and no amount of "
            "click data separates them without knowing or estimating the propensities."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the click simulator.")
    parser.add_argument("--n-queries", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=Path("outputs/simulator_calibration.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = calibrate(n_queries=args.n_queries)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'eta':>5} {'clickrate':>10} {'ctr@1':>8} {'ctr@last':>9} {'ratio':>9} {'CV':>7}")
    sweep_rows: list[dict[str, float]] = report["sweep"]  # type: ignore[assignment]
    for row in sweep_rows:
        print(
            f"{row['eta']:>5.1f} {row['overall_click_rate']:>10.4f} {row['ctr_rank_1']:>8.4f} "
            f"{row['ctr_rank_last']:>9.4f} {row['top_to_bottom_ratio']:>9.2f} "
            f"{row['implied_relevance_cv']:>7.4f}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()


__all__ = ["calibrate"]
