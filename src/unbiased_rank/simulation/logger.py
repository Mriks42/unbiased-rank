"""Impression and click-log generation.

Simulates a production search system serving results and recording clicks:

1. A *logging policy* (by default BM25 -- a plausible pre-ML production ranker)
   orders each query's candidates.
2. The top `top_k` are "displayed".
3. Clicks are sampled under the position-bias and click models.
4. Everything is recorded, including the propensity at each displayed rank.

The propensity is logged because it is what IPS needs. A real system would have
to estimate it; recording the true value lets the experiment separate "does
correction work at all" (oracle propensities) from "does it survive
estimation error" (the interesting question).

Clicks are simulated, never observed. Every conclusion is conditional on these
models being a reasonable description of user behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from unbiased_rank.ranking.candidates import CandidateSet
from unbiased_rank.simulation.click_model import ClickModel
from unbiased_rank.simulation.position_bias import PositionBiasModel

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

DEFAULT_TOP_K: Final[int] = 20
IMPRESSIONS_SWEEP: Final[tuple[int, ...]] = (1, 5, 20)


@dataclass(frozen=True)
class LogConfig:
    """Simulation settings for one click log."""

    top_k: int = DEFAULT_TOP_K
    impressions_per_query: int = 5
    seed: int = 0
    randomize_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        if self.impressions_per_query < 1:
            raise ValueError(
                f"impressions_per_query must be at least 1, got {self.impressions_per_query}"
            )
        if not 0.0 <= self.randomize_fraction <= 1.0:
            raise ValueError(
                f"randomize_fraction must be in [0, 1], got {self.randomize_fraction}"
            )


def display_order(
    candidate: CandidateSet, policy_scores: FloatArray, top_k: int
) -> tuple[IntArray, IntArray]:
    """Apply the logging policy and truncate to the displayed window.

    Returns:
        (product_rows, grades) for the displayed items, in display order.
    """
    order = np.lexsort((np.arange(policy_scores.size), -policy_scores))[:top_k]
    return candidate.product_rows[order], candidate.grades[order]


def simulate_click_log(
    candidate_sets: list[CandidateSet],
    policy_scores: list[FloatArray],
    bias: PositionBiasModel,
    clicks: ClickModel,
    config: LogConfig | None = None,
) -> pd.DataFrame:
    """Generate a click log over the given candidate sets.

    Returns a frame with one row per (impression, displayed rank):

        query_id, impression, rank, product_row, grade, propensity, clicked,
        randomized

    `grade` is retained purely for diagnostics and evaluation. Training arms that
    consume clicks must not read it -- that would be the leak the whole
    experiment exists to avoid.

    When `randomize_fraction > 0`, that share of impressions displays the
    candidates in a *random* order instead of the policy's. This is intervention
    harvesting: because document placement is then independent of relevance,
    click-through rate by rank becomes proportional to the propensity alone,
    which is what makes propensity estimation possible without knowing the truth.
    Real systems pay for this in degraded user experience, which is why the
    fraction is small.
    """
    cfg = config if config is not None else LogConfig()
    rng = np.random.default_rng(cfg.seed)

    frames: list[pd.DataFrame] = []
    for candidate, scores in zip(candidate_sets, policy_scores, strict=True):
        rows, grades = display_order(candidate, scores, cfg.top_k)
        propensities = bias.propensities(rows.size)
        ranks = np.arange(1, rows.size + 1, dtype=np.int64)

        for impression in range(cfg.impressions_per_query):
            randomized = bool(rng.random() < cfg.randomize_fraction)
            if randomized:
                shuffle = rng.permutation(rows.size)
                shown_rows, shown_grades = rows[shuffle], grades[shuffle]
            else:
                shown_rows, shown_grades = rows, grades

            clicked = clicks.sample_clicks(shown_grades, propensities, rng)
            frames.append(
                pd.DataFrame(
                    {
                        "query_id": np.full(shown_rows.size, candidate.query_id, dtype=np.int64),
                        "impression": np.full(shown_rows.size, impression, dtype=np.int64),
                        "rank": ranks,
                        "product_row": shown_rows,
                        "grade": shown_grades,
                        "propensity": propensities,
                        "clicked": clicked,
                        "randomized": np.full(shown_rows.size, randomized, dtype=bool),
                    }
                )
            )

    log = pd.concat(frames, ignore_index=True) if frames else _empty_log()
    logger.info(
        "simulated %d impressions over %d queries: %d rows, %.4f click rate (eta=%.2f, noise=%.2f)",
        cfg.impressions_per_query,
        len(candidate_sets),
        len(log),
        float(log["clicked"].mean()) if len(log) else 0.0,
        bias.eta,
        clicks.noise,
    )
    return log


def observed_click_rate_by_rank(log: pd.DataFrame) -> pd.DataFrame:
    """Empirical click-through rate per displayed rank.

    Used to calibrate the simulator: with relevance roughly balanced across
    ranks under a fixed policy, the CTR curve should track the propensity curve.
    A mismatch means the simulation is not doing what the model says.
    """
    grouped = log.groupby("rank", sort=True).agg(
        clicks=("clicked", "sum"),
        impressions=("clicked", "size"),
        propensity=("propensity", "first"),
    )
    grouped["ctr"] = grouped["clicks"] / grouped["impressions"]
    return grouped.reset_index()


def clicked_pairs(log: pd.DataFrame) -> pd.DataFrame:
    """Collapse a log to per-(query, product) click counts and exposure.

    This is the shape training consumes: how often an item was shown, how often
    it was clicked, and the propensity it was shown at. Grades are deliberately
    dropped here so a clicks-trained arm cannot accidentally read them.
    """
    return (
        log.groupby(["query_id", "product_row"], sort=True)
        .agg(
            impressions=("clicked", "size"),
            clicks=("clicked", "sum"),
            mean_propensity=("propensity", "mean"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
    )


def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": pd.Series(dtype="int64"),
            "impression": pd.Series(dtype="int64"),
            "rank": pd.Series(dtype="int64"),
            "product_row": pd.Series(dtype="int64"),
            "grade": pd.Series(dtype="int64"),
            "propensity": pd.Series(dtype="float64"),
            "clicked": pd.Series(dtype="bool"),
            "randomized": pd.Series(dtype="bool"),
        }
    )


__all__ = [
    "DEFAULT_TOP_K",
    "IMPRESSIONS_SWEEP",
    "LogConfig",
    "clicked_pairs",
    "display_order",
    "observed_click_rate_by_rank",
    "simulate_click_log",
]
