"""Label construction for the four experimental arms.

The arms differ *only* in where their training labels and weights come from.
Everything else -- features, model capacity, seeds, evaluation -- is held fixed,
so any measured difference is attributable to the label source.

| Arm | Labels | Weights |
|---|---|---|
| A. ceiling | true ESCI grades | none |
| B. floor | (no training; BM25 score used directly) | n/a |
| C. naive | click counts from the log | none |
| D. corrected | click counts from the log | 1 / propensity |

Arm D is the object of study. Arm A bounds what any correction could achieve;
arm C is what a team gets by training on logs without thinking about position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from unbiased_rank.ranking.candidates import CandidateSet

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LabelBlocks:
    """Per-query labels and optional weights, aligned to candidate order."""

    labels: list[FloatArray]
    weights: list[FloatArray] | None = None


def grade_labels(candidate_sets: list[CandidateSet]) -> LabelBlocks:
    """Arm A: true graded relevance. The ceiling."""
    return LabelBlocks(labels=[c.grades.astype(np.float64) for c in candidate_sets])


def click_labels(
    candidate_sets: list[CandidateSet],
    log: pd.DataFrame,
    propensity_weights: bool = False,
    clip: float | None = None,
) -> LabelBlocks:
    """Arms C and D: labels from the click log.

    Args:
        candidate_sets: candidate sets, defining row order per query.
        log: click log from `simulation.logger.simulate_click_log`.
        propensity_weights: when True, weight each row by 1/propensity (arm D).
        clip: propensity floor. Unclipped IPS is unbiased but a few
            low-propensity rows can dominate; clipping trades that variance for
            bias. Swept in the experiment, and both the mean *and* the variance
            are reported because reporting only the mean hides the trade-off.

    Notes:
        Candidates never displayed get label 0 and weight 1. That is the honest
        representation of a real log: an item that was never shown produced no
        evidence, and pretending otherwise would leak information the logging
        policy never revealed.
    """
    observed = (
        log.groupby(["query_id", "product_row"], sort=False)
        .agg(clicks=("clicked", "sum"), propensity=("propensity", "mean"))
        .reset_index()
    )
    click_lookup: dict[tuple[int, int], tuple[float, float]] = {
        (int(q), int(p)): (float(c), float(pr))
        for q, p, c, pr in zip(
            observed["query_id"],
            observed["product_row"],
            observed["clicks"],
            observed["propensity"],
            strict=True,
        )
    }

    label_blocks: list[FloatArray] = []
    weight_blocks: list[FloatArray] = []
    for candidate in candidate_sets:
        labels = np.zeros(len(candidate), dtype=np.float64)
        weights = np.ones(len(candidate), dtype=np.float64)
        for i, row in enumerate(candidate.product_rows.tolist()):
            found = click_lookup.get((candidate.query_id, int(row)))
            if found is None:
                continue
            clicks, propensity = found
            labels[i] = clicks
            if propensity_weights:
                floor = propensity if clip is None else max(propensity, clip)
                weights[i] = 1.0 / max(floor, 1e-12)
        label_blocks.append(labels)
        weight_blocks.append(weights)

    return LabelBlocks(
        labels=label_blocks, weights=weight_blocks if propensity_weights else None
    )


def binarise(blocks: LabelBlocks, threshold: float = 1.0) -> LabelBlocks:
    """Collapse click counts to 0/1.

    LambdaMART treats labels as graded gains, so raw click counts from many
    impressions would make a 5-click item worth 2^5 - 1 = 31 times a 1-click
    item under exponential gain. Binarising keeps the label scale comparable to
    the graded arm.
    """
    return LabelBlocks(
        labels=[(block >= threshold).astype(np.float64) for block in blocks.labels],
        weights=blocks.weights,
    )


__all__ = ["LabelBlocks", "binarise", "click_labels", "grade_labels"]
