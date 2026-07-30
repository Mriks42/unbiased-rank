"""Propensity estimation.

IPS correction needs `p_k` -- the probability an item at rank `k` was examined.
The simulation knows the true value; a production system does not, and must
estimate it. That gap is the interesting part of this project: "correction works
when you know the bias exactly" is unsurprising, while "correction breaks past
*this* much estimation error" is useful.

Four estimators, chosen to span the practical spectrum:

* **Oracle** -- the true curve. An upper bound on what any correction could
  achieve, not a deployable method.
* **Misspecified** -- the true functional form with the wrong severity. Models
  the realistic case of a team assuming a standard bias curve.
* **Randomization** -- from deliberately shuffled impressions. Principled, and
  what a company willing to degrade a slice of traffic can actually do.
* **RegressionEM** -- from ordinary logs with no intervention. Cheapest to
  deploy and the most likely to be wrong.

All return a length-`n_positions` array, normalised so `p_1 = 1`. Normalisation
matters: IPS is invariant to a constant scaling of the propensities (it changes
every weight by the same factor), so only the *shape* is identifiable from data.
Pinning `p_1 = 1` makes estimates comparable to the true curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from unbiased_rank.simulation.position_bias import PositionBiasModel, propensity_curve

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

# Floor applied to every estimate. A zero or negative propensity would produce an
# infinite IPS weight and destroy training.
MIN_PROPENSITY: Final[float] = 1e-3


class PropensityEstimator(Protocol):
    """Estimates the examination curve from a click log."""

    name: str

    def estimate(self, log: pd.DataFrame, n_positions: int) -> FloatArray: ...


def _normalise(curve: FloatArray) -> FloatArray:
    """Scale so p_1 = 1 and floor the result.

    Only the shape of the curve is identifiable from click data, because scaling
    every propensity by a constant scales every IPS weight equally and leaves
    the trained model unchanged.
    """
    floored: FloatArray = np.maximum(np.asarray(curve, dtype=np.float64), MIN_PROPENSITY)
    if floored.size == 0:
        return floored
    scaled: FloatArray = np.maximum(floored / floored[0], MIN_PROPENSITY)
    return scaled


@dataclass
class OraclePropensity:
    """The true curve. Upper bound, not a deployable method."""

    bias: PositionBiasModel
    name: str = "oracle"

    def estimate(self, log: pd.DataFrame, n_positions: int) -> FloatArray:
        return _normalise(self.bias.propensities(n_positions))


@dataclass
class MisspecifiedPropensity:
    """Right functional form, wrong severity.

    The realistic failure mode: a team adopts a published bias curve without
    measuring their own. `assumed_eta` is what they believe; the log was
    generated under something else.
    """

    assumed_eta: float
    name: str = "misspecified"

    def estimate(self, log: pd.DataFrame, n_positions: int) -> FloatArray:
        return _normalise(propensity_curve(n_positions, self.assumed_eta))


@dataclass
class RandomizationPropensity:
    """Estimate from randomly-ordered impressions (intervention harvesting).

    In randomised impressions, placement is independent of relevance, so
    expected relevance is the same at every rank and

        CTR(k) = p_k * E[relevance]

    Dividing by CTR(1) cancels the unknown constant and recovers p_k / p_1.

    This is the principled option, and it costs real user experience: the
    randomised slice shows people deliberately worse results.
    """

    name: str = "randomization"

    def estimate(self, log: pd.DataFrame, n_positions: int) -> FloatArray:
        if "randomized" not in log.columns:
            raise KeyError(
                "log has no 'randomized' column; simulate with "
                "LogConfig(randomize_fraction > 0) to enable this estimator"
            )
        randomized = log[log["randomized"]]
        if randomized.empty:
            raise ValueError(
                "no randomised impressions in the log. This estimator needs an "
                "intervention slice; without it the propensity is not identifiable "
                "from CTR, because rank and relevance are confounded."
            )

        ctr = randomized.groupby("rank", sort=True)["clicked"].mean()
        curve = np.full(n_positions, np.nan, dtype=np.float64)
        ranks = np.asarray(ctr.index, dtype=np.int64) - 1
        values = np.asarray(ctr.to_numpy(), dtype=np.float64)
        in_range = (ranks >= 0) & (ranks < n_positions)
        curve[ranks[in_range]] = values[in_range]

        # Ranks with no randomised data fall back to the observed trend rather
        # than to an arbitrary constant.
        if np.isnan(curve).any():
            observed = ~np.isnan(curve)
            if not observed.any():
                raise ValueError("no usable randomised observations")
            curve = np.interp(
                np.arange(n_positions), np.flatnonzero(observed), curve[observed]
            )
        return _normalise(curve)


@dataclass
class RegressionEMPropensity:
    """Estimate from ordinary logs via Expectation-Maximisation.

    Fits P(click) = p_k * r_qd with no intervention, alternating between:

    * **E-step** -- for a non-click, infer the probability the item was examined
      anyway: P(E=1 | C=0) = p_k (1 - r) / (1 - p_k r). A click implies
      examination, so those contribute 1.
    * **M-step** -- re-estimate p_k as the mean inferred examination at rank k,
      and r_qd as the click rate among impressions where examination is inferred.

    Cheapest to deploy and the most likely to be wrong: relevance and position
    are only weakly separable from observational data, so the fit is sensitive
    to initialisation and to how often each item is shown at different ranks.
    """

    max_iterations: int = 30
    tolerance: float = 1e-4
    name: str = "regression_em"

    def estimate(self, log: pd.DataFrame, n_positions: int) -> FloatArray:
        ranks = log["rank"].to_numpy(dtype=np.int64) - 1
        clicked = log["clicked"].to_numpy(dtype=bool)

        # Item identity: (query, product). Relevance is estimated per item.
        item_codes, _ = pd.factorize(
            pd.Series(list(zip(log["query_id"], log["product_row"], strict=True)))
        )
        item_codes = np.asarray(item_codes, dtype=np.int64)
        n_items = int(item_codes.max()) + 1 if item_codes.size else 0
        if n_items == 0:
            raise ValueError("empty log")

        propensity = np.full(n_positions, 0.5, dtype=np.float64)
        propensity[0] = 1.0
        relevance = np.full(n_items, 0.2, dtype=np.float64)

        for iteration in range(self.max_iterations):
            p_row = propensity[ranks]
            r_row = relevance[item_codes]

            # E-step.
            denominator = np.maximum(1.0 - p_row * r_row, 1e-12)
            examined = np.where(clicked, 1.0, p_row * (1.0 - r_row) / denominator)
            relevant = np.where(clicked, 1.0, r_row * (1.0 - p_row) / denominator)

            # M-step.
            new_propensity = _grouped_mean(examined, ranks, n_positions, fallback=propensity)
            new_relevance = _grouped_mean(relevant, item_codes, n_items, fallback=relevance)

            shift = float(np.max(np.abs(new_propensity - propensity)))
            propensity, relevance = new_propensity, new_relevance
            if shift < self.tolerance:
                logger.debug("regression-EM converged after %d iterations", iteration + 1)
                break

        return _normalise(propensity)


def _grouped_mean(
    values: FloatArray, groups: npt.NDArray[np.int64], n_groups: int, fallback: FloatArray
) -> FloatArray:
    """Mean of `values` per group, falling back where a group has no data."""
    totals = np.bincount(groups, weights=values, minlength=n_groups)
    counts = np.bincount(groups, minlength=n_groups)
    result = np.where(counts > 0, totals / np.maximum(counts, 1), fallback)
    return np.clip(result, 1e-6, 1.0)


def estimation_error(estimated: FloatArray, truth: FloatArray) -> dict[str, float]:
    """Compare an estimated curve against the truth.

    Reported alongside downstream ranking quality so the two can be related:
    a large estimation error that barely moves NDCG is a different finding from
    a small one that wrecks it.
    """
    estimated = _normalise(estimated)
    truth = _normalise(truth)
    absolute = np.abs(estimated - truth)
    # Log-space error, since propensities span orders of magnitude and an
    # absolute gap at rank 20 means far more than the same gap at rank 2.
    log_error = np.abs(np.log(estimated) - np.log(truth))
    return {
        "mean_absolute_error": float(absolute.mean()),
        "max_absolute_error": float(absolute.max()),
        "mean_log_error": float(log_error.mean()),
    }


__all__ = [
    "MIN_PROPENSITY",
    "MisspecifiedPropensity",
    "OraclePropensity",
    "PropensityEstimator",
    "RandomizationPropensity",
    "RegressionEMPropensity",
    "estimation_error",
]
