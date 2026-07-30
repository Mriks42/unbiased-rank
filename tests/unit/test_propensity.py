"""Tests for propensity estimation.

The estimators are checked for *recovery*: given a log generated under a known
curve, does the estimator get close to it? An estimator that returns
plausible-looking numbers without tracking the truth would silently produce a
correction that corrects nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unbiased_rank.propensity.estimators import (
    MIN_PROPENSITY,
    MisspecifiedPropensity,
    OraclePropensity,
    RandomizationPropensity,
    RegressionEMPropensity,
    estimation_error,
)
from unbiased_rank.ranking.candidates import CandidateSet
from unbiased_rank.simulation.click_model import ClickModel
from unbiased_rank.simulation.logger import LogConfig, simulate_click_log
from unbiased_rank.simulation.position_bias import PositionBiasModel


def _log(
    eta: float,
    n_queries: int = 800,
    n_candidates: int = 10,
    impressions: int = 20,
    randomize_fraction: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a click log under a known propensity curve."""
    rng = np.random.default_rng(seed)
    sets = [
        CandidateSet(
            query_id=q,
            query_text=f"q{q}",
            product_rows=np.arange(n_candidates, dtype=np.int64) + q * n_candidates,
            grades=rng.integers(0, 4, size=n_candidates).astype(np.int64),
        )
        for q in range(n_queries)
    ]
    policy = [rng.random(n_candidates) for _ in sets]
    return simulate_click_log(
        sets,
        policy,
        PositionBiasModel(eta=eta),
        ClickModel(noise=0.0),
        LogConfig(
            top_k=n_candidates,
            impressions_per_query=impressions,
            seed=seed,
            randomize_fraction=randomize_fraction,
        ),
    )


class TestOracle:
    def test_returns_the_true_curve(self) -> None:
        bias = PositionBiasModel(eta=1.0)
        got = OraclePropensity(bias).estimate(_log(1.0, n_queries=10), 10)
        assert np.allclose(got, bias.propensities(10))

    def test_is_normalised_to_one_at_rank_one(self) -> None:
        got = OraclePropensity(PositionBiasModel(eta=2.0)).estimate(_log(2.0, n_queries=10), 8)
        assert got[0] == pytest.approx(1.0)


class TestMisspecified:
    def test_wrong_eta_gives_a_wrong_curve(self) -> None:
        truth = PositionBiasModel(eta=1.5).propensities(10)
        got = MisspecifiedPropensity(assumed_eta=1.0).estimate(_log(1.5, n_queries=10), 10)

        assert got[0] == pytest.approx(1.0)
        assert not np.allclose(got, truth)
        # Assuming milder bias than reality means over-estimating deep propensities.
        assert np.all(got[1:] > truth[1:])

    def test_correct_eta_recovers_the_truth(self) -> None:
        truth = PositionBiasModel(eta=1.0).propensities(10)
        got = MisspecifiedPropensity(assumed_eta=1.0).estimate(_log(1.0, n_queries=10), 10)
        assert np.allclose(got, truth)

    def test_error_grows_with_the_size_of_the_mistake(self) -> None:
        truth = PositionBiasModel(eta=1.0).propensities(10)
        log = _log(1.0, n_queries=10)

        small = estimation_error(MisspecifiedPropensity(1.2).estimate(log, 10), truth)
        large = estimation_error(MisspecifiedPropensity(2.0).estimate(log, 10), truth)

        assert large["mean_log_error"] > small["mean_log_error"]


class TestRandomization:
    def test_recovers_the_curve_from_randomised_impressions(self) -> None:
        """The estimator's core claim: with random placement, CTR by rank is
        proportional to the propensity."""
        truth = PositionBiasModel(eta=1.0).propensities(10)
        log = _log(1.0, n_queries=1500, impressions=20, randomize_fraction=1.0, seed=3)

        got = RandomizationPropensity().estimate(log, 10)
        error = estimation_error(got, truth)

        assert error["mean_absolute_error"] < 0.03

    def test_works_from_a_small_randomised_slice(self) -> None:
        """Realistic deployment: only a fraction of traffic is randomised."""
        truth = PositionBiasModel(eta=1.0).propensities(10)
        log = _log(1.0, n_queries=2000, impressions=20, randomize_fraction=0.1, seed=4)

        error = estimation_error(RandomizationPropensity().estimate(log, 10), truth)
        assert error["mean_absolute_error"] < 0.06

    def test_requires_randomised_impressions(self) -> None:
        """Without an intervention slice the propensity is not identifiable:
        rank and relevance are confounded."""
        log = _log(1.0, n_queries=100, randomize_fraction=0.0)
        with pytest.raises(ValueError, match="no randomised impressions"):
            RandomizationPropensity().estimate(log, 10)

    def test_requires_the_randomized_column(self) -> None:
        log = _log(1.0, n_queries=50, randomize_fraction=0.5).drop(columns=["randomized"])
        with pytest.raises(KeyError, match="randomized"):
            RandomizationPropensity().estimate(log, 10)

    def test_detects_severity_differences(self) -> None:
        mild = RandomizationPropensity().estimate(
            _log(0.5, n_queries=1200, randomize_fraction=1.0, seed=5), 10
        )
        severe = RandomizationPropensity().estimate(
            _log(2.0, n_queries=1200, randomize_fraction=1.0, seed=5), 10
        )
        assert severe[-1] < mild[-1]


class TestRegressionEM:
    def test_produces_a_decreasing_curve_under_bias(self) -> None:
        got = RegressionEMPropensity().estimate(_log(1.0, n_queries=800, seed=6), 10)
        assert got[0] == pytest.approx(1.0)
        # Monotonicity is not enforced by the estimator, so check the trend.
        assert got[-1] < got[0]

    def test_ranks_severity_correctly(self) -> None:
        """Even if the curve is imprecise, harsher bias must read as harsher."""
        mild = RegressionEMPropensity().estimate(_log(0.5, n_queries=800, seed=7), 10)
        severe = RegressionEMPropensity().estimate(_log(2.0, n_queries=800, seed=7), 10)
        assert severe[-1] < mild[-1]

    def test_output_is_bounded(self) -> None:
        got = RegressionEMPropensity().estimate(_log(1.5, n_queries=400, seed=8), 10)
        assert np.all(got >= MIN_PROPENSITY)
        assert np.all(got <= 1.0 + 1e-9)

    def test_rejects_empty_log(self) -> None:
        empty = _log(1.0, n_queries=10).iloc[0:0]
        with pytest.raises(ValueError, match="empty log"):
            RegressionEMPropensity().estimate(empty, 10)


class TestEstimationError:
    def test_zero_error_for_identical_curves(self) -> None:
        truth = PositionBiasModel(eta=1.0).propensities(10)
        error = estimation_error(truth, truth)
        assert error["mean_absolute_error"] == pytest.approx(0.0)
        assert error["mean_log_error"] == pytest.approx(0.0)

    def test_scaling_invariance(self) -> None:
        """IPS is invariant to a constant scaling of the propensities, so the
        error metric must be too -- otherwise it would penalise estimates that
        are functionally identical."""
        truth = PositionBiasModel(eta=1.0).propensities(10)
        error = estimation_error(truth * 0.5, truth)
        assert error["mean_absolute_error"] == pytest.approx(0.0, abs=1e-9)

    def test_log_error_weights_deep_ranks(self) -> None:
        """An absolute gap at rank 20 matters more than the same gap at rank 2,
        because propensities span orders of magnitude."""
        truth = PositionBiasModel(eta=1.0).propensities(20)
        shallow = truth.copy()
        shallow[1] += 0.05
        deep = truth.copy()
        deep[19] += 0.05

        assert (
            estimation_error(deep, truth)["mean_log_error"]
            > estimation_error(shallow, truth)["mean_log_error"]
        )
