"""Tests for the click simulator.

The simulator is the foundation of M4: if it does not behave as the model says,
every downstream number is wrong while still looking entirely plausible. So
these tests check *calibration* -- that simulated behaviour matches the stated
probabilities -- not merely that functions return arrays of the right shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from unbiased_rank.ranking.candidates import CandidateSet
from unbiased_rank.simulation.click_model import (
    COMPRESSED_RELEVANCE,
    DEFAULT_RELEVANCE,
    ClickModel,
)
from unbiased_rank.simulation.logger import (
    LogConfig,
    clicked_pairs,
    display_order,
    observed_click_rate_by_rank,
    simulate_click_log,
)
from unbiased_rank.simulation.position_bias import (
    ETA_SWEEP,
    PositionBiasModel,
    propensity_curve,
)


class TestPropensityCurve:
    def test_eta_zero_is_unbiased(self) -> None:
        """The control condition: every position examined equally."""
        assert np.allclose(propensity_curve(10, eta=0.0), 1.0)

    def test_first_rank_is_always_one(self) -> None:
        for eta in ETA_SWEEP:
            assert propensity_curve(5, eta)[0] == pytest.approx(1.0)

    def test_propensity_decreases_with_rank(self) -> None:
        curve = propensity_curve(10, eta=1.0)
        assert np.all(np.diff(curve) < 0)

    def test_larger_eta_biases_harder(self) -> None:
        mild = propensity_curve(10, eta=0.5)
        severe = propensity_curve(10, eta=2.0)
        assert np.all(severe[1:] < mild[1:])

    def test_hand_computed_values(self) -> None:
        """eta=1 gives the harmonic sequence 1, 1/2, 1/3, ..."""
        assert np.allclose(propensity_curve(4, eta=1.0), [1.0, 0.5, 1 / 3, 0.25])

    def test_propensities_stay_in_unit_interval(self) -> None:
        for eta in ETA_SWEEP:
            curve = propensity_curve(50, eta)
            assert np.all((curve > 0.0) & (curve <= 1.0))

    def test_zero_positions_gives_empty(self) -> None:
        assert propensity_curve(0, eta=1.0).size == 0

    def test_negative_inputs_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            propensity_curve(-1, eta=1.0)
        with pytest.raises(ValueError, match="eta must be non-negative"):
            propensity_curve(5, eta=-0.5)


class TestPositionBiasModel:
    def test_is_unbiased_flag(self) -> None:
        assert PositionBiasModel(eta=0.0).is_unbiased
        assert not PositionBiasModel(eta=1.0).is_unbiased

    def test_inverse_weights_invert_propensities(self) -> None:
        model = PositionBiasModel(eta=1.0)
        weights = model.inverse_propensity_weights(5)
        assert np.allclose(weights * model.propensities(5), 1.0)

    def test_unbiased_model_gives_unit_weights(self) -> None:
        assert np.allclose(PositionBiasModel(eta=0.0).inverse_propensity_weights(8), 1.0)

    def test_clipping_bounds_the_weights(self) -> None:
        """Clipping is the bias-variance knob; it must actually cap weights."""
        model = PositionBiasModel(eta=2.0)
        unclipped = model.inverse_propensity_weights(20)
        clipped = model.inverse_propensity_weights(20, clip=0.05)

        assert unclipped.max() > clipped.max()
        assert clipped.max() == pytest.approx(20.0)  # 1 / 0.05

    def test_invalid_clip_rejected(self) -> None:
        with pytest.raises(ValueError, match="clip must be in"):
            PositionBiasModel().inverse_propensity_weights(5, clip=0.0)
        with pytest.raises(ValueError, match="clip must be in"):
            PositionBiasModel().inverse_propensity_weights(5, clip=1.5)


class TestClickModel:
    def test_relevance_increases_with_grade(self) -> None:
        model = ClickModel()
        relevance = model.relevance(np.array([0, 1, 2, 3]))
        assert np.all(np.diff(relevance) > 0)

    def test_relevance_matches_the_configured_table(self) -> None:
        model = ClickModel()
        assert np.allclose(
            model.relevance(np.array([0, 3])), [DEFAULT_RELEVANCE[0], DEFAULT_RELEVANCE[3]]
        )

    def test_click_probability_factorises(self) -> None:
        """The defining PBM property: P(click) = p_k * r(grade)."""
        model = ClickModel(noise=0.0)
        grades = np.array([3, 3])
        propensities = np.array([1.0, 0.5])
        probability = model.click_probability(grades, propensities)

        assert probability[0] == pytest.approx(DEFAULT_RELEVANCE[3])
        assert probability[1] == pytest.approx(DEFAULT_RELEVANCE[3] * 0.5)

    def test_noise_pulls_probability_toward_a_coin_flip(self) -> None:
        grades = np.array([3])
        propensities = np.array([1.0])
        clean = ClickModel(noise=0.0).click_probability(grades, propensities)[0]
        noisy = ClickModel(noise=1.0).click_probability(grades, propensities)[0]

        assert clean == pytest.approx(0.90)
        assert noisy == pytest.approx(0.50)

    def test_sampled_click_rate_matches_probability(self) -> None:
        """Calibration: sampling must reproduce the stated probability."""
        model = ClickModel(noise=0.0)
        rng = np.random.default_rng(0)
        grades = np.full(200_000, 2, dtype=np.int64)  # Substitute -> 0.50
        propensities = np.full(200_000, 0.5)

        clicked = model.sample_clicks(grades, propensities, rng)

        assert clicked.mean() == pytest.approx(0.25, abs=0.005)

    def test_non_monotone_relevance_rejected(self) -> None:
        """A better grade must never be less clickable."""
        with pytest.raises(ValueError, match="non-decreasing"):
            ClickModel(relevance_by_grade=(0.9, 0.5, 0.2, 0.05))

    def test_out_of_range_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="noise must be in"):
            ClickModel(noise=1.5)
        with pytest.raises(ValueError, match="must be in"):
            ClickModel(relevance_by_grade=(0.0, 0.5, 0.9, 1.5))

    def test_compressed_variant_is_valid_and_narrower(self) -> None:
        default_spread = DEFAULT_RELEVANCE[3] - DEFAULT_RELEVANCE[0]
        compressed_spread = COMPRESSED_RELEVANCE[3] - COMPRESSED_RELEVANCE[0]
        assert compressed_spread < default_spread
        ClickModel(relevance_by_grade=COMPRESSED_RELEVANCE)  # must not raise

    def test_misaligned_inputs_rejected(self) -> None:
        with pytest.raises(ValueError, match="must align"):
            ClickModel().sample_clicks(
                np.array([1, 2]), np.array([1.0]), np.random.default_rng(0)
            )


def _candidates(n_queries: int = 30, n_candidates: int = 25) -> list[CandidateSet]:
    """Candidate sets with a fixed grade pattern, for controlled simulation."""
    rng = np.random.default_rng(11)
    sets = []
    for q in range(n_queries):
        grades = rng.integers(0, 4, size=n_candidates).astype(np.int64)
        sets.append(
            CandidateSet(
                query_id=q,
                query_text=f"query {q}",
                product_rows=np.arange(n_candidates, dtype=np.int64) + q * n_candidates,
                grades=grades,
            )
        )
    return sets


class TestDisplayOrder:
    def test_truncates_to_top_k(self) -> None:
        candidate = _candidates(1, 30)[0]
        rows, grades = display_order(candidate, np.random.default_rng(0).random(30), top_k=20)
        assert rows.size == 20
        assert grades.size == 20

    def test_orders_by_policy_score(self) -> None:
        candidate = CandidateSet(1, "q", np.array([10, 11, 12]), np.array([0, 1, 2]))
        rows, grades = display_order(candidate, np.array([0.1, 0.9, 0.5]), top_k=3)
        assert list(rows) == [11, 12, 10]
        assert list(grades) == [1, 2, 0]

    def test_short_candidate_sets_are_not_padded(self) -> None:
        candidate = CandidateSet(1, "q", np.array([10, 11]), np.array([0, 1]))
        rows, _ = display_order(candidate, np.array([0.5, 0.9]), top_k=20)
        assert rows.size == 2


class TestSimulateClickLog:
    def test_log_shape_and_columns(self) -> None:
        sets = _candidates(10, 25)
        scores = [np.random.default_rng(i).random(len(c)) for i, c in enumerate(sets)]
        log = simulate_click_log(
            sets,
            scores,
            PositionBiasModel(1.0),
            ClickModel(),
            LogConfig(top_k=20, impressions_per_query=3),
        )

        assert set(log.columns) == {
            "query_id",
            "impression",
            "rank",
            "product_row",
            "grade",
            "propensity",
            "clicked",
        }
        assert len(log) == 10 * 20 * 3

    def test_ranks_start_at_one_per_impression(self) -> None:
        sets = _candidates(2, 25)
        scores = [np.random.default_rng(i).random(len(c)) for i, c in enumerate(sets)]
        log = simulate_click_log(
            sets, scores, PositionBiasModel(1.0), ClickModel(), LogConfig(impressions_per_query=2)
        )
        for _, group in log.groupby(["query_id", "impression"]):
            assert group["rank"].min() == 1
            assert list(group["rank"]) == sorted(group["rank"])

    def test_logged_propensity_matches_the_model(self) -> None:
        """IPS reads this column; a wrong value silently breaks the correction."""
        sets = _candidates(3, 25)
        scores = [np.random.default_rng(i).random(len(c)) for i, c in enumerate(sets)]
        bias = PositionBiasModel(eta=1.5)
        log = simulate_click_log(sets, scores, bias, ClickModel(), LogConfig(top_k=20))

        expected = bias.propensities(20)
        for rank, propensity in log.groupby("rank")["propensity"].first().items():
            assert propensity == pytest.approx(expected[int(rank) - 1])

    def test_simulation_is_reproducible(self) -> None:
        sets = _candidates(5, 25)
        scores = [np.random.default_rng(i).random(len(c)) for i, c in enumerate(sets)]
        args = (sets, scores, PositionBiasModel(1.0), ClickModel())

        first = simulate_click_log(*args, LogConfig(seed=42))
        second = simulate_click_log(*args, LogConfig(seed=42))
        assert first.equals(second)

    def test_different_seeds_give_different_clicks(self) -> None:
        sets = _candidates(20, 25)
        scores = [np.random.default_rng(i).random(len(c)) for i, c in enumerate(sets)]
        args = (sets, scores, PositionBiasModel(1.0), ClickModel())

        a = simulate_click_log(*args, LogConfig(seed=1))
        b = simulate_click_log(*args, LogConfig(seed=2))
        assert not a["clicked"].equals(b["clicked"])

    def test_unbiased_model_gives_flat_ctr_by_rank(self) -> None:
        """Control condition: with eta=0, CTR must not decline with rank.

        A declining curve here would mean bias is leaking in from somewhere
        other than the propensity model.
        """
        sets = _candidates(400, 25)
        # A policy uncorrelated with relevance, so rank carries no relevance signal.
        scores = [np.random.default_rng(1000 + i).random(len(c)) for i, c in enumerate(sets)]
        log = simulate_click_log(
            sets,
            scores,
            PositionBiasModel(eta=0.0),
            ClickModel(),
            LogConfig(top_k=20, impressions_per_query=10, seed=3),
        )
        ctr = observed_click_rate_by_rank(log)

        top_half = ctr.iloc[:10]["ctr"].mean()
        bottom_half = ctr.iloc[10:]["ctr"].mean()
        assert abs(top_half - bottom_half) < 0.02

    def test_biased_model_produces_declining_ctr(self) -> None:
        sets = _candidates(400, 25)
        scores = [np.random.default_rng(2000 + i).random(len(c)) for i, c in enumerate(sets)]
        log = simulate_click_log(
            sets,
            scores,
            PositionBiasModel(eta=1.0),
            ClickModel(),
            LogConfig(top_k=20, impressions_per_query=10, seed=4),
        )
        ctr = observed_click_rate_by_rank(log)

        assert ctr.iloc[0]["ctr"] > 3 * ctr.iloc[-1]["ctr"]

    def test_observed_ctr_tracks_the_propensity_curve(self) -> None:
        """The core calibration check.

        Under a relevance-independent policy, CTR at rank k should be
        propensity(k) times a constant (the mean relevance). So the ratio
        CTR(k)/propensity(k) must be roughly flat -- that is what confirms the
        simulator implements the model it claims to.
        """
        sets = _candidates(600, 25)
        scores = [np.random.default_rng(3000 + i).random(len(c)) for i, c in enumerate(sets)]
        log = simulate_click_log(
            sets,
            scores,
            PositionBiasModel(eta=1.0),
            ClickModel(noise=0.0),
            LogConfig(top_k=10, impressions_per_query=20, seed=5),
        )
        ctr = observed_click_rate_by_rank(log)
        implied_relevance = ctr["ctr"] / ctr["propensity"]

        # Flat to within 10% of its own mean across all ten ranks.
        assert implied_relevance.std() / implied_relevance.mean() < 0.10

    def test_empty_input_gives_an_empty_log(self) -> None:
        log = simulate_click_log([], [], PositionBiasModel(), ClickModel())
        assert len(log) == 0
        assert "clicked" in log.columns


class TestClickedPairs:
    def test_aggregates_impressions_and_clicks(self) -> None:
        sets = _candidates(4, 25)
        scores = [np.random.default_rng(i).random(len(c)) for i, c in enumerate(sets)]
        log = simulate_click_log(
            sets,
            scores,
            PositionBiasModel(1.0),
            ClickModel(),
            LogConfig(top_k=20, impressions_per_query=5),
        )
        pairs = clicked_pairs(log)

        assert (pairs["impressions"] == 5).all()
        assert (pairs["clicks"] <= pairs["impressions"]).all()

    def test_grades_are_dropped(self) -> None:
        """Training arms consume this; exposing grades would be the leak the
        whole experiment exists to avoid."""
        sets = _candidates(3, 25)
        scores = [np.random.default_rng(i).random(len(c)) for i, c in enumerate(sets)]
        log = simulate_click_log(sets, scores, PositionBiasModel(1.0), ClickModel())

        assert "grade" not in clicked_pairs(log).columns
