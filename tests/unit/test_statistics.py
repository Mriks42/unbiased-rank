"""Tests for the statistics layer.

These check calibration, not just that the functions return numbers: a
bootstrap that produced intervals of the wrong width would still look
plausible in a report, so coverage is verified by simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from unbiased_rank.evaluation.statistics import (
    benjamini_hochberg,
    bootstrap_mean,
    minimum_detectable_effect,
    paired_bootstrap_difference,
    paired_difference_sd,
    required_sample_size,
)


def test_bootstrap_mean_centres_on_the_sample_mean() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.1, size=500)
    est = bootstrap_mean(values, n_resamples=2_000, seed=1)

    assert est.value == pytest.approx(values.mean())
    assert est.ci_low < est.value < est.ci_high
    assert est.n == 500


def test_bootstrap_interval_narrows_as_n_grows() -> None:
    rng = np.random.default_rng(0)
    small = bootstrap_mean(rng.normal(0, 1, size=100), n_resamples=2_000, seed=1)
    large = bootstrap_mean(rng.normal(0, 1, size=10_000), n_resamples=2_000, seed=1)

    assert large.half_width < small.half_width


def test_bootstrap_coverage_is_approximately_nominal() -> None:
    """A 95% interval should contain the true mean about 95% of the time.

    This is the test that would catch a genuinely wrong bootstrap; checking
    only that an interval exists would not.
    """
    rng = np.random.default_rng(20260727)
    true_mean = 0.3
    covered = 0
    trials = 200

    for i in range(trials):
        sample = rng.normal(true_mean, 0.15, size=300)
        est = bootstrap_mean(sample, n_resamples=800, seed=i)
        if est.ci_low <= true_mean <= est.ci_high:
            covered += 1

    assert 0.90 <= covered / trials <= 0.99


def test_paired_bootstrap_detects_a_real_shift() -> None:
    rng = np.random.default_rng(0)
    control = rng.normal(0.40, 0.10, size=2_000)
    treatment = control + rng.normal(0.02, 0.01, size=2_000)  # small but real

    est = paired_bootstrap_difference(treatment, control, n_resamples=2_000, seed=1)

    assert est.value == pytest.approx(0.02, abs=0.005)
    assert est.excludes_zero()


def test_paired_bootstrap_reports_no_effect_when_there_is_none() -> None:
    """Negative control: identical arms must not produce a significant result."""
    rng = np.random.default_rng(0)
    values = rng.normal(0.4, 0.1, size=1_000)

    est = paired_bootstrap_difference(values, values.copy(), n_resamples=2_000, seed=1)

    assert est.value == pytest.approx(0.0)
    assert not est.excludes_zero()


def test_pairing_is_tighter_than_treating_arms_independently() -> None:
    """The whole reason for a paired design: correlated arms give a smaller CI."""
    rng = np.random.default_rng(0)
    difficulty = rng.normal(0.4, 0.25, size=1_500)  # shared query difficulty
    control = difficulty + rng.normal(0, 0.02, size=1_500)
    treatment = difficulty + rng.normal(0.01, 0.02, size=1_500)

    paired = paired_bootstrap_difference(treatment, control, n_resamples=2_000, seed=1)
    # Unpaired equivalent: variance of each arm separately, added.
    unpaired_sd = np.sqrt(np.var(treatment, ddof=1) + np.var(control, ddof=1))
    paired_sd = paired_difference_sd(treatment, control)

    assert paired_sd < unpaired_sd
    assert paired.excludes_zero()


def test_paired_bootstrap_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        paired_bootstrap_difference(np.zeros(5), np.zeros(4))


def test_bootstrap_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_mean(np.array([]))


def test_required_sample_size_matches_hand_calculation() -> None:
    """n = 7.849 * sigma_d^2 / mde^2 at alpha=0.05, power=0.80."""
    n = required_sample_size(sigma_d=0.12, mde=0.01)
    assert n == pytest.approx(1130, rel=0.02)


def test_required_sample_size_scales_inversely_with_squared_effect() -> None:
    """Halving the detectable effect should roughly quadruple the sample."""
    big = required_sample_size(sigma_d=0.12, mde=0.02)
    small = required_sample_size(sigma_d=0.12, mde=0.01)
    assert small / big == pytest.approx(4.0, rel=0.05)


def test_mde_and_required_sample_size_are_inverses() -> None:
    sigma_d = 0.12
    n = required_sample_size(sigma_d=sigma_d, mde=0.005)
    assert minimum_detectable_effect(n=n, sigma_d=sigma_d) == pytest.approx(0.005, rel=0.01)


def test_power_analysis_rejects_nonsense_inputs() -> None:
    with pytest.raises(ValueError, match="sigma_d must be positive"):
        required_sample_size(sigma_d=0.0, mde=0.01)
    with pytest.raises(ValueError, match="mde must be positive"):
        required_sample_size(sigma_d=0.1, mde=0.0)


def test_benjamini_hochberg_rejects_only_small_p_values() -> None:
    p = np.array([0.001, 0.008, 0.4, 0.6, 0.9])
    rejected = benjamini_hochberg(p, q=0.05)
    assert list(rejected) == [True, True, False, False, False]


def test_benjamini_hochberg_is_stricter_than_uncorrected_testing() -> None:
    """With many null tests, BH must reject far fewer than a raw 0.05 cut."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=500)  # all null

    naive = int(np.count_nonzero(p < 0.05))
    corrected = int(np.count_nonzero(benjamini_hochberg(p, q=0.05)))

    assert naive > corrected
    assert corrected <= 2


def test_benjamini_hochberg_handles_empty_input() -> None:
    assert benjamini_hochberg(np.array([])).size == 0
