"""Statistical machinery for comparing ranking arms.

The experiment reports two *separate* sources of uncertainty, and conflating
them would misstate confidence:

1. **Query-sampling variance** — we evaluate on a finite sample of queries.
   Handled by the paired bootstrap here.
2. **Simulation variance** — clicks are generated stochastically, so a whole
   run is one draw. Handled by repeating runs across seeds (see the experiment
   layer), not by anything in this module.

Comparisons are *paired*: every arm is evaluated on the same queries, so the
quantity of interest is the per-query difference. Pairing removes query
difficulty from the variance and is what makes small effects detectable at a
realistic sample size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]

DEFAULT_RESAMPLES: Final[int] = 10_000
DEFAULT_ALPHA: Final[float] = 0.05
DEFAULT_POWER: Final[float] = 0.80


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a confidence interval."""

    value: float
    ci_low: float
    ci_high: float
    n: int

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0

    def excludes_zero(self) -> bool:
        """True when the interval does not straddle zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def __str__(self) -> str:
        return f"{self.value:.4f} [{self.ci_low:.4f}, {self.ci_high:.4f}] (n={self.n})"


def bootstrap_mean(
    values: FloatArray,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for the mean of per-query values."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Estimate(float(values.mean()), float(low), float(high), int(values.size))


def paired_bootstrap_difference(
    treatment: FloatArray,
    control: FloatArray,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for the mean paired difference (treatment - control).

    Queries are resampled as units, so the two arms are always resampled
    together. Bootstrapping the arms independently would destroy the pairing
    and inflate the interval.
    """
    treatment = np.asarray(treatment, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if treatment.shape != control.shape:
        raise ValueError(
            f"paired comparison needs equal shapes, got {treatment.shape} and {control.shape}"
        )
    return bootstrap_mean(treatment - control, n_resamples=n_resamples, alpha=alpha, seed=seed)


def paired_difference_sd(treatment: FloatArray, control: FloatArray) -> float:
    """Sample SD of per-query differences.

    This is `sigma_d` in the power analysis. It must be *measured* on a pilot
    rather than assumed, because it depends on the metric, the cutoff and the
    query mix, and a wrong assumption silently mis-sizes the whole experiment.
    """
    diffs = np.asarray(treatment, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    if diffs.size < 2:
        raise ValueError("need at least two paired observations to estimate a standard deviation")
    return float(np.std(diffs, ddof=1))


def required_sample_size(
    sigma_d: float,
    mde: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> int:
    """Queries needed to detect a paired difference of `mde`.

        n >= (z_{1-alpha/2} + z_{power})^2 * sigma_d^2 / mde^2

    Uses the normal approximation, which is appropriate at the sample sizes
    this experiment operates at (thousands of queries).
    """
    if sigma_d <= 0.0:
        raise ValueError("sigma_d must be positive")
    if mde <= 0.0:
        raise ValueError("mde must be positive")

    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    n = (z_alpha + z_power) ** 2 * sigma_d**2 / mde**2
    return int(np.ceil(n))


def minimum_detectable_effect(
    n: int,
    sigma_d: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> float:
    """Smallest paired difference detectable with `n` queries.

    The inverse of `required_sample_size`; reported in FINDINGS.md so readers
    can see what the experiment was and was not powered to detect.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if sigma_d <= 0.0:
        raise ValueError("sigma_d must be positive")

    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    return float((z_alpha + z_power) * sigma_d / np.sqrt(n))


def benjamini_hochberg(p_values: FloatArray, q: float = 0.05) -> npt.NDArray[np.bool_]:
    """Benjamini-Hochberg FDR control.

    The sweep runs hundreds of comparisons; without correction a handful would
    clear p<0.05 by chance alone. Returns a boolean mask of rejected nulls.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(p)
    ranks = np.arange(1, p.size + 1)
    thresholds = q * ranks / p.size
    passing = p[order] <= thresholds

    rejected = np.zeros(p.size, dtype=bool)
    if passing.any():
        # Reject everything up to the largest index that passes.
        cutoff = int(np.flatnonzero(passing)[-1])
        rejected[order[: cutoff + 1]] = True
    return rejected


__all__ = [
    "Estimate",
    "benjamini_hochberg",
    "bootstrap_mean",
    "minimum_detectable_effect",
    "paired_bootstrap_difference",
    "paired_difference_sd",
    "required_sample_size",
]
