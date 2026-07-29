"""Relevance-to-click generation.

Converts ESCI graded judgments into click probabilities, then samples clicks
conditioned on examination.

The grade-to-relevance mapping is a *modelling assumption*, not a measurement.
ESCI tells us a product is "Exact" for a query; it does not tell us how often a
shopper would click it. The chosen values are plausible and monotone in grade,
and the experiment reports a robustness variant with a compressed mapping to
confirm conclusions are not an artifact of the spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Grade (from metrics.GRADE_MAP: E=3, S=2, C=1, I=0) -> P(click | examined).
DEFAULT_RELEVANCE: Final[tuple[float, float, float, float]] = (0.05, 0.20, 0.50, 0.90)

# Compressed variant for the robustness check: same ordering, smaller spread.
COMPRESSED_RELEVANCE: Final[tuple[float, float, float, float]] = (0.10, 0.30, 0.55, 0.80)

NOISE_SWEEP: Final[tuple[float, ...]] = (0.0, 0.1, 0.2)


@dataclass(frozen=True)
class ClickModel:
    """Maps grades to click probabilities and samples clicks.

    Attributes:
        relevance_by_grade: P(click | examined) indexed by integer grade 0..3.
        noise: with this probability an outcome is replaced by a coin flip,
            modelling clicks unexplained by relevance (misclicks, curiosity).
    """

    relevance_by_grade: tuple[float, float, float, float] = DEFAULT_RELEVANCE
    noise: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError(f"noise must be in [0, 1], got {self.noise}")
        if any(not 0.0 <= p <= 1.0 for p in self.relevance_by_grade):
            raise ValueError(
                f"relevance probabilities must be in [0, 1], got {self.relevance_by_grade}"
            )
        if list(self.relevance_by_grade) != sorted(self.relevance_by_grade):
            raise ValueError(
                f"relevance must be non-decreasing in grade, got {self.relevance_by_grade}. "
                "A non-monotone mapping would make a better-graded product less "
                "clickable, which no downstream conclusion could survive."
            )

    def relevance(self, grades: IntArray) -> FloatArray:
        """P(click | examined) per item."""
        table = np.asarray(self.relevance_by_grade, dtype=np.float64)
        return table[np.asarray(grades, dtype=np.int64)]

    def click_probability(self, grades: IntArray, propensities: FloatArray) -> FloatArray:
        """P(click) = p_k * r(grade), then blended with noise.

        Noise is applied to the *probability*, which is equivalent to flipping a
        coin with probability `noise` and is cheaper than sampling twice.
        """
        relevance = self.relevance(grades)
        clean = propensities * relevance
        if self.noise == 0.0:
            return clean
        return (1.0 - self.noise) * clean + self.noise * 0.5

    def sample_clicks(
        self,
        grades: IntArray,
        propensities: FloatArray,
        rng: np.random.Generator,
    ) -> BoolArray:
        """Draw clicks for one impression."""
        if grades.size != propensities.size:
            raise ValueError(
                f"grades ({grades.size}) and propensities ({propensities.size}) must align; "
                "propensities are per displayed rank, so grades must already be in rank order"
            )
        probability = self.click_probability(grades, propensities)
        return rng.random(probability.size) < probability


__all__ = [
    "COMPRESSED_RELEVANCE",
    "DEFAULT_RELEVANCE",
    "NOISE_SWEEP",
    "ClickModel",
]
