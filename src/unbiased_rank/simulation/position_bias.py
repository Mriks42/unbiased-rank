"""Position-based examination model.

Under the Position-Based Model (PBM) a click factorises into examination and
relevance:

    P(C = 1 | q, d, k)  =  P(E = 1 | k) * P(R = 1 | q, d)
                        =  p_k          * r(q, d)

`p_k` is the *propensity* -- the probability a user examines the item shown at
rank `k` at all. It depends only on position, never on the item. That
independence is the whole reason inverse propensity weighting can work: it makes
the bias a known function of something we control (where we showed things)
rather than of something we are trying to learn (relevance).

Parameterisation follows the standard form in the unbiased learning-to-rank
literature:

    p_k = (1 / k) ** eta

`eta = 0` gives no bias at all (`p_k = 1` everywhere) and is the control
condition that validates the whole simulation harness. `eta = 1` is the usual
default. Larger values concentrate examination at the top.

PBM is an assumption, not a fact. It ignores cascade behaviour (users stopping
after a satisfying click) and trust bias (users clicking top results *because*
they are top). Those are real effects, and their absence is a stated threat to
validity rather than a claim that they do not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Sweep points used by the experiment. eta=0 is the control; eta=1 the default.
ETA_SWEEP: Final[tuple[float, ...]] = (0.0, 0.5, 1.0, 1.5, 2.0)
DEFAULT_ETA: Final[float] = 1.0


def propensity_curve(n_positions: int, eta: float) -> FloatArray:
    """Examination probability per rank, for ranks 1..n_positions.

    Args:
        n_positions: number of displayed slots.
        eta: bias severity. 0 means unbiased.

    Returns:
        Array of length `n_positions`; element `i` is the propensity at rank
        `i + 1`. Always starts at 1.0, since rank 1 is examined by definition
        under this parameterisation.
    """
    if n_positions < 0:
        raise ValueError(f"n_positions must be non-negative, got {n_positions}")
    if eta < 0.0:
        raise ValueError(f"eta must be non-negative, got {eta}")

    ranks = np.arange(1, n_positions + 1, dtype=np.float64)
    return np.power(1.0 / ranks, eta)


@dataclass(frozen=True)
class PositionBiasModel:
    """A propensity curve plus the clipping used when inverting it."""

    eta: float = DEFAULT_ETA

    def propensities(self, n_positions: int) -> FloatArray:
        return propensity_curve(n_positions, self.eta)

    @property
    def is_unbiased(self) -> bool:
        """True when this model introduces no position bias.

        Used by the harness-validation gate: with an unbiased model and no click
        noise, a ranker trained on simulated clicks must match one trained on
        true grades. If it does not, the simulator is wrong and nothing
        downstream can be trusted.
        """
        return self.eta == 0.0

    def inverse_propensity_weights(
        self, n_positions: int, clip: float | None = None
    ) -> FloatArray:
        """Weights `1 / max(p_k, clip)` for IPS correction.

        Clipping trades bias against variance: unclipped IPS is unbiased but
        can be dominated by a handful of low-propensity observations, while
        aggressive clipping is stable and reintroduces bias. The experiment
        sweeps `clip` and reports both the mean *and* the variance, because
        reporting only the mean hides the entire trade-off.
        """
        propensities = self.propensities(n_positions)
        if clip is not None:
            if not 0.0 < clip <= 1.0:
                raise ValueError(f"clip must be in (0, 1], got {clip}")
            propensities = np.maximum(propensities, clip)
        return 1.0 / propensities


__all__ = [
    "DEFAULT_ETA",
    "ETA_SWEEP",
    "PositionBiasModel",
    "propensity_curve",
]
