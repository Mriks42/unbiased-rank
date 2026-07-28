"""Reciprocal Rank Fusion.

RRF combines rankings by position rather than by score:

    RRF(d) = sum_i 1 / (k + rank_i(d))

Score-based fusion would require BM25 scores (unbounded, corpus-dependent) and
cosine similarities (bounded [-1, 1]) to be put on a common scale, which needs
per-corpus calibration that is itself a tunable knob. RRF sidesteps that
entirely by discarding magnitudes, which is why it is the usual default for
hybrid retrieval.

The constant k damps the influence of top ranks; 60 is the value from the
original paper and is used here as a documented default rather than a tuned one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DEFAULT_RRF_K: Final[float] = 60.0


def ranks_from_scores(scores: FloatArray) -> npt.NDArray[np.int64]:
    """Convert scores to 1-based ranks, highest score first.

    Ties resolve by original position so fusion is deterministic; otherwise two
    identical runs could fuse differently and the difference would look real.
    """
    order = np.lexsort((np.arange(scores.size), -scores))
    ranks = np.empty(scores.size, dtype=np.int64)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks


def reciprocal_rank_fusion(
    score_lists: Sequence[FloatArray], k: float = DEFAULT_RRF_K
) -> FloatArray:
    """Fuse several score vectors over the *same* candidate set.

    Args:
        score_lists: score vectors, all aligned to one candidate ordering.
        k: RRF damping constant.

    Returns:
        Fused scores, aligned to the same candidate ordering. Higher is better.
    """
    if not score_lists:
        raise ValueError("need at least one score list to fuse")

    sizes = {s.size for s in score_lists}
    if len(sizes) != 1:
        raise ValueError(f"all score lists must cover the same candidates, got sizes {sizes}")

    fused = np.zeros(score_lists[0].size, dtype=np.float64)
    for scores in score_lists:
        fused += 1.0 / (k + ranks_from_scores(np.asarray(scores, dtype=np.float64)))
    return fused


__all__ = ["DEFAULT_RRF_K", "ranks_from_scores", "reciprocal_rank_fusion"]
