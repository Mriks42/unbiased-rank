"""LambdaMART ranker over the shared feature set.

Wraps LightGBM's `lambdarank` objective. The wrapper exists to pin the parts of
the setup the experiment depends on:

* **Group boundaries.** LambdaMART optimises a listwise objective, so it must be
  told which rows belong to which query. Getting this wrong silently trains a
  pointwise model on shuffled groups and still produces plausible metrics.
* **Sample weights.** IPS correction enters here, as a per-row weight. Keeping
  it a weight rather than a modified label means the naive and corrected arms
  differ in exactly one place.
* **Determinism.** Fixed seed and single-threaded histogram construction, so
  reruns reproduce.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt

from unbiased_rank.ranking.features import N_FEATURES

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class RankerParams:
    """LambdaMART hyperparameters.

    Defaults are modest on purpose. The experiment compares *label sources*, so
    model capacity is held fixed; a heavily tuned ranker would lift every arm
    and add a confounder.
    """

    n_estimators: int = 200
    learning_rate: float = 0.1
    num_leaves: int = 31
    min_child_samples: int = 20
    lambdarank_truncation_level: int = 20
    seed: int = 0

    def to_lightgbm(self) -> dict[str, Any]:
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "lambdarank_truncation_level": self.lambdarank_truncation_level,
            "seed": self.seed,
            "deterministic": True,
            "num_threads": 0,
            "verbosity": -1,
        }


@dataclass
class TrainingData:
    """Flattened per-query feature blocks plus group boundaries."""

    features: FloatArray
    labels: FloatArray
    group_sizes: IntArray
    weights: FloatArray | None = field(default=None)

    def __post_init__(self) -> None:
        n_rows = self.features.shape[0]
        if self.labels.size != n_rows:
            raise ValueError(f"features have {n_rows} rows but labels have {self.labels.size}")
        if int(self.group_sizes.sum()) != n_rows:
            raise ValueError(
                f"group sizes sum to {int(self.group_sizes.sum())} "
                f"but features have {n_rows} rows. "
                "Mismatched groups would train a listwise objective over the wrong "
                "query boundaries and still report plausible metrics."
            )
        if self.weights is not None and self.weights.size != n_rows:
            raise ValueError(f"weights have {self.weights.size} entries, expected {n_rows}")
        if self.features.shape[1] != N_FEATURES:
            raise ValueError(
                f"expected {N_FEATURES} features, got {self.features.shape[1]}"
            )

    @property
    def n_queries(self) -> int:
        return int(self.group_sizes.size)


def stack_training_data(
    feature_blocks: list[FloatArray],
    label_blocks: list[FloatArray],
    weight_blocks: list[FloatArray] | None = None,
) -> TrainingData:
    """Concatenate per-query blocks into LightGBM's flat + group format."""
    if not feature_blocks:
        raise ValueError("no queries to train on")

    group_sizes = np.array([block.shape[0] for block in feature_blocks], dtype=np.int64)
    weights = np.concatenate(weight_blocks) if weight_blocks is not None else None
    return TrainingData(
        features=np.vstack(feature_blocks),
        labels=np.concatenate(label_blocks),
        group_sizes=group_sizes,
        weights=weights,
    )


class LambdaMartRanker:
    """Trained LambdaMART model."""

    def __init__(self, params: RankerParams | None = None) -> None:
        self.params = params if params is not None else RankerParams()
        self._booster: lgb.Booster | None = None

    @property
    def is_fitted(self) -> bool:
        return self._booster is not None

    def fit(self, data: TrainingData) -> LambdaMartRanker:
        """Train on the given labels and optional sample weights."""
        dataset = lgb.Dataset(
            data.features,
            label=data.labels,
            group=data.group_sizes,
            weight=data.weights,
            free_raw_data=False,
        )
        logger.info(
            "training LambdaMART: %d queries, %d rows, weighted=%s",
            data.n_queries,
            data.features.shape[0],
            data.weights is not None,
        )
        self._booster = lgb.train(
            self.params.to_lightgbm(),
            dataset,
            num_boost_round=self.params.n_estimators,
        )
        return self

    def score(self, features: FloatArray) -> FloatArray:
        """Score a feature block. Higher is better."""
        if self._booster is None:
            raise RuntimeError("ranker is not fitted; call fit() first")
        return np.asarray(self._booster.predict(features), dtype=np.float64)

    def score_blocks(self, feature_blocks: list[FloatArray]) -> list[FloatArray]:
        """Score per-query blocks, preserving the block structure."""
        return [self.score(block) for block in feature_blocks]

    def feature_importance(self) -> FloatArray:
        if self._booster is None:
            raise RuntimeError("ranker is not fitted; call fit() first")
        return np.asarray(
            self._booster.feature_importance(importance_type="gain"), dtype=np.float64
        )

    def describe(self) -> dict[str, Any]:
        return {"params": asdict(self.params), "fitted": self.is_fitted}


__all__ = [
    "LambdaMartRanker",
    "RankerParams",
    "TrainingData",
    "stack_training_data",
]
