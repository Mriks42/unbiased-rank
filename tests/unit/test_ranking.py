"""Tests for features, LambdaMART wiring and arm label construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unbiased_rank.ranking.candidates import CandidateSet
from unbiased_rank.ranking.features import (
    FEATURE_NAMES,
    N_FEATURES,
    ProductText,
    extract_features,
    feature_importance_frame,
)
from unbiased_rank.ranking.labels import binarise, click_labels, grade_labels
from unbiased_rank.ranking.lambdamart import (
    LambdaMartRanker,
    RankerParams,
    TrainingData,
    stack_training_data,
)


@pytest.fixture
def catalogue() -> ProductText:
    return ProductText.from_catalogue(
        pd.DataFrame(
            {
                "product_title": [
                    "red running shoes",
                    "blue hat",
                    "red running shoes for men extra long title with padding",
                    "",
                ],
                "product_brand": ["Acme", "Nike", "Acme", ""],
            }
        )
    )


class TestFeatures:
    def test_shape_and_names_agree(self, catalogue: ProductText) -> None:
        features = extract_features(
            "red shoes",
            np.array([0, 1], dtype=np.int64),
            np.array([1.0, 0.5]),
            np.array([0.8, 0.2]),
            catalogue,
        )
        assert features.shape == (2, N_FEATURES)
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_retrieval_scores_pass_through(self, catalogue: ProductText) -> None:
        features = extract_features(
            "red", np.array([0], dtype=np.int64), np.array([3.5]), np.array([0.42]), catalogue
        )
        assert features[0, 0] == pytest.approx(3.5)
        assert features[0, 1] == pytest.approx(0.42)

    def test_token_coverage_is_query_normalised(self, catalogue: ProductText) -> None:
        """Coverage is asymmetric on purpose: a long title containing the whole
        query is a good match, and Jaccard alone would penalise its length."""
        features = extract_features(
            "red running shoes",
            np.array([2], dtype=np.int64),  # long title, contains all query tokens
            np.array([1.0]),
            np.array([1.0]),
            catalogue,
        )
        coverage = features[0, FEATURE_NAMES.index("token_coverage")]
        jaccard = features[0, FEATURE_NAMES.index("token_jaccard")]
        assert coverage == pytest.approx(1.0)
        assert jaccard < 1.0

    def test_exact_prefix_flag(self, catalogue: ProductText) -> None:
        idx = FEATURE_NAMES.index("exact_title_prefix")
        hit = extract_features(
            "red running",
            np.array([0], dtype=np.int64),
            np.array([1.0]),
            np.array([1.0]),
            catalogue,
        )
        miss = extract_features(
            "blue", np.array([0], dtype=np.int64), np.array([1.0]), np.array([1.0]), catalogue
        )
        assert hit[0, idx] == 1.0
        assert miss[0, idx] == 0.0

    def test_brand_match_flag(self, catalogue: ProductText) -> None:
        idx = FEATURE_NAMES.index("brand_match")
        hit = extract_features(
            "acme shoes", np.array([0], dtype=np.int64), np.array([1.0]), np.array([1.0]), catalogue
        )
        miss = extract_features(
            "red shoes", np.array([0], dtype=np.int64), np.array([1.0]), np.array([1.0]), catalogue
        )
        assert hit[0, idx] == 1.0
        assert miss[0, idx] == 0.0

    def test_empty_title_does_not_crash(self, catalogue: ProductText) -> None:
        features = extract_features(
            "anything", np.array([3], dtype=np.int64), np.array([0.0]), np.array([0.0]), catalogue
        )
        assert np.all(np.isfinite(features))

    def test_empty_query_does_not_divide_by_zero(self, catalogue: ProductText) -> None:
        features = extract_features(
            "", np.array([0], dtype=np.int64), np.array([0.0]), np.array([0.0]), catalogue
        )
        assert np.all(np.isfinite(features))

    def test_misaligned_scores_rejected(self, catalogue: ProductText) -> None:
        with pytest.raises(ValueError, match="must align"):
            extract_features(
                "q",
                np.array([0, 1], dtype=np.int64),
                np.array([1.0]),
                np.array([1.0, 2.0]),
                catalogue,
            )

    def test_importance_frame_is_labelled_and_sorted(self) -> None:
        frame = feature_importance_frame(np.arange(N_FEATURES, dtype=np.float64))
        assert list(frame["feature"])[0] == FEATURE_NAMES[-1]
        assert frame["importance"].is_monotonic_decreasing

    def test_importance_frame_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            feature_importance_frame(np.zeros(3))


class TestTrainingData:
    def test_group_sizes_must_sum_to_rows(self) -> None:
        """A group mismatch would train a listwise objective over the wrong
        query boundaries and still report plausible metrics."""
        with pytest.raises(ValueError, match="group sizes sum to"):
            TrainingData(
                features=np.zeros((10, N_FEATURES)),
                labels=np.zeros(10),
                group_sizes=np.array([3, 3], dtype=np.int64),
            )

    def test_label_length_must_match(self) -> None:
        with pytest.raises(ValueError, match="labels have"):
            TrainingData(
                features=np.zeros((5, N_FEATURES)),
                labels=np.zeros(4),
                group_sizes=np.array([5], dtype=np.int64),
            )

    def test_wrong_feature_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            TrainingData(
                features=np.zeros((5, 3)),
                labels=np.zeros(5),
                group_sizes=np.array([5], dtype=np.int64),
            )

    def test_stack_builds_group_boundaries(self) -> None:
        blocks = [np.zeros((4, N_FEATURES)), np.zeros((7, N_FEATURES))]
        labels = [np.zeros(4), np.zeros(7)]
        data = stack_training_data(blocks, labels)

        assert list(data.group_sizes) == [4, 7]
        assert data.features.shape[0] == 11
        assert data.n_queries == 2

    def test_stack_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="no queries"):
            stack_training_data([], [])


def _synthetic_training(n_queries: int = 60, n_candidates: int = 20, seed: int = 0):  # type: ignore[no-untyped-def]
    """Feature blocks where feature 0 genuinely predicts the label."""
    rng = np.random.default_rng(seed)
    feature_blocks, label_blocks = [], []
    for _ in range(n_queries):
        labels = rng.integers(0, 4, size=n_candidates).astype(np.float64)
        features = rng.normal(size=(n_candidates, N_FEATURES))
        features[:, 0] = labels + rng.normal(scale=0.3, size=n_candidates)  # informative
        feature_blocks.append(features)
        label_blocks.append(labels)
    return feature_blocks, label_blocks


class TestLambdaMartRanker:
    def test_unfitted_ranker_refuses_to_score(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            LambdaMartRanker().score(np.zeros((2, N_FEATURES)))

    def test_learns_an_informative_feature(self) -> None:
        features, labels = _synthetic_training()
        ranker = LambdaMartRanker(RankerParams(n_estimators=40)).fit(
            stack_training_data(features, labels)
        )

        scores = ranker.score(features[0])
        # Rank correlation with the true labels should be strongly positive.
        correlation = np.corrcoef(scores, labels[0])[0, 1]
        assert correlation > 0.5

    def test_training_is_reproducible(self) -> None:
        features, labels = _synthetic_training()
        data = stack_training_data(features, labels)

        params = RankerParams(n_estimators=30, seed=5)
        first = LambdaMartRanker(params).fit(data).score(features[0])
        second = LambdaMartRanker(params).fit(data).score(features[0])
        assert np.allclose(first, second)

    def test_sample_weights_change_the_model(self) -> None:
        """IPS enters as a weight, so weights must actually influence training."""
        features, labels = _synthetic_training()
        rng = np.random.default_rng(1)
        weights = [rng.uniform(0.1, 10.0, size=block.shape[0]) for block in features]

        unweighted = LambdaMartRanker(RankerParams(n_estimators=30)).fit(
            stack_training_data(features, labels)
        )
        weighted = LambdaMartRanker(RankerParams(n_estimators=30)).fit(
            stack_training_data(features, labels, weights)
        )

        assert not np.allclose(unweighted.score(features[0]), weighted.score(features[0]))

    def test_importance_has_one_entry_per_feature(self) -> None:
        features, labels = _synthetic_training(n_queries=20)
        ranker = LambdaMartRanker(RankerParams(n_estimators=20)).fit(
            stack_training_data(features, labels)
        )
        assert ranker.feature_importance().size == N_FEATURES

    def test_score_blocks_preserves_structure(self) -> None:
        features, labels = _synthetic_training(n_queries=10)
        ranker = LambdaMartRanker(RankerParams(n_estimators=10)).fit(
            stack_training_data(features, labels)
        )
        scored = ranker.score_blocks(features)

        assert len(scored) == len(features)
        assert all(s.size == f.shape[0] for s, f in zip(scored, features, strict=True))


def _sets() -> list[CandidateSet]:
    return [
        CandidateSet(1, "q1", np.array([10, 11, 12]), np.array([3, 1, 0])),
        CandidateSet(2, "q2", np.array([20, 21]), np.array([2, 0])),
    ]


def _log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2],
            "product_row": [10, 11, 12, 20, 21],
            "clicked": [True, False, False, True, False],
            "propensity": [1.0, 0.5, 0.25, 1.0, 0.5],
        }
    )


class TestLabels:
    def test_grade_labels_use_true_relevance(self) -> None:
        blocks = grade_labels(_sets())
        assert list(blocks.labels[0]) == [3.0, 1.0, 0.0]
        assert blocks.weights is None

    def test_click_labels_count_clicks(self) -> None:
        blocks = click_labels(_sets(), _log())
        assert list(blocks.labels[0]) == [1.0, 0.0, 0.0]
        assert blocks.weights is None

    def test_click_labels_differ_from_grades(self) -> None:
        """The premise of the experiment: clicks are not grades."""
        grades = grade_labels(_sets()).labels[0]
        clicks = click_labels(_sets(), _log()).labels[0]
        assert not np.array_equal(grades, clicks)

    def test_ips_weights_apply_to_clicked_rows_only(self) -> None:
        """The bias is in which items were *observed as positive*.

        A non-click is absence of evidence, not evidence of irrelevance, so
        up-weighting it amplifies noise rather than correcting bias. Weighting
        every row was a real bug here: with oracle propensities it drove the
        corrected arm below both the naive arm and the no-learning floor.
        """
        # In _log(), only rows at propensity 1.0 were clicked.
        blocks = click_labels(_sets(), _log(), propensity_weights=True)
        assert blocks.weights is not None
        # Row 0 clicked at propensity 1.0 -> weight 1. Rows 1-2 not clicked -> 1.
        assert list(blocks.weights[0]) == pytest.approx([1.0, 1.0, 1.0])

    def test_deep_click_gets_upweighted(self) -> None:
        sets = [CandidateSet(1, "q", np.array([10, 11]), np.array([3, 3]))]
        log = pd.DataFrame(
            {
                "query_id": [1, 1],
                "product_row": [10, 11],
                "clicked": [False, True],  # click at the low-propensity position
                "propensity": [1.0, 0.25],
            }
        )
        blocks = click_labels(sets, log, propensity_weights=True)

        assert blocks.weights is not None
        assert list(blocks.weights[0]) == pytest.approx([1.0, 4.0])

    def test_clipping_caps_weights(self) -> None:
        sets = [CandidateSet(1, "q", np.array([10]), np.array([3]))]
        log = pd.DataFrame(
            {"query_id": [1], "product_row": [10], "clicked": [True], "propensity": [0.1]}
        )
        blocks = click_labels(sets, log, propensity_weights=True, clip=0.5)

        assert blocks.weights is not None
        assert blocks.weights[0][0] == pytest.approx(2.0)  # 1 / 0.5, not 1 / 0.1

    def test_undisplayed_candidates_get_no_evidence(self) -> None:
        """An item never shown produced no evidence; label 0, weight 1."""
        sets = [CandidateSet(1, "q1", np.array([10, 99]), np.array([3, 3]))]
        blocks = click_labels(sets, _log(), propensity_weights=True)

        assert blocks.labels[0][1] == 0.0
        assert blocks.weights is not None
        assert blocks.weights[0][1] == 1.0

    def test_binarise_collapses_counts(self) -> None:
        raw = click_labels(_sets(), _log())
        counted = type(raw)(labels=[np.array([5.0, 0.0, 2.0])], weights=None)
        assert list(binarise(counted).labels[0]) == [1.0, 0.0, 1.0]

    def test_binarise_preserves_weights(self) -> None:
        blocks = click_labels(_sets(), _log(), propensity_weights=True)
        assert binarise(blocks).weights is blocks.weights
