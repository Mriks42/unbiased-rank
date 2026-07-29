"""Tests for tokenization and BM25.

The scoring tests check BM25's defining behaviours (saturation, length
normalisation, corpus-wide IDF) rather than exact score values, so they stay
meaningful if k1/b defaults are ever retuned.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from unbiased_rank.indexing.lexical import BM25Index, BM25Params
from unbiased_rank.indexing.text import product_text, tokenize


class TestTokenize:
    def test_lowercases_and_splits(self) -> None:
        assert tokenize("Revent 80 CFM Fan") == ["revent", "80", "cfm", "fan"]

    def test_drops_punctuation_and_short_tokens(self) -> None:
        assert tokenize("a bc, d-ef!") == ["bc", "ef"]

    def test_empty_and_none_like_input(self) -> None:
        assert tokenize("") == []
        assert tokenize("!!! ??") == []

    def test_product_text_includes_brand(self) -> None:
        assert product_text("Widget", "Acme") == "Widget Acme"
        assert product_text("Widget", None) == "Widget"


class TestBM25:
    @pytest.fixture
    def index(self) -> BM25Index:
        return BM25Index(
            [
                "red running shoes",          # 0
                "blue running shoes",         # 1
                "red hat",                    # 2
                "running running running",    # 3 repeated term
                "red shoes for running with extra padding and laces here",  # 4 long
            ]
        )

    def test_exact_match_outscores_partial(self, index: BM25Index) -> None:
        scores = index.score("red running shoes", np.array([0, 2], dtype=np.int64))
        assert scores[0] > scores[1]

    def test_unmatched_query_scores_zero(self, index: BM25Index) -> None:
        scores = index.score("zzz nonexistent", np.array([0, 1], dtype=np.int64))
        assert np.allclose(scores, 0.0)

    def test_empty_candidate_set_returns_empty(self, index: BM25Index) -> None:
        assert index.score("red", np.array([], dtype=np.int64)).size == 0

    def test_rare_term_outweighs_common_term(self, index: BM25Index) -> None:
        """IDF: 'hat' appears once, 'running' in four documents."""
        rare = index.score("hat", np.array([2], dtype=np.int64))[0]
        common = index.score("running", np.array([0], dtype=np.int64))[0]
        assert rare > common

    def test_term_frequency_saturates(self, index: BM25Index) -> None:
        """Three occurrences must not score three times one occurrence.

        Saturation is the property that separates BM25 from raw tf-idf.
        """
        one = index.score("running", np.array([0], dtype=np.int64))[0]
        three = index.score("running", np.array([3], dtype=np.int64))[0]
        assert one < three < 3 * one

    def test_length_normalisation_penalises_long_documents(self) -> None:
        """Same term count, longer document, lower score."""
        index = BM25Index(["red shoes", "red shoes plus many additional filler words here now"])
        scores = index.score("red", np.array([0, 1], dtype=np.int64))
        assert scores[0] > scores[1]

    def test_b_zero_disables_length_normalisation(self) -> None:
        index = BM25Index(
            ["red shoes", "red shoes plus many additional filler words here now"],
            params=BM25Params(b=0.0),
        )
        scores = index.score("red", np.array([0, 1], dtype=np.int64))
        assert scores[0] == pytest.approx(scores[1])

    def test_idf_uses_whole_corpus_not_candidate_subset(self) -> None:
        """A term's weight must not depend on which candidates are scored.

        Computing IDF over candidates would let a query with few candidates
        inflate its own scores -- a subtle leak that would show up as an
        apparent ranking improvement.
        """
        index = BM25Index(["alpha", "alpha", "alpha", "alpha", "beta"])
        alone = index.score("alpha", np.array([0], dtype=np.int64))[0]
        with_others = index.score("alpha", np.array([0, 1, 2, 3], dtype=np.int64))[0]
        assert alone == pytest.approx(with_others)

    def test_scores_are_order_aligned_with_candidates(self, index: BM25Index) -> None:
        forward = index.score("red", np.array([0, 2], dtype=np.int64))
        reverse = index.score("red", np.array([2, 0], dtype=np.int64))
        assert forward[0] == pytest.approx(reverse[1])
        assert forward[1] == pytest.approx(reverse[0])

    def test_score_batch_matches_individual_scoring(self, index: BM25Index) -> None:
        queries = ["red", "running"]
        candidates = [np.array([0, 2], dtype=np.int64), np.array([1, 3], dtype=np.int64)]
        batched = index.score_batch(queries, candidates)
        for got, (q, c) in zip(batched, zip(queries, candidates, strict=True), strict=True):
            assert np.allclose(got, index.score(q, c))

    def test_vocabulary_and_document_counts(self, index: BM25Index) -> None:
        assert index.n_documents == 5
        assert index.vocabulary_size > 0

    def test_empty_corpus_does_not_divide_by_zero(self) -> None:
        index = BM25Index([])
        assert index.n_documents == 0
        assert index.average_doc_length == 1.0

    def test_documents_with_no_tokens_are_handled(self) -> None:
        index = BM25Index(["", "!!!", "red shoes"])
        scores = index.score("red", np.array([0, 1, 2], dtype=np.int64))
        assert scores[0] == 0.0
        assert scores[1] == 0.0
        assert scores[2] > 0.0

    def test_scoring_cost_is_independent_of_corpus_size(self) -> None:
        """Scoring a fixed candidate set must not scale with the corpus.

        Regression guard. An earlier implementation densified one full corpus
        column per query term, allocating n_documents floats per lookup. It was
        correct but scaled with the corpus, which at 1.2M products turned a
        seconds-long evaluation into an hours-long one. Correctness tests alone
        would not have caught that.
        """
        common = ["red running shoes size ten"]
        small = BM25Index(common * 500)
        large = BM25Index(common * 20_000)
        candidates = np.arange(20, dtype=np.int64)

        def elapsed(index: BM25Index) -> float:
            start = time.perf_counter()
            for _ in range(200):
                index.score("red running shoes", candidates)
            return time.perf_counter() - start

        small_time = elapsed(small)
        large_time = elapsed(large)

        # A corpus 40x larger must not cost anywhere near 40x. The bound is
        # loose because wall-clock timing is noisy; the broken version exceeded
        # it by orders of magnitude.
        assert large_time < max(small_time * 5.0, 0.5)
