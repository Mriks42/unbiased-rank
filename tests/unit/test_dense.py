"""Tests for dense encoding, caching and indexing.

The encoder itself is stubbed in most tests: loading a real transformer would
make the suite slow and network-dependent, and what needs verifying here is the
*cache invalidation* logic, not that sentence-transformers works. One test
marked `requires_model` exercises the real model and skips when unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from unbiased_rank.indexing.dense import (
    DenseEncoder,
    EmbeddingFingerprint,
    cosine_scores,
    fingerprint_texts,
)


class StubEncoder(DenseEncoder):
    """Encoder producing deterministic vectors without loading a model."""

    def __init__(self, dimension: int = 8, model_name: str = "stub-model") -> None:
        super().__init__(model_name=model_name)
        self.dimension = dimension
        self.encode_calls = 0

    def encode(self, texts, show_progress: bool = False):  # type: ignore[no-untyped-def]
        self.encode_calls += 1
        rng = np.random.default_rng(abs(hash(tuple(texts))) % (2**32))
        raw = rng.normal(size=(len(texts), self.dimension)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return np.ascontiguousarray(raw / np.clip(norms, 1e-12, None), dtype=np.float32)


class TestFingerprint:
    def test_same_corpus_gives_same_fingerprint(self) -> None:
        texts = [f"product {i}" for i in range(100)]
        assert fingerprint_texts(texts, "m").matches(fingerprint_texts(texts, "m"))

    def test_changed_corpus_changes_fingerprint(self) -> None:
        a = fingerprint_texts(["a", "b", "c"], "m")
        b = fingerprint_texts(["a", "b", "d"], "m")
        assert not a.matches(b)

    def test_different_length_changes_fingerprint(self) -> None:
        a = fingerprint_texts(["a", "b"], "m")
        b = fingerprint_texts(["a", "b", "c"], "m")
        assert not a.matches(b)

    def test_different_model_changes_fingerprint(self) -> None:
        texts = ["a", "b"]
        assert not fingerprint_texts(texts, "model-1").matches(fingerprint_texts(texts, "model-2"))

    def test_empty_corpus_is_handled(self) -> None:
        assert fingerprint_texts([], "m").matches(fingerprint_texts([], "m"))


class TestEncodeCached:
    def test_first_call_encodes_and_writes_cache(self, tmp_path: Path) -> None:
        encoder = StubEncoder()
        texts = ["alpha", "beta", "gamma"]
        cache = tmp_path / "emb.npy"

        out = encoder.encode_cached(texts, cache, show_progress=False)

        assert encoder.encode_calls == 1
        assert out.shape == (3, 8)
        assert cache.exists()
        assert cache.with_suffix(".meta.json").exists()

    def test_second_call_reuses_cache(self, tmp_path: Path) -> None:
        encoder = StubEncoder()
        texts = ["alpha", "beta"]
        cache = tmp_path / "emb.npy"

        first = encoder.encode_cached(texts, cache, show_progress=False)
        second = encoder.encode_cached(texts, cache, show_progress=False)

        assert encoder.encode_calls == 1  # not re-encoded
        assert np.array_equal(first, second)

    def test_changed_corpus_invalidates_cache(self, tmp_path: Path) -> None:
        """The failure this guards against is silent: a stale cache reused after
        the corpus changed would corrupt every downstream number while the run
        looked completely healthy."""
        encoder = StubEncoder()
        cache = tmp_path / "emb.npy"

        encoder.encode_cached(["alpha", "beta"], cache, show_progress=False)
        encoder.encode_cached(["alpha", "gamma"], cache, show_progress=False)

        assert encoder.encode_calls == 2

    def test_changed_model_invalidates_cache(self, tmp_path: Path) -> None:
        texts = ["alpha", "beta"]
        cache = tmp_path / "emb.npy"

        first = StubEncoder(model_name="model-a")
        first.encode_cached(texts, cache, show_progress=False)

        second = StubEncoder(model_name="model-b")
        second.encode_cached(texts, cache, show_progress=False)

        assert second.encode_calls == 1  # re-encoded despite an existing cache

    def test_cache_metadata_records_identity(self, tmp_path: Path) -> None:
        encoder = StubEncoder(dimension=8, model_name="stub-model")
        cache = tmp_path / "emb.npy"
        encoder.encode_cached(["a", "b", "c"], cache, show_progress=False)

        meta = json.loads(cache.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["model_name"] == "stub-model"
        assert meta["n_texts"] == 3
        assert meta["dimension"] == 8

    def test_fingerprint_roundtrips_through_json(self, tmp_path: Path) -> None:
        encoder = StubEncoder()
        cache = tmp_path / "emb.npy"
        encoder.encode_cached(["a"], cache, show_progress=False)

        meta = json.loads(cache.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert isinstance(EmbeddingFingerprint(**meta), EmbeddingFingerprint)


class TestCosineScores:
    def test_identical_vectors_score_one(self) -> None:
        vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_scores(vector, vector.reshape(1, -1))[0] == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        candidates = np.array([[0.0, 1.0]], dtype=np.float32)
        assert cosine_scores(query, candidates)[0] == pytest.approx(0.0)

    def test_scores_align_with_candidate_order(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        candidates = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        scores = cosine_scores(query, candidates)
        assert scores[1] > scores[0]

    def test_empty_candidates_returns_empty(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        assert cosine_scores(query, np.zeros((0, 2), dtype=np.float32)).size == 0


class TestDenseIndex:
    def test_exact_search_finds_the_identical_vector(self) -> None:
        from unbiased_rank.indexing.dense import DenseIndex

        embeddings = np.eye(4, dtype=np.float32)
        index = DenseIndex(embeddings)
        scores, indices = index.search(embeddings[2], k=1)

        assert indices[0][0] == 2
        assert scores[0][0] == pytest.approx(1.0)

    def test_index_reports_shape(self) -> None:
        from unbiased_rank.indexing.dense import DenseIndex

        index = DenseIndex(np.eye(5, dtype=np.float32))
        assert index.n_documents == 5
        assert index.dimension == 5

    def test_one_dimensional_input_is_rejected(self) -> None:
        from unbiased_rank.indexing.dense import DenseIndex

        with pytest.raises(ValueError, match="2-D embedding matrix"):
            DenseIndex(np.ones(4, dtype=np.float32))


@pytest.mark.requires_model
def test_real_encoder_produces_normalised_embeddings() -> None:
    """Exercises the actual model; skipped when it cannot be loaded."""
    pytest.importorskip("sentence_transformers")
    encoder = DenseEncoder()
    try:
        embeddings = encoder.encode(["red running shoes", "blue hat"])
    except Exception as exc:  # noqa: BLE001 - model download may be unavailable
        pytest.skip(f"encoder unavailable: {exc}")

    assert embeddings.shape[0] == 2
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
