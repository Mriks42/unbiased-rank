"""Dense retrieval with a sentence-transformer bi-encoder.

Encoding the ESCI catalogue is the single most expensive step in the project
(~1.2M products, CPU-only on this machine), so embeddings are cached to disk.
The cache carries a fingerprint of the model name, corpus size and a digest of
the input text: a stale cache silently reused after the corpus or model changed
would corrupt every downstream number while looking perfectly healthy.

Embeddings are L2-normalised at encode time, so inner product *is* cosine
similarity and FAISS `IndexFlatIP` needs no further scaling.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]

DEFAULT_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE: Final[int] = 256
_FINGERPRINT_SAMPLE: Final[int] = 1000


@dataclass(frozen=True)
class EmbeddingFingerprint:
    """Identity of a cached embedding matrix."""

    model_name: str
    n_texts: int
    dimension: int
    text_digest: str

    def matches(self, other: EmbeddingFingerprint) -> bool:
        return (
            self.model_name == other.model_name
            and self.n_texts == other.n_texts
            and self.text_digest == other.text_digest
        )


def fingerprint_texts(
    texts: Sequence[str], model_name: str, dimension: int = 0
) -> EmbeddingFingerprint:
    """Fingerprint a corpus without hashing all of it.

    Hashing 1.2M strings on every startup is wasteful, so the digest covers the
    corpus length plus an evenly spaced sample. That detects a changed or
    reordered corpus in practice while staying cheap.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(len(texts)).encode())
    if texts:
        step = max(1, len(texts) // _FINGERPRINT_SAMPLE)
        for i in range(0, len(texts), step):
            digest.update(texts[i].encode("utf-8", errors="replace"))
    return EmbeddingFingerprint(model_name, len(texts), dimension, digest.hexdigest())


class DenseEncoder:
    """Batch text encoder with a disk cache."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_seq_length: int | None = None  # None keeps the model default.
        self._model: object | None = None  # Loaded lazily; import is slow.

    def _load_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading encoder %s on %s", self.model_name, self.device)
            model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_seq_length is not None:
                model.max_seq_length = self.max_seq_length
            self._model = model
        return self._model

    def _cache_identity(self) -> str:
        """Model identity for the cache fingerprint.

        Includes max_seq_length, since a different truncation produces
        different embeddings from the same model and must invalidate the cache.
        """
        return f"{self.model_name}@msl={self.max_seq_length}"

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> FloatArray:
        """Encode texts to L2-normalised float32 embeddings."""
        model = self._load_model()
        embeddings = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def encode_cached(
        self, texts: Sequence[str], cache_path: Path, show_progress: bool = True
    ) -> FloatArray:
        """Encode with a disk cache guarded by a fingerprint."""
        meta_path = cache_path.with_suffix(".meta.json")
        expected = fingerprint_texts(texts, self._cache_identity())

        if cache_path.exists() and meta_path.exists():
            stored = EmbeddingFingerprint(**json.loads(meta_path.read_text(encoding="utf-8")))
            if stored.matches(expected):
                logger.info("reusing cached embeddings at %s", cache_path)
                cached: FloatArray = np.load(cache_path)
                return cached
            logger.warning(
                "embedding cache at %s is stale (model/corpus changed); re-encoding", cache_path
            )

        logger.info("encoding %d texts with %s", len(texts), self.model_name)
        embeddings = self.encode(texts, show_progress=show_progress)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        written = EmbeddingFingerprint(
            model_name=self._cache_identity(),
            n_texts=len(texts),
            dimension=int(embeddings.shape[1]),
            text_digest=expected.text_digest,
        )
        meta_path.write_text(json.dumps(asdict(written), indent=2), encoding="utf-8")
        return embeddings


class DenseIndex:
    """Exact inner-product index over normalised embeddings.

    `IndexFlatIP` is exhaustive rather than approximate. At 1.2M x 384 that is
    ~1.8 GB and a few hundred ms per query -- slower than an HNSW index, but
    exact. The experiment needs retrieval quality to be a property of the
    *embeddings*, not of an ANN structure's recall/latency tuning, so
    approximate search is deliberately avoided here.
    """

    def __init__(self, embeddings: FloatArray) -> None:
        import faiss

        if embeddings.ndim != 2:
            raise ValueError(f"expected a 2-D embedding matrix, got shape {embeddings.shape}")
        self.dimension = int(embeddings.shape[1])
        self.n_documents = int(embeddings.shape[0])
        self._index = faiss.IndexFlatIP(self.dimension)
        self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    def search(
        self, queries: FloatArray, k: int
    ) -> tuple[FloatArray, npt.NDArray[np.int64]]:
        """Return (scores, document indices) for the top k per query."""
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        scores, indices = self._index.search(
            np.ascontiguousarray(queries, dtype=np.float32), k
        )
        return scores.astype(np.float32), indices.astype(np.int64)


def cosine_scores(query_embedding: FloatArray, candidate_embeddings: FloatArray) -> FloatArray:
    """Cosine similarity of one query against its candidates.

    Valid because embeddings are normalised at encode time, so the dot product
    already is the cosine.
    """
    if candidate_embeddings.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(candidate_embeddings @ query_embedding, dtype=np.float32)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MODEL",
    "DenseEncoder",
    "DenseIndex",
    "EmbeddingFingerprint",
    "cosine_scores",
    "fingerprint_texts",
]
