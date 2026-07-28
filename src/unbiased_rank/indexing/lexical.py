"""BM25 over a sparse term-document matrix.

Implemented directly rather than pulled from a library for two reasons:

* `rank_bm25` scores every document in the corpus per query in pure Python.
  At 1.2M products and tens of thousands of queries that is intractable.
  A CSC term-document matrix lets a query touch only the postings for its own
  terms, which is the standard inverted-index access pattern.
* IDF must be computed over the *whole* corpus even when only a candidate
  subset is scored. Computing it over candidates would make a term's weight
  depend on which documents were retrieved -- a subtle leak that inflates
  scores for queries with small candidate sets.

Scoring follows Robertson/Sparck Jones BM25:

    score(q, d) = sum_t IDF(t) * tf(t,d) * (k1 + 1)
                  / (tf(t,d) + k1 * (1 - b + b * len(d) / avgdl))

    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import sparse

from unbiased_rank.indexing.text import tokenize

FloatArray = npt.NDArray[np.float64]

DEFAULT_K1: Final[float] = 1.2
DEFAULT_B: Final[float] = 0.75


@dataclass(frozen=True)
class BM25Params:
    """BM25 free parameters.

    Defaults are the standard values from the literature. They are exposed so
    the choice is visible and tunable rather than buried as magic numbers.
    """

    k1: float = DEFAULT_K1
    b: float = DEFAULT_B


class BM25Index:
    """Sparse BM25 index over a document collection."""

    def __init__(self, documents: Sequence[str], params: BM25Params | None = None) -> None:
        self.params = params if params is not None else BM25Params()
        self._vocabulary: dict[str, int] = {}
        self._matrix = self._build_matrix(documents)
        self.n_documents = self._matrix.shape[0]

        doc_lengths = np.asarray(self._matrix.sum(axis=1)).ravel().astype(np.float64)
        self.doc_lengths = doc_lengths
        # Guard the empty-corpus and all-empty-documents cases so the length
        # normalisation term cannot divide by zero.
        self.average_doc_length = float(doc_lengths.mean()) if doc_lengths.size else 0.0
        if self.average_doc_length == 0.0:
            self.average_doc_length = 1.0

        self._idf = self._compute_idf()
        # CSC gives O(nnz in column) access to a single term's postings.
        self._csc = self._matrix.tocsc()

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocabulary)

    def _build_matrix(self, documents: Sequence[str]) -> sparse.csr_matrix:
        """Build the term-frequency matrix, assigning term ids on first sight."""
        indptr: list[int] = [0]
        indices: list[int] = []
        values: list[int] = []

        for document in documents:
            counts: dict[int, int] = {}
            for token in tokenize(document):
                term_id = self._vocabulary.setdefault(token, len(self._vocabulary))
                counts[term_id] = counts.get(term_id, 0) + 1
            indices.extend(counts.keys())
            values.extend(counts.values())
            indptr.append(len(indices))

        shape = (len(documents), max(len(self._vocabulary), 1))
        return sparse.csr_matrix(
            (
                np.array(values, dtype=np.float64),
                np.array(indices, dtype=np.int64),
                np.array(indptr, dtype=np.int64),
            ),
            shape=shape,
        )

    def _compute_idf(self) -> FloatArray:
        """IDF per term, computed over the full corpus."""
        document_frequency = np.asarray((self._matrix > 0).sum(axis=0)).ravel().astype(np.float64)
        n = float(self.n_documents)
        return np.log(1.0 + (n - document_frequency + 0.5) / (document_frequency + 0.5))

    def score(self, query: str, candidates: npt.NDArray[np.int64]) -> FloatArray:
        """BM25 score of `query` against the given document row indices.

        Scoring a candidate subset rather than the whole corpus is what makes
        the re-ranking setting cheap; IDF still comes from the full corpus.
        """
        scores = np.zeros(candidates.size, dtype=np.float64)
        if candidates.size == 0:
            return scores

        k1, b = self.params.k1, self.params.b
        norm = 1.0 - b + b * self.doc_lengths[candidates] / self.average_doc_length

        for token in tokenize(query):
            term_id = self._vocabulary.get(token)
            if term_id is None:
                continue  # Term absent from the corpus contributes nothing.
            tf = self._term_frequencies(term_id, candidates)
            scores += self._idf[term_id] * tf * (k1 + 1.0) / (tf + k1 * norm)
        return scores

    def _term_frequencies(
        self, term_id: int, candidates: npt.NDArray[np.int64]
    ) -> FloatArray:
        """Term frequency of one term across the candidate documents."""
        column = self._csc[:, term_id]
        # `.toarray()` on a single sparse column is dense in n_documents, which
        # is fine at 1.2M rows and avoids a Python-level postings intersection.
        return np.asarray(column.todense()).ravel()[candidates]

    def score_batch(
        self, queries: Iterable[str], candidate_sets: Iterable[npt.NDArray[np.int64]]
    ) -> list[FloatArray]:
        """Score many (query, candidates) pairs."""
        return [self.score(q, c) for q, c in zip(queries, candidate_sets, strict=True)]


__all__ = ["BM25Index", "BM25Params", "DEFAULT_B", "DEFAULT_K1"]
