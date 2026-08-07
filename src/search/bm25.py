"""Character-n-gram BM25 retrieval index, with a pickle-free on-disk format.

The scoring is BM25 term-frequency saturation with IDF pinned to 1 — see
:class:`BM25Index`.  This module owns two things beyond that: an
:meth:`~BM25Index.save`/:meth:`~BM25Index.load` pair that uses ``.npz`` rather
than ``pickle``, and a scorer that reproduces ``rank_bm25``'s arithmetic exactly
while being fast enough to serve.
"""

from pathlib import Path
from typing import TypeVar

import numpy as np
from rank_bm25 import BM25Okapi

from src.search.tokenizer import char_ngrams

T = TypeVar("T")

# Bumped when the on-disk array layout changes, so a stale file is detected
# rather than misread. Format 6 is the first pickle-free one: 1-5 were
# `pickle.dumps` of this object graph.
#
# `k1` and `b` are not pinned here on purpose: they are saved with each index
# and read back from it, so a file built under different values scores under
# those values rather than being silently reinterpreted under today's.
INDEX_FORMAT = 6


class _BM25Uniform(BM25Okapi):
    """BM25Okapi with IDF fixed to 1 — pure TF-based n-gram matching."""

    def _calc_idf(self, nd: dict) -> None:
        self.idf = {term: 1.0 for term in nd}


class BM25Index:
    """Character-n-gram BM25 retrieval index over place-name strings.

    Documents are tokenized into character n-grams via the shared
    :func:`char_ngrams` tokenizer, then indexed with BM25Okapi whose IDF is
    pinned to 1: every shared n-gram counts equally, with no rare-term
    weighting.  Scoring is BM25's term-frequency saturation over those
    n-grams, so partial and fuzzy name overlaps still rank.

    Internally the postings are held as a CSR matrix keyed by n-gram rather
    than as ``rank_bm25``'s list-of-dicts.  That is a storage and speed change
    only: :meth:`get_scores` reproduces ``BM25Okapi.get_scores`` exactly,
    including its per-token accumulation order (see there for why that
    matters).
    """

    def __init__(self, documents: list[str]) -> None:
        self._build(_BM25Uniform([char_ngrams(doc) for doc in documents]))

    def _build(self, bm25: BM25Okapi) -> None:
        """Transpose ``rank_bm25``'s per-document dicts into a per-term CSR."""
        self.corpus_size: int = bm25.corpus_size
        self.avgdl: float = bm25.avgdl
        self.k1: float = bm25.k1
        self.b: float = bm25.b
        self.doc_len = np.asarray(bm25.doc_len, dtype=np.int64)

        # Term -> (documents containing it, their frequencies). Sorting the
        # vocabulary makes the saved file a function of the corpus contents
        # alone, not of dict insertion order.
        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, freqs in enumerate(bm25.doc_freqs):
            for term, freq in freqs.items():
                postings.setdefault(term, []).append((doc_id, freq))

        vocab = sorted(postings)
        indptr = np.zeros(len(vocab) + 1, dtype=np.int64)
        indices: list[int] = []
        data: list[int] = []
        for i, term in enumerate(vocab):
            entries = postings[term]
            indptr[i + 1] = indptr[i] + len(entries)
            indices.extend(doc_id for doc_id, _ in entries)
            data.extend(freq for _, freq in entries)

        self._vocab: dict[str, int] = {term: i for i, term in enumerate(vocab)}
        self._indptr = indptr
        self._indices = np.asarray(indices, dtype=np.int64)
        self._data = np.asarray(data, dtype=np.float64)

        # The per-document half of the saturation denominator does not depend
        # on the query, so it is computed once here rather than per token.
        self._denom_base = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Score every document against *query_tokens*.

        Mirrors ``BM25Okapi.get_scores`` term for term.  Two details of that
        implementation are load-bearing and deliberately preserved:

        * the loop runs over the query **as given**, with no de-duplication, so
          a repeated n-gram contributes twice;
        * accumulation happens per token into one float64 array, so the
          floating-point summation order is unchanged.

        Both matter because ``retriever_score`` is a *trained* feature of the
        CatBoost reranker (:mod:`src.rerank.features`): a scoring change that
        is merely close, rather than identical, silently shifts the model off
        the distribution it was fitted on.

        What does change is how the per-token document frequencies are found.
        ``rank_bm25`` builds ``[doc.get(q, 0) for doc in self.doc_freqs]``,
        touching all ~1.2M document dicts for every token; here the term's CSR
        row is gathered directly, so the work is proportional to the number of
        documents that actually contain the n-gram.
        """
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        for token in query_tokens:
            term_id = self._vocab.get(token)
            if term_id is None:
                # IDF is pinned to 1 for known terms and 0 for unknown ones, so
                # an out-of-vocabulary n-gram contributes nothing at all —
                # exactly `(self.idf.get(q) or 0)` in the original.
                continue
            start, end = self._indptr[term_id], self._indptr[term_id + 1]
            docs = self._indices[start:end]
            freqs = self._data[start:end]
            scores[docs] += freqs * (self.k1 + 1.0) / (freqs + self._denom_base[docs])
        return scores

    def get_top_n(self, query: str, payloads: list[T], n: int) -> list[tuple[T, float]]:
        """Return the top-*n* ``(payload, bm25_score)`` pairs, best first.

        *payloads* is parallel to the documents the index was built from: one
        entry per name.  The score is computed per query — unlike population,
        it isn't a property of the corpus alone — so it's handed back here
        instead of being cached on the payload.
        """
        scores = self.get_scores(char_ngrams(query))
        top_n = np.argsort(scores)[::-1][:n]
        return [(payloads[i], float(scores[i])) for i in top_n]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the index to *path* as a ``.npz`` of plain arrays.

        Not pickle.  The previous format was ``pickle.dumps`` of this object
        graph, which is arbitrary code execution on load — acceptable for a
        file this process just wrote, not for one fetched from a public
        HuggingFace repo, where it also trips the Hub's pickle scanner.
        Everything here is a numpy array or a scalar, so :meth:`load` can pass
        ``allow_pickle=False`` and mean it.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The vocabulary rides as a sorted array whose position *is* the term
        # id, so the dict is rebuilt on load rather than serialised.
        vocab = np.empty(len(self._vocab), dtype=object)
        for term, term_id in self._vocab.items():
            vocab[term_id] = term
        np.savez_compressed(
            path,
            format=np.int64(INDEX_FORMAT),
            vocab=np.asarray(vocab.tolist(), dtype=np.str_),
            indptr=self._indptr,
            indices=self._indices,
            data=self._data,
            doc_len=self.doc_len,
            avgdl=np.float64(self.avgdl),
            corpus_size=np.int64(self.corpus_size),
            k1=np.float64(self.k1),
            b=np.float64(self.b),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """Read an index written by :meth:`save`.

        Raises :class:`StaleIndexError` when the file predates the current
        layout, so a mismatch surfaces at startup with an actionable message
        rather than as wrong scores at query time.
        """
        with np.load(path, allow_pickle=False) as payload:
            on_disk = int(payload["format"])
            if on_disk != INDEX_FORMAT:
                raise StaleIndexError(
                    f"{path} is index format {on_disk}, but this build expects "
                    f"{INDEX_FORMAT} — rebuild it with `make artifacts`."
                )
            index = cls.__new__(cls)
            index._vocab = {
                str(term): i for i, term in enumerate(payload["vocab"].tolist())
            }
            index._indptr = payload["indptr"]
            index._indices = payload["indices"]
            index._data = payload["data"]
            index.doc_len = payload["doc_len"]
            index.avgdl = float(payload["avgdl"])
            index.corpus_size = int(payload["corpus_size"])
            index.k1 = float(payload["k1"])
            index.b = float(payload["b"])
        index._denom_base = index.k1 * (
            1.0 - index.b + index.b * index.doc_len / index.avgdl
        )
        return index


class StaleIndexError(RuntimeError):
    """The index file on disk was written by an incompatible layout."""
