from typing import TypeVar

import numpy as np
from rank_bm25 import BM25Okapi

from src.search.tokenizer import char_ngrams

T = TypeVar("T")


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
    """

    def __init__(self, documents: list[str]) -> None:
        tokenized = [char_ngrams(doc) for doc in documents]
        self._bm25 = _BM25Uniform(tokenized)

    def get_top_n(self, query: str, payloads: list[T], n: int) -> list[tuple[T, float]]:
        """Return the top-*n* ``(payload, bm25_score)`` pairs, best first.

        *payloads* is parallel to the documents the index was built from: one
        entry per name.  The score is computed per query — unlike population,
        it isn't a property of the corpus alone — so it's handed back here
        instead of being cached on the payload.
        """
        scores = self._bm25.get_scores(char_ngrams(query))
        top_n = np.argsort(scores)[::-1][:n]
        return [(payloads[i], float(scores[i])) for i in top_n]


if __name__ == "__main__":
    # ponytail: smallest check that IDF override + tokenization + flatten wire up
    docs = ["Moscow", "Moscow Oblast", "Saint Petersburg"]
    ids = [[1, 10], [2], [3]]
    index = BM25Index(docs)
    top = index.get_top_n("Moskva", payloads=ids, n=2)
    top_payloads = [payload for payload, _ in top]
    assert [1, 10] in top_payloads, top  # "Moscow" group should surface, ids kept grouped
    assert len(top) == 2, top
    print("ok", top)
