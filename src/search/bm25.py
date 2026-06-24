from rank_bm25 import BM25Okapi

from src.search.tokenizer import char_ngrams


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

    def get_top_n(self, query: str, ids: list[list[int]], n: int) -> list[int]:
        """Return geonameids of the top-*n* matching names, flattened.

        Each document is one name that may carry several geonameids (homonyms,
        asciiname == alternate name).  BM25 ranks names; the id-lists of the top
        *n* names are flattened into a single id list — the notebook's
        ``sum(res, [])``.  So *n* names yield *at least* *n* ids.
        """
        groups = self._bm25.get_top_n(char_ngrams(query), ids, n=n)
        return [gid for group in groups for gid in group]


if __name__ == "__main__":
    # ponytail: smallest check that IDF override + tokenization + flatten wire up
    docs = ["Moscow", "Moscow Oblast", "Saint Petersburg"]
    ids = [[1, 10], [2], [3]]
    index = BM25Index(docs)
    top = index.get_top_n("Moskva", ids=ids, n=2)
    assert 1 in top, top  # "Moscow" group should surface, flattened to ints
    assert all(isinstance(x, int) for x in top), top
    print("ok", top)
