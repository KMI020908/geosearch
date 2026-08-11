"""The CSR index must score *identically* to rank_bm25, not merely closely.

``retriever_score`` decides which candidates survive retrieval's top-*k* cutoff
and is the final ranking score whenever no reranker is loaded
(:mod:`src.search.engine`), so a scoring change that drifts in the last bits
still silently shifts results — with no error anywhere. These tests pin the
arithmetic rather than a tolerance.
"""

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from src.search.bm25 import INDEX_FORMAT, BM25Index, StaleIndexError, _BM25Uniform
from src.search.tokenizer import char_ngrams

# Multilingual on purpose: the CJK names exercise the branch of `char_ngrams`
# where a whole word is shorter than n, and the transliteration pairs are the
# fuzzy-overlap case the uniform IDF exists to serve.
DOCUMENTS = [
    "Moscow",
    "Moskva",
    "Москва",
    "莫斯科",
    "Moscow Oblast",
    "Moscow Idaho",
    "Saint Petersburg",
    "Sankt-Peterburg",
    "İstanbul",
    "Istanbul",
    "Kazan",
    "Казань",
    "New York",
    "York",
]

QUERIES = [
    "Moscow",
    "Moskva",
    "Москва",
    "莫斯科",
    "Moscow Moscow",  # repeated tokens — must count twice, not once
    "istanbul",
    "New York York",
    "Ыгыатта",  # entirely out of vocabulary
    "",  # no tokens at all
]


def _reference_scores(query: str) -> np.ndarray:
    """What the original implementation produced, computed the original way."""
    return _BM25Uniform([char_ngrams(d) for d in DOCUMENTS]).get_scores(
        char_ngrams(query)
    )


@pytest.fixture(scope="module")
def index() -> BM25Index:
    return BM25Index(DOCUMENTS)


@pytest.mark.parametrize("query", QUERIES)
def test_scores_match_rank_bm25_exactly(index: BM25Index, query: str) -> None:
    """Bit-for-bit, not approximately — see the module docstring."""
    np.testing.assert_array_equal(
        index.get_scores(char_ngrams(query)), _reference_scores(query)
    )


@pytest.mark.parametrize("query", QUERIES)
def test_scores_survive_a_save_load_round_trip(
    index: BM25Index, query: str, tmp_path
) -> None:
    path = tmp_path / "index.npz"
    index.save(path)
    np.testing.assert_array_equal(
        BM25Index.load(path).get_scores(char_ngrams(query)),
        index.get_scores(char_ngrams(query)),
    )


def test_repeated_query_token_contributes_twice(index: BM25Index) -> None:
    """rank_bm25 does not de-duplicate the query, so neither may we.

    Asserted directly rather than left to the equality test above, because a
    de-duplicating implementation would still pass every *other* query here.

    ``allclose`` rather than ``array_equal`` only because the right-hand side
    is a different arithmetic: the index accumulates each token's contribution
    in sequence, while ``once * 2`` rounds the finished sum once. The exactness
    that matters — against ``rank_bm25`` on this same doubled query — is
    covered by :func:`test_scores_match_rank_bm25_exactly`.
    """
    once = index.get_scores(char_ngrams("Moscow"))
    twice = index.get_scores(char_ngrams("Moscow Moscow"))
    np.testing.assert_allclose(twice, once * 2.0, rtol=0, atol=1e-12)
    assert (twice[once > 0] > once[once > 0]).all()


def test_out_of_vocabulary_token_scores_zero(index: BM25Index) -> None:
    """Unknown n-grams contribute nothing — `(self.idf.get(q) or 0)` upstream."""
    assert not index.get_scores(char_ngrams("Ыгыатта")).any()


def test_top_n_payloads_and_scores_round_trip(index: BM25Index, tmp_path) -> None:
    payloads = [((i, i * 10),) for i in range(len(DOCUMENTS))]
    path = tmp_path / "index.npz"
    index.save(path)

    before = index.get_top_n("Moskva", payloads, n=3)
    after = BM25Index.load(path).get_top_n("Moskva", payloads, n=3)

    assert after == before
    assert before[0][0] == payloads[DOCUMENTS.index("Moskva")]


def test_saved_file_contains_no_pickle(index: BM25Index, tmp_path) -> None:
    """The whole point of format 6: loadable with allow_pickle=False."""
    path = tmp_path / "index.npz"
    index.save(path)
    with np.load(path, allow_pickle=False) as payload:
        assert int(payload["format"]) == INDEX_FORMAT
        assert payload["corpus_size"] == len(DOCUMENTS)


def test_stale_format_is_rejected(index: BM25Index, tmp_path) -> None:
    """A wrong layout must fail loudly, not be misread as the current one."""
    path = tmp_path / "index.npz"
    index.save(path)
    with np.load(path, allow_pickle=False) as payload:
        fields = dict(payload)
    fields["format"] = np.int64(INDEX_FORMAT - 1)
    np.savez_compressed(path, **fields)

    with pytest.raises(StaleIndexError, match="make artifacts"):
        BM25Index.load(path)


def test_vocabulary_order_is_independent_of_insertion(tmp_path) -> None:
    """The file is a function of corpus *contents*, so it is reproducible.

    `_load_corpus`'s SQL UNION has no ORDER BY, so the document order it hands
    over is not stable between runs; the vocabulary must not inherit that.
    """
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    BM25Index(DOCUMENTS).save(a)
    BM25Index(list(reversed(DOCUMENTS))).save(b)

    with np.load(a, allow_pickle=False) as fa, np.load(b, allow_pickle=False) as fb:
        np.testing.assert_array_equal(fa["vocab"], fb["vocab"])


def test_matches_stock_bm25okapi_shape() -> None:
    """Guard the assumption that only _calc_idf was overridden upstream."""
    tokenized = [char_ngrams(d) for d in DOCUMENTS]
    stock = BM25Okapi(tokenized)
    uniform = _BM25Uniform(tokenized)
    assert stock.corpus_size == uniform.corpus_size
    assert stock.avgdl == uniform.avgdl
    assert set(uniform.idf.values()) == {1.0}
